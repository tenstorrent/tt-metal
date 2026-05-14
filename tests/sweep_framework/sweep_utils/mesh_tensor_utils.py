# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for handling mesh devices and tensor placements in sweep tests.
Supports creating tensors on mesh devices with proper placement (Shard/Replicate).

Environment Variables:
    MESH_DEVICE_SHAPE: Mesh shape to use (e.g., "1x2", "2x4", "4x8")
                       If not set, uses single device (default)
                       If set, creates mesh device with that shape
                       Tests will fail naturally if mesh shape exceeds hardware
"""

import os
import torch
import ttnn
from typing import Optional, Dict, Tuple
import ast


def parse_placement_from_traced(tensor_placement: Optional[Dict]) -> Optional[ttnn.TensorMemoryLayout]:
    """
    Parse tensor placement from traced config and return appropriate mesh mapper.

    Args:
        tensor_placement: Dict with 'placement', 'distribution_shape', 'mesh_device_shape'
                         e.g., {'placement': "['PlacementShard(2)', 'PlacementShard(3)']", ...}

    Returns:
        Mesh mapper object (ShardTensor2dMesh or ReplicateTensorToMesh) or None
    """
    if not tensor_placement:
        return None

    try:
        placement_raw = tensor_placement.get("placement", "")
        placement_str = str(placement_raw) if not isinstance(placement_raw, str) else placement_raw

        # Check if it's a replicate placement
        if "PlacementReplicate" in placement_str:
            return ttnn.ReplicateTensorToMesh

        # Check if it's a shard placement
        if "PlacementShard" in placement_str:
            # Extract shard dimensions
            # e.g., "['PlacementShard(2)', 'PlacementShard(3)']" -> shard on dims 2,3
            import re

            shard_dims = re.findall(r"PlacementShard\((?:dim=)?(-?\d+)\)", placement_str)

            if shard_dims:
                # For 2D mesh, we typically shard on the last dimension(s)
                # Return a shard mapper - the specific implementation depends on the operation
                mesh_shape_str = tensor_placement.get("mesh_device_shape", "[1, 1]")
                mesh_shape = ast.literal_eval(mesh_shape_str) if isinstance(mesh_shape_str, str) else mesh_shape_str

                # For now, return ShardTensor2dMesh which will shard based on mesh shape
                return ttnn.ShardTensor2dMesh(
                    mesh_device=None,  # Will be set later
                    dim=int(shard_dims[-1]) if shard_dims else -1,
                    mesh_shape=ttnn.MeshShape(*mesh_shape) if len(mesh_shape) == 2 else None,
                )
    except Exception as e:
        print(f"⚠️ Warning: Failed to parse tensor placement: {e}")
        return None

    return None


def get_mesh_shape_from_machine_info(machine_info: Optional[Dict]) -> Optional[Tuple[int, int]]:
    """
    Extract mesh device shape from traced machine_info.

    Args:
        machine_info: Dict with 'mesh_device_shape', 'device_count', etc.

    Returns:
        Tuple of (rows, cols) or None if no mesh info
    """
    if not machine_info:
        return None

    mesh_shape = machine_info.get("mesh_device_shape")
    if not mesh_shape:
        return None

    # Handle both list and string formats
    if isinstance(mesh_shape, str):
        mesh_shape = ast.literal_eval(mesh_shape)

    if isinstance(mesh_shape, list) and len(mesh_shape) == 2:
        return tuple(mesh_shape)

    return None


def create_mesh_device(
    mesh_shape: Tuple[int, int],
    device_ids: Optional[list] = None,
    l1_small_size: int = 79104,
) -> ttnn.MeshDevice:
    """
    Create a mesh device with the specified shape.

    Args:
        mesh_shape: Tuple of (rows, cols) for mesh shape
        device_ids: Optional list of device IDs (deprecated, not used by API)
        l1_small_size: L1 small buffer size (default 79104 to prevent OOM in model-traced sweeps)

    Returns:
        ttnn.MeshDevice instance

    Dispatch axis selection (in priority order):
      1. `TTNN_DISPATCH_AXIS=col|row` env var — explicit override. Used by
         the two-pass workflow when a single op has master configs that
         straddle both axes (e.g. linear has both y=9 and x=7 masters).
      2. Auto-detect from master JSON (legacy behaviour) — works when an
         op's masters all need the same axis.
      3. Default to COL.
    """
    # 1. Env-var override — but ETH dispatch overrides when 8x8 grid is needed.
    _env_axis = os.environ.get("TTNN_DISPATCH_AXIS", "").strip().lower()

    # Auto-discover master JSON if env var not set
    if not os.environ.get("TTNN_MASTER_JSON_PATH"):
        _auto_master = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "model_tracer",
            "traced_operations",
            "ttnn_operations_master.json",
        )
        if os.path.isfile(_auto_master):
            os.environ["TTNN_MASTER_JSON_PATH"] = os.path.abspath(_auto_master)

    # 2. Auto-detect from master configs (legacy path).
    # ROW dispatch gives compute_with_storage_grid_size = (8, 9): valid y in [0, 8].
    # COL dispatch gives (7, 10): valid y in [0, 9], valid x in [0, 6].
    # Master traces from deepseek_v3 use either layout depending on which dispatch
    # was active when traced. Default to ROW; switch to COL only if any of the
    # op's master shard_specs requires y=9 cores. Both cases: cores with x=7 fall
    # outside COL but inside ROW — those configs need ROW.
    needs_col = False
    needs_row_only = False
    try:
        # Try to derive the op name from the runner's --module-name arg, e.g.
        # "model_traced.linear_model_traced" -> "ttnn.linear"
        op_name = os.environ.get("TTNN_SWEEP_OP_NAME", "")
        if not op_name:
            import sys as _sys_d

            for _i, _a in enumerate(_sys_d.argv):
                if _a == "--module-name" and _i + 1 < len(_sys_d.argv):
                    _m = _sys_d.argv[_i + 1]
                    if _m.startswith("model_traced."):
                        _stem = _m.split(".", 1)[1].replace("_model_traced", "")
                        # Check experimental + transformer prefixes by probing master json
                        op_name = _stem  # bare; we'll match flexibly below
                    break
        master_json = os.environ.get("TTNN_MASTER_JSON_PATH", "")
        if op_name and master_json and os.path.isfile(master_json):
            import json as _json_d

            with open(master_json) as _f:
                _m = _json_d.load(_f)
            # Try multiple forms: "ttnn.X", "ttnn.experimental.X", "ttnn.transformer.X"
            _candidates = [
                op_name,
                f"ttnn.{op_name}",
                f"ttnn.experimental.{op_name}",
                f"ttnn.transformer.{op_name}",
            ]
            _matching_op = None
            _ops_dict = _m.get("operations", {})
            for _c in _candidates:
                if _c in _ops_dict:
                    _matching_op = _c
                    break
            if _matching_op is None:
                _matching_op = op_name
            for _cfg in _ops_dict.get(_matching_op, {}).get("configurations", []):
                for _arg in _cfg.get("arguments", {}).values():
                    if not isinstance(_arg, dict):
                        continue
                    _val = str(_arg.get("value", ""))
                    if "compute_with_storage_grid_size=8-8" in _val:
                        needs_row_only = True
                    _ss = (_arg.get("memory_config") or {}).get("shard_spec")
                    if not isinstance(_ss, dict):
                        _ss = _arg.get("shard_spec")
                    if not isinstance(_ss, dict):
                        continue
                    for _g in _ss.get("grid", []):
                        for _key in ("start", "end"):
                            _p = _g.get(_key, {})
                            if _p.get("y") == 9:
                                needs_col = True
                            if _p.get("x") == 7:
                                needs_row_only = True
    except Exception:
        needs_col = False
        needs_row_only = False

    # Default: COL (gives compute grid 7x10) since most lead_models traces use
    # cores in the 7-wide pattern with y up to 9. Switch to ROW only if any of
    # the op's master shard_specs uses x=7 (which COL excludes).
    # When x=7 or 8-8 grid is needed, use ETH dispatch so all 8x8
    # compute cores are available. ETH takes priority over the env-var
    # because the op cannot run at all without the full grid.
    if needs_row_only:
        try:
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*mesh_shape),
                l1_small_size=l1_small_size,
                dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
            )
        except Exception:
            pass

    # 3. Env-var override (when ETH not needed).
    if _env_axis in ("col", "row"):
        _override_axis = ttnn.DispatchCoreAxis.COL if _env_axis == "col" else ttnn.DispatchCoreAxis.ROW
        try:
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*mesh_shape),
                l1_small_size=l1_small_size,
                dispatch_core_config=ttnn.DispatchCoreConfig(axis=_override_axis),
            )
        except Exception:
            pass

    # 4. Default to COL.
    use_axis = ttnn.DispatchCoreAxis.COL

    try:
        return ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*mesh_shape),
            l1_small_size=l1_small_size,
            dispatch_core_config=ttnn.DispatchCoreConfig(axis=use_axis),
        )
    except Exception:
        pass

    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        l1_small_size=l1_small_size,
        dispatch_core_config=ttnn.DispatchCoreConfig(),
    )


def _parse_shard_dim(placement_str: str) -> int:
    """Extract shard dimension from placement string, handling negative dims."""
    import re

    shard_dims = re.findall(r"PlacementShard\((?:dim=)?(-?\d+)\)", placement_str)
    return int(shard_dims[-1]) if shard_dims else -1


def _is_shard_placement(tensor_placement: Optional[Dict], num_devices: int) -> bool:
    """Check if placement is a shard placement with multiple devices."""
    if not tensor_placement:
        return False
    placement_str = tensor_placement.get("placement", "")
    # If it has both Replicate and Shard, check which comes first or treat as replicate
    if "PlacementReplicate" in placement_str and "PlacementShard" not in placement_str:
        return False
    return "PlacementShard" in placement_str and num_devices > 1


def get_mesh_composer(mesh_device, tensor_placement: Optional[Dict] = None):
    """
    Create a mesh composer matching the tensor placement for converting back to torch.

    For sharded tensors, returns a ConcatMesh2dToTensor that reassembles shards.
    For replicated tensors, returns None (caller should use device 0 extraction).

    Args:
        mesh_device: The mesh device
        tensor_placement: Placement info from traced config

    Returns:
        Mesh composer or None
    """
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if not _is_shard_placement(tensor_placement, num_devices):
        return None

    placement_str = tensor_placement.get("placement", "")
    shard_dim = _parse_shard_dim(placement_str)

    try:
        mesh_shape = (1, num_devices)
        return ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape, dims=shard_dim)
    except (TypeError, RuntimeError):
        return None


def _restore_topology(
    tensor: ttnn.Tensor,
    placement_entries: list,
    dist_parsed: list,
    mesh_shape_tuple: tuple,
) -> None:
    """Restore correct TensorTopology on a device tensor to match the master trace.

    The C++ factory methods may create a topology that doesn't match what the
    master trace recorded (e.g., flattening 2D to 1D, or setting 2D when the
    master had 1D).  This helper reconstructs the exact topology from the vector
    config's distribution_shape and placement info.
    """
    import re

    ndim = len(dist_parsed)

    if ndim >= 2:
        # 2D (or higher) distribution — e.g. [4, 8]
        dist_shape = ttnn.MeshShape(*dist_parsed[:2])

        placements = []
        for entry in placement_entries or []:
            shard_match = re.search(r"PlacementShard\((-?\d+)\)", entry)
            if shard_match:
                placements.append(ttnn.PlacementShard(int(shard_match.group(1))))
            else:
                placements.append(ttnn.PlacementReplicate())
        while len(placements) < 2:
            placements.append(ttnn.PlacementReplicate())

        rows, cols = dist_parsed[0], dist_parsed[1]
        mesh_coords = [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]
    elif ndim == 1:
        # 1D distribution — e.g. [32].  Keep as 1D to match the master trace.
        dist_shape = ttnn.MeshShape(shape=dist_parsed)

        placements = []
        for entry in placement_entries or []:
            shard_match = re.search(r"PlacementShard\((-?\d+)\)", entry)
            if shard_match:
                placements.append(ttnn.PlacementShard(int(shard_match.group(1))))
            else:
                placements.append(ttnn.PlacementReplicate())
        if not placements:
            placements.append(ttnn.PlacementReplicate())

        total = dist_parsed[0]
        mesh_coords = [ttnn.MeshCoordinate(coords=[i]) for i in range(total)]
    else:
        return  # Nothing to restore

    topology = ttnn.TensorTopology(dist_shape, placements, mesh_coords)
    tensor.update_tensor_topology(topology)


def apply_tensor_placement_topology(tensor, tensor_placement, mesh_shape_tuple):
    """Apply topology from a tensor_placement config dict to a device tensor.

    Use this for tensors created outside of ``create_tensor_on_mesh`` (e.g. in
    decode-mode paths that use ``from_torch`` + ``interleaved_to_sharded``).
    """
    import ast as _ast
    import re

    if not tensor_placement:
        return
    dist_raw = tensor_placement.get("distribution_shape", "")
    if isinstance(dist_raw, str):
        try:
            dist_parsed = _ast.literal_eval(dist_raw)
        except (ValueError, SyntaxError):
            return
    else:
        dist_parsed = list(dist_raw) if dist_raw else []
    if not dist_parsed:
        return
    placement_str = str(tensor_placement.get("placement", ""))
    entries = re.findall(r"Placement(?:Shard\((?:dim=)?-?\d+\)|Replicate)", placement_str)
    try:
        _restore_topology(tensor, entries, dist_parsed, mesh_shape_tuple)
    except Exception:
        # Best-effort: topology restore is a soft annotation on the tensor.
        # Failure here doesn't affect numeric correctness — the tensor is
        # still valid for downstream use.
        pass


def replicate_with_topology(
    torch_tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config: ttnn.MemoryConfig,
    tensor_placement: Optional[Dict] = None,
) -> ttnn.Tensor:
    """Create a replicated tensor on mesh and restore the master trace's topology.

    Use this when the master model creates a tensor per-device (replicated) but the
    traced topology has shard-like placement.  This keeps the per-device shape as the
    logical shape while restoring the correct topology metadata so the operation
    tracer captures placement info matching the master trace.

    Args:
        torch_tensor: Input torch tensor (per-device shape)
        mesh_device: Mesh device to create tensor on
        dtype: TTNN data type
        layout: TTNN layout (TILE/ROW_MAJOR)
        memory_config: Memory configuration
        tensor_placement: Optional placement info from traced config

    Returns:
        TTNN tensor on mesh device with replicated data and restored topology
    """
    import ast as _ast
    import re

    # Create tensor with target layout. When memory_config is sharded, going
    # straight to from_torch tile-pads logical_shape to match the shard height
    # (e.g. 8 -> 32), mismatching master where the production tensor preserved
    # logical shape. Create in DRAM first, then to_memory_config preserves it.
    def _is_sharded(mc):
        if mc is None:
            return False
        try:
            return getattr(mc, "is_sharded", lambda: False)()
        except Exception:
            return False

    if _is_sharded(memory_config):
        tensor = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        try:
            tensor = ttnn.to_memory_config(tensor, memory_config)
        except Exception:
            # Best-effort upcast to the requested memory_config (e.g. L1
            # sharded). On failure we keep the DRAM-resident tensor — sweeps
            # tolerate the placement difference and the kernel still runs.
            pass
    else:
        tensor = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    if tensor_placement:
        placement_raw = tensor_placement.get("placement", "")
        placement_str = str(placement_raw) if not isinstance(placement_raw, str) else placement_raw
        entries = re.findall(r"Placement(?:Shard\(-?\d+\)|Replicate)", placement_str)

        dist_raw = tensor_placement.get("distribution_shape", "")
        if isinstance(dist_raw, str):
            try:
                dist_parsed = _ast.literal_eval(dist_raw)
            except Exception:
                dist_parsed = []
        else:
            dist_parsed = list(dist_raw) if dist_raw else []

        mesh_shape_raw = tensor_placement.get("mesh_device_shape", "[1, 1]")
        if isinstance(mesh_shape_raw, str):
            mesh_shape_raw = _ast.literal_eval(mesh_shape_raw)
        mesh_shape_tuple = tuple(mesh_shape_raw) if isinstance(mesh_shape_raw, (list, tuple)) else (1, 1)
        if len(mesh_shape_tuple) == 0:
            mesh_shape_tuple = (1, 1)
        elif len(mesh_shape_tuple) == 1:
            mesh_shape_tuple = (mesh_shape_tuple[0], 1)
        if dist_parsed:
            try:
                _restore_topology(tensor, entries, dist_parsed, mesh_shape_tuple)
            except Exception:
                pass  # Best-effort; don't block sweep execution

    return tensor


def vector_required_axis(op_kwargs, named_mcs=None):
    """Return 'col' / 'row' / None for the dispatch axis a single sweep
    vector logically needs.

    Detection scans every memory_config (kwarg or function-named param) and
    looks at its shard_spec grid:
      - any L1 shard_spec core at y=9 -> needs COL (only COL exposes y=9)
      - any L1 shard_spec core at x=7 -> needs ROW (only ROW exposes x=7)
      - program_config compute_with_storage_grid_size with x>=8 -> ROW
      - program_config compute_with_storage_grid_size with y>=10 -> COL
    Returns None if neither axis is required (vector fits both).
    """
    import re as _re_a

    _y9_pat = _re_a.compile(r"""['"]y['"]\s*:\s*9(?!\d)""")
    _x7_pat = _re_a.compile(r"""['"]x['"]\s*:\s*7(?!\d)""")
    _grid_x_pat = _re_a.compile(r"x\s*=\s*(\d+)")
    _grid_y_pat = _re_a.compile(r"y\s*=\s*(\d+)")

    needs_y9 = False
    needs_x7 = False

    def _walk_mc(_obj):
        nonlocal needs_y9, needs_x7
        if _obj is None:
            return
        if isinstance(_obj, dict):
            _bt = str(_obj.get("buffer_type", ""))
            if "DRAM" in _bt:
                return
            _ss = _obj.get("shard_spec")
            if not _ss or _ss == "None":
                return
            _r = repr(_ss)
            if _y9_pat.search(_r):
                needs_y9 = True
            if _x7_pat.search(_r):
                needs_x7 = True
            return
        _r = repr(_obj)
        if "BufferType::DRAM" in _r:
            return
        if "shard_spec" not in _r:
            return
        if _y9_pat.search(_r):
            needs_y9 = True
        if _x7_pat.search(_r):
            needs_x7 = True

    _all_mcs = dict(op_kwargs) if op_kwargs else {}
    for _name, _val in named_mcs or []:
        if _val is not None:
            _all_mcs[_name] = _val
    for _key, _v in _all_mcs.items():
        if "memory_config" not in _key:
            continue
        _walk_mc(_v)

    # program_config grid (SDPA-style): x>=8 -> ROW, y>=10 -> COL.
    _pc = (op_kwargs or {}).get("program_config")
    if _pc is not None:
        _pc_text = ""
        if isinstance(_pc, dict):
            _pc_text = str(_pc.get("value", "")) or str(_pc.get("repr", ""))
        else:
            _pc_text = repr(_pc)
        if "compute_with_storage_grid_size" in _pc_text:
            _idx = _pc_text.find("compute_with_storage_grid_size")
            _section = _pc_text[_idx : _idx + 80]
            _xm = _grid_x_pat.search(_section)
            _ym = _grid_y_pat.search(_section)
            if _xm and int(_xm.group(1)) >= 8:
                needs_x7 = True
            if _ym and int(_ym.group(1)) >= 10:
                needs_y9 = True

    if needs_y9:
        return "col"
    if needs_x7:
        return "row"
    return None


def current_device_axis(device):
    """Return 'col' / 'row' / None inferred from the device's
    compute_with_storage_grid_size."""
    try:
        g = device.compute_with_storage_grid_size() if hasattr(device, "compute_with_storage_grid_size") else None
    except Exception:
        return None
    if g is None:
        return None
    # COL -> (7, 10); ROW -> (8, 9). Other meshes (N150 etc.) return None.
    if g.x == 7 and g.y == 10:
        return "col"
    if g.x == 8 and g.y == 9:
        return "row"
    return None


def vector_axis_matches(device, op_kwargs, named_mcs=None):
    """True if the vector either has no required axis or matches the device."""
    required = vector_required_axis(op_kwargs, named_mcs)
    if required is None:
        return True
    actual = current_device_axis(device)
    if actual is None:
        return True  # unknown / non-Galaxy mesh — let the test run.
    return required == actual


def create_tensor_on_mesh(
    torch_tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config,
    tensor_placement: Optional[Dict] = None,
) -> ttnn.Tensor:
    """
    Create a TTNN tensor on a mesh device with optional placement.

    Args:
        torch_tensor: Input torch tensor
        mesh_device: Mesh device to create tensor on
        dtype: TTNN data type
        layout: TTNN layout (TILE/ROW_MAJOR)
        memory_config: Memory configuration (MemoryConfig object or dict)
        tensor_placement: Optional placement info from traced config

    Returns:
        TTNN tensor on mesh device with proper placement
    """
    if isinstance(memory_config, dict):
        from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

        memory_config = dict_to_memory_config(memory_config) or ttnn.DRAM_MEMORY_CONFIG
    # Trace-validation path: when placement has Shard, the master records the
    # per-chip .shape (which equals the torch input shape). Going through
    # ShardTensor2dMesh would re-shard the input and produce a smaller per-chip
    # shape, so delegate to replicate_with_topology, which keeps .shape =
    # input shape and stamps the correct sharded topology metadata.
    if tensor_placement:
        _placement_str = str(tensor_placement.get("placement", ""))
        if "PlacementShard" in _placement_str:
            try:
                actual_mesh = mesh_device.shape
                _ar, _ac = actual_mesh[0], actual_mesh[1]
            except Exception:
                _ar, _ac = 1, 1
            import ast as _ast0

            _ms_raw = tensor_placement.get("mesh_device_shape", "[1, 1]")
            if isinstance(_ms_raw, str):
                try:
                    _ms_raw = _ast0.literal_eval(_ms_raw)
                except Exception:
                    _ms_raw = [1, 1]
            _tr = _ms_raw[0] if len(_ms_raw) > 0 else 1
            _tc = _ms_raw[1] if len(_ms_raw) > 1 else 1
            if _ar >= _tr and _ac >= _tc:
                return replicate_with_topology(
                    torch_tensor, mesh_device, dtype, layout, memory_config, tensor_placement
                )

    # Determine mesh mapper based on placement
    if tensor_placement:
        import re
        import ast as _ast

        placement_raw = tensor_placement.get("placement", "")
        placement_str = str(placement_raw) if not isinstance(placement_raw, str) else placement_raw

        mesh_shape_raw = tensor_placement.get("mesh_device_shape", "[1, 1]")
        if isinstance(mesh_shape_raw, str):
            mesh_shape_raw = _ast.literal_eval(mesh_shape_raw)
        mesh_shape_tuple = tuple(mesh_shape_raw) if isinstance(mesh_shape_raw, (list, tuple)) else (1, 1)
        # Default empty/short mesh_shape to (1, 1)
        if len(mesh_shape_tuple) == 0:
            mesh_shape_tuple = (1, 1)
        elif len(mesh_shape_tuple) == 1:
            mesh_shape_tuple = (mesh_shape_tuple[0], 1)

        # Check if the actual device mesh can support the traced mesh shape.
        # If not (e.g., traced on Galaxy 4x8 but running on N150 1x1), fall back to replicate.
        try:
            actual_mesh = mesh_device.shape
            actual_rows, actual_cols = actual_mesh[0], actual_mesh[1]
        except Exception:
            actual_rows, actual_cols = 1, 1
        traced_rows = mesh_shape_tuple[0]
        traced_cols = mesh_shape_tuple[1] if len(mesh_shape_tuple) > 1 else 1
        mesh_compatible = actual_rows >= traced_rows and actual_cols >= traced_cols

        entries = re.findall(r"Placement(?:Shard\((?:dim=)?-?\d+\)|Replicate)", placement_str)

        dist_raw = tensor_placement.get("distribution_shape", "")
        if isinstance(dist_raw, str):
            try:
                dist_parsed = _ast.literal_eval(dist_raw)
            except Exception:
                dist_parsed = []
        else:
            dist_parsed = list(dist_raw) if dist_raw else []
        is_2d_distribution = len(dist_parsed) >= 2

        if not mesh_compatible or not entries or "PlacementShard" not in placement_str:
            if is_2d_distribution and mesh_compatible:
                mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape_tuple)
            else:
                mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        elif len(entries) >= 2:
            dims = []
            for entry in entries[:2]:
                shard_match = re.search(r"PlacementShard\((?:dim=)?(-?\d+)\)", entry)
                if shard_match:
                    dims.append(int(shard_match.group(1)))
                else:
                    dims.append(None)
            dims_tuple = tuple(dims)

            # Traced shapes are global (pre-shard); ShardTensor2dMesh splits
            # them across the mesh internally.
            mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims_tuple, mesh_shape=mesh_shape_tuple)
        elif len(entries) == 1:
            shard_match = re.search(r"PlacementShard\((?:dim=)?(-?\d+)\)", entries[0])
            if shard_match:
                dim = int(shard_match.group(1))
                dims_tuple = (None, dim)

                # Traced shapes are global; ShardTensor2dMesh splits the
                # tensor across the mesh internally.
                mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims_tuple, mesh_shape=mesh_shape_tuple)
            else:
                if is_2d_distribution:
                    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape_tuple)
                else:
                    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        else:
            if is_2d_distribution:
                mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape_tuple)
            else:
                mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # Create tensor on mesh. When memory_config is sharded, route via DRAM
    # then to_memory_config to preserve logical shape (avoid tile-pad to shard
    # height).
    def _ctom_is_sharded(mc):
        if mc is None:
            return False
        try:
            return getattr(mc, "is_sharded", lambda: False)()
        except Exception:
            return False

    if _ctom_is_sharded(memory_config):
        result = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        try:
            result = ttnn.to_memory_config(result, memory_config)
        except Exception:
            # Best-effort upcast to the requested memory_config. On failure we
            # keep the DRAM-resident tensor — sweeps tolerate the placement
            # difference and the kernel still runs.
            pass
    else:
        result = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )

    # Restore correct tensor topology from vector placement info.
    # The C++ factory methods may create a topology that doesn't match the
    # master trace (e.g., flattening [4,8] to [32] or vice versa).
    # We reconstruct and re-apply the exact topology so that the operation
    # tracer captures it accurately (matching the master trace).
    if tensor_placement and dist_parsed:
        try:
            _restore_topology(result, entries, dist_parsed, mesh_shape_tuple)
        except Exception:
            pass  # Best-effort; don't block sweep execution

    return result


def get_mesh_shape() -> Optional[Tuple[int, int]]:
    """
    Get mesh shape from environment variable or auto-detect from hardware.

    Returns:
        Tuple of (rows, cols) or None if using single device

    Environment variable format:
        MESH_DEVICE_SHAPE="1x2" -> (1, 2)
        MESH_DEVICE_SHAPE="2x4" -> (2, 4)
        Not set -> auto-detect from available hardware
    """
    mesh_env = os.environ.get("MESH_DEVICE_SHAPE", "").strip()

    if mesh_env:
        # Parse "NxM" format
        if "x" in mesh_env.lower():
            try:
                parts = mesh_env.lower().split("x")
                rows, cols = int(parts[0]), int(parts[1])
                return (rows, cols)
            except (ValueError, IndexError):
                print(f"⚠️ Invalid MESH_DEVICE_SHAPE format: {mesh_env}, expected NxM (e.g., 1x2)")
                return None
        return None

    # Auto-detect mesh shape from available hardware when env var not set.
    # Model-traced sweeps need the correct mesh topology to reproduce the
    # tensor placement metadata recorded during model tracing.
    try:
        num_devices = ttnn.get_num_devices()
        if num_devices >= 32:
            return (4, 8)  # Galaxy
        elif num_devices >= 8:
            return (1, 8)  # T3000
        elif num_devices >= 2:
            return (1, num_devices)
    except Exception:
        # ttnn may not be initialized yet (env var path is preferred).
        # Fall through to None so the caller can pick a default.
        pass

    return None


def get_model_traced_mesh_shape() -> Tuple[int, int]:
    """Get mesh shape for model-traced sweep modules.

    Model traces are always captured on a mesh device (even 1x1 on single
    chips).  The tracer records 2-D ``tensor_placement`` metadata that is
    only reproduced when the sweep re-executes on a mesh device.

    Returns ``MESH_DEVICE_SHAPE`` from the environment when set, otherwise
    auto-detects from available hardware so that the sweep device topology
    matches the trace topology.
    """
    # Read mesh shape from master JSON matching current hardware.
    # The master may contain configs from multiple devices (N300 + BH).
    # Only use mesh shape from configs whose device_series matches this machine.
    try:
        _master_path = os.environ.get("TTNN_MASTER_JSON_PATH")
        if not _master_path:
            for _base in [
                os.path.join(os.path.dirname(__file__), "..", "..", ".."),
                os.environ.get("TT_METAL_HOME", ""),
            ]:
                if not _base:
                    continue
                _auto = os.path.join(
                    _base,
                    "model_tracer",
                    "traced_operations",
                    "ttnn_operations_master.json",
                )
                if os.path.isfile(_auto):
                    _master_path = _auto
                    break
        if _master_path and os.path.isfile(_master_path):
            import json as _json_ms

            # Detect current arch to filter configs
            _current_arch = os.environ.get("ARCH_NAME", "")
            _is_bh = "blackhole" in _current_arch.lower()
            _is_wh = "wormhole" in _current_arch.lower()

            with open(_master_path) as _f_ms:
                _m_ms = _json_ms.load(_f_ms)
            for _op_ms in _m_ms.get("operations", {}).values():
                for _cfg_ms in _op_ms.get("configurations", []):
                    _mi_ms = _cfg_ms.get("traced_machine_info") or {}
                    if not _mi_ms:
                        _execs = _cfg_ms.get("executions", [])
                        if _execs and isinstance(_execs[0], dict):
                            _mi_ms = _execs[0].get("machine_info", {})
                    # Filter: only use configs matching current arch
                    _board = str(_mi_ms.get("board_type", "")).lower()
                    if _is_bh and "blackhole" not in _board and "wormhole" not in _board:
                        pass  # no board info, use anyway
                    elif _is_bh and "wormhole" in _board:
                        continue  # skip N300/WH configs on BH
                    elif _is_wh and "blackhole" in _board:
                        continue  # skip BH configs on WH

                    _ms_val = _mi_ms.get("mesh_device_shape")
                    if _ms_val:
                        import ast as _ast_ms

                        if isinstance(_ms_val, str):
                            _ms_val = _ast_ms.literal_eval(_ms_val)
                        if isinstance(_ms_val, list) and len(_ms_val) == 2:
                            return tuple(_ms_val)
    except Exception:
        pass  # Intentionally ignored: master config parsing is best-effort, fall through to env var / auto-detect
    # Env var override (used when master JSON is not available)
    shape = get_mesh_shape()
    if shape:
        return shape
    # Auto-detect mesh shape from available hardware when env var not set.
    # This ensures model-traced sweeps on Galaxy (32 devices) create a [4, 8]
    # mesh matching the topology used during model tracing.
    try:
        num_devices = ttnn.get_num_devices()
        if num_devices >= 32:
            return (4, 8)  # Galaxy
        elif num_devices >= 8:
            return (1, 8)  # T3000
        elif num_devices >= 2:
            return (1, num_devices)
    except Exception:
        # ttnn may not be initialized yet (env var path is preferred).
        # Fall through to a 1x1 default for non-mesh runs.
        pass
    return (1, 1)


def mesh_tensor_to_torch(ttnn_tensor, mesh_device=None, mesh_composer=None) -> torch.Tensor:
    """Convert a TTNN mesh tensor back to torch, reassembling shards by topology.

    Replicated tensors return device 0. Sharded tensors are reassembled
    according to the tensor's TensorTopology placements. Mixed
    [Replicate, Shard(d)] cases concatenate only the unique row/col of devices
    along the shard dim. A caller-supplied mesh_composer overrides this.
    """

    def _get_torch_dtype(t):
        try:
            dt = t.dtype
            if dt == ttnn.uint16:
                return torch.int32
            if dt == ttnn.uint32:
                return torch.int64
        except Exception:
            pass
        return None

    def _to_torch_safe(t):
        torch_dtype = _get_torch_dtype(t)
        if torch_dtype is not None:
            return ttnn.to_torch(t).to(torch_dtype)
        return ttnn.to_torch(t)

    try:
        device = ttnn_tensor.device()
    except Exception:
        device = None

    is_mesh = device is not None and hasattr(device, "get_num_devices")

    if not is_mesh:
        # Host tensor brought back from a mesh device (e.g. via from_device on a
        # replicated multi-device tensor) keeps multiple per-device buffers but
        # reports device()==None.  Plain ttnn.to_torch then asserts buffers==1.
        # Mirror the on-mesh non-shard path: take the first replica.
        try:
            topology = ttnn_tensor.tensor_topology()
            placements = list(topology.placements())
        except Exception:
            placements = []
        if placements and not any(type(p).__name__ == "PlacementShard" for p in placements):
            try:
                device_tensors = ttnn.get_device_tensors(ttnn_tensor)
            except Exception:
                device_tensors = []
            if len(device_tensors) > 1:
                return _to_torch_safe(device_tensors[0])
        return _to_torch_safe(ttnn_tensor)

    if mesh_composer is not None:
        result = ttnn.to_torch(ttnn_tensor, mesh_composer=mesh_composer)
        torch_dtype = _get_torch_dtype(ttnn_tensor)
        return result.to(torch_dtype) if torch_dtype is not None else result

    try:
        topology = ttnn_tensor.tensor_topology()
        placements = list(topology.placements())
        dist_shape = topology.distribution_shape()
        dist_dims = [int(d) for d in list(dist_shape)]
        mesh_coords = list(topology.mesh_coords())
    except Exception:
        placements = []
        dist_dims = []
        mesh_coords = []

    def _is_shard(p):
        return type(p).__name__ == "PlacementShard"

    has_shard = any(_is_shard(p) for p in placements)

    device_tensors = ttnn.get_device_tensors(ttnn_tensor)

    if not has_shard:
        if device_tensors:
            return _to_torch_safe(device_tensors[0])
        return _to_torch_safe(ttnn_tensor)

    # Validate shard dims against per-device tensor rank. A reshape may reduce
    # rank below the topology's shard dim (e.g. 4D->2D with PlacementShard(dim=2)).
    # When that happens the shard axis no longer exists, so each device already
    # holds a full copy along that axis -- treat as replicated.
    per_dev_ndim = None
    if device_tensors:
        try:
            per_dev_ndim = len(device_tensors[0].shape)
        except Exception:
            pass  # Intentionally ignored: shape query may fail on deallocated tensors, treat as unknown

    if per_dev_ndim is not None:
        if any(_is_shard(p) and p.dim >= per_dev_ndim for p in placements):
            return _to_torch_safe(device_tensors[0])

    if len(placements) == 2 and len(dist_dims) == 2 and all(_is_shard(p) for p in placements):
        try:
            d0 = placements[0].dim
            d1 = placements[1].dim
            comp = ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(dist_dims), dims=(d0, d1))
            result = ttnn.to_torch(ttnn_tensor, mesh_composer=comp)
            torch_dtype = _get_torch_dtype(ttnn_tensor)
            return result.to(torch_dtype) if torch_dtype is not None else result
        except Exception:
            pass

    if len(placements) == 2 and len(dist_dims) == 2:
        rows, cols = dist_dims[0], dist_dims[1]
        if _is_shard(placements[0]) and not _is_shard(placements[1]):
            shard_dim = placements[0].dim
            picks = []
            for r in range(rows):
                for i, mc in enumerate(mesh_coords):
                    coord = list(mc)
                    if len(coord) == 2 and coord[0] == r and coord[1] == 0:
                        picks.append(i)
                        break
            shards = [_to_torch_safe(device_tensors[i]) for i in picks]
            return torch.cat(shards, dim=shard_dim)
        elif _is_shard(placements[1]) and not _is_shard(placements[0]):
            shard_dim = placements[1].dim
            picks = []
            for c in range(cols):
                for i, mc in enumerate(mesh_coords):
                    coord = list(mc)
                    if len(coord) == 2 and coord[0] == 0 and coord[1] == c:
                        picks.append(i)
                        break
            shards = [_to_torch_safe(device_tensors[i]) for i in picks]
            return torch.cat(shards, dim=shard_dim)

    shard_p = next((p for p in placements if _is_shard(p)), None)
    if shard_p is not None:
        try:
            comp = ttnn.ConcatMeshToTensor(device, dim=shard_p.dim)
            result = ttnn.to_torch(ttnn_tensor, mesh_composer=comp)
            torch_dtype = _get_torch_dtype(ttnn_tensor)
            return result.to(torch_dtype) if torch_dtype is not None else result
        except Exception:
            pass

    if device_tensors:
        return _to_torch_safe(device_tensors[0])
    return _to_torch_safe(ttnn_tensor)


def broadcast_torch_inputs_to_global(
    torch_a: torch.Tensor,
    placement_a: Optional[Dict],
    torch_b: torch.Tensor,
    placement_b: Optional[Dict],
):
    """Reconcile torch shapes for elementwise ops with mismatched global shapes.

    Uses placement (Replicate/Shard) and distribution_shape to derive per-chip
    sizes for each tensor dim. Per-dim expansion rules when shapes mismatch:
      - per-chip sizes equal: smaller operand is "replicated full"; tile by mesh
        factor of the larger side using torch.repeat.
      - smaller operand has per-chip size 1 along this dim: it broadcasts within
        each chip; expand using torch.repeat_interleave by the per-chip size of
        the larger side (so each per-chip element of the smaller side is
        replicated to fill its corresponding chunk of the larger side).
    Falls back to plain torch.repeat (legacy behavior) when placement info is
    missing, when ndims differ, or when no clean integer-ratio expansion exists.
    """
    if torch_a.shape == torch_b.shape:
        return torch_a, torch_b
    if torch_a.ndim != torch_b.ndim:
        return torch_a, torch_b

    def _parse_placement_str(plac_val):
        if plac_val is None:
            return None
        if isinstance(plac_val, (list, tuple)):
            parts = [str(x).strip().strip("'") for x in plac_val]
        else:
            s_inner = str(plac_val).strip()
            if s_inner.startswith("[") and s_inner.endswith("]"):
                s_inner = s_inner[1:-1]
            parts = [x.strip().strip("'") for x in s_inner.split(",")]
        out = []
        for x in parts:
            if not x:
                continue
            if x.startswith("PlacementShard("):
                d = int(x[len("PlacementShard(") : -1])
                out.append(("S", d))
            elif x.startswith("PlacementReplicate"):
                out.append(("R", None))
            else:
                out.append(("?", None))
        return out

    def _parse_dist_str(dist_val):
        if dist_val is None:
            return None
        if isinstance(dist_val, (list, tuple)):
            return [int(x) for x in dist_val]
        s_inner = str(dist_val).strip()
        if s_inner.startswith("[") and s_inner.endswith("]"):
            s_inner = s_inner[1:-1]
        return [int(x.strip()) for x in s_inner.split(",") if x.strip()]

    def _factors(p, ndim):
        if not isinstance(p, dict):
            return [1] * ndim
        plac = _parse_placement_str(p.get("placement"))
        dist = _parse_dist_str(p.get("distribution_shape"))
        if plac is None or dist is None:
            return [1] * ndim
        f = [1] * ndim
        for (kind, dim), n in zip(plac, dist):
            if kind == "S" and dim is not None:
                d = dim if dim >= 0 else dim + ndim
                if 0 <= d < ndim:
                    f[d] *= n
        return f

    def _tile(t, dim, n):
        if n == 1:
            return t
        repeats = [1] * t.ndim
        repeats[dim] = n
        return t.repeat(*repeats)

    fa = _factors(placement_a, torch_a.ndim)
    fb = _factors(placement_b, torch_b.ndim)

    def _try_broadcast(a, b):
        try:
            return torch.broadcast_tensors(a, b)
        except RuntimeError:
            return None

    def _fallback_global_broadcast():
        """Try expanding per-chip sharded operands to their gathered global shape."""
        tiled_a = tile_torch_to_global(torch_a, placement_a)
        tiled_b = tile_torch_to_global(torch_b, placement_b)
        for cand_a, cand_b in (
            (torch_a, tiled_b),
            (tiled_a, torch_b),
            (tiled_a, tiled_b),
            (torch_a, torch_b),
        ):
            result = _try_broadcast(cand_a, cand_b)
            if result is not None:
                return result
        return torch_a, torch_b

    new_a, new_b = torch_a, torch_b
    for d in range(torch_a.ndim):
        sa = new_a.shape[d]
        sb = new_b.shape[d]
        if sa == sb:
            continue
        # Per-chip sizes derived from current shape and placement factor.
        per_chip_a = sa // fa[d] if fa[d] > 0 and sa % fa[d] == 0 else None
        per_chip_b = sb // fb[d] if fb[d] > 0 and sb % fb[d] == 0 else None
        if per_chip_a is None or per_chip_b is None:
            # Fall back to a single tile attempt below.
            per_chip_a = sa
            per_chip_b = sb

        if per_chip_a == per_chip_b:
            # Both operands carry the same per-chip slice; the smaller is
            # replicated across the mesh while the larger is sharded. Tile the
            # smaller by the larger side's mesh factor.
            if sa < sb and sb % sa == 0:
                new_a = _tile(new_a, d, sb // sa)
            elif sb < sa and sa % sb == 0:
                new_b = _tile(new_b, d, sa // sb)
            else:
                return _fallback_global_broadcast()
        elif per_chip_a == 1 and per_chip_b > 1:
            # a's per-chip element broadcasts to all per_chip_b elements on
            # each chip; globally that is repeat_interleave.
            if sb % sa == 0:
                new_a = new_a.repeat_interleave(per_chip_b, dim=d)
            else:
                return _fallback_global_broadcast()
        elif per_chip_b == 1 and per_chip_a > 1:
            if sa % sb == 0:
                new_b = new_b.repeat_interleave(per_chip_a, dim=d)
            else:
                return _fallback_global_broadcast()
        else:
            # Mixed per-chip sizes neither equal nor singleton: try plain tile.
            if sa < sb and sb % sa == 0:
                new_a = _tile(new_a, d, sb // sa)
            elif sb < sa and sa % sb == 0:
                new_b = _tile(new_b, d, sa // sb)
            else:
                return _fallback_global_broadcast()
    result = _try_broadcast(new_a, new_b)
    return result if result is not None else _fallback_global_broadcast()


def tile_torch_to_global(torch_tensor: torch.Tensor, tensor_placement: Optional[Dict]) -> torch.Tensor:
    """Expand a per-chip torch tensor to its global shape based on placement.

    For each PlacementShard(d) entry in `placement` paired with a factor N from
    `distribution_shape`, repeat the tensor along dim d by N. PlacementReplicate
    entries are no-ops. Returns the input unchanged when placement is missing
    or has no Shard entries.

    This mirrors the gather semantics of mesh_tensor_to_torch: a sweep that
    generates a per-chip golden via torch.op(per_chip_a, per_chip_b) needs the
    result tiled along the sharded dims so it matches the gathered global
    output shape used for PCC.
    """
    if not isinstance(tensor_placement, dict):
        return torch_tensor

    plac_val = tensor_placement.get("placement")
    dist_val = tensor_placement.get("distribution_shape")
    if plac_val is None or dist_val is None:
        return torch_tensor

    if isinstance(plac_val, (list, tuple)):
        plac_parts = [str(x).strip().strip("'") for x in plac_val]
    else:
        s = str(plac_val).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        plac_parts = [x.strip().strip("'") for x in s.split(",")]

    plac_entries = []
    for x in plac_parts:
        if not x:
            continue
        if x.startswith("PlacementShard("):
            plac_entries.append(("S", int(x[len("PlacementShard(") : -1])))
        elif x.startswith("PlacementReplicate"):
            plac_entries.append(("R", None))
        else:
            plac_entries.append(("?", None))

    if isinstance(dist_val, (list, tuple)):
        dist_factors = [int(x) for x in dist_val]
    else:
        s = str(dist_val).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        dist_factors = [int(x.strip()) for x in s.split(",") if x.strip()]

    ndim = torch_tensor.ndim
    out = torch_tensor
    for (kind, dim), n in zip(plac_entries, dist_factors):
        if kind != "S" or dim is None or n <= 1:
            continue
        d = dim if dim >= 0 else dim + ndim
        if d >= ndim:
            # Some traced transformer helper ops collapse a sharded input dim
            # into the last output dim (for example QKV/head reshape paths).
            # Preserve the shard factor by applying it to the innermost dim.
            d = ndim - 1
        if d < 0:
            continue
        repeats = [1] * out.ndim
        repeats[d] = n
        out = out.repeat(*repeats)
    return out


def reconcile_golden_to_actual(
    torch_golden: torch.Tensor,
    actual_global: torch.Tensor,
    *placements: Optional[Dict],
) -> torch.Tensor:
    """Tile a per-chip torch golden along sharded dims so it matches the gathered actual shape.

    Try strategies in order:

    1. Shapes already match: return as-is.
    2. Per-dim integer-ratio tile: every dim of `actual` is an integer
       multiple of the corresponding dim of `golden`, with at least one
       dim > 1. This handles the common trace-validation case where the
       inputs were produced via `replicate_with_topology` (so all chips
       hold identical data) and the device op's mesh-aware stitching
       only tiles along a subset of dims (e.g. concat-style ops that
       reassemble along one mesh axis but not the other). Picking up the
       actual's per-dim repeat factor works regardless of which mesh
       axis the device chose to stitch along.
    3. Original placement-driven tile via tile_torch_to_global: relies
       on the master's recorded `placement` + `distribution_shape` to
       repeat by the per-axis Shard factor. This is correct for genuine
       sharded inputs (inputs split across the mesh, each chip computing
       its slice) but produces the wrong shape when the mesh stitch
       only fired along a subset of axes.

    Strategy 2 is tried first because the trace-validation framework's
    default is to replicate inputs and rely on stitch-driven tiling on
    the output.
    """
    if torch_golden.shape == actual_global.shape:
        return torch_golden

    # Strategy 2: per-dim integer-ratio tile.
    if torch_golden.ndim == actual_global.ndim:
        repeats = []
        ok = True
        for d in range(torch_golden.ndim):
            g = torch_golden.shape[d]
            a = actual_global.shape[d]
            if g == 0 or a % g != 0:
                ok = False
                break
            repeats.append(a // g)
        if ok and any(r > 1 for r in repeats):
            tiled = torch_golden.repeat(*repeats)
            if tiled.shape == actual_global.shape:
                return tiled

    # Strategy 3: placement-driven tile (legacy path).
    out = torch_golden
    for plac in placements:
        if out.shape == actual_global.shape:
            break
        out = tile_torch_to_global(out, plac)
    return out
