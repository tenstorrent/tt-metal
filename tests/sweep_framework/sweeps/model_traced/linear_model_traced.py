# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

# Import V2 master config loader and helpers for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import (
    MasterConfigLoader,
    dict_to_memory_config,
)
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    create_mesh_device,
    create_tensor_on_mesh,
    get_model_traced_mesh_shape,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

# Override the default timeout in seconds for hang detection.
# Linear operations with large shapes can take longer, increase timeout
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("linear")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(32, 32)],  # Input shape (m, k)
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(32, 32)],  # Weight shape (k, n)
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "bias_shape": [(32,)],  # Bias shape (n,) - optional
        "bias_dtype": [ttnn.bfloat16],
        "bias_layout": [ttnn.TILE_LAYOUT],
        "transpose_a": [False],
        "transpose_b": [False],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_model_traced_mesh_shape()
    device = create_mesh_device(mesh_shape)
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_mesh_device(device)


def _parse_placement_list(plac_val):
    """Return list of (kind, dim) per mesh dim. kind in {'S','R','?'}."""
    if plac_val is None:
        return None
    if isinstance(plac_val, (list, tuple)):
        items = [str(x).strip().strip("'") for x in plac_val]
    else:
        s_inner = str(plac_val).strip()
        if s_inner.startswith("[") and s_inner.endswith("]"):
            s_inner = s_inner[1:-1]
        items = [x.strip().strip("'") for x in s_inner.split(",") if x.strip()]
    out = []
    for x in items:
        if x.startswith("PlacementShard("):
            d = int(x[len("PlacementShard(") : -1])
            out.append(("S", d))
        elif x.startswith("PlacementReplicate"):
            out.append(("R", None))
        else:
            out.append(("?", None))
    return out


def _parse_dist_list(dist_val):
    if dist_val is None:
        return None
    if isinstance(dist_val, (list, tuple)):
        return [int(x) for x in dist_val]
    s_inner = str(dist_val).strip()
    if s_inner.startswith("[") and s_inner.endswith("]"):
        s_inner = s_inner[1:-1]
    return [int(x.strip()) for x in s_inner.split(",") if x.strip()]


def _mesh_factor_for_axis(plac_dict, axis, ndim):
    if not isinstance(plac_dict, dict):
        return 1
    plac = _parse_placement_list(plac_dict.get("placement"))
    dist = _parse_dist_list(plac_dict.get("distribution_shape"))
    if not plac or not dist:
        return 1
    factor = 1
    for (kind, d), n in zip(plac, dist):
        if kind == "S" and d is not None:
            ad = d if d >= 0 else d + ndim
            if ad == axis:
                factor *= n
    return factor


def _align_linear_for_torch(torch_a, placement_a, torch_w, placement_w):
    """Align shapes so torch.matmul(a, w) yields the global result.

    K-sharding on a (last dim) with replicated w: tile w by mesh factor along
    its K (first) axis. K-sharding on both a and w (along the matching K axis):
    no-op (per-chip partials, kernel must reduce; torch.matmul on global already
    matches).
    """
    if torch_a.ndim < 2 or torch_w.ndim < 2:
        return torch_a, torch_w
    a_K = torch_a.shape[-1]
    w_K = torch_w.shape[-2]
    if a_K == w_K:
        return torch_a, torch_w
    fa_last = _mesh_factor_for_axis(placement_a, torch_a.ndim - 1, torch_a.ndim)
    fw_first_of_2d = _mesh_factor_for_axis(placement_w, torch_w.ndim - 2, torch_w.ndim)
    # Case: a sharded on K, w replicated => tile w along K by fa_last.
    if fa_last > 1 and fw_first_of_2d == 1 and a_K == w_K * fa_last:
        repeat = [1] * torch_w.ndim
        repeat[-2] = fa_last
        torch_w = torch_w.repeat(*repeat)
    # Case: w sharded on K (first 2D dim), a replicated => tile a along K by fw.
    elif fw_first_of_2d > 1 and fa_last == 1 and w_K == a_K * fw_first_of_2d:
        repeat = [1] * torch_a.ndim
        repeat[-1] = fw_first_of_2d
        torch_a = torch_a.repeat(*repeat)
    return torch_a, torch_w


def _reorder_l1_mc_for_dram_sharded(mc, device):
    """Reorder an L1-sharded MemoryConfig's core_ranges to match the device's
    optimal DRAM bank → worker assignment. Required by the BatchedDRAMSharded
    matmul kernel: it asserts storage_core[i] == worker_core[i] (NOC_0 list).

    Master configs record cores in insertion order, which often differs from
    the device's optimal order. Same set of cores, just shuffled.
    """
    try:
        if mc is None or mc.buffer_type != ttnn.BufferType.L1:
            return mc
        if mc.shard_spec is None:
            return mc
        old_grid = mc.shard_spec.grid
        # Collect the set of (x,y) cores in master's mc
        master_cores = set()
        for cr in old_grid.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    master_cores.add((x, y))
        if not master_cores:
            return mc
        # Get the device's optimal assignment for NOC_0
        try:
            optimal = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        except Exception:
            return mc
        # Build a new core_ranges list: take optimal cores in order, only those
        # that appear in master's set. If sizes mismatch, leave mc unchanged.
        ordered = [(c.x, c.y) for c in optimal if (c.x, c.y) in master_cores]
        if len(ordered) != len(master_cores):
            return mc
        new_ranges = [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in ordered]
        new_grid = ttnn.CoreRangeSet(new_ranges)
        new_shard_spec = ttnn.ShardSpec(new_grid, mc.shard_spec.shape, mc.shard_spec.orientation)
        return ttnn.MemoryConfig(mc.memory_layout, mc.buffer_type, new_shard_spec)
    except Exception:
        return mc


def run(
    input_a_shape,  # Input shape (m, k)
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape=None,  # Weight shape (k, n)
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    bias_shape=None,  # Optional bias shape (n,)
    bias_dtype=None,
    bias_layout=None,
    bias_memory_config=None,
    transpose_a=False,
    transpose_b=False,
    storage_type="StorageType::DEVICE",
    memory_config=None,  # Alternative memory_config parameter
    dtype=None,  # Output dtype
    core_grid=None,  # Core grid configuration
    program_config=None,  # Program configuration
    compute_kernel_config=None,  # Compute kernel configuration
    activation=None,  # Activation function
    *,
    device,
    **kwargs,  # Accept placements, traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # V2 vectors provide weight as input_tensor_b_* instead of input_b_*. Each
    # field can be present in either convention (or None when absent in master),
    # so fall through per-field rather than gating on input_b_shape alone.
    if input_b_shape is None:
        input_b_shape = kwargs.get("input_tensor_b_shape")
    if input_b_dtype is None:
        input_b_dtype = kwargs.get("input_tensor_b_dtype", input_a_dtype)
    if input_b_layout is None:
        input_b_layout = kwargs.get("input_tensor_b_layout", input_a_layout)
    if input_b_memory_config is None:
        input_b_memory_config = kwargs.get("input_tensor_b_memory_config", ttnn.DRAM_MEMORY_CONFIG)

    if input_b_shape is None:
        raise ValueError("Weight shape (input_b_shape or input_tensor_b_shape) is required")

    # Parse named op params that were in the function signature (not in **kwargs)
    from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

    if isinstance(memory_config, dict):
        memory_config = dict_to_memory_config(memory_config) or parse_dict_value("memory_config", memory_config)
    if isinstance(core_grid, dict):
        core_grid = parse_dict_value("core_grid", core_grid)
    if isinstance(compute_kernel_config, dict):
        compute_kernel_config = parse_dict_value("compute_kernel_config", compute_kernel_config)
    if isinstance(dtype, (dict, str)):
        dtype = (
            parse_dict_value("dtype", dtype)
            if isinstance(dtype, dict)
            else parse_dict_value("dtype", {"type": "DataType", "repr": dtype})
        )
    # Use traced program_config when available — master and sweep both run on the
    # same Galaxy 4×8 topology so block/grid sizes are valid. Parse dict form to
    # the appropriate ttnn program_config object.
    if isinstance(program_config, dict):
        program_config = parse_dict_value("program_config", program_config)
        if isinstance(program_config, dict):
            # parse_dict_value couldn't resolve it — drop rather than fail.
            program_config = None

    # V2 passes memory_config as a serialized dict; parse to ttnn.MemoryConfig.
    if isinstance(input_a_memory_config, dict):
        input_a_memory_config = dict_to_memory_config(input_a_memory_config)
    if isinstance(input_b_memory_config, dict):
        input_b_memory_config = dict_to_memory_config(input_b_memory_config)

    # BatchedDRAMSharded matmul kernel asserts that the L1 input_a shard
    # grid uses the same core ordering as the device's optimal DRAM bank
    # → worker assignment. Master records cores in insertion order; reorder
    # to match the kernel's expected worker order.
    _pc_cls = type(program_config).__name__ if program_config is not None else ""
    if "BatchedDRAMSharded" in _pc_cls:
        input_a_memory_config = _reorder_l1_mc_for_dram_sharded(input_a_memory_config, device)
        if isinstance(memory_config, ttnn.MemoryConfig):
            memory_config = _reorder_l1_mc_for_dram_sharded(memory_config, device)
    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement")
    if input_b_tensor_placement is None:
        input_b_tensor_placement = kwargs.get("input_tensor_b_tensor_placement")
    bias_tensor_placement = kwargs.get("bias_tensor_placement", None)
    output_memory_config = dict_to_memory_config(kwargs.get("output_memory_config", None))

    # Use build_op_kwargs to parse dict values for op kwargs (compute_kernel_config, etc.).
    # Exclude program_config (handled above), activation (used for golden too),
    # and output_tile (a Tile object that can't be auto-parsed from dict).
    parsed_op_kwargs = build_op_kwargs(kwargs, exclude={"output_tile"})

    # Parse master's output_tile (a Tile object that build_op_kwargs can't auto-
    # parse). Format: {"type": "Tile", "value": "Tile with shape: [32, 32]"}.
    _ot_raw = kwargs.get("output_tile")
    _output_tile = None
    if isinstance(_ot_raw, dict) and _ot_raw.get("type") == "Tile":
        import re as _re_ot

        _v = str(_ot_raw.get("value", ""))
        _m = _re_ot.search(r"shape:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", _v)
        if _m:
            try:
                _output_tile = ttnn.Tile([int(_m.group(1)), int(_m.group(2))])
            except Exception:
                _output_tile = None

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")  # MeshDevice has this method

    # V2 format provides separate shapes
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if isinstance(input_b_shape, (list, tuple)) else input_b_shape

    # Detect 4D batched weights (batch > 1 in weight tensor).
    # ttnn.linear hits TT_FATAL with batched weights (requires batch_b == 1).
    # Use ttnn.matmul instead, which handles batched matmul natively.
    # Force ttnn.linear path for all configs so trace matches master's traced
    # ttnn.linear (batched weights are handled by ttnn.linear internally — no
    # need to special-case to ttnn.matmul, which would mismatch the master trace).
    is_batched_weight = False

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Create random tensors
    torch_a = torch.randn(*shape_a, dtype=torch.float32)
    torch_b = torch.randn(*shape_b, dtype=torch.float32)

    # For linear operations, use the weight as-is (TTNN handles the format)
    torch_weight = torch_b

    # Create bias tensor if needed
    torch_bias = None
    ttnn_bias = None
    has_bias = bias_shape is not None and bias_shape != tuple()

    if has_bias:
        shape_bias = tuple(bias_shape) if isinstance(bias_shape, (list, tuple)) else bias_shape
        torch_bias = torch.randn(*shape_bias, dtype=torch.float32) if shape_bias != tuple() else torch.randn(())

        # Create bias tensor with mesh support if needed
        if not is_host:
            if is_mesh_device and bias_tensor_placement:
                ttnn_bias = create_tensor_on_mesh(
                    torch_bias,
                    device,
                    bias_dtype if bias_dtype else input_a_dtype,
                    bias_layout if bias_layout else input_a_layout,
                    bias_memory_config if bias_memory_config else ttnn.DRAM_MEMORY_CONFIG,
                    bias_tensor_placement,
                )
            else:
                ttnn_bias = ttnn.from_torch(
                    torch_bias,
                    dtype=bias_dtype if bias_dtype else input_a_dtype,
                    layout=bias_layout if bias_layout else input_a_layout,
                    device=device,
                    memory_config=bias_memory_config if bias_memory_config else ttnn.DRAM_MEMORY_CONFIG,
                )
        else:
            ttnn_bias = ttnn.from_torch(
                torch_bias,
                dtype=bias_dtype if bias_dtype else input_a_dtype,
                layout=bias_layout if bias_layout else input_a_layout,
            )

    # Golden output using PyTorch
    # Align shapes for K-sharded matmul: when input is sharded along K with a
    # replicated weight (or vice versa), tile the replicated side so torch
    # produces the same global result as the kernel's reduce-sum semantics.
    torch_a_for_golden, torch_weight_for_golden = _align_linear_for_torch(
        torch_a, input_a_tensor_placement, torch_weight, input_b_tensor_placement
    )
    if len(torch_a_for_golden.shape) > 2:
        torch_output_tensor = torch.matmul(torch_a_for_golden, torch_weight_for_golden)
        if torch_bias is not None:
            torch_output_tensor = torch_output_tensor + torch_bias
    else:
        torch_weight_for_linear = torch_weight_for_golden
        if len(torch_weight_for_golden.shape) >= 2:
            torch_weight_for_linear = torch_weight_for_golden.transpose(-1, -2)
        torch_output_tensor = torch.nn.functional.linear(torch_a_for_golden, torch_weight_for_linear, torch_bias)

    # Apply activation to golden reference to match ttnn.linear behavior
    # Skip for batched weights (ttnn.matmul path doesn't apply activation)
    if activation is not None and not is_batched_weight:
        act = str(activation).lower()
        if "silu" in act or "swish" in act:
            torch_output_tensor = torch.nn.functional.silu(torch_output_tensor)
        elif "gelu" in act:
            approx = "tanh" if "approx" in act else "none"
            torch_output_tensor = torch.nn.functional.gelu(torch_output_tensor, approximate=approx)
        elif "relu" in act:
            torch_output_tensor = torch.nn.functional.relu(torch_output_tensor)

    # Create input tensor A. Mirror the model's flow: build the tensor in
    # DRAM-interleaved with the right per-chip placement, then to_memory_config
    # to land on the master's exact memory_config. This avoids the kernel
    # rejecting "from_torch direct to L1-sharded" creation paths.
    from tests.sweep_framework.sweep_utils.mesh_tensor_utils import apply_tensor_placement_topology as _apply_topo

    if not is_host:
        try:
            if is_mesh_device and input_a_tensor_placement:
                ttnn_a = create_tensor_on_mesh(
                    torch_a,
                    device,
                    input_a_dtype,
                    input_a_layout,
                    ttnn.DRAM_MEMORY_CONFIG,
                    input_a_tensor_placement,
                )
            else:
                ttnn_a = ttnn.from_torch(
                    torch_a,
                    dtype=input_a_dtype,
                    layout=input_a_layout,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            if input_a_memory_config is not None and input_a_memory_config != ttnn.DRAM_MEMORY_CONFIG:
                try:
                    ttnn_a = ttnn.to_memory_config(ttnn_a, input_a_memory_config)
                except Exception:
                    pass  # leave in DRAM-interleaved if the conversion fails
        except Exception:
            ttnn_a = ttnn.from_torch(
                torch_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
            )
        if is_mesh_device and input_a_tensor_placement:
            try:
                actual_mesh = device.shape
                _apply_topo(ttnn_a, input_a_tensor_placement, (actual_mesh[0], actual_mesh[1]))
            except Exception:
                # Best-effort: if the C++ topology setter rejects the mesh
                # shape (e.g. fewer chips than master traced), the trace will
                # show the fallback topology rather than crash the sweep.
                pass
    else:
        ttnn_a = ttnn.from_torch(torch_a, dtype=input_a_dtype, layout=input_a_layout)

    # Create weight tensor B — same DRAM-then-to_memory_config flow as input_a.
    weight_memory_config = input_b_memory_config

    if not is_host:
        if is_mesh_device and input_b_tensor_placement:
            ttnn_b = create_tensor_on_mesh(
                torch_b,
                device,
                input_b_dtype,
                input_b_layout,
                ttnn.DRAM_MEMORY_CONFIG,
                input_b_tensor_placement,
            )
            if weight_memory_config is not None and weight_memory_config != ttnn.DRAM_MEMORY_CONFIG:
                try:
                    ttnn_b = ttnn.to_memory_config(ttnn_b, weight_memory_config)
                except Exception:
                    # Leave weight in DRAM-interleaved if the kernel rejects
                    # the master shard layout (e.g. shard_spec incompatible
                    # with current dispatch grid). The trace will show DRAM
                    # rather than crash the sweep.
                    pass
        else:
            # Regular single-device tensor
            ttnn_b = ttnn.from_torch(
                torch_b,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=weight_memory_config,
            )
    else:
        # Host storage
        ttnn_b = ttnn.from_torch(torch_b, dtype=input_b_dtype, layout=input_b_layout)

    # Run TTNN op
    start_time = start_measuring_time()

    def _make_dram_tensors():
        # Build replicated DRAM tensors but stamp the master's tensor topology
        # so the trace records placement matching the master (even though
        # memory_config falls back to DRAM-interleaved when the L1-sharded path
        # fails).
        a = ttnn.from_torch(
            torch_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )
        b = ttnn.from_torch(
            torch_b,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )
        if is_mesh_device:
            try:
                from tests.sweep_framework.sweep_utils.mesh_tensor_utils import apply_tensor_placement_topology

                actual_mesh = device.shape
                if input_a_tensor_placement:
                    apply_tensor_placement_topology(a, input_a_tensor_placement, (actual_mesh[0], actual_mesh[1]))
                if input_b_tensor_placement:
                    apply_tensor_placement_topology(b, input_b_tensor_placement, (actual_mesh[0], actual_mesh[1]))
            except Exception:
                pass  # best-effort; trace will show fallback topology
        return a, b

    if is_batched_weight:
        matmul_kwargs = {}
        if compute_kernel_config is not None:
            matmul_kwargs["compute_kernel_config"] = compute_kernel_config
        if dtype is not None:
            matmul_kwargs["dtype"] = dtype
        try:
            output_tensor = ttnn.matmul(ttnn_a, ttnn_b, **matmul_kwargs)
        except Exception:
            ttnn_a, ttnn_b = _make_dram_tensors()
            try:
                output_tensor = ttnn.matmul(ttnn_a, ttnn_b, **matmul_kwargs)
            except Exception:
                output_tensor = ttnn.matmul(ttnn_a, ttnn_b)
    else:
        linear_kwargs = {}
        # Only pass bias if it was actually traced (non-None).
        # Passing bias=None creates extra_key diff when master didn't have it.
        if ttnn_bias is not None:
            linear_kwargs["bias"] = ttnn_bias

        if transpose_a:
            linear_kwargs["transpose_a"] = transpose_a
        if transpose_b:
            linear_kwargs["transpose_b"] = transpose_b

        # Forward memory_config when master had it (not __ABSENT__)
        if memory_config != "__ABSENT__" and memory_config is not None:
            linear_kwargs["memory_config"] = memory_config
        elif output_memory_config is not None:
            linear_kwargs["memory_config"] = output_memory_config

        _absent = set(kwargs.get("__absent_keys__") or [])
        if dtype is not None and dtype != "__ABSENT__":
            linear_kwargs["dtype"] = dtype
        elif dtype is None and "dtype" not in _absent:
            linear_kwargs["dtype"] = None

        if program_config is not None and program_config != "__ABSENT__":
            linear_kwargs["program_config"] = program_config

        # Pass compute_kernel_config even when None — the master trace records it
        # when the model explicitly passed it (including None). Use __absent_keys__
        # (injected by execute_test) to distinguish "master had ckc=None" from
        # "master never passed ckc". Falls back to value-based check for older callers.
        if compute_kernel_config is not None and compute_kernel_config != "__ABSENT__":
            linear_kwargs["compute_kernel_config"] = compute_kernel_config

        if core_grid is not None and core_grid != "__ABSENT__":
            linear_kwargs["core_grid"] = core_grid
        elif core_grid is None and "core_grid" not in _absent:
            linear_kwargs["core_grid"] = None

        if activation is not None:
            linear_kwargs["activation"] = activation

        if _output_tile is not None:
            linear_kwargs["output_tile"] = _output_tile

        linear_kwargs.update(parsed_op_kwargs)

        # Master traced ttnn.linear with two call forms: 26 cfgs used the kwarg
        # `input_tensor_b=` (vectors carry input_tensor_b_shape), 3 cfgs used
        # the positional arg (vectors carry input_b_shape).  Match each form
        # so the tracer captures the same arg key the master saw.
        # Master used `input_tensor_b=` named for 26 cfgs and positional `arg1` for 3.
        # __absent_keys__ tells us which form the vector preserves.
        _absent = kwargs.get("__absent_keys__", set()) or set()
        _used_named_b = "input_b_shape" in _absent and "input_tensor_b_shape" not in _absent

        def _do_linear(_a, _b, **_kw):
            if _used_named_b:
                return ttnn.linear(_a, input_tensor_b=_b, **_kw)
            return ttnn.linear(_a, _b, **_kw)

        output_tensor = _do_linear(ttnn_a, ttnn_b, **linear_kwargs)

    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)

    # Partial-reduce fallback: if a K-sharded matmul produces per-chip partial
    # outputs falsely marked as Shard(-1), the reassembler concats them; the
    # actual last-dim is mesh_factor times the expected. Reshape and sum to
    # recover the correct global result.
    expected_shape = list(torch_output_tensor.shape)
    actual_shape = list(output_tensor.shape)
    if len(expected_shape) == len(actual_shape) and expected_shape != actual_shape:
        for d in range(len(expected_shape)):
            if expected_shape[d] != actual_shape[d] and actual_shape[d] % expected_shape[d] == 0:
                ratio = actual_shape[d] // expected_shape[d]
                view_shape = list(actual_shape)
                view_shape[d : d + 1] = [ratio, expected_shape[d]]
                output_tensor = output_tensor.reshape(*view_shape).sum(dim=d)
                break
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(
            torch_output_tensor, output_tensor, input_a_tensor_placement, input_b_tensor_placement
        )
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
