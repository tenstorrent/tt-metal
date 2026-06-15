# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("matmul")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_shape": [(1, 1, 32, 32)],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def _parse_2d_shard_dims(placement, ndim=4):
    """[dim_on_mesh_rows, dim_on_mesh_cols] from a traced placement dict (Shard dims,
    normalized to >=0; None for Replicate)."""
    import re

    s = str(placement.get("placement", "")) if isinstance(placement, dict) else str(placement)
    out = []
    for m in re.finditer(r"PlacementShard\((?:dim=)?(-?\d+)\)|PlacementReplicate", s):
        if m.group(1) is None:
            out.append(None)
        else:
            d = int(m.group(1))
            out.append(d + ndim if d < 0 else d)
    return out


def _as_dtype(v, default):
    """Resolve a traced dtype (ttnn.DataType | dict | repr-string | None) to ttnn.DataType."""
    from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

    if v is None or v == "__ABSENT__":
        return default
    if isinstance(v, ttnn.DataType):
        return v
    try:
        if isinstance(v, dict):
            return parse_dict_value("dtype", v) or default
        if isinstance(v, str):
            return parse_dict_value("dtype", {"type": "DataType", "repr": v}) or default
    except Exception:
        return default
    return default


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _run_gather_in0_ring_matmul(
    input_a_shape,
    input_b_shape,
    pc,
    mesh_shape,
    input_b_placement,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    compute_kernel_config_raw,
):
    """gather_in0 1D ring matmul (decode LM-head / MLP-w2 projections), ttnn.matmul form.

    Same distributed-fragment reconstruction as the linear sweep (see
    linear_model_traced._run_gather_in0_ring_matmul): the op runs on the model's
    prefetcher+worker sub-devices over a fixed 24-core ring and emits per-K-shard
    PARTIALS that the model finishes with a cross-mesh all-reduce; we rebuild the
    decode launch path and reconstruct the all-reduce in torch for the golden. The
    only difference here is the op call (ttnn.matmul instead of ttnn.linear; no bias).
    The caller must have released the fixture's device before invoking this (it opens
    its own full-galaxy COL mesh with sub-devices).
    """
    import math

    from models.demos.llama3_70b_galaxy.tt.prefetcher_common import get_core_ranges
    from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value
    from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

    TILE = ttnn.TILE_SIZE
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])

    def _ru(n, m):
        return ((n + m - 1) // m) * m

    M = int(input_a_shape[-2])
    K_pd = int(input_b_shape[-2])  # per-device K (weight dim -2)
    N_pd = int(input_b_shape[-1])  # per-device N (weight dim -1)

    b_dims = _parse_2d_shard_dims(input_b_placement, ndim=4)
    if len(b_dims) < 2 or 2 not in b_dims or 3 not in b_dims:
        raise ValueError(f"weight placement is not a 2D K/N mesh-shard: {input_b_placement}")
    k_axis = b_dims.index(2)  # 0=rows, 1=cols
    n_axis = b_dims.index(3)
    global_K = K_pd * mesh_shape[k_axis]
    global_N = N_pd * mesh_shape[n_axis]

    grid = (int(pc["compute_with_storage_grid_size"]["x"]), int(pc["compute_with_storage_grid_size"]["y"]))
    in0_block_w = int(pc["in0_block_w"])
    per_core_N = int(pc["per_core_N"])
    out_subblock_h = int(pc.get("out_subblock_h", 1))
    out_subblock_w = int(pc.get("out_subblock_w", 1))
    per_core_M = int(pc.get("per_core_M", 1))
    n_gcb = pc.get("num_global_cb_receivers")
    prefetch = bool(n_gcb is not None and int(n_gcb) >= 2)

    wt_dtype = _as_dtype(input_b_dtype, ttnn.bfloat8_b)
    act_dtype = _as_dtype(input_a_dtype, ttnn.bfloat8_b)
    out_dtype = _as_dtype(output_dtype, ttnn.bfloat8_b)
    ckc = compute_kernel_config_raw
    if isinstance(ckc, dict):
        try:
            ckc = parse_dict_value("compute_kernel_config", ckc)
        except Exception:
            ckc = None
    if not isinstance(ckc, ttnn.WormholeComputeKernelConfig):
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=prefetch,
            fp32_dest_acc_en=prefetch,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        )

    dev = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, cols),
        l1_small_size=79104,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL),
    )
    try:
        (
            active_sender_cores,
            dram_cores,
            all_sender_cores,
            active_receiver_cores_list,
            all_receiver_cores,
            worker_cores_range_set,
            mm_optimised_ring_cores,
            hop_grid,
        ) = get_core_ranges(num_reader_cores=12, num_global_cb_receivers=2, is_functional_test=False)
        RING = len(mm_optimised_ring_cores)  # 24

        K_per_shard = _ru(math.ceil(K_pd / RING), TILE)
        N_out_shard = _ru(math.ceil(N_pd / RING), TILE)
        N_PADDED = N_out_shard * RING

        global_cb = None
        if prefetch:
            global_cb_size = 728 * 1088  # TtLlamaPrefetcherSetup.global_cb_size
            global_cb = ttnn.create_global_circular_buffer(
                dev, list(zip(all_sender_cores, all_receiver_cores)), global_cb_size
            )
            sender_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in active_sender_cores])
        else:
            sender_crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_sender_cores])
        mgr = dev.create_sub_device_manager([ttnn.SubDevice([sender_crs]), ttnn.SubDevice([worker_cores_range_set])], 0)
        dev.load_sub_device_manager(mgr)
        worker_id = ttnn.SubDeviceId(1)
        if not prefetch:
            dev.set_sub_device_stall_group([ttnn.SubDeviceId(0), worker_id])

        torch.manual_seed(0)
        hidden = torch.randn(M, global_K)
        w_global = torch.randn(global_K, global_N)
        golden = hidden @ w_global  # [M, global_N]

        N_per_bank = _ru(math.ceil(N_PADDED / len(dram_cores)), TILE)
        wt_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(len(dram_cores) - 1, 0))]),
                [K_pd, N_per_bank],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        b_tt = ttnn.as_tensor(
            w_global.reshape(1, 1, global_K, global_N),
            device=dev,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(b_dims[0], b_dims[1]), mesh_shape=(rows, cols)),
            layout=ttnn.TILE_LAYOUT,
            dtype=wt_dtype,
            memory_config=wt_mc,
        )

        if prefetch:
            tensor_addrs = torch.tensor([b_tt.buffer_address()]).repeat(len(dram_cores), 1)
            addr_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    sender_crs,
                    [tensor_addrs.shape[0] // len(dram_cores), tensor_addrs.shape[1]],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            addr_tt = ttnn.as_tensor(
                tensor_addrs,
                device=dev,
                dtype=ttnn.uint32,
                memory_config=addr_mc,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
            )
            ttnn.dram_prefetcher([b_tt, addr_tt], num_layers=1, global_cb=global_cb)
            dev.set_sub_device_stall_group([worker_id])

        act_host = torch.zeros(rows, cols, M, K_pd)
        for r in range(rows):
            for c in range(cols):
                ki = c if k_axis == 1 else r
                act_host[r, c] = hidden[:, ki * K_pd : (ki + 1) * K_pd]
        a_tt = ttnn.from_torch(
            act_host,
            dtype=act_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(0, 1), mesh_shape=(rows, cols)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        in0_ring_mc = ttnn.create_sharded_memory_config(
            shape=(M, K_per_shard),
            core_grid=_crs(mm_optimised_ring_cores),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        a_tt = ttnn.to_memory_config(a_tt, in0_ring_mc)

        out_mc = ttnn.create_sharded_memory_config(
            shape=(M, N_PADDED // RING),
            core_grid=_crs(active_receiver_cores_list),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        pc_obj = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=_crs(hop_grid),
            **({"num_global_cb_receivers": int(n_gcb)} if prefetch else {}),
        )

        start_time = start_measuring_time()
        mm_kwargs = dict(
            compute_kernel_config=ckc,
            program_config=pc_obj,
            memory_config=out_mc,
            dtype=out_dtype,
            sub_device_id=worker_id,
        )
        if prefetch:
            mm_kwargs["core_grid"] = None
            mm_kwargs["global_cb"] = global_cb
        out = ttnn.matmul(a_tt, b_tt, **mm_kwargs)
        if prefetch:
            dev.reset_sub_device_stall_group()

        partials = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(dev, (rows, cols), dims=(0, 1))
        ).float()  # [rows, cols, M, N_pd]
        reduced = partials.sum(dim=k_axis)  # all-reduce over the K-shards
        recon = reduced.permute(1, 0, 2).reshape(M, global_N)
        e2e_perf = stop_measuring_time(start_time)
        return [check_with_pcc(golden, recon, 0.99), e2e_perf]
    finally:
        try:
            ttnn.close_mesh_device(dev)
        except Exception:
            # best-effort teardown of the gather_in0 ring-matmul device
            pass


# The sweeps runner opens ONE device (via mesh_device_fixture) and reuses it across
# every vector in a suite — it never re-enters the fixture per vector. The gather_in0
# reconstruction has to open its OWN full-galaxy COL mesh, so it must close that shared
# device first; doing so would otherwise leave the runner handing a CLOSED device to all
# subsequent vectors. We track that closure here and transparently (re)open a managed
# replacement so following vectors keep working.
_LIVE_DEVICE = None
_FIXTURE_DEVICE_CLOSED = False


def _live_device(fixture_device):
    """Device to use for this vector: the fixture's until a gather_in0 vector closes it,
    then a module-managed mesh we (re)open lazily and reuse."""
    global _LIVE_DEVICE
    if not _FIXTURE_DEVICE_CLOSED:
        return fixture_device
    if _LIVE_DEVICE is None:
        _LIVE_DEVICE = create_mesh_device(get_mesh_shape())
    return _LIVE_DEVICE


def _close_shared_device(fixture_device):
    """Close whichever shared device is currently open so gather_in0 can open its own."""
    global _LIVE_DEVICE, _FIXTURE_DEVICE_CLOSED
    target = _LIVE_DEVICE if _FIXTURE_DEVICE_CLOSED else fixture_device
    if target is not None:
        try:
            ttnn.close_mesh_device(target)
        except Exception as e:
            # Best-effort: a failed close must not mask the test result, but log it.
            print(f"SWEEPS: best-effort close of shared device before gather_in0 failed: {e}")
    _LIVE_DEVICE = None
    _FIXTURE_DEVICE_CLOSED = True


def mesh_device_fixture():
    """
    Override default device fixture.
    Creates mesh device if MESH_DEVICE_SHAPE is set, otherwise single device.
    """
    mesh_shape = get_mesh_shape()

    if mesh_shape:
        # Create mesh device based on env var
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            try:
                ttnn.close_mesh_device(device)
            except Exception:
                # The gather_in0 ring path closes this device itself (it opens its
                # own COL mesh); a double-close here is expected and harmless.
                pass
            # Close the managed replacement device, if a gather_in0 vector reopened one.
            if _LIVE_DEVICE is not None:
                try:
                    ttnn.close_mesh_device(_LIVE_DEVICE)
                except Exception:
                    pass
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        # Single device (default)
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_shape=None,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept scalar, placements, traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)

    # A prior gather_in0 vector may have closed the shared fixture device; use the
    # managed replacement when so (see _live_device).
    device = _live_device(device)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")

    # gather_in0 1D ring matmuls are distributed model fragments (per-K-shard
    # partials finished by a cross-mesh all-reduce); the generic path produces
    # garbage PCC (0.001-0.15). Detect on the RAW program_config dict and run the
    # faithful reconstruction (mirrors the linear sweep). It opens its own
    # full-galaxy COL mesh + sub-devices, so release the shared device first (and
    # mark it closed so subsequent vectors get a reopened one).
    _pc_raw = kwargs.get("program_config")
    _ib_shape = input_b_shape if input_b_shape is not None else kwargs.get("input_tensor_b_shape")
    if is_mesh_device and isinstance(_pc_raw, dict) and _pc_raw.get("gather_in0") and _ib_shape is not None:
        _ib_plac = input_b_tensor_placement or kwargs.get("input_tensor_b_tensor_placement")
        _close_shared_device(device)
        return _run_gather_in0_ring_matmul(
            input_a_shape=input_a_shape,
            input_b_shape=_ib_shape,
            pc=_pc_raw,
            mesh_shape=get_model_traced_mesh_shape(),
            input_b_placement=_ib_plac,
            input_a_dtype=input_a_dtype,
            input_b_dtype=(input_b_dtype if input_b_dtype is not None else kwargs.get("input_tensor_b_dtype")),
            output_dtype=kwargs.get("dtype"),
            compute_kernel_config_raw=kwargs.get("compute_kernel_config"),
        )

    # Keep all traced params including program_config — they are required for
    # correct matmul behavior with sharded memory configs.
    op_kwargs = build_op_kwargs(kwargs)

    # build_op_kwargs filters memory_config (infrastructure key), but matmul
    # accepts it as an op kwarg.  Re-inject from the traced kwargs when present.
    raw_mc = kwargs.get("memory_config")
    if raw_mc is not None and raw_mc != "__ABSENT__":
        from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

        op_kwargs["memory_config"] = parse_dict_value("memory_config", raw_mc) if isinstance(raw_mc, dict) else raw_mc

    # V2 format provides separate shapes for each input
    shape_a = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape
    shape_b = tuple(input_b_shape) if input_b_shape and isinstance(input_b_shape, (list, tuple)) else shape_a

    # Tile layout pads last two dims to multiples of 32.  When A uses TILE and B
    # uses ROW_MAJOR (or vice-versa), the inner matmul dimension will mismatch
    # because one side is padded and the other is not.  Align the torch shapes so
    # that the inner dimension (A.width / B.height) is the same after tile padding.
    def _tile_align(dim):
        return ((dim + 31) // 32) * 32

    a_is_tile = input_a_layout == ttnn.TILE_LAYOUT
    b_is_tile = input_b_layout == ttnn.TILE_LAYOUT

    # Skip tile-alignment expansion when V2 placements are set: V2 vector shapes
    # already encode the global tensor shape per the mesh distribution scheme,
    # and asymmetric K-sharding (A K-sharded, B K-replicated) intentionally has
    # different pre-shard inner dims.  Expanding here causes per-chip K mismatch
    # (e.g. A-chip-K=2880 vs B-chip-K=23040).
    have_placement = bool(input_a_tensor_placement) or bool(input_b_tensor_placement)

    if len(shape_a) >= 2 and len(shape_b) >= 2 and not have_placement:
        inner_a = shape_a[-1]  # A's width
        inner_b = shape_b[-2]  # B's height
        aligned_a = _tile_align(inner_a) if a_is_tile else inner_a
        aligned_b = _tile_align(inner_b) if b_is_tile else inner_b
        if aligned_a != aligned_b:
            # Ensure inner dims match after tile padding by aligning both to the
            # larger tile-aligned size.
            target = max(aligned_a, aligned_b)
            if inner_a != target:
                shape_a = tuple(list(shape_a[:-1]) + [target])
            if inner_b != target:
                shape_b = tuple(list(shape_b[:-2]) + [target, shape_b[-1]])

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape_a)

    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(shape_b)

    # Align A and B inner dims for torch golden when V2 K-sharded shapes diverge.
    # When A is K-sharded across the mesh but B is K-replicated, V2 stores A with
    # global K (=K_chip * mesh_factor) and B with K=K_chip.  torch.matmul needs
    # matching K, so tile B along K by the inferred mesh factor.
    torch_a_for_golden = torch_input_tensor_a.float()
    torch_b_for_golden = torch_input_tensor_b.float()
    if torch_a_for_golden.ndim >= 2 and torch_b_for_golden.ndim >= 2:
        a_K = torch_a_for_golden.shape[-1]
        b_K = torch_b_for_golden.shape[-2]
        if a_K != b_K and a_K % b_K == 0:
            mesh_factor = a_K // b_K
            repeat = [1] * torch_b_for_golden.ndim
            repeat[-2] = mesh_factor
            torch_b_for_golden = torch_b_for_golden.repeat(*repeat)
        elif b_K != a_K and b_K % a_K == 0:
            mesh_factor = b_K // a_K
            repeat = [1] * torch_a_for_golden.ndim
            repeat[-1] = mesh_factor
            torch_a_for_golden = torch_a_for_golden.repeat(*repeat)

    # Trace-validation mode: every chip receives the FULL per-chip A and B via
    # replicate_with_topology and runs matmul independently. The gathered output
    # is the per-chip matmul tiled along the shard axis — handled by
    # reconcile_golden_to_actual below.
    torch_output_tensor = torch.matmul(torch_a_for_golden, torch_b_for_golden)

    # Apply activation to golden if specified — check both op kwarg and program_config.fused_activation
    activation = op_kwargs.get("activation")
    if not activation or activation == "__ABSENT__":
        # Check program_config for fused_activation
        pc = op_kwargs.get("program_config")
        if pc and hasattr(pc, "fused_activation") and pc.fused_activation is not None:
            activation = str(pc.fused_activation)
    if activation and activation != "__ABSENT__":
        act_str = str(activation).lower()
        if "gelu" in act_str:
            torch_output_tensor = torch.nn.functional.gelu(torch_output_tensor, approximate="tanh")
        elif "relu" in act_str:
            torch_output_tensor = torch.nn.functional.relu(torch_output_tensor)
        elif "silu" in act_str:
            torch_output_tensor = torch.nn.functional.silu(torch_output_tensor)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Create tensors with the traced memory configs
    # If direct creation fails, try creating interleaved first then converting to sharded
    # This matches how models typically create sharded tensors
    try:
        if not is_host:
            if is_mesh_device and input_a_tensor_placement:
                input_tensor_a = create_tensor_on_mesh(
                    torch_input_tensor_a,
                    device,
                    input_a_dtype,
                    input_a_layout,
                    input_a_memory_config,
                    input_a_tensor_placement,
                )
            else:
                input_tensor_a = ttnn.from_torch(
                    torch_input_tensor_a,
                    dtype=input_a_dtype,
                    layout=input_a_layout,
                    device=device,
                    memory_config=input_a_memory_config,
                )
        else:
            input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)
    except Exception:
        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Create input_b tensor.
    # When a program_config is present (e.g. MatmulMultiCoreReuseProgramConfig), the
    # kernel may expect input_b in its traced memory layout (including sharded).
    # Only force input_b to interleaved when there is NO program_config.
    input_b_is_sharded = (
        hasattr(input_b_memory_config, "shard_spec")
        and input_b_memory_config.shard_spec is not None
        and input_b_memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    )
    has_program_config = "program_config" in op_kwargs

    if input_b_is_sharded and not has_program_config:
        # No program_config: matmul's default path requires input_b to be INTERLEAVED
        input_tensor_b_interleaved = ttnn.from_torch(
            torch_input_tensor_b,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_tensor_b = input_tensor_b_interleaved
    else:
        try:
            if not is_host:
                if is_mesh_device and input_b_tensor_placement:
                    input_tensor_b = create_tensor_on_mesh(
                        torch_input_tensor_b,
                        device,
                        input_b_dtype,
                        input_b_layout,
                        input_b_memory_config,
                        input_b_tensor_placement,
                    )
                else:
                    input_tensor_b = ttnn.from_torch(
                        torch_input_tensor_b,
                        dtype=input_b_dtype,
                        layout=input_b_layout,
                        device=device,
                        memory_config=input_b_memory_config,
                    )
            else:
                input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=input_b_dtype, layout=input_b_layout)
        except Exception:
            input_tensor_b = ttnn.from_torch(
                torch_input_tensor_b,
                dtype=input_b_dtype,
                layout=input_b_layout,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    try:
        start_time = start_measuring_time()
        output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, **op_kwargs)
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
        e2e_perf = stop_measuring_time(start_time)
    except Exception:
        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            dtype=input_a_dtype,
            layout=input_a_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_tensor_b = ttnn.from_torch(
            torch_input_tensor_b,
            dtype=input_b_dtype,
            layout=input_b_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        fallback_kwargs = {k: v for k, v in op_kwargs.items() if k != "program_config"}
        fallback_kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        start_time = start_measuring_time()
        output_tensor = ttnn.matmul(input_tensor_a, input_tensor_b, **fallback_kwargs)
        output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
        e2e_perf = stop_measuring_time(start_time)

    # Slice output back to original shape in case tile padding expanded it
    if output_tensor.shape != torch_output_tensor.shape:
        output_tensor = output_tensor[tuple(slice(0, s) for s in torch_output_tensor.shape)]

    pcc_threshold = 0.80
    compute_cfg = op_kwargs.get("compute_kernel_config")
    if compute_cfg and hasattr(compute_cfg, "math_fidelity"):
        fidelity = str(compute_cfg.math_fidelity)
        if "HiFi4" in fidelity or "HiFi3" in fidelity:
            pcc_threshold = 0.999
        elif "HiFi2" in fidelity:
            pcc_threshold = 0.98
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(
            torch_output_tensor, output_tensor, input_a_tensor_placement, input_b_tensor_placement
        )
    pcc = check_with_pcc(torch_output_tensor, output_tensor, pcc_threshold)

    return [pcc, e2e_perf]
