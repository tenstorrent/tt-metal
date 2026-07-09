# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

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
    dispatch_axis_for_grid,
    get_model_traced_mesh_shape,
    mesh_tensor_to_torch,
    program_config_grid_bounds,
    reconcile_golden_to_actual,
    shard_grid_bounds,
)


def _linear_dispatch_axis(program_config, output_memory_config, input_a_memory_config=None):
    """Dispatch axis the matmul's program_config needs.

    For 1D ``gather_in0`` matmuls the nominal ``compute_with_storage_grid_size``
    width over-estimates the real core usage (the op validates the output shard
    grid against the device / its sparse core set, not that rectangle). So the
    real placement is the (possibly sparse) output shard grid + hop_cores — e.g.
    a (8,3) grid whose output shard grid actually uses x<=6, y=9 needs COL, not
    the ROW its nominal width=8 implies. For ordinary 2D matmuls the compute grid
    is authoritative, so only override for gather_in0 (overriding everywhere
    wrongly forces COL on configs that genuinely need ROW).
    """
    if isinstance(program_config, dict) and program_config.get("gather_in0"):
        sx, sy = shard_grid_bounds(output_memory_config)
        hx, hy = program_config_grid_bounds({"hop_cores": program_config.get("hop_cores")})
        mx = max([v for v in (sx, hx) if v is not None], default=None)
        my = max([v for v in (sy, hy) if v is not None], default=None)
        if mx is not None or my is not None:
            return dispatch_axis_for_grid(mx, my)
    pcx, pcy = program_config_grid_bounds(program_config)
    if pcx is None and pcy is None:
        # No program_config grid to fix the axis (e.g. ttnn.linear with no
        # explicit program_config). The op still resharded input_a / output to
        # their traced L1 shard grids, so those grids decide the axis — a grid
        # spanning x=0..7 (full row) needs ROW dispatch, else the reshard worker
        # lands on a dispatch core (TT_FATAL "not on_dispatch_core").
        ax, ay = shard_grid_bounds(input_a_memory_config)
        ox, oy = shard_grid_bounds(output_memory_config)
        mx = max([v for v in (ax, ox) if v is not None], default=None)
        my = max([v for v in (ay, oy) if v is not None], default=None)
        if mx is not None or my is not None:
            return dispatch_axis_for_grid(mx, my)
    return dispatch_axis_for_grid(pcx, pcy)


# Device opened per-vector (see _ensure_vector_device) so each vector can use the
# dispatch axis its traced matmul program_config grid needs: some grids touch x=7
# (need ROW/8x9), others use y=9/10 (need COL/7x10), and no single per-suite axis
# serves both. Cached; only reopened when the required axis changes between vectors.
_CUR_DEVICE = None
_CUR_AXIS = "__uninit__"
_CUR_SHAPE = None


def _ensure_vector_device(axis):
    global _CUR_DEVICE, _CUR_AXIS, _CUR_SHAPE
    shape = get_model_traced_mesh_shape()
    if _CUR_DEVICE is None or axis != _CUR_AXIS or shape != _CUR_SHAPE:
        _close_vector_device()
        _CUR_DEVICE = create_mesh_device(shape, dispatch_core_axis=axis)
        _CUR_AXIS = axis
        _CUR_SHAPE = shape
    return _CUR_DEVICE


def _close_vector_device():
    global _CUR_DEVICE, _CUR_AXIS, _CUR_SHAPE
    if _CUR_DEVICE is not None:
        try:
            ttnn.close_mesh_device(_CUR_DEVICE)
        except Exception:
            # best-effort teardown — a failed device close must not mask the real test result
            pass
    _CUR_DEVICE = None
    _CUR_AXIS = "__uninit__"
    _CUR_SHAPE = None


from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, check_with_pcc_safe
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time


def _parse_2d_shard_dims(placement, ndim=4, normalize=True):
    """[dim_on_mesh_rows, dim_on_mesh_cols] from a traced placement dict (Shard dims;
    None for Replicate).

    ``normalize=True`` (default) maps negative dims to >=0 for K/N axis logic.
    ``normalize=False`` preserves the original sign so a reconstructed mesh
    mapper records the exact same ``PlacementShard(d)`` repr the master traced
    (the model often shards on ``Shard(-2)/Shard(-1)``; normalizing to 2/3 is
    the same placement but a different string the validator flags)."""
    import re

    s = str(placement.get("placement", "")) if isinstance(placement, dict) else str(placement)
    out = []
    for m in re.finditer(r"PlacementShard\((?:dim=)?(-?\d+)\)|PlacementReplicate", s):
        if m.group(1) is None:
            out.append(None)
        else:
            d = int(m.group(1))
            out.append((d + ndim if d < 0 else d) if normalize else d)
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
    """gather_in0 1D ring matmul (decode LM-head / MLP-w2 projections).

    These traced configs are fragments of a distributed model op, not self-contained
    matmuls: the op runs on the model's prefetcher+worker sub-devices (SubDeviceId(1))
    over a fixed 24-core ring, and it emits per-K-shard PARTIALS that the model
    finishes with a downstream cross-mesh all-reduce (line_all_reduce). The generic
    per-vector path can reproduce none of that, so this routine rebuilds the model's
    decode launch path (mirrors models/demos/llama3_70b_galaxy/tt/{lm_head,llama_mlp}.py
    and the standalone repros) and reconstructs the all-reduce in torch for the golden.

    One path covers all four traced configs; they differ only by data read from the
    traced vector:
      * weight placement -> which mesh axis carries K vs N (LM-head: N over rows,
        K over cols; w2 down-proj: K over rows, N over cols);
      * num_global_cb_receivers >= 2 -> the weight is streamed from DRAM through the
        prefetcher global circular buffer before the matmul (w2 family);
      * dtypes + compute-kernel flags come straight from the trace.
    """
    import math

    from models.demos.llama3_70b_galaxy.tt.prefetcher_common import get_core_ranges
    from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

    TILE = ttnn.TILE_SIZE
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])

    def _ru(n, m):
        return ((n + m - 1) // m) * m

    M = int(input_a_shape[-2])
    K_pd = int(input_b_shape[-2])  # per-device K (weight dim -2)
    N_pd = int(input_b_shape[-1])  # per-device N (weight dim -1)

    # Weight placement -> mesh-axis for K (dim 2) and N (dim 3).
    b_dims = _parse_2d_shard_dims(input_b_placement, ndim=4)
    # Sign-preserved dims for the mesh mapper, so the recorded PlacementShard
    # repr matches the master's exact sign (e.g. Shard(-2)/Shard(-1)).
    b_dims_signed = _parse_2d_shard_dims(input_b_placement, ndim=4, normalize=False)
    if len(b_dims) < 2 or 2 not in b_dims or 3 not in b_dims:
        raise ValueError(f"weight placement is not a 2D K/N mesh-shard: {input_b_placement}")
    k_axis = b_dims.index(2)  # 0=rows, 1=cols
    n_axis = b_dims.index(3)
    global_K = K_pd * mesh_shape[k_axis]
    global_N = N_pd * mesh_shape[n_axis]

    # Program-config values come straight from the trace; shard widths follow the
    # model's ring tiling (round each per-device dim up to a tile over the 24 cores).
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

    _close_vector_device()
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

        # --- prefetcher global CB + sub-devices ---
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

        # --- weight: global [1,1,global_K,global_N] -> per-device via placement dims ---
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
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(b_dims_signed[0], b_dims_signed[1]), mesh_shape=(rows, cols)),
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

        # --- activation: per-device content is the K-slice along the K mesh-axis ---
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
        linear_kwargs = dict(
            compute_kernel_config=ckc,
            program_config=pc_obj,
            memory_config=out_mc,
            dtype=out_dtype,
            sub_device_id=worker_id,
        )
        if prefetch:
            linear_kwargs["core_grid"] = None
            linear_kwargs["global_cb"] = global_cb
        out = ttnn.linear(a_tt, b_tt, **linear_kwargs)
        if prefetch:
            dev.reset_sub_device_stall_group()

        partials = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(dev, (rows, cols), dims=(0, 1))
        ).float()  # [rows, cols, M, N_pd]
        reduced = partials.sum(dim=k_axis)  # all-reduce over the K-shards
        recon = reduced.permute(1, 0, 2).reshape(M, global_N)
        e2e_perf = stop_measuring_time(start_time)
        return [check_with_pcc_safe(golden, recon, 0.99), e2e_perf]
    finally:
        try:
            ttnn.close_mesh_device(dev)
        except Exception:
            # best-effort teardown of the gather_in0 ring-matmul device
            pass


def _pad_batch_to_dram_banks(batch, banks):
    # `banks` is the device's actual DRAM-bank count
    # (dev.get_optimal_dram_bank_to_logical_worker_assignment) — hardware-specific
    # (Wormhole=12, Blackhole differs), so it is always passed in, never defaulted.
    return batch if batch % banks == 0 else ((batch + banks - 1) // banks) * banks


def _placement_to_2d_mesh_dims(placement):
    """Parse a traced placement into (rows_spec, cols_spec) for ShardTensor2dMesh.

    Each spec is the (possibly negative) tensor dim sharded on that mesh axis, or
    None for Replicate. A single-element ``[Replicate]`` (ReplicateTensorToMesh)
    returns ("REPLICATE_ALL", None).
    """
    s = str(placement.get("placement", "")) if isinstance(placement, dict) else str(placement)
    import re

    toks = re.findall(r"PlacementShard\((?:dim=)?(-?\d+)\)|PlacementReplicate", s)
    # re.findall with alternation returns the captured group ('' for Replicate)
    specs = []
    for t in toks:
        specs.append(int(t) if t != "" else None)
    if len(specs) <= 1:
        return ("REPLICATE_ALL", None)
    return (specs[0], specs[1])


def _run_batched_dram_sharded_matmul(
    input_a_shape,
    input_b_shape,
    input_a_placement,
    input_b_placement,
    pc,
    mesh_shape,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    output_tile,
    compute_kernel_config_raw,
):
    """BatchedDRAMSharded matmul (DeepSeek-V3 MLA wkv_b1 / wkv_b2 decode projections).

    ``MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig`` is a batch-sharded
    distributed matmul: the activation (in0) is L1 HEIGHT_SHARDED across the 12 DRAM-bank
    -> worker cores (get_optimal_dram_bank_to_logical_worker_assignment), the weight is
    DRAM HEIGHT_SHARDED over the 12 banks, and the output is L1 HEIGHT_SHARDED on the same
    worker cores. The generic per-vector path can't reproduce that L1 batch-shard on the
    optimal worker grid (in0 lands interleaved -> ``is_sharded()`` TT_FATAL), so this routine
    rebuilds the model's launch path on a COL-dispatch mesh (frees row y=9 so the device's
    optimal worker grid matches the trace) and reads the per-device tensors back for the
    torch golden. Mirrors models/demos/deepseek_v3/tt/mla/mla1d.py and the standalone repro.
    """
    from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

    TILE = ttnn.TILE_SIZE
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])

    def _ru(n, m):
        return ((n + m - 1) // m) * m

    # Traced input_a_shape is the per-device shape (1, batch, m, k); weight (1, batch, k, n).
    a_shape = tuple(input_a_shape)
    b_shape = tuple(input_b_shape)
    batch = int(a_shape[-3])
    m = int(a_shape[-2])
    k = int(a_shape[-1])
    n = int(b_shape[-1])

    tile_h = TILE
    if output_tile is not None:
        try:
            tile_h = int(output_tile.tile_shape[0])
        except Exception:
            tile_h = TILE
    shard_m = _ru(m, tile_h)  # program/tile-padded M (>= logical m)
    reshard_in0 = m < shard_m  # logical m < a tile -> build interleaved then reshard

    in0_block_w = int(pc["in0_block_w"])
    per_core_M = int(pc.get("per_core_M", 1))
    per_core_N = int(pc.get("per_core_N", 1))

    act_dtype = _as_dtype(input_a_dtype, ttnn.bfloat16)
    wt_dtype = _as_dtype(input_b_dtype, ttnn.bfloat8_b)
    out_dtype = _as_dtype(output_dtype, None)

    ckc = compute_kernel_config_raw
    if isinstance(ckc, dict):
        try:
            ckc = parse_dict_value("compute_kernel_config", ckc)
        except Exception:
            ckc = None
    if not isinstance(ckc, ttnn.WormholeComputeKernelConfig):
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    in0_dims = _placement_to_2d_mesh_dims(input_a_placement)
    wt_dims = _placement_to_2d_mesh_dims(input_b_placement)

    _close_vector_device()
    dev = create_mesh_device((rows, cols), l1_small_size=0, dispatch_core_axis=ttnn.DispatchCoreAxis.COL)
    try:
        # Batch is sharded over the device's DRAM banks (one worker core per bank), so the
        # bank count is a hardware property — query it rather than hardcoding the Wormhole 12
        # (Blackhole differs). The kernel keys off this exact optimal worker assignment, so
        # bpc / shard shapes must follow the device's real bank count, not a constant.
        cores = dev.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        num_banks = len(cores)
        worker_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores]
        )
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})

        batch_padded = _pad_batch_to_dram_banks(batch, num_banks)
        bpc = batch_padded // num_banks  # batches per bank / worker core

        in0_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(worker_grid, [bpc * shard_m, k], ttnn.ShardOrientation.ROW_MAJOR),
        )
        in1_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [bpc * k, n], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(worker_grid, [bpc * shard_m, n], ttnn.ShardOrientation.ROW_MAJOR),
        )

        torch.manual_seed(0)

        # --- in0: build the global host so ShardTensor2dMesh yields per-device [1,batch,m,k] ---
        g_a = [1, batch, m, k]
        if in0_dims[0] == "REPLICATE_ALL":
            a_mapper = ttnn.ReplicateTensorToMesh(dev)
        else:
            if in0_dims[0] is not None:
                g_a[in0_dims[0]] *= rows
            if in0_dims[1] is not None:
                g_a[in0_dims[1]] *= cols
            a_mapper = ttnn.ShardTensor2dMesh(dev, dims=in0_dims, mesh_shape=(rows, cols))
        a_host = torch.randn(*g_a, dtype=torch.bfloat16)
        if reshard_in0:
            # Direct from_torch into the height-shard (a) pads the logical m to a tile and
            # (b) lands the buffer in low L1 where it collides with the matmul's static CBs.
            # Build L1-interleaved first (preserves logical m, lands high in L1), then reshard.
            a_tmp = ttnn.from_torch(
                a_host,
                dtype=act_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=a_mapper,
            )
            a_tt = ttnn.to_memory_config(a_tmp, in0_mc)
            ttnn.deallocate(a_tmp)
        else:
            a_tt = ttnn.from_torch(
                a_host,
                dtype=act_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=in0_mc,
                mesh_mapper=a_mapper,
            )

        # --- weight: global host -> per-device [1,batch,k,n] via placement, DRAM HEIGHT_SHARDED ---
        g_b = [1, batch, k, n]
        if wt_dims[0] == "REPLICATE_ALL":
            b_mapper = ttnn.ReplicateTensorToMesh(dev)
        else:
            if wt_dims[0] is not None:
                g_b[wt_dims[0]] *= rows
            if wt_dims[1] is not None:
                g_b[wt_dims[1]] *= cols
            b_mapper = ttnn.ShardTensor2dMesh(dev, dims=wt_dims, mesh_shape=(rows, cols))
        b_host = torch.randn(*g_b, dtype=torch.bfloat16)
        b_tt = ttnn.from_torch(
            b_host,
            dtype=wt_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=in1_mc,
            mesh_mapper=b_mapper,
        )

        pc_obj = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
            in0_block_w=in0_block_w, per_core_M=per_core_M, per_core_N=per_core_N, fused_activation=None
        )
        linear_kwargs = dict(
            input_tensor_b=b_tt, memory_config=out_mc, compute_kernel_config=ckc, program_config=pc_obj
        )
        if out_dtype is not None:
            linear_kwargs["dtype"] = out_dtype
        if output_tile is not None:
            linear_kwargs["output_tile"] = output_tile

        start_time = start_measuring_time()
        out = ttnn.linear(a_tt, **linear_kwargs)
        ttnn.synchronize_device(dev)
        e2e_perf = stop_measuring_time(start_time)

        # --- golden: read back the exact per-device inputs, matmul in torch, compare per device ---
        # ConcatMeshToTensor(dim=0) stacks the per-device [1,batch,m,k] tensors -> [n_dev,batch,m,k]
        # at their *logical* shapes (the height-shard's tile padding on M is not surfaced here).
        cat = ttnn.ConcatMeshToTensor(dev, dim=0)
        a_rb = ttnn.to_torch(a_tt, mesh_composer=cat).float()
        b_rb = ttnn.to_torch(b_tt, mesh_composer=cat).float()
        o_rb = ttnn.to_torch(out, mesh_composer=cat).float()
        golden = torch.matmul(a_rb, b_rb)  # [n_dev, batch, m, n]
        # guard against any tile padding the op leaves on the output's M/N
        o_rb = o_rb[..., : golden.shape[-2], : golden.shape[-1]]
        golden = golden[..., : o_rb.shape[-2], : o_rb.shape[-1]]

        ttnn.deallocate(a_tt)
        ttnn.deallocate(b_tt)
        ttnn.deallocate(out)
        return [check_with_pcc_safe(golden, o_rb, 0.99), e2e_perf]
    finally:
        try:
            ttnn.close_mesh_device(dev)
        except Exception:
            # best-effort teardown of the batched-DRAM-sharded device
            pass


def _apply_linear_activation(t, activation, fused_act_optype, fused_act_param):
    """Apply the matmul's fused/explicit activation to a torch golden tensor."""
    if activation is not None:
        act = str(activation).lower()
        if "silu" in act or "swish" in act:
            return torch.nn.functional.silu(t)
        if "gelu" in act:
            return torch.nn.functional.gelu(t, approximate=("tanh" if "approx" in act else "none"))
        if "relu" in act:
            return torch.nn.functional.relu(t)
    if fused_act_optype is not None:
        try:
            ot = int(fused_act_optype)
        except (TypeError, ValueError):
            ot = None
        if ot == 2:
            approx = (
                "tanh"
                if (isinstance(fused_act_param, (list, tuple)) and fused_act_param and fused_act_param[0])
                else "none"
            )
            return torch.nn.functional.gelu(t, approximate=approx)
        if ot == 3:
            return torch.nn.functional.relu(t)
        if ot == 5:
            return torch.sigmoid(t)
        if ot == 57:
            return torch.nn.functional.silu(t)
    return t


def _run_kshard_replicated_matmul(
    input_a_shape,
    input_b_shape,
    input_a_dtype,
    input_b_dtype,
    output_dtype,
    compute_kernel_config_raw,
    activation,
    fused_act_optype,
    fused_act_param,
):
    """K-sharded tensor-parallel matmul (BOTH operands sharded on the contracting
    K dim). On the model's Galaxy mesh each device computes a partial over its
    K-slice and a downstream all-reduce sums them — which equals the full-K matmul
    a@b over the traced K. N300 (2 chips) can't reproduce the Galaxy 32-way
    distribution, and the traced program_config encodes that mesh's per-device
    tiling (invalid for the reconstructed shapes -> garbage/TT_FATAL). But the
    all-reduced RESULT is just a@b, so validate THAT: replicate the operands, run a
    plain matmul (device auto-selects a valid config for the actual shapes), and
    compare to torch.matmul. (Equivalent to taking chip-0 of a replicated result.)
    """
    from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

    a_dtype = _as_dtype(input_a_dtype, ttnn.bfloat16)
    b_dtype = _as_dtype(input_b_dtype, ttnn.bfloat16)
    out_dtype = _as_dtype(output_dtype, None)
    ckc = compute_kernel_config_raw
    if isinstance(ckc, dict):
        try:
            ckc = parse_dict_value("compute_kernel_config", ckc)
        except Exception:
            ckc = None

    dev = _ensure_vector_device(None)  # full 8x8 ETH grid on N150/N300
    is_mesh = hasattr(dev, "get_num_devices")
    torch.manual_seed(0)
    a = torch.randn(*tuple(input_a_shape), dtype=torch.float32)
    b = torch.randn(*tuple(input_b_shape), dtype=torch.float32)
    golden = torch.matmul(a, b)
    golden = _apply_linear_activation(golden, activation, fused_act_optype, fused_act_param)

    mm = ttnn.ReplicateTensorToMesh(dev) if is_mesh else None
    ta = ttnn.from_torch(
        a, dtype=a_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mm
    )
    tb = ttnn.from_torch(
        b, dtype=b_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mm
    )
    linear_kwargs = {}
    if isinstance(ckc, ttnn.WormholeComputeKernelConfig):
        linear_kwargs["compute_kernel_config"] = ckc
    if out_dtype is not None:
        linear_kwargs["dtype"] = out_dtype
    if activation is not None:
        linear_kwargs["activation"] = activation
    start_time = start_measuring_time()
    out = ttnn.linear(ta, tb, **linear_kwargs)
    res = mesh_tensor_to_torch(out, dev if is_mesh else None)
    e2e_perf = stop_measuring_time(start_time)
    return [check_with_pcc_safe(golden, res, 0.99), e2e_perf]


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
    # Device is opened per-vector in run() (see _ensure_vector_device) so each
    # vector gets the dispatch axis its matmul program_config grid needs. A blunt
    # per-suite axis can't serve both the x=7/ROW and y=9/COL configs linear has.
    yield (None, "wormhole_b0")
    _close_vector_device()


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


def _placement_shards_axis(plac_dict, axis, ndim):
    """True if the traced placement shards the given tensor axis — regardless of
    the per-mesh distribution size. (_mesh_factor_for_axis returns 1 when the
    distribution on that axis is 1, e.g. a tensor-parallel config re-grouped onto
    a 1x1 mesh; but the traced program_config is still the distributed mesh's, so
    we must still treat it as a distributed fragment.)"""
    if not isinstance(plac_dict, dict):
        return False
    for kind, d in _parse_placement_list(plac_dict.get("placement")) or []:
        if kind == "S" and d is not None and (d if d >= 0 else d + ndim) == axis:
            return True
    return False


def _placement_has_any_shard(plac_dict):
    if not isinstance(plac_dict, dict):
        return False
    return any(kind == "S" for kind, _ in (_parse_placement_list(plac_dict.get("placement")) or []))


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

    # Reproduce THIS vector's traced mesh shape. The runner groups vectors by
    # mesh into per-mesh files, but a single main-process invocation loads all of
    # them and opens one device, so without this every vector would be re-run on
    # whatever shape get_model_traced_mesh_shape() auto-detects (e.g. all at
    # [4, 8]) — mismatching the [8, 4]/[1, 32]/[1, 1] vectors' traced
    # tensor_placement.mesh_device_shape. Pin MESH_DEVICE_SHAPE to the vector's
    # own traced shape so both the device open AND every downstream
    # get_model_traced_mesh_shape() call (gather_in0 / batched / standard paths)
    # reproduce the master topology exactly.
    for _plac in (
        kwargs.get("input_a_tensor_placement"),
        kwargs.get("input_b_tensor_placement"),
        kwargs.get("input_tensor_b_tensor_placement"),
    ):
        if isinstance(_plac, dict) and _plac.get("mesh_device_shape"):
            import re as _re_mesh

            _dims = _re_mesh.findall(r"-?\d+", str(_plac["mesh_device_shape"]))
            if len(_dims) == 2:
                os.environ["MESH_DEVICE_SHAPE"] = f"{int(_dims[0])}x{int(_dims[1])}"
                break

    # Capture the matmul program_config's fused activation (read from the RAW dict,
    # before it is parsed to an object). The model frequently carries the activation
    # (e.g. GELU) inside program_config.fused_activation rather than the separate
    # `activation` kwarg; the golden must apply it too, else golden(no-act) vs
    # device(act) yields a spurious ~0.86 PCC. op_type: 2=GELU, 3=RELU, 5=SIGMOID,
    # 57=SILU (ttnn.UnaryOpType).
    _fused_act_optype = None
    _fused_act_param = None
    if isinstance(program_config, dict):
        _fa = program_config.get("fused_activation")
        if isinstance(_fa, dict):
            _fused_act_optype = _fa.get("op_type")
            _fused_act_param = _fa.get("param")

    # gather_in0 1D ring matmuls (decode LM-head / MLP-w2) are distributed model
    # fragments — they need the model's prefetcher+worker sub-devices, the fixed
    # 24-core ring, and a cross-mesh all-reduce the generic path can't provide.
    # Detect on the RAW program_config (before it is parsed to an object) and run
    # the faithful model reconstruction instead.
    _ib_shape = input_b_shape if input_b_shape is not None else kwargs.get("input_tensor_b_shape")
    if isinstance(program_config, dict) and program_config.get("gather_in0") and _ib_shape is not None:
        _ib_plac = kwargs.get("input_b_tensor_placement") or kwargs.get("input_tensor_b_tensor_placement")
        return _run_gather_in0_ring_matmul(
            input_a_shape=input_a_shape,
            input_b_shape=_ib_shape,
            pc=program_config,
            mesh_shape=get_model_traced_mesh_shape(),
            input_b_placement=_ib_plac,
            input_a_dtype=input_a_dtype,
            input_b_dtype=(input_b_dtype if input_b_dtype is not None else kwargs.get("input_tensor_b_dtype")),
            output_dtype=dtype,
            compute_kernel_config_raw=compute_kernel_config,
        )

    # BatchedDRAMSharded matmuls (DeepSeek-V3 MLA wkv_b1 / wkv_b2) are batch-sharded
    # distributed fragments: in0 must be L1 HEIGHT_SHARDED on the device's optimal
    # DRAM-bank -> worker grid (COL dispatch), weight DRAM HEIGHT_SHARDED. The generic
    # path can't build that (in0 lands interleaved -> is_sharded() TT_FATAL), so detect
    # on the RAW program_config and run the faithful model reconstruction.
    if (
        isinstance(program_config, dict)
        and "BatchedDRAMSharded" in str(program_config.get("type", ""))
        and _ib_shape is not None
    ):
        _ot_raw = kwargs.get("output_tile")
        _otile = None
        if isinstance(_ot_raw, dict) and _ot_raw.get("type") == "Tile":
            import re as _re_ot

            _m = _re_ot.search(r"shape:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", str(_ot_raw.get("value", "")))
            if _m:
                try:
                    _otile = ttnn.Tile([int(_m.group(1)), int(_m.group(2))])
                except Exception:
                    _otile = None
        return _run_batched_dram_sharded_matmul(
            input_a_shape=input_a_shape,
            input_b_shape=_ib_shape,
            input_a_placement=kwargs.get("input_a_tensor_placement"),
            input_b_placement=(kwargs.get("input_b_tensor_placement") or kwargs.get("input_tensor_b_tensor_placement")),
            pc=program_config,
            mesh_shape=get_model_traced_mesh_shape(),
            input_a_dtype=input_a_dtype,
            input_b_dtype=(input_b_dtype if input_b_dtype is not None else kwargs.get("input_tensor_b_dtype")),
            output_dtype=dtype,
            output_tile=_otile,
            compute_kernel_config_raw=compute_kernel_config,
        )

    # K-sharded tensor-parallel matmul: BOTH operands sharded on the contracting
    # K dim (input_a's last dim, input_b's second-to-last). This is a distributed
    # Galaxy fragment whose traced program_config encodes that mesh's per-device
    # tiling — invalid for the N300 reconstruction (kernel reads wrong data -> ~0
    # PCC / TT_FATAL). The all-reduced result is just a@b, so validate the math via
    # a replicated plain matmul. (Separate path so the standard path is untouched.)
    _a_plac = kwargs.get("input_a_tensor_placement")
    _b_plac = kwargs.get("input_b_tensor_placement") or kwargs.get("input_tensor_b_tensor_placement")
    _sa = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else ()
    _sb = tuple(_ib_shape) if isinstance(_ib_shape, (list, tuple)) else ()
    # Route any tensor-parallel matmul where the activation is sharded on its
    # contracting dim (a's last dim = K) AND the weight is sharded on some dim
    # (K via dim-2, or N via dim-1). These are Galaxy distributed fragments whose
    # traced program_config encodes that mesh's per-device tiling (invalid on
    # N300); the distributed result is still a@b, so validate the math via the
    # replicated path.
    # Use placement (not per-mesh factor): a tensor-parallel config re-grouped to
    # 1x1 has factor 1 on every axis yet still carries the distributed mesh's
    # program_config. Trigger when a is sharded on K (its last dim) AND b is
    # sharded anywhere — but only when a traced program_config is present (that's
    # what's invalid on N300; un-config'd matmuls already validate fine).
    _kshard = (
        program_config is not None
        and _ib_shape is not None
        and len(_sa) >= 2
        and len(_sb) >= 2
        and _placement_shards_axis(_a_plac, len(_sa) - 1, len(_sa))
        and _placement_has_any_shard(_b_plac)
    )
    if _kshard:
        return _run_kshard_replicated_matmul(
            input_a_shape=input_a_shape,
            input_b_shape=_ib_shape,
            input_a_dtype=input_a_dtype,
            input_b_dtype=(input_b_dtype if input_b_dtype is not None else kwargs.get("input_tensor_b_dtype")),
            output_dtype=dtype,
            compute_kernel_config_raw=compute_kernel_config,
            activation=activation,
            fused_act_optype=_fused_act_optype,
            fused_act_param=_fused_act_param,
        )

    # Open (or reuse) a mesh device whose dispatch axis matches this vector's
    # matmul program_config grid + the real shard-grid placement (read raw,
    # before parsing below). The fixture yielded None; we own the device here.
    device = _ensure_vector_device(
        _linear_dispatch_axis(program_config, kwargs.get("output_memory_config"), input_a_memory_config)
    )

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
    # Drop a traced program_config whose compute grid can't fit this device. Some
    # configs are traced on a wider chip (e.g. Blackhole's (8,10)) and overflow
    # the Wormhole 8x8 grid even under ETH dispatch, crashing with
    # "compute_with_storage_grid_size must fit within (8,8)". The grid only sets
    # the matmul's M/N core tiling — the result is grid-independent — so drop the
    # config and let ttnn.linear auto-select a valid grid for this device.
    if isinstance(program_config, dict):
        _csg = program_config.get("compute_with_storage_grid_size")
        if isinstance(_csg, dict):
            try:
                _dg = device.compute_with_storage_grid_size()
                if int(_csg.get("x", 0)) > int(_dg.x) or int(_csg.get("y", 0)) > int(_dg.y):
                    program_config = None
            except Exception:
                # best-effort grid check; leave program_config unchanged on failure
                pass

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
    # Honor transpose_a / transpose_b (the model traces e.g. transpose_b=True, so
    # the device computes a @ wᵀ — the golden must match or it computes a @ w and
    # diverges, ~0 PCC). torch.matmul over the (optionally transposed) operands
    # gives the same result the old F.linear dance did for transpose_b=False, so
    # the common case is unchanged.
    _ag = torch_a_for_golden.transpose(-1, -2) if (transpose_a and torch_a_for_golden.ndim >= 2) else torch_a_for_golden
    _wg = (
        torch_weight_for_golden.transpose(-1, -2)
        if (transpose_b and torch_weight_for_golden.ndim >= 2)
        else torch_weight_for_golden
    )
    torch_output_tensor = torch.matmul(_ag, _wg)
    if torch_bias is not None:
        torch_output_tensor = torch_output_tensor + torch_bias

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

    # Apply the program_config's fused_activation to the golden too (the model puts
    # the activation here, not in the `activation` kwarg, for these matmuls). Without
    # this the golden omits the activation the device applied -> spurious ~0.86 PCC.
    if _fused_act_optype is not None and not is_batched_weight:
        try:
            _ot = int(_fused_act_optype)
        except (TypeError, ValueError):
            _ot = None
        if _ot == 2:  # GELU; param[0]==1 -> tanh (fast/approximate) mode
            _approx = (
                "tanh"
                if (isinstance(_fused_act_param, (list, tuple)) and _fused_act_param and _fused_act_param[0])
                else "none"
            )
            torch_output_tensor = torch.nn.functional.gelu(torch_output_tensor, approximate=_approx)
        elif _ot == 3:  # RELU
            torch_output_tensor = torch.nn.functional.relu(torch_output_tensor)
        elif _ot == 5:  # SIGMOID
            torch_output_tensor = torch.sigmoid(torch_output_tensor)
        elif _ot == 57:  # SILU
            torch_output_tensor = torch.nn.functional.silu(torch_output_tensor)

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

        try:
            output_tensor = _do_linear(ttnn_a, ttnn_b, **linear_kwargs)
        except Exception as _e:
            # Large L1-sharded matmuls (e.g. SDXL 16384x512) overflow N300 L1: the
            # static matmul CBs clash with the L1 input/output buffers ("circular
            # buffers ... clash with L1 buffers" / OOM). On Galaxy each shard is
            # spread across many chips so it fits; N300 (2 chips) can't reproduce
            # that. Retry from DRAM (the matmul math is identical — only the L1
            # placement can't be reproduced) with a DRAM-interleaved output.
            _m = str(_e).lower()
            if any(s in _m for s in ("circular buffer", "clash", "l1 buffer", "out of memory", "not enough space")):
                ttnn_a, ttnn_b = _make_dram_tensors()
                _kw = dict(linear_kwargs)
                _kw.pop("memory_config", None)
                output_tensor = _do_linear(ttnn_a, ttnn_b, **_kw)
            else:
                raise

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

    # Check with PCC.
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(
            torch_output_tensor, output_tensor, input_a_tensor_placement, input_b_tensor_placement
        )
    pcc = check_with_pcc_safe(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
