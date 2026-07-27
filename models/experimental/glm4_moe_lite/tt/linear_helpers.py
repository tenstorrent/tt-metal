# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul / linear projection helpers for GLM-4.7-Flash decode and prefill.

Extracted from decoder_layer_tt.py. These were originally nested closures inside
the 2000+ line decode function. Now they take explicit arguments (device, config)
instead of capturing outer scope variables.
"""

from __future__ import annotations

import math
import os
from typing import Any

import ttnn
from models.experimental.glm4_moe_lite.tt.runtime_config import Glm4RuntimeConfig

# Width of the GlobalCB prefetcher's worker SubDevice (columns 0-5 of the 8x9 grid).
# Matches prefetcher_setup.get_glm_core_ranges; kept here so matmul program configs can
# be clamped without importing the prefetcher module on the hot path.
WORKER_GRID_X = 6

# Memoised worker CoreRangeSets, keyed by device grid. Built once per grid shape rather
# than per call: the decode path asks for this at ~30 sites x 47 layers per token.
_WORKER_SCG_CACHE: dict[tuple[int, int], Any] = {}


def worker_sub_core_grids(device: Any, cfg: Glm4RuntimeConfig) -> Any | None:
    """CoreRangeSet of the prefetcher's worker region, or None when prefetch is off.

    Pass as `sub_core_grids=` to permute/transpose. Those ops size their grid from the
    full device grid, which under the prefetcher's SubDevice is rejected with
    "Kernel group cores do not match sub device cores"; sub_core_grids was added to ttnn
    for this. Returning None keeps the default full-grid behaviour when prefetch is off,
    so call sites need no conditional.
    """
    if not cfg.prefetch:
        return None
    grid = device.compute_with_storage_grid_size()
    key = (int(grid.x), int(grid.y))
    scg = _WORKER_SCG_CACHE.get(key)
    if scg is None:
        scg = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(WORKER_GRID_X, key[0]) - 1, key[1] - 1))]
        )
        _WORKER_SCG_CACHE[key] = scg
    return scg


def compute_1d_prog_cfg(
    device: Any,
    b_weight: ttnn.Tensor,
    m_total: int,
    *,
    in0_block_w: int | None = None,
    max_grid_x: int | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Compute 1D multicast program config for decode matmuls (M <= 1 tile).

    max_grid_x: clamp the grid's width. Required under the GlobalCB prefetcher's worker
      SubDevice (columns 0-5): this config is otherwise derived from the FULL device grid
      (8x9 on Galaxy), so the resulting kernel group spills onto the sender columns and
      every such matmul fails with
          TT_FATAL: Kernel group cores do not match sub device cores for TENSIX
      Clamping to 6 keeps the grid inside the worker rectangle.
    """
    K = int(b_weight.shape[-2])
    N = int(b_weight.shape[-1])
    m_tiles = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    k_tiles = (K + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    n_tiles = (N + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    if max_grid_x is not None:
        grid_x = min(grid_x, int(max_grid_x))
    num_cores = grid_x * grid_y

    if n_tiles < grid_x:
        grid_x = n_tiles
        grid_y = 1
        num_cores = n_tiles
    elif n_tiles < num_cores:
        grid_y = max(1, n_tiles // grid_x)
        num_cores = grid_x * grid_y

    per_core_N = max(1, math.ceil(n_tiles / num_cores))

    in0_bw = 1
    requested_in0_bw = (
        int(in0_block_w)
        if in0_block_w is not None
        else int(os.environ.get("GLM4_MOE_LITE_EXPLICIT_IN0_BLOCK_W", "0").strip() or "0")
    )
    # All current GLM decode projection K dimensions are divisible by 8. This
    # block width measured 54.2 -> 53.4 ms/token on Galaxy B1 versus width 4.
    candidates = (requested_in0_bw,) if requested_in0_bw > 0 else (8, 4, 3, 2)
    for candidate in candidates:
        if k_tiles % candidate == 0:
            in0_bw = candidate
            break

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def mlp_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """General-purpose linear for MLP and small matmuls.

    Applies the MLP compute kernel config and optional 1D program config
    for decode-sized (M=1) matmuls.
    """
    kwargs: dict[str, object] = {}
    mc = memory_config if memory_config is not None else cfg.decode_act_mc
    if mc is not None:
        kwargs["memory_config"] = mc
    ckc = cfg.mlp_compute_kernel_config()
    kwargs["compute_kernel_config"] = ckc
    if cfg.explicit_prog_cfg:
        m_total = 1
        for i in range(len(a.shape) - 1):
            m_total *= int(a.shape[i])
        b_batch = 1
        for i in range(len(b.shape) - 2):
            b_batch *= int(b.shape[i])
        if m_total <= ttnn.TILE_SIZE and b_batch == 1:
            # WORKER_GRID_X when prefetch is on: the prefetcher's worker SubDevice is
            # columns 0-5, and an unclamped grid spills onto the sender columns.
            kwargs["program_config"] = compute_1d_prog_cfg(
                device, b, m_total, max_grid_x=WORKER_GRID_X if cfg.prefetch else None
            )
    return ttnn.linear(a, b, **kwargs)


def tp_row_parallel_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
) -> ttnn.Tensor:
    """Row-parallel matmul for TP-sharded weights.

    Partitions activation along last dim across TP axis, does local matmul,
    then all_reduce to sum partial dot products.
    """
    a_tp = ttnn.mesh_partition(a, dim=3, cluster_axis=cfg.tp_axis)
    out = mlp_linear(a_tp, b, device=device, cfg=cfg)
    ttnn.deallocate(a_tp, force=False)
    out_reduced = ttnn.all_reduce(
        out,
        num_links=cfg.ccl_num_links,
        topology=cfg.ccl_topology,
        cluster_axis=cfg.tp_axis,
        memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(out, force=False)
    return out_reduced


# ---------------------------------------------------------------------------
# DRAM-sharded matmul helpers (decode-only perf optimization from DeepSeek V3)
# ---------------------------------------------------------------------------

_DS_BATCH = 32  # padded batch size for decode (TILE_SIZE)

_DS_CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def _ds_act_mc(device: Any, width: int) -> ttnn.MemoryConfig:
    """Create WIDTH_SHARDED L1 activation config for DRAM-sharded matmul."""
    from models.demos.deepseek_v3.utils.config_helpers import get_activation_sharding_core_counts_for_dram_matmul

    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    cores = max(get_activation_sharding_core_counts_for_dram_matmul(width, max_cores))
    return ttnn.create_sharded_memory_config_(
        shape=(_DS_BATCH, width // cores),
        core_grid=ttnn.num_cores_to_corerangeset(
            cores,
            ttnn.CoreCoord(grid.x, grid.y),
            row_wise=True,
        ),
        strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
        use_height_and_width_as_shard_shape=True,
    )


def dram_sharded_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
) -> ttnn.Tensor:
    """DRAM-sharded matmul for decode. Weight b must be in DRAM WIDTH_SHARDED format."""
    from models.demos.deepseek_v3.utils.config_helpers import (
        get_activation_sharding_core_counts_for_dram_matmul,
        get_dram_sharded_matmul_config,
    )

    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    K = int(b.shape[2])
    N = int(b.shape[3])
    input_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K, max_cores))
    output_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N, max_cores))

    a_sharded = ttnn.to_memory_config(a, _ds_act_mc(device, K))

    prog_cfg = get_dram_sharded_matmul_config(
        m=_DS_BATCH,
        k=K,
        n=N,
        input_num_shards=input_cores,
        output_num_shards=output_cores,
    )

    result = ttnn.linear(
        a_sharded,
        b,
        program_config=prog_cfg,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=_DS_CKC,
    )
    ttnn.deallocate(a_sharded, force=False)
    result_dram = ttnn.to_memory_config(result, cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(result, force=False)
    return result_dram


def dram_sharded_mlp(
    x: ttnn.Tensor,
    w_gate: ttnn.Tensor,
    w_up: ttnn.Tensor,
    w_down: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
) -> ttnn.Tensor:
    """Fused gate->silu->up->mul->down MLP entirely in L1 WIDTH_SHARDED.

    Follows DeepSeek V3 decode MLP pattern: reshard input once, keep all
    intermediates in L1 WIDTH_SHARDED, only move final output to DRAM.
    """
    from models.demos.deepseek_v3.utils.config_helpers import (
        get_activation_sharding_core_counts_for_dram_matmul,
        get_dram_sharded_matmul_config,
    )

    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    K_gate = int(w_gate.shape[2])
    N_gate = int(w_gate.shape[3])
    K_down = int(w_down.shape[2])
    N_down = int(w_down.shape[3])

    input_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K_gate, max_cores))
    inner_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N_gate, max_cores))
    output_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N_down, max_cores))

    x_sharded = ttnn.to_memory_config(x, _ds_act_mc(device, K_gate))

    gate_cfg = get_dram_sharded_matmul_config(
        m=_DS_BATCH,
        k=K_gate,
        n=N_gate,
        input_num_shards=input_cores,
        output_num_shards=inner_cores,
    )

    gate = ttnn.linear(
        x_sharded,
        w_gate,
        program_config=gate_cfg,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=_DS_CKC,
    )

    up = ttnn.linear(
        x_sharded,
        w_up,
        program_config=gate_cfg,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=_DS_CKC,
    )
    ttnn.deallocate(x_sharded, force=False)

    x_ff = ttnn.mul(
        gate, up, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
    )
    ttnn.deallocate(gate, force=False)
    ttnn.deallocate(up, force=False)

    down_cfg = get_dram_sharded_matmul_config(
        m=_DS_BATCH,
        k=K_down,
        n=N_down,
        input_num_shards=inner_cores,
        output_num_shards=output_cores,
    )
    result = ttnn.linear(
        x_ff,
        w_down,
        program_config=down_cfg,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=_DS_CKC,
    )
    ttnn.deallocate(x_ff, force=False)

    result_dram = ttnn.to_memory_config(result, cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(result, force=False)
    return result_dram


def ring_prefetch_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    program_config: Any,
    input_mem_cfg: ttnn.MemoryConfig,
    output_mem_cfg: ttnn.MemoryConfig,
    global_cb: Any,
    sub_device_id: Any,
    compute_kernel_config: Any,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """gather_in0 ring matmul consuming weights streamed into a GlobalCB.

    Three things differ from the ordinary decode path and all three matter:

    - `ttnn.matmul`, NOT `ttnn.linear`. gather_in0 has no bias support; any bias must
      be applied separately afterwards. (Flash's attention projections are bias-free,
      so there is nothing to re-apply here.)
    - the activation must be resharded to WIDTH_SHARDED on exactly the ring cores, so
      the matmul's CB core set is a subset of global_cb.all_cores().
    - the output memory config is explicit and also ring-sharded.

    The program config must come from Glm4MoeLitePrefetcherSetup.make_ring_config,
    which enforces that the ring width divides both matmul dimensions. A ring that
    does not divide deadlocks on device rather than raising -- never hand-build one.
    """
    a_ring = ttnn.to_memory_config(a, input_mem_cfg)
    out = ttnn.matmul(
        a_ring,
        b,
        program_config=program_config,
        memory_config=output_mem_cfg,
        compute_kernel_config=compute_kernel_config,
        global_cb=global_cb,
        sub_device_id=sub_device_id,
        dtype=dtype,
    )
    ttnn.deallocate(a_ring, force=False)
    return out


def attn_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    force_no_tp: bool = False,
    prefetch: Any | None = None,
) -> ttnn.Tensor:
    """Attention projection linear. Routes to DRAM-sharded or standard path based on config.

    When force_no_tp=True, skip mesh_partition and all_reduce (weight is replicated).

    `prefetch` is a Glm4MoeLitePrefetcherSetup whose GlobalCB holds this weight; when
    supplied the ring path is used. Only wired for o_proj today -- the caller decides,
    because only some weights admit a legal ring (see ring_feasibility).
    """
    use_tp = cfg.tp_enabled and not force_no_tp
    if prefetch is not None:
        assert not use_tp, "ring prefetch is not supported with TP (weights must be replicated)"
        return ring_prefetch_linear(
            a,
            b,
            program_config=prefetch.oproj_program_config,
            input_mem_cfg=prefetch.oproj_input_mem_cfg,
            output_mem_cfg=prefetch.oproj_output_mem_cfg,
            global_cb=prefetch.global_circular_buffer,
            sub_device_id=prefetch.worker_sub_device_id,
            compute_kernel_config=cfg.mlp_compute_kernel_config(),
        )
    if cfg.dram_sharded_attn:
        if use_tp:
            a_tp = ttnn.mesh_partition(a, dim=3, cluster_axis=cfg.tp_axis)
            out = dram_sharded_linear(a_tp, b, device=device, cfg=cfg)
            ttnn.deallocate(a_tp, force=False)
            out_reduced = ttnn.all_reduce(
                out,
                num_links=cfg.ccl_num_links,
                topology=cfg.ccl_topology,
                cluster_axis=cfg.tp_axis,
                memory_config=cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(out, force=False)
            return out_reduced
        else:
            return dram_sharded_linear(a, b, device=device, cfg=cfg)
    else:
        if use_tp:
            return tp_row_parallel_linear(a, b, device=device, cfg=cfg)
        else:
            return mlp_linear(a, b, device=device, cfg=cfg)
