# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul / linear projection helpers for GLM-4.7-Flash decode and prefill.

Extracted from decoder_layer_tt.py. These were originally nested closures inside
the 2000+ line decode function. Now they take explicit arguments (device, config)
instead of capturing outer scope variables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import ttnn
from models.experimental.glm4_moe_lite.tt.runtime_config import Glm4RuntimeConfig

# Shared-expert down proj (e.g. 32 x 10240 x 2048): 64 N-tiles / 32 cores => per_core_N=2,
# which allows out_subblock_w=2 (perf report recommends out_subblock_h * out_subblock_w >= 2).
_DOWN_MATMUL_NUM_CORES = 32
_DOWN_OUT_SUBBLOCK_W = 2

# ---------------------------------------------------------------------------
# w_o output projection decode matmul tuning — M=32, K=5120, N=2048
# Default 64-core path gives per_core_N=1 → out_subblock_w=1 (SLOW in tracy).
# 32 cores → per_core_N=2 → out_subblock_w=2, matching the down-proj pattern.
#
# Output sharding strategy (WIDTH_SHARDED):
#   With mcast_in0=True the 1D multicast kernel width-parallelizes across N:
#   each of the 32 compute cores independently produces its
#   [per_core_M*TILE × per_core_N*TILE] = [32 × 64] output slice in its own DST
#   registers.  Setting memory_config to WIDTH_SHARDED lets each core write that
#   result directly to its local L1 bank (core-local store, no NOC hop) instead
#   of routing it to DRAM via cross-chip NOC writes.  The net effect:
#     • matmul writes: 32 local L1 writes of 4 KB each (fast)   vs.
#                      32 NOC→DRAM writes of 4 KB each (slow, bandwidth-limited)
#     • one follow-up to_memory_config gathers the 32 L1 shards to downstream
#       format (DRAM or L1-interleaved), amortized across the whole output tensor
#   Activation (in0) is NOT width-sharded — it is multicast via mcast_in0=True.
#   K is the reduction dimension and is not partitioned.
#   Weights stay DRAM-backed; 20 MB (K=5120×N=2048×2 B) does not fit in L1.
#
# Edit these constants directly to retune without touching function bodies.
# ---------------------------------------------------------------------------
_WO_NUM_CORES = 32  # total cores for 1D multicast; 32 → per_core_N=2 for N=2048
_WO_PER_CORE_N = 2  # N=2048 → 64 N-tiles / 32 cores
_WO_PER_CORE_M = 1  # M=32 → 1 M-tile (decode batch fits in one tile row)
_WO_IN0_BLOCK_W = 4  # K=5120 → 160 K-tiles; 160 % 4 == 0 ✓
_WO_OUT_SUBBLOCK_H = 1
_WO_OUT_SUBBLOCK_W = 2  # avoids out_subblock_h * out_subblock_w == 1 penalty
_WO_MCAST_IN0 = True  # broadcast 1-tile activation to all cores (weight-stationary)
_WO_ACT_IN_L1 = True  # move v to L1 interleaved before matmul; eliminates DRAM read


@dataclass(frozen=True)
class Matmul1dProgOverrides:
    """Optional overrides for ``compute_1d_prog_cfg`` (``None`` = auto)."""

    in0_block_w: int | None = None
    per_core_M: int | None = None
    per_core_N: int | None = None
    out_subblock_w: int | None = None
    out_subblock_h: int | None = None


# LM-head 1D multicast tuning (edit in source; ``None`` = auto heuristics).
# Defaults for GLM-4.7-Flash full vocab on 110 cores: in0_block_w=4, per_core_M=1 (decode),
# per_core_N=44, out_subblock_h=1, out_subblock_w=4.
LM_HEAD_MATMUL_OVERRIDES = Matmul1dProgOverrides(
    in0_block_w=None,
    per_core_M=None,
    per_core_N=None,
    out_subblock_w=None,
    out_subblock_h=None,
)


def _auto_in0_block_w(k_tiles: int) -> int:
    for candidate in (4, 3, 2):
        if k_tiles % candidate == 0:
            return candidate
    return 1


def _auto_out_subblock_w(*, per_core_N: int, per_core_M: int, fp32_dest_acc_en: bool) -> tuple[int, int]:
    """Pick output subblock geometry (h, w) within DST register limits."""
    out_subblock_h = 1
    max_subblock_w = 2 if fp32_dest_acc_en else 4
    out_subblock_w = 1
    for cand in range(min(max_subblock_w, per_core_N), 0, -1):
        if per_core_N % cand == 0:
            out_subblock_w = cand
            break
    if per_core_M > 1:
        for cand in range(min(max_subblock_w, per_core_M), 0, -1):
            if per_core_M % cand == 0:
                out_subblock_h = cand
                break
    return out_subblock_h, out_subblock_w


def compute_1d_prog_cfg(
    device: Any,
    b_weight: ttnn.Tensor,
    m_total: int,
    *,
    fp32_dest_acc_en: bool = False,
    overrides: Matmul1dProgOverrides | None = None,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """Compute 1D multicast program config for decode-sized matmuls.

    Heuristics target bandwidth-bound ops (e.g. LM head ``32 x 2048 x 154880``):
    ``in0_block_w`` up to 4 when ``k_tiles`` allows, and ``out_subblock_w`` up to 4
    when ``per_core_N`` divides evenly (perf report: avoid ``out_subblock_h * w == 1``).
    """
    ov = overrides or Matmul1dProgOverrides()
    K = int(b_weight.shape[-2])
    N = int(b_weight.shape[-1])
    m_tiles = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    k_tiles = (K + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    n_tiles = (N + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    num_cores = grid_x * grid_y

    if n_tiles < num_cores:
        grid_y = max(1, n_tiles // grid_x)
        num_cores = grid_x * grid_y

    if ov.per_core_N is not None:
        per_core_N = int(ov.per_core_N)
        if per_core_N <= 0:
            raise ValueError(f"per_core_N must be positive, got {per_core_N}")
    else:
        per_core_N = max(1, math.ceil(n_tiles / num_cores))

    per_core_M = int(ov.per_core_M) if ov.per_core_M is not None else m_tiles

    if ov.in0_block_w is not None:
        in0_bw = int(ov.in0_block_w)
        if k_tiles % in0_bw != 0:
            raise ValueError(f"in0_block_w={in0_bw} must divide k_tiles={k_tiles}")
    else:
        in0_bw = _auto_in0_block_w(k_tiles)

    if ov.out_subblock_w is not None:
        out_subblock_h = int(ov.out_subblock_h) if ov.out_subblock_h is not None else 1
        out_subblock_w = int(ov.out_subblock_w)
    else:
        out_subblock_h, out_subblock_w = _auto_out_subblock_w(
            per_core_N=per_core_N, per_core_M=per_core_M, fp32_dest_acc_en=fp32_dest_acc_en
        )

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_bw,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def lm_head_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    memory_config: ttnn.MemoryConfig | None = None,
    overrides: Matmul1dProgOverrides | None = None,
) -> ttnn.Tensor:
    """LM head: interleaved activations + weights with tuned 1D multicast program config.

    Setting ``program_config`` flips ttnn.linear's implicit fidelity default from HiFi2 to LoFi
    (matmul_device_operation.cpp::create_matmul_attributes). For the LM-head reduction
    over hidden_size that drop is enough to flip top-1 token selection and produce
    template-style drift, so we pin an explicit HiFi4 + fp32 DST accumulation kernel config
    here. ``fp32_dest_acc_en`` is also threaded into ``compute_1d_prog_cfg`` so the
    out-subblock heuristic respects the smaller (4-tile) DST budget.
    """
    m_total = 1
    for i in range(len(a.shape) - 1):
        m_total *= int(a.shape[i])
    kwargs: dict[str, object] = {
        "program_config": compute_1d_prog_cfg(
            device,
            b,
            m_total,
            overrides=overrides or LM_HEAD_MATMUL_OVERRIDES,
            fp32_dest_acc_en=True,
        ),
        "compute_kernel_config": _lm_head_compute_kernel_config(),
    }
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    return ttnn.linear(a, b, **kwargs)


def _lm_head_compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def compute_1d_mlp_down_prog_cfg(
    device: Any, b_weight: ttnn.Tensor, m_total: int
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """1D multicast program config for MLP down projections (K=intermediate, N=hidden).

    Uses exactly 32 cores with per_core_N=2 and out_subblock_w=2 when N-tiles are
    divisible by 32 (2048 hidden => 64 tiles). Falls back to compute_1d_prog_cfg otherwise.
    """
    K = int(b_weight.shape[-2])
    N = int(b_weight.shape[-1])
    m_tiles = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    k_tiles = (K + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    n_tiles = (N + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    num_cores = _DOWN_MATMUL_NUM_CORES

    if n_tiles % num_cores != 0:
        return compute_1d_prog_cfg(device, b_weight, m_total)

    per_core_N = n_tiles // num_cores
    if per_core_N % _DOWN_OUT_SUBBLOCK_W != 0:
        return compute_1d_prog_cfg(device, b_weight, m_total)

    grid = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)
    core_x, core_y = max_x, max_y
    found = False
    for gx in range(min(max_x, num_cores), 0, -1):
        if num_cores % gx != 0:
            continue
        gy = num_cores // gx
        if gy <= max_y:
            core_x, core_y = gx, gy
            found = True
            break
    if not found:
        return compute_1d_prog_cfg(device, b_weight, m_total)

    in0_bw = 1
    for candidate in (4, 3, 2):
        if k_tiles % candidate == 0:
            in0_bw = candidate
            break

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=_DOWN_OUT_SUBBLOCK_W,
        out_block_h=1,
        out_block_w=_DOWN_OUT_SUBBLOCK_W,
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
            kwargs["program_config"] = compute_1d_prog_cfg(device, b, m_total, fp32_dest_acc_en=cfg.moe_fp32_acc)
    return ttnn.linear(a, b, **kwargs)


def mlp_down_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Linear for MLP down projections with a fixed 32-core / out_subblock_w=2 program config."""
    kwargs: dict[str, object] = {}
    mc = memory_config if memory_config is not None else cfg.decode_act_mc
    if mc is not None:
        kwargs["memory_config"] = mc
    kwargs["compute_kernel_config"] = cfg.mlp_compute_kernel_config()
    m_total = 1
    for i in range(len(a.shape) - 1):
        m_total *= int(a.shape[i])
    kwargs["program_config"] = compute_1d_mlp_down_prog_cfg(device, b, m_total)
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


def attn_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    force_no_tp: bool = False,
) -> ttnn.Tensor:
    """Attention projection linear. Routes to DRAM-sharded or standard path based on config.

    When force_no_tp=True, skip mesh_partition and all_reduce (weight is replicated).
    """
    use_tp = cfg.tp_enabled and not force_no_tp
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


def attn_wo_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
) -> ttnn.Tensor:
    """Tuned w_o output projection for MLA decode: M=32, K=5120, N=2048.

    Program config: 32-core 1D multicast, per_core_N=2, out_subblock_w=2
    (see _WO_* constants), avoiding the out_subblock_h*w==1 SLOW penalty.

    Output layout: WIDTH_SHARDED across the same 32-core compute grid.
    Each core stores its [32 × 64] (= per_core_M*TILE × per_core_N*TILE) result
    in its own L1 — a core-local write with no NOC hop, replacing the previous
    cross-NOC DRAM writes.  One to_memory_config call afterwards materializes the
    32 L1 shards into the downstream interleaved format.

    Activation (in0) is NOT width-sharded; mcast_in0=True multicasts the full
    [32 × 5120] activation from L1 to all 32 cores unchanged.

    DRAM-sharded and TP execution modes route to their existing helpers.
    """
    if cfg.dram_sharded_attn:
        return dram_sharded_linear(a, b, device=device, cfg=cfg)
    if cfg.tp_enabled and not cfg.attn_dp:
        return tp_row_parallel_linear(a, b, device=device, cfg=cfg)

    # Resolve compute grid: fit _WO_NUM_CORES into the physical grid dimensions.
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)
    core_x, core_y = max_x, max_y
    for gx in range(min(max_x, _WO_NUM_CORES), 0, -1):
        if _WO_NUM_CORES % gx != 0:
            continue
        gy = _WO_NUM_CORES // gx
        if gy <= max_y:
            core_x, core_y = gx, gy
            break

    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(core_x, core_y),
        in0_block_w=_WO_IN0_BLOCK_W,
        out_subblock_h=_WO_OUT_SUBBLOCK_H,
        out_subblock_w=_WO_OUT_SUBBLOCK_W,
        per_core_M=_WO_PER_CORE_M,
        per_core_N=_WO_PER_CORE_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=_WO_MCAST_IN0,
    )

    # WIDTH_SHARDED output: shard shape [per_core_M*TILE, per_core_N*TILE] = [32, 64].
    # Core grid matches the compute grid exactly so each core owns its output slice.
    # ShardStrategy.WIDTH distributes along the N (column) dimension — the natural
    # decomposition of mcast_in0=True where each core independently computes its
    # N-column slice of the result.
    out_shard_h = _WO_PER_CORE_M * ttnn.TILE_SIZE  # 32 rows
    out_shard_w = _WO_PER_CORE_N * ttnn.TILE_SIZE  # 64 cols
    out_mc = ttnn.create_sharded_memory_config(
        shape=(out_shard_h, out_shard_w),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_x - 1, core_y - 1))]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    act = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG) if _WO_ACT_IN_L1 else a
    out_sharded = ttnn.linear(
        act,
        b,
        program_config=prog_cfg,
        compute_kernel_config=cfg.mlp_compute_kernel_config(),
        memory_config=out_mc,
    )
    if _WO_ACT_IN_L1:
        ttnn.deallocate(act, force=False)

    # Materialize WIDTH_SHARDED L1 output to the downstream interleaved format.
    # This is a single flat gather (each core reads its 4 KB shard and writes to
    # DRAM/L1-interleaved) — cheaper than the 32 cross-NOC DRAM writes the matmul
    # would have issued if the output memory config were DRAM directly.
    out = ttnn.to_memory_config(out_sharded, cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out_sharded, force=False)
    return out
