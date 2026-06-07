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

# Down proj tuning: 32 cores, per_core_N=2, out_subblock_w=2 (avoids subblock_h*w==1 penalty).
_DOWN_MATMUL_NUM_CORES = 32
_DOWN_OUT_SUBBLOCK_W = 2

# w_o decode tuning: 32 cores, per_core_N=2, out_subblock_w=2, WIDTH_SHARDED L1 output.
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
LM_HEAD_MATMUL_OVERRIDES = Matmul1dProgOverrides(
    in0_block_w=None,
    per_core_M=None,
    per_core_N=None,
    out_subblock_w=None,
    out_subblock_h=None,
)


# Default compute kernel config for prefill 1D+ws matmuls.
# Without an explicit CKC, TTNN defaults to LoFi when a program_config is given,
# which degrades accuracy.  HiFi2 matches what ttnn.linear auto-selects for
# BFP8 weights without an explicit program config.
_PREFILL_CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
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


def _lm_head_dram_sharded(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """DRAM-sharded LM head for M=32 decode (b must be DRAM WIDTH_SHARDED).

    Eliminates the NOC weight multicast used by 1D configs: each compute core reads
    directly from its DRAM bank, achieving higher parallel DRAM bandwidth utilization.
    HiFi4 + fp32_dest_acc is preserved to avoid top-1 logit flip vs the 1D path.
    """
    from models.demos.deepseek_v3.utils.config_helpers import (
        get_activation_sharding_core_counts_for_dram_matmul,
        get_dram_sharded_matmul_config,
    )

    grid = device.compute_with_storage_grid_size()
    max_cores = int(grid.x) * int(grid.y)
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
        compute_kernel_config=_lm_head_compute_kernel_config(),
    )
    ttnn.deallocate(a_sharded, force=False)
    downstream_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.to_memory_config(result, downstream_mc)
    ttnn.deallocate(result, force=False)
    return out


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

    When the weight ``b`` is DRAM WIDTH_SHARDED (set via GLM4_MOE_LITE_DRAM_SHARDED_LM_HEAD=1),
    routes to ``_lm_head_dram_sharded`` which uses parallel direct-DRAM reads instead of the
    NOC weight multicast, achieving higher bandwidth for the large M=32×K=2048×N=vocab matmul.
    """
    if b.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return _lm_head_dram_sharded(a, b, device=device, memory_config=memory_config)
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
    a_l1 = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG)
    result = ttnn.linear(a_l1, b, **kwargs)
    ttnn.deallocate(a_l1, force=False)
    return result


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


def _prefill_1d_prog_and_ws_mc(device: Any, b_weight: ttnn.Tensor, m_total: int) -> tuple:
    """1D multicast program config + WIDTH_SHARDED output MC for prefill (M > TILE_SIZE).

    Grid heuristic derived from sweep (test_prefill_matmul_sweep.py) across all 6
    GLM-4 MoE Lite prefill shapes:
      - Largest num_cores | Nt where per_core_N % out_subblock_w == 0 for some
        out_subblock_w ∈ {4, 3, 2}.  Avoids per_core_N=5 (only subblock_w=1,
        severe SLOW penalty) and per_core_N=1 (no parallelism).
      - WIDTH_SHARDED L1 output: each core writes to its own L1 bank (no NOC hop
        during the matmul).  Sweep showed 10-15% improvement across all shapes.
    """
    K = int(b_weight.shape[-2])
    N = int(b_weight.shape[-1])
    mt = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    kt = (K + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    nt = (N + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE

    grid = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)
    max_cores = max_x * max_y

    # Find largest nc | nt where per_core_N ≥ 2 and per_core_N has a good subblock
    # (divisible by 4, 3, or 2).  Iterating top-down means the first hit is the winner.
    best_nc, best_pcn, best_osw = 1, nt, 1
    for nc in range(min(nt, max_cores), 0, -1):
        if nt % nc != 0:
            continue
        pcn = nt // nc
        if pcn < 2:
            continue
        osw = next((w for w in (4, 3, 2) if pcn % w == 0), 1)
        if osw >= 2:
            best_nc, best_pcn, best_osw = nc, pcn, osw
            break
    # Fallback: any nc with per_core_N ≥ 2 (no subblock constraint)
    if best_osw == 1:
        for nc in range(min(nt, max_cores), 0, -1):
            if nt % nc == 0 and nt // nc >= 2:
                best_nc, best_pcn = nc, nt // nc
                best_osw = next((w for w in (4, 3, 2) if best_pcn % w == 0), 1)
                break

    # Fit best_nc into the physical grid (maximize gx to spread across columns).
    core_x, core_y = 1, best_nc
    for gx in range(min(max_x, best_nc), 0, -1):
        if best_nc % gx != 0:
            continue
        gy = best_nc // gx
        if gy <= max_y:
            core_x, core_y = gx, gy
            break

    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=_auto_in0_block_w(kt),
        out_subblock_h=1,
        out_subblock_w=best_osw,
        per_core_M=mt,
        per_core_N=best_pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    out_mc = ttnn.create_sharded_memory_config(
        shape=(1, 1, mt * ttnn.TILE_SIZE, best_pcn * ttnn.TILE_SIZE),
        core_grid=ttnn.CoreGrid(y=core_y, x=core_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return prog_cfg, out_mc


def _prefill_linear_ws_out(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    downstream_mc = memory_config if memory_config is not None else (cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    return prefill_linear_ws_out(
        a, b, device=device, compute_kernel_config=cfg.mlp_compute_kernel_config(), memory_config=downstream_mc
    )


def prefill_linear_ws_out(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    compute_kernel_config: Any = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Prefill linear: 1D multicast program config + WIDTH_SHARDED L1 output.

    Avoids per-core DRAM writes during the matmul by having each core write
    its result to a local L1 shard.  A single to_memory_config gather follows.
    Only applied when the weight tensor has no batch dimension (b_batch == 1)
    and M > TILE_SIZE (prefill mode).  Safe to call from code that doesn't
    have access to Glm4RuntimeConfig.
    """
    m_total = 1
    for i in range(len(a.shape) - 1):
        m_total *= int(a.shape[i])
    prog_cfg, ws_mc = _prefill_1d_prog_and_ws_mc(device, b, m_total)
    ckc = compute_kernel_config if compute_kernel_config is not None else _PREFILL_CKC
    # Move activation to L1 interleaved so the matmul reads from L1 instead of DRAM.
    # Sweep confirmed l1/dram/ws beats dram/dram/ws for all 6 GLM-4 prefill shapes.
    a_l1 = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG)
    out_sharded = ttnn.linear(a_l1, b, program_config=prog_cfg, memory_config=ws_mc, compute_kernel_config=ckc)
    ttnn.deallocate(a_l1, force=False)
    downstream_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.to_memory_config(out_sharded, downstream_mc)
    ttnn.deallocate(out_sharded, force=False)
    return out


def prefill_per_head_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    compute_kernel_config: Any = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Per-head batched prefill linear for [1,H,M,K]×[1,H,K,N] (fuse_batch=False).

    Uses 1D mcast_in0 with nc=Nt (one N-tile/core, max N parallelism) and
    per_core_M=Mt.  Activation is staged in L1; output gathered to memory_config.

    Sweep results for GLM-4 MoE Lite (test_prefill_batched_matmul_sweep.py):
      kv_b1 Nt=16 → nc=16 (8×2), bw=2  72.90µs  6.90 TFLOPs (+1.29× vs auto ~94µs)
      kv_b2 Nt=8  → nc=8  (8×1), bw=4  121.38µs 5.53 TFLOPs (+1.80× vs auto ~219µs)
    """
    mt = max(1, (int(a.shape[-2]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    kt = (int(b.shape[-2]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    nt = (int(b.shape[-1]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE

    grid = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)

    # nc = Nt (one N-tile per core); reduce until nc fits on the hardware grid.
    nc = nt
    while nc > 1 and (nt % nc != 0 or nc > max_x * max_y):
        nc -= 1

    core_x, core_y = 1, nc
    for gx in range(min(max_x, nc), 0, -1):
        if nc % gx == 0 and (nc // gx) <= max_y:
            core_x, core_y = gx, nc // gx
            break

    per_core_N = max(1, nt // nc)
    out_subblock_h, out_subblock_w = _auto_out_subblock_w(
        per_core_N=per_core_N,
        per_core_M=mt,
        fp32_dest_acc_en=False,
    )

    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=_auto_in0_block_w(kt),
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=mt,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )

    ckc = compute_kernel_config if compute_kernel_config is not None else _PREFILL_CKC
    a_l1 = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG)
    out_l1 = ttnn.linear(
        a_l1, b, program_config=prog_cfg, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc
    )
    ttnn.deallocate(a_l1, force=False)
    downstream_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.to_memory_config(out_l1, downstream_mc)
    ttnn.deallocate(out_l1, force=False)
    return out


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
    m_total = 1
    for i in range(len(a.shape) - 1):
        m_total *= int(a.shape[i])
    b_batch = 1
    for i in range(len(b.shape) - 2):
        b_batch *= int(b.shape[i])
    if b_batch > 1 and m_total > ttnn.TILE_SIZE:
        return prefill_per_head_linear(
            a,
            b,
            device=device,
            compute_kernel_config=ckc,
            memory_config=memory_config,
        )
    if b_batch == 1:
        if m_total > ttnn.TILE_SIZE:
            return _prefill_linear_ws_out(a, b, device=device, cfg=cfg, memory_config=memory_config)
        if cfg.explicit_prog_cfg and m_total <= ttnn.TILE_SIZE:
            # Thread fp32-DST into the subblock heuristic so out_subblock_w respects the 4-tile DST budget.
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
    if m_total > ttnn.TILE_SIZE:
        return _prefill_linear_ws_out(a, b, device=device, cfg=cfg, memory_config=memory_config)
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


# Gate/up decode tuning: per_core_N=2, out_subblock_w=2, WIDTH_SHARDED L1 output.
_GATE_UP_PER_CORE_N = 2
_GATE_UP_OUT_SUBBLOCK_W = 2


def mlp_gate_up_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Gate/up projection linear, optimized for decode (M <= TILE_SIZE).

    When M <= TILE_SIZE, N-tiles divisible by _GATE_UP_PER_CORE_N, and the
    resulting core count fits the physical grid: runs with per_core_N=2,
    out_subblock_w=2, mcast_in0=True, and WIDTH_SHARDED L1 output.
    Each compute core writes its output shard to its own L1 bank (no NOC
    hop during the matmul); one to_memory_config then gathers the shards
    to the downstream format.  Falls back to mlp_linear if any constraint
    is not satisfied (prefill, large N, or grid too small).
    """
    m_total = 1
    for i in range(len(a.shape) - 1):
        m_total *= int(a.shape[i])
    b_batch = 1
    for i in range(len(b.shape) - 2):
        b_batch *= int(b.shape[i])

    N = int(b.shape[-1])
    K = int(b.shape[-2])
    n_tiles = (N + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    k_tiles = (K + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    m_tiles = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)

    if (
        m_total > ttnn.TILE_SIZE
        or b_batch != 1
        or n_tiles % _GATE_UP_PER_CORE_N != 0
        or _GATE_UP_PER_CORE_N % _GATE_UP_OUT_SUBBLOCK_W != 0
    ):
        return mlp_linear(a, b, device=device, cfg=cfg, memory_config=memory_config)

    num_cores = n_tiles // _GATE_UP_PER_CORE_N
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)
    core_x, core_y = None, None
    for gx in range(min(max_x, num_cores), 0, -1):
        if num_cores % gx != 0:
            continue
        gy = num_cores // gx
        if gy <= max_y:
            core_x, core_y = gx, gy
            break

    if core_x is None:
        return mlp_linear(a, b, device=device, cfg=cfg, memory_config=memory_config)

    in0_bw = 1
    for candidate in (4, 3, 2):
        if k_tiles % candidate == 0:
            in0_bw = candidate
            break

    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=_GATE_UP_OUT_SUBBLOCK_W,
        out_block_h=1,
        out_block_w=_GATE_UP_OUT_SUBBLOCK_W,
        per_core_M=m_tiles,
        per_core_N=_GATE_UP_PER_CORE_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    # WIDTH_SHARDED output: each core writes its result shard to local L1; to_memory_config gathers downstream.
    out_shard_h = m_tiles * ttnn.TILE_SIZE
    out_shard_w = _GATE_UP_PER_CORE_N * ttnn.TILE_SIZE
    out_mc = ttnn.create_sharded_memory_config(
        shape=(out_shard_h, out_shard_w),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_x - 1, core_y - 1))]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_sharded = ttnn.linear(
        a,
        b,
        program_config=prog_cfg,
        compute_kernel_config=cfg.mlp_compute_kernel_config(),
        memory_config=out_mc,
    )

    downstream_mc = memory_config if memory_config is not None else (cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out_sharded, downstream_mc)
    ttnn.deallocate(out_sharded, force=False)
    return out


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


# w_kv_a decode tuning: 21 cores (7×3), per_core_N=2, out_subblock_w=2, WIDTH_SHARDED L1 output.
_KVA_NUM_CORES = 21  # N=1344 → 42 N-tiles / per_core_N=2 = 21 cores
_KVA_PER_CORE_N = 2  # N-tiles per core; must divide n_tiles (42 % 2 == 0)
_KVA_PER_CORE_M = 1  # M=32 → 1 M-tile (decode batch fits in one tile row)
_KVA_IN0_BLOCK_W = 4  # K=2048 → 64 k-tiles; 64 % 4 == 0 ✓
_KVA_OUT_SUBBLOCK_H = 1
_KVA_OUT_SUBBLOCK_W = 2  # avoids out_subblock_h * out_subblock_w == 1 penalty
_KVA_MCAST_IN0 = True  # broadcast activation to all cores (weight-stationary)


def attn_kva_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    force_no_tp: bool = False,
) -> ttnn.Tensor:
    """w_kv_a / w_q_kv_a decode linear: M=32, K=2048, N=1344.

    21-core 1D multicast (7×3 grid), per_core_N=2, out_subblock_w=2,
    WIDTH_SHARDED L1 output.  Avoids the per_core_N=1 / out_subblock_h*w==1
    SLOW penalty from the default 64-core heuristic.

    Output is gathered to the downstream interleaved format via a single
    to_memory_config call so that downstream ttnn.slice remains unchanged.

    DRAM-sharded, TP, and prefill (M > TILE_SIZE) routes fall back to their
    existing helpers.  If _KVA_NUM_CORES does not fit the physical grid the
    function also falls back gracefully.
    """
    if cfg.dram_sharded_attn:
        if cfg.tp_enabled and not force_no_tp:
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
        return dram_sharded_linear(a, b, device=device, cfg=cfg)

    if cfg.tp_enabled and not force_no_tp:
        return tp_row_parallel_linear(a, b, device=device, cfg=cfg)

    # Decode heuristic: only optimize for M fitting in a single tile row.
    m_total = 1
    for i in range(len(a.shape) - 1):
        m_total *= int(a.shape[i])
    if m_total > ttnn.TILE_SIZE:
        return mlp_linear(a, b, device=device, cfg=cfg)

    # Fit _KVA_NUM_CORES into the physical grid.
    grid = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)
    core_x, core_y = max_x, max_y
    found = False
    for gx in range(min(max_x, _KVA_NUM_CORES), 0, -1):
        if _KVA_NUM_CORES % gx != 0:
            continue
        gy = _KVA_NUM_CORES // gx
        if gy <= max_y:
            core_x, core_y = gx, gy
            found = True
            break
    if not found:
        return mlp_linear(a, b, device=device, cfg=cfg)

    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(core_x, core_y),
        in0_block_w=_KVA_IN0_BLOCK_W,
        out_subblock_h=_KVA_OUT_SUBBLOCK_H,
        out_subblock_w=_KVA_OUT_SUBBLOCK_W,
        per_core_M=_KVA_PER_CORE_M,
        per_core_N=_KVA_PER_CORE_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=_KVA_MCAST_IN0,
    )

    # WIDTH_SHARDED output: each core stores its [32×64] result in local L1 (no NOC hop).
    out_shard_h = _KVA_PER_CORE_M * ttnn.TILE_SIZE
    out_shard_w = _KVA_PER_CORE_N * ttnn.TILE_SIZE
    out_mc = ttnn.create_sharded_memory_config(
        shape=(out_shard_h, out_shard_w),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_x - 1, core_y - 1))]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_sharded = ttnn.linear(
        a,
        b,
        program_config=prog_cfg,
        compute_kernel_config=cfg.mlp_compute_kernel_config(),
        memory_config=out_mc,
    )

    # Gather WIDTH_SHARDED L1 shards to interleaved; avoids DRAM round-trip before nope/rope splits.
    out = ttnn.to_memory_config(out_sharded, cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out_sharded, force=False)
    return out


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

    # WIDTH_SHARDED output: shard [32, 64], core grid matches compute grid, distributes along N.
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

    # Gather WIDTH_SHARDED L1 output to downstream interleaved (cheaper than per-core NOC→DRAM writes).
    out = ttnn.to_memory_config(out_sharded, cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out_sharded, force=False)
    return out
