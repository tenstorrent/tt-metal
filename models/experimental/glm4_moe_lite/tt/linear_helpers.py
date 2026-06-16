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
# Tuned via test_decode_matmul_sweep.py on Blackhole p300c (11×10=110 cores):
# full grid, per_core_N=11 (Nt=1210/110), in0_block_w=2, out_subblock_w=1 → 146.6us
# (vs 169us auto, 1.15x). The shape is DRAM-bandwidth-bound so config gains are modest;
# the DRAM-sharded path (GLM4_MOE_LITE_DRAM_SHARDED_LM_HEAD) is the larger lever.
LM_HEAD_MATMUL_OVERRIDES = Matmul1dProgOverrides(
    in0_block_w=2,
    per_core_M=None,
    per_core_N=11,
    out_subblock_w=1,
    out_subblock_h=1,
)


@dataclass(frozen=True)
class _DecodeTuned:
    """Swept-optimal 1D mcast config for a specific decode (K, N) matmul."""

    num_cores: int
    in0_block_w: int
    per_core_N: int
    out_subblock_w: int
    in0_dram: bool = False  # True → stream in0 from DRAM (else L1-resident)


# Decode matmul (K, N) → tuned config, from test_decode_matmul_sweep.py winners
# (Blackhole p300c, M=32, LoFi).  These shapes otherwise fall through to ttnn's
# auto config in mlp_linear (the per_core_N=1 "SLOW" path); consulting this table
# in mlp_linear's decode branch applies the swept config + WIDTH_SHARDED L1 output.
# Keyed (K, N) so only the exact measured shapes change; all others are untouched.
_DECODE_MATMUL_TUNED: dict[tuple[int, int], _DecodeTuned] = {
    (2048, 768): _DecodeTuned(num_cores=8, in0_block_w=4, per_core_N=3, out_subblock_w=3),  # q_a: 28→13us (2.1x)
    (768, 5120): _DecodeTuned(num_cores=40, in0_block_w=4, per_core_N=4, out_subblock_w=4),  # q_b: stage in0 in L1
    (1280, 2048): _DecodeTuned(
        num_cores=32, in0_block_w=4, per_core_N=2, out_subblock_w=2
    ),  # w_o (head-parallel): 23→11us (2.2x)
    # Head-parallel w_o per-shard (K=heads_per_dev*v_head_dim=384, N=hidden); removes the
    # per_core_N=1 SLOW penalty on the auto 64-core config (4.27→3.72us, 1.15x).
    (384, 2048): _DecodeTuned(num_cores=16, in0_block_w=4, per_core_N=4, out_subblock_w=4),
}


# Default compute kernel config for prefill 1D+ws matmuls (LoFi + BFP4 weights).
_PREFILL_CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# MoE router prefill CKC (matches moe_topk_tt).
_ROUTER_PREFILL_CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
# Fused in router matmul (moe_topk_tt); bias add stays a separate op after scores.
ROUTER_FUSED_SIGMOID = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)

# Decode kv_b2 per-head matmul (b={H/tp} x 32 x 512 x 256): HiFi2 + tuned 1D prog cfg.
# Sweep baseline (test_prefill_batched_matmul_sweep.py): nc=8 bw=4; decode uses nc=4 pcN=2
# so out_subblock h*w >= 2 (avoids the auto-config HiFi4 / no-prog_cfg penalty).
_KVB2_DECODE_CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def prefill_matmul_tuned_enabled() -> bool:
    return os.environ.get("GLM4_MOE_LITE_PREFILL_MATMUL_TUNED", "").strip() == "1"


def _prefill_1d_ws_max_mt() -> int:
    """Max M-tiles safe for 1D WIDTH_SHARDED prefill matmul (per_core_M=mt) on L1."""
    raw = os.environ.get("GLM4_MOE_LITE_PREFILL_1D_WS_MAX_MT", "").strip()
    if raw:
        return max(1, int(raw))
    return 32  # 1024 tokens; empirically safe on Blackhole 1×4


def _cap_subblock_hw(h: int, w: int, *, fp32_dest_acc_en: bool) -> tuple[int, int]:
    max_hw = 16 if fp32_dest_acc_en else 8
    while h * w > max_hw:
        if w > 1:
            w -= 1
        elif h > 1:
            h -= 1
        else:
            break
    return h, w


def prepare_sparse_moe_matmul_in0(
    a: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    consume_input: bool = False,
) -> tuple[ttnn.Tensor, bool]:
    """Cast sparse MoE matmul activations to BFP8 (sparse expert path only).

    Returns (tensor, deallocate_after_use).
    """
    if a.dtype == ttnn.bfloat8_b:
        return a, False
    a_mc = a.memory_config()
    if a_mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return a, False
    mc = memory_config if memory_config is not None else a_mc
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        mc = ttnn.L1_MEMORY_CONFIG
    a_work = a
    gathered = False
    if not _is_l1_interleaved(a_mc) and a_mc.buffer_type == ttnn.BufferType.L1:
        a_work = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG)
        gathered = True
    out = ttnn.typecast(a_work, dtype=ttnn.bfloat8_b, memory_config=mc)
    if gathered:
        ttnn.deallocate(a_work, force=False)
    elif consume_input:
        ttnn.deallocate(a, force=False)
    return out, True


def _weight_tile_shape(b: ttnn.Tensor) -> tuple[int, int]:
    k = int(b.shape[-2])
    n = int(b.shape[-1])
    return (k + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE, (n + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE


def _tuned_shared_down_1d_prog_and_ws(*, m_total: int) -> tuple:
    """Sweep winner: 1D 8×4 ibw4 ws — shared expert down (K=384 N=512 TP shard)."""
    mt = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    grid_x, grid_y = 8, 4
    per_core_M = mt
    per_core_N = 2
    in0_block_w = 4
    osh, osw = ensure_min_subblock_area(per_core_M, per_core_N, fp32_dest_acc_en=False, width_sharded_out=True)
    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    ws_mc = ttnn.create_sharded_memory_config(
        shape=(1, 1, mt * ttnn.TILE_SIZE, per_core_N * ttnn.TILE_SIZE),
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return prog_cfg, ws_mc


def prefill_linear_2d_bs_out(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    grid_x: int,
    grid_y: int,
    in0_block_w: int,
    per_core_M: int,
    per_core_N: int,
    compute_kernel_config: Any = None,
    memory_config: ttnn.MemoryConfig | None = None,
    fp32_dest_acc: bool = False,
    fused_activation: ttnn.UnaryWithParam | None = None,
) -> ttnn.Tensor:
    """Prefill linear using 2D multicast + BLOCK_SHARDED L1 output, then gather."""
    m_total = 1
    for i in range(len(a.shape) - 1):
        m_total *= int(a.shape[i])
    m = int(a.shape[-2]) if len(a.shape) >= 1 else m_total
    n = int(b.shape[-1])
    osh, osw = ensure_min_subblock_area(
        per_core_M,
        per_core_N,
        fp32_dest_acc_en=fp32_dest_acc,
        width_sharded_out=False,
    )
    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=osh,
        out_subblock_w=osw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=fused_activation,
        transpose_mcast=False,
    )
    bs_mc = ttnn.create_sharded_memory_config(
        (1, 1, m, n),
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    ckc = compute_kernel_config if compute_kernel_config is not None else _PREFILL_CKC
    a_l1, copied = _to_l1_if_needed(a)
    out_bs = ttnn.linear(a_l1, b, program_config=prog_cfg, memory_config=bs_mc, compute_kernel_config=ckc)
    if copied:
        ttnn.deallocate(a_l1, force=False)
    downstream_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.to_memory_config(out_bs, downstream_mc)
    ttnn.deallocate(out_bs, force=False)
    return out


def tuned_moe_router_prefill_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Sweep winner: 2D 2×4 ibw4 bs — MoE router gate prefill (K=2048 N=64)."""
    return prefill_linear_2d_bs_out(
        a,
        b,
        device=a.device(),
        grid_x=2,
        grid_y=4,
        in0_block_w=4,
        per_core_M=1,
        per_core_N=1,
        compute_kernel_config=_ROUTER_PREFILL_CKC,
        memory_config=memory_config,
        fp32_dest_acc=True,
        fused_activation=ROUTER_FUSED_SIGMOID,
    )


def _auto_in0_block_w(k_tiles: int) -> int:
    for candidate in (4, 3, 2):
        if k_tiles % candidate == 0:
            return candidate
    return 1


def _in0_block_w_for_sharded_input(k_tiles: int, input_shard_w_tiles: int) -> int:
    """Pick in0_block_w valid for both K-tiles and the WIDTH_SHARDED activation shard width."""
    for candidate in (4, 3, 2, 1):
        if k_tiles % candidate == 0 and input_shard_w_tiles % candidate == 0:
            return candidate
    return 1


def _is_l1_width_sharded(tensor: ttnn.Tensor) -> bool:
    mc = tensor.memory_config()
    return mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED


def _decode_activation_m_total(x: ttnn.Tensor) -> int:
    """Flattened M for decode activations (product of all dims except hidden)."""
    shape = x.shape
    if len(shape) < 2:
        return 1
    m = 1
    for i in range(len(shape) - 1):
        m *= int(shape[i])
    return m


def _width_shard_mc_matches(tensor: ttnn.Tensor, expected_mc: ttnn.MemoryConfig) -> bool:
    """True when ``tensor`` is already in the same WIDTH_SHARDED L1 layout as ``expected_mc``."""
    if not _is_l1_width_sharded(tensor):
        return False
    mc = tensor.memory_config()
    if mc.memory_layout != expected_mc.memory_layout or mc.buffer_type != expected_mc.buffer_type:
        return False
    spec = mc.shard_spec
    exp = expected_mc.shard_spec
    if spec is None or exp is None:
        return False
    if spec.shape != exp.shape or spec.orientation != exp.orientation:
        return False
    return int(spec.grid.num_cores()) == int(exp.grid.num_cores())


def _shard_grid_fits_prog(sharded_tensor: ttnn.Tensor, prog_core_x: int, prog_core_y: int) -> bool:
    """True when the activation shard grid fits inside the matmul program compute grid."""
    spec = sharded_tensor.memory_config().shard_spec
    if spec is None:
        return False
    in_x, in_y = _shard_grid_xy(spec.grid)
    return in_x <= int(prog_core_x) and in_y <= int(prog_core_y)


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


def ensure_min_subblock_area(
    per_core_M: int,
    per_core_N: int,
    *,
    fp32_dest_acc_en: bool = False,
    width_sharded_out: bool = False,
) -> tuple[int, int]:
    """Pick (out_subblock_h, out_subblock_w) with h*w >= 2 when possible.

    Profiler marks 1x1 subblocks as SLOW.  WIDTH_SHARDED outputs require h==1
    (TTNN kernel constraint); only widen along N in that case.
    """
    h, w = _auto_out_subblock_w(per_core_N=per_core_N, per_core_M=per_core_M, fp32_dest_acc_en=fp32_dest_acc_en)
    if h * w >= 2:
        if width_sharded_out and h != 1:
            for cand in (4, 3, 2):
                if per_core_N >= cand and per_core_N % cand == 0:
                    return _cap_subblock_hw(1, cand, fp32_dest_acc_en=fp32_dest_acc_en)
            return 1, 1
        return _cap_subblock_hw(h, w, fp32_dest_acc_en=fp32_dest_acc_en)
    if width_sharded_out:
        for cand in (4, 3, 2):
            if per_core_N >= cand and per_core_N % cand == 0:
                return _cap_subblock_hw(1, cand, fp32_dest_acc_en=fp32_dest_acc_en)
        return 1, 1
    if per_core_M >= 2:
        for cand in (4, 3, 2):
            if per_core_M % cand == 0:
                return _cap_subblock_hw(cand, 1, fp32_dest_acc_en=fp32_dest_acc_en)
    if per_core_N >= 2:
        for cand in (4, 3, 2):
            if per_core_N % cand == 0:
                return _cap_subblock_hw(cand, 1, fp32_dest_acc_en=fp32_dest_acc_en)
    return _cap_subblock_hw(h, w, fp32_dest_acc_en=fp32_dest_acc_en)


def compute_1d_prog_cfg(
    device: Any,
    b_weight: ttnn.Tensor,
    m_total: int,
    *,
    fp32_dest_acc_en: bool = False,
    fused_activation: ttnn.UnaryWithParam | None = None,
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
        out_subblock_h, out_subblock_w = ensure_min_subblock_area(
            per_core_M=per_core_M, per_core_N=per_core_N, fp32_dest_acc_en=fp32_dest_acc_en
        )

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_bw,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=fused_activation,
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
            fp32_dest_acc_en=False,
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
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
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
    # Decode (m_tiles==1): per_core_N=2 → 2× the cores (32 for N=2048) which, combined with
    # the larger decode K-block above, beats the 16-core/per_core_N=4 config on this weight-BW
    # -bound matmul (sweep moe_down 2560x2048: 8x4 w16 11.5µs vs 16-core 15µs). Prefill keeps
    # per_core_N=4 (fewer cores, bigger per-core tile work amortizes launch over the long M).
    if m_tiles == 1 and n_tiles % 2 == 0:
        per_core_N, out_subblock_w = 2, 2
    elif n_tiles % 4 == 0:
        per_core_N, out_subblock_w = 4, 4
    elif n_tiles % _DOWN_MATMUL_NUM_CORES == 0 and (n_tiles // _DOWN_MATMUL_NUM_CORES) % _DOWN_OUT_SUBBLOCK_W == 0:
        per_core_N, out_subblock_w = n_tiles // _DOWN_MATMUL_NUM_CORES, _DOWN_OUT_SUBBLOCK_W
    else:
        return compute_1d_prog_cfg(device, b_weight, m_total)
    num_cores = n_tiles // per_core_N

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
    # Decode (m_tiles==1): allow a much larger K-block. _auto-style caps at 4, leaving
    # this weight-BW-bound down matmul K-loop-bound (sweep moe_down 2560x2048: bw=4 20µs
    # → larger bw ~halves it). Prefill keeps the small cap (large per_core_M × big bw
    # would overflow L1).
    bw_candidates = (16, 8, 4, 3, 2) if m_tiles == 1 else (4, 3, 2)
    for candidate in bw_candidates:
        if k_tiles % candidate == 0:
            in0_bw = candidate
            break

    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=out_subblock_w,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def _common_ws_nc(nt_a: int, nt_b: int, max_cores: int) -> int:
    """Largest core count ≤ max_cores that divides both N-tile counts (for sharded-in0 chains)."""
    limit = min(int(nt_a), int(nt_b), int(max_cores))
    for nc in range(limit, 0, -1):
        if nt_a % nc == 0 and nt_b % nc == 0:
            return nc
    return 1


def _prefill_ws_grid(
    device: Any,
    m_total: int,
    n_width: int,
    *,
    k_width: int | None = None,
    forced_nc: int | None = None,
) -> tuple:
    """WIDTH_SHARDED L1 grid + memory config for prefill (M > TILE_SIZE).

    Returns (core_x, core_y, best_pcn, mt, best_osw, ws_mc).
    Shared by matmul (``_prefill_1d_prog_and_ws_mc``) and sharded RMSNorm.

    When ``forced_nc`` is set, use that core count instead of the perf heuristic
    (e.g. w_q_a grid must divide w_q_b N-tiles for matched sharded-in0).
    """
    mt = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    nt = (int(n_width) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE

    grid = device.compute_with_storage_grid_size()
    max_x, max_y = int(grid.x), int(grid.y)
    max_cores = max_x * max_y

    best_nc, best_pcn, best_osw = 1, nt, 1
    if forced_nc is not None:
        nc = int(forced_nc)
        if nc > 0 and nt % nc == 0 and nc <= max_cores:
            best_nc = nc
            best_pcn = nt // nc
            best_osw = next((w for w in (4, 3, 2) if best_pcn % w == 0), 1)
    else:
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
        if best_osw == 1:
            for nc in range(min(nt, max_cores), 0, -1):
                if nt % nc == 0 and nt // nc >= 2:
                    best_nc, best_pcn = nc, nt // nc
                    best_osw = next((w for w in (4, 3, 2) if best_pcn % w == 0), 1)
                    break

    core_x, core_y = 1, best_nc
    for gx in range(min(max_x, best_nc), 0, -1):
        if best_nc % gx != 0:
            continue
        gy = best_nc // gx
        if gy <= max_y:
            core_x, core_y = gx, gy
            break

    out_mc = ttnn.create_sharded_memory_config(
        shape=(1, 1, mt * ttnn.TILE_SIZE, best_pcn * ttnn.TILE_SIZE),
        core_grid=ttnn.CoreGrid(y=core_y, x=core_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return core_x, core_y, best_pcn, mt, best_osw, out_mc


def _norm_subblock_w(block_w: int) -> int:
    for w in (4, 3, 2, 1):
        if block_w % w == 0:
            return w
    return 1


def _shard_grid_xy(crs: ttnn.CoreRangeSet) -> tuple[int, int]:
    min_x = min_y = 10**9
    max_x = max_y = -1
    for cr in crs.ranges():
        for coord in (cr.start, cr.end):
            min_x = min(min_x, int(coord.x))
            max_x = max(max_x, int(coord.x))
            min_y = min(min_y, int(coord.y))
            max_y = max(max_y, int(coord.y))
    return max_x - min_x + 1, max_y - min_y + 1


def prefill_norm_config_from_width_sharded_tensor(tensor: ttnn.Tensor) -> dict:
    """RMSNorm config matching an existing WIDTH_SHARDED L1 activation (no resharding)."""
    mc = tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        raise ValueError(f"expected WIDTH_SHARDED tensor, got {mc.memory_layout}")
    spec = mc.shard_spec
    if spec is None:
        raise ValueError("WIDTH_SHARDED tensor missing shard_spec")
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    block_h = shard_h // ttnn.TILE_SIZE
    block_w = shard_w // ttnn.TILE_SIZE
    core_x, core_y = _shard_grid_xy(spec.grid)
    prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[core_x, core_y],
        subblock_w=_norm_subblock_w(block_w),
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )
    return {
        "sharded_program_config": prog,
        "sharded_output_config": mc,
    }


def decode_width_sharded_norm_input_config(device: Any, batch: int, width: int) -> ttnn.MemoryConfig:
    """WIDTH_SHARDED L1 memory config for decode embed output / input RMSNorm."""
    cfg = prefill_width_sharded_norm_config(device, int(batch), int(width))
    return cfg["sharded_output_config"]


def prefill_width_sharded_norm_config(device: Any, m_total: int, width: int, *, ws_nc_hint: int | None = None) -> dict:
    """RMSNorm config for WIDTH_SHARDED prefill activations (``in_sharded`` + ``out_sharded``)."""
    core_x, core_y, best_pcn, mt, _, ws_mc = _prefill_ws_grid(device, m_total, width, forced_nc=ws_nc_hint)
    prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[core_x, core_y],
        subblock_w=_norm_subblock_w(best_pcn),
        block_h=mt,
        block_w=best_pcn,
        inplace=False,
    )
    return {
        "sharded_program_config": prog,
        "sharded_output_config": ws_mc,
    }


def _norm_kva_sharded_in0_compatible(*, k_width: int, n_width: int) -> bool:
    """True when norm WIDTH_SHARDED output can feed attn_kva without an input gather.

    Requires a core count that divides both K-tiles (norm width) and N-tiles (matmul
    output).  For GLM w_q_kv_a (K=2048, N=1344) gcd(64,42)=2 only — too small.
    """
    kt = (int(k_width) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    nt = (int(n_width) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    g = math.gcd(kt, nt)
    return g >= 8


def sharded_decode_norm(
    norm_fn: Any,
    x: ttnn.Tensor,
    *,
    device: Any,
    width: int,
    downstream_mc: ttnn.MemoryConfig,
    return_sharded: bool = False,
    ws_nc_hint: int | None = None,
):
    """Run a decode RMSNorm width-sharded across cores instead of on a single core.

    ``ttnn.rms_norm`` on an interleaved [B, width] decode activation runs on ONE
    core (~30us for width=2048).  Width-sharding the input lets the LayerNorm
    sharded multi-core kernel parallelize the reduction; one to_memory_config
    gathers the result back to the downstream interleaved format unless
    ``return_sharded=True``.  Falls back to the plain single-core norm if the
    width is not shardable on this grid.
    """
    m_total = _decode_activation_m_total(x)
    try:
        norm_cfg = prefill_width_sharded_norm_config(device, m_total, int(width), ws_nc_hint=ws_nc_hint)
        if _width_shard_mc_matches(x, norm_cfg["sharded_output_config"]):
            x_sharded = x
            x_owned = False
        else:
            x_sharded = ttnn.to_memory_config(x, norm_cfg["sharded_output_config"])
            x_owned = x_sharded is not x
    except Exception:
        return norm_fn(x, mode="decode")
    out_sharded = norm_fn(x_sharded, mode="decode", in_sharded=True, out_sharded=True, norm_config=norm_cfg)
    if x_owned:
        ttnn.deallocate(x_sharded, force=False)
    if return_sharded:
        return out_sharded
    out = ttnn.to_memory_config(out_sharded, downstream_mc)
    ttnn.deallocate(out_sharded, force=False)
    return out


def _prefill_1d_prog_and_ws_mc_matched_in0(
    b_weight: ttnn.Tensor,
    m_total: int,
    sharded_in0: ttnn.Tensor,
) -> tuple | None:
    """1D matmul prog + WIDTH_SHARDED output MC on the same cores as sharded in0.

    Returns None when N-tiles are not evenly divisible across the input shard grid
    (caller must gather in0 and use the default output grid).
    """
    spec = sharded_in0.memory_config().shard_spec
    if spec is None:
        return None

    K = int(b_weight.shape[-2])
    N = int(b_weight.shape[-1])
    kt = (K + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    nt = (N + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    mt = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)

    num_cores = int(spec.grid.num_cores())
    if num_cores <= 0 or nt % num_cores != 0:
        return None

    input_shard_h = int(spec.shape[0]) // ttnn.TILE_SIZE
    input_shard_w_tiles = int(spec.shape[1]) // ttnn.TILE_SIZE
    if input_shard_h != mt or input_shard_w_tiles * num_cores != kt:
        return None

    per_core_N = nt // num_cores
    core_x, core_y = _shard_grid_xy(spec.grid)
    in0_block_w = _in0_block_w_for_sharded_input(kt, input_shard_w_tiles)
    osw = next((w for w in (4, 3, 2) if per_core_N % w == 0), 1)

    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=mt,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    out_mc = ttnn.create_sharded_memory_config(
        shape=(1, 1, mt * ttnn.TILE_SIZE, per_core_N * ttnn.TILE_SIZE),
        core_grid=ttnn.CoreGrid(y=core_y, x=core_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=spec.orientation,
        use_height_and_width_as_shard_shape=True,
    )
    return prog_cfg, out_mc


def _prefill_1d_prog_and_ws_mc(
    device: Any,
    b_weight: ttnn.Tensor,
    m_total: int,
    *,
    sharded_in0: ttnn.Tensor | None = None,
    ws_nc_hint: int | None = None,
) -> tuple:
    """1D multicast program config + WIDTH_SHARDED output MC for prefill (M > TILE_SIZE)."""
    K = int(b_weight.shape[-2])
    N = int(b_weight.shape[-1])
    kt = (K + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    core_x, core_y, best_pcn, mt, best_osw, out_mc = _prefill_ws_grid(device, m_total, N, forced_nc=ws_nc_hint)

    if sharded_in0 is not None:
        spec = sharded_in0.memory_config().shard_spec
        input_shard_w_tiles = int(spec.shape[1]) // ttnn.TILE_SIZE
        in0_block_w = _in0_block_w_for_sharded_input(kt, input_shard_w_tiles)
    else:
        in0_block_w = _auto_in0_block_w(kt)

    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=best_osw,
        per_core_M=mt,
        per_core_N=best_pcn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    return prog_cfg, out_mc


def _to_l1_if_needed(a: ttnn.Tensor) -> tuple[ttnn.Tensor, bool]:
    """Return (a_l1, was_copied). Skip the copy when a is already L1 interleaved."""
    mc = a.memory_config()
    if mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return a, False
    return ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG), True


def _is_l1_interleaved(mc: ttnn.MemoryConfig) -> bool:
    return mc.buffer_type == ttnn.BufferType.L1 and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


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
    return_sharded: bool = False,
    sharded_in0: bool = False,
    ws_nc_hint: int | None = None,
) -> ttnn.Tensor:
    """Prefill linear: 1D multicast program config + WIDTH_SHARDED L1 output.

    Avoids per-core DRAM writes during the matmul by having each core write
    its result to a local L1 shard.  A single to_memory_config gather follows.
    Only applied when the weight tensor has no batch dimension (b_batch == 1)
    and M > TILE_SIZE (prefill mode).  Safe to call from code that doesn't
    have access to Glm4RuntimeConfig.

    When ``return_sharded=True`` the WIDTH_SHARDED L1 shard tensor is returned
    directly without gathering.  The caller owns it and must deallocate when done.
    Use this when the immediate consumer is an elementwise op (e.g. silu+mul)
    whose two inputs share the same shard spec, so no gather is needed.

    When the activation is already WIDTH_SHARDED L1 (e.g. sharded RMSNorm output),
    it is fed directly into the matmul with an ``in0_block_w`` aligned to the
    input shard width — no ``ShardedToInterleaved`` gather on the input side.
    """
    m_total = 1
    for i in range(len(a.shape) - 1):
        m_total *= int(a.shape[i])
    mt = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    max_ws_mt = _prefill_1d_ws_max_mt()
    ckc = compute_kernel_config if compute_kernel_config is not None else _PREFILL_CKC

    def _prefill_dram_linear_fallback() -> ttnn.Tensor:
        a_l1, copied = _to_l1_if_needed(a)
        downstream_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.linear(a_l1, b, compute_kernel_config=ckc, memory_config=downstream_mc)
        if copied:
            ttnn.deallocate(a_l1, force=False)
        return out

    def _prefill_2d_m_split_fallback() -> ttnn.Tensor | None:
        """2D multicast with M split across grid rows (avoids 1D per_core_M=mt L1 clash)."""
        kt, nt = _weight_tile_shape(b)
        grid = device.compute_with_storage_grid_size()
        max_x, max_y = int(grid.x), int(grid.y)
        for grid_y in (4, 2, 1):
            if grid_y > max_y:
                continue
            per_core_M = max(1, (mt + grid_y - 1) // grid_y)
            if per_core_M > max_ws_mt:
                continue
            for grid_x in range(min(8, max_x), 0, -1):
                cores = grid_x * grid_y
                if cores <= 0 or nt % cores != 0:
                    continue
                per_core_N = nt // cores
                if per_core_N < 1:
                    continue
                return prefill_linear_2d_bs_out(
                    a,
                    b,
                    device=device,
                    grid_x=grid_x,
                    grid_y=grid_y,
                    in0_block_w=_auto_in0_block_w(kt),
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.L1_MEMORY_CONFIG if return_sharded else memory_config,
                )
        return None

    if prefill_matmul_tuned_enabled() and m_total > ttnn.TILE_SIZE:
        kt, nt = _weight_tile_shape(b)
        # L1 shared expert fused gate+up: 2D 8×4 ibw4 bs (gather after matmul).
        if kt == 64 and nt == 24:
            mt = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
            out = prefill_linear_2d_bs_out(
                a,
                b,
                device=device,
                grid_x=8,
                grid_y=4,
                in0_block_w=4,
                per_core_M=max(1, mt // 4),
                per_core_N=3,
                compute_kernel_config=ckc,
                memory_config=ttnn.L1_MEMORY_CONFIG if return_sharded else memory_config,
            )
            return out
        # kv_a (K=2048 N=576): 2D 6×4 ibw4 pcN3 (sweep winner ~18.7µs vs ~25.7µs for the
        # 9-core 1D ws path).  test_prefill_matmul_sweep.py::kv_a on Blackhole p300c.
        if kt == 64 and nt == 18:
            mt = max(1, (m_total + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
            out = prefill_linear_2d_bs_out(
                a,
                b,
                device=device,
                grid_x=6,
                grid_y=4,
                in0_block_w=4,
                per_core_M=max(1, mt // 4),
                per_core_N=3,
                compute_kernel_config=ckc,
                memory_config=ttnn.L1_MEMORY_CONFIG if return_sharded else memory_config,
            )
            return out
        # L1 shared expert down: 1D 8×4 ibw4 ws (mt must fit L1 budget).
        if kt == 12 and nt == 16:
            if mt > max_ws_mt:
                out_2d = _prefill_2d_m_split_fallback()
                if out_2d is not None:
                    return out_2d
                if return_sharded:
                    raise RuntimeError(
                        f"prefill_linear_ws_out shared_down: mt={mt} > max_ws_mt={max_ws_mt}; "
                        "set GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE"
                    )
                return _prefill_dram_linear_fallback()
            prog_cfg, ws_mc = _tuned_shared_down_1d_prog_and_ws(m_total=m_total)
            a_l1, copied = _to_l1_if_needed(a)
            out_sharded = ttnn.linear(a_l1, b, program_config=prog_cfg, memory_config=ws_mc, compute_kernel_config=ckc)
            if copied:
                ttnn.deallocate(a_l1, force=False)
            if return_sharded:
                return out_sharded
            downstream_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
            out = ttnn.to_memory_config(out_sharded, downstream_mc)
            ttnn.deallocate(out_sharded, force=False)
            return out

    # WIDTH_SHARDED activations are gathered L1→L1 interleaved before matmul unless
    # sharded_in0 is enabled (per-call or GLM4_MOE_LITE_PREFILL_SHARDED_MATMUL_IN0=1) and
    # the activation shard grid fits inside the output program grid.
    use_sharded_in0 = False
    sharded_in0_enabled = sharded_in0 or os.environ.get("GLM4_MOE_LITE_PREFILL_SHARDED_MATMUL_IN0", "0").strip() == "1"
    matched = None
    if sharded_in0_enabled and _is_l1_width_sharded(a) and mt <= max_ws_mt:
        matched = _prefill_1d_prog_and_ws_mc_matched_in0(b, m_total, a)
        use_sharded_in0 = matched is not None
    else:
        use_sharded_in0 = False
    if use_sharded_in0:
        prog_cfg, ws_mc = matched
        a_l1, _copied = a, False
    elif mt > max_ws_mt:
        out_2d = _prefill_2d_m_split_fallback()
        if out_2d is not None:
            return out_2d
        if return_sharded:
            raise RuntimeError(
                f"prefill_linear_ws_out: M={m_total} (mt={mt}) exceeds L1 1D ws budget "
                f"(max_ws_mt={max_ws_mt}) and no 2D split config matched; "
                "enable chunked prefill via GLM4_MOE_LITE_MAX_PREFILL_CHUNK_SIZE"
            )
        return _prefill_dram_linear_fallback()
    else:
        prog_cfg, ws_mc = _prefill_1d_prog_and_ws_mc(device, b, m_total, ws_nc_hint=ws_nc_hint)
        a_l1, _copied = _to_l1_if_needed(a)
    out_sharded = ttnn.linear(a_l1, b, program_config=prog_cfg, memory_config=ws_mc, compute_kernel_config=ckc)
    if _copied:
        ttnn.deallocate(a_l1, force=False)
    if return_sharded:
        return out_sharded
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
    in0_block_w: int | None = None,
) -> ttnn.Tensor:
    """Per-head batched prefill linear for [1,H,M,K]×[1,H,K,N] (fuse_batch=False).

    Uses 1D mcast_in0 with nc=Nt (one N-tile/core, max N parallelism) and
    per_core_M=Mt.  Activation is staged in L1; output gathered to memory_config.

    ``in0_block_w`` overrides the K-block (default ``_auto_in0_block_w(kt)``, which
    caps at 4).  Decode kv_b1 passes the full Kt=6 — see the caller in ``mlp_linear``.

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

    bw = in0_block_w if (in0_block_w is not None and kt % in0_block_w == 0) else _auto_in0_block_w(kt)
    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=bw,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=mt,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )

    ckc = compute_kernel_config if compute_kernel_config is not None else _PREFILL_CKC
    # Caller may already stage activations in L1 interleaved (e.g. q_nope after L1 slice).
    a_mc = a.memory_config()
    if _is_l1_interleaved(a_mc):
        a_l1, _copied = a, False
    else:
        a_l1, _copied = _to_l1_if_needed(a)
    out_l1 = ttnn.linear(
        a_l1, b, program_config=prog_cfg, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc
    )
    if _copied:
        ttnn.deallocate(a_l1, force=False)
    downstream_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    out_l1_mc = out_l1.memory_config()
    if _is_l1_interleaved(downstream_mc) and _is_l1_interleaved(out_l1_mc):
        return out_l1
    out = ttnn.to_memory_config(out_l1, downstream_mc)
    ttnn.deallocate(out_l1, force=False)
    return out


def _decode_kvb2_per_head_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """Decode kv_b2: [1,H,32,512] × [1,H,512,256], fuse_batch=False, HiFi2 + tuned prog cfg."""
    mt = max(1, (int(a.shape[-2]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 1),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=mt,
        per_core_N=2,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    a_l1, copied = _to_l1_if_needed(a)
    out_l1 = ttnn.linear(
        a_l1,
        b,
        program_config=prog_cfg,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=_KVB2_DECODE_CKC,
    )
    if copied:
        ttnn.deallocate(a_l1, force=False)
    # Default: keep L1 interleaved for the kv_b2 → head-flatten → w_o chain (avoids a
    # DRAM gather Copy before permute/reshape).  Caller may override via memory_config.
    downstream_mc = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
    out_l1_mc = out_l1.memory_config()
    if _is_l1_interleaved(downstream_mc) and _is_l1_interleaved(out_l1_mc):
        return out_l1
    if downstream_mc.buffer_type == out_l1_mc.buffer_type and downstream_mc.memory_layout == out_l1_mc.memory_layout:
        return out_l1
    out = ttnn.to_memory_config(out_l1, downstream_mc)
    ttnn.deallocate(out_l1, force=False)
    return out


def _resolve_grid(num_cores: int, max_x: int, max_y: int) -> tuple[int, int] | None:
    """Largest (gx, gy) with gx*gy == num_cores that fits the physical grid."""
    for gx in range(min(max_x, num_cores), 0, -1):
        if num_cores % gx == 0 and num_cores // gx <= max_y:
            return gx, num_cores // gx
    return None


def _tuned_decode_linear(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    device: Any,
    cfg: Glm4RuntimeConfig,
    memory_config: ttnn.MemoryConfig | None,
    tuned: _DecodeTuned,
) -> ttnn.Tensor | None:
    """Run a decode matmul with a swept-optimal 1D config + WIDTH_SHARDED L1 output.

    Mirrors the attn_wo_linear / mlp_gate_up_linear pattern: each core writes its
    output shard to local L1 (no NOC hop), then one to_memory_config gathers the
    shards to the downstream format.  Returns None if the grid cannot host the
    config (caller falls back to the default path).
    """
    grid = device.compute_with_storage_grid_size()
    g = _resolve_grid(tuned.num_cores, int(grid.x), int(grid.y))
    if g is None:
        return None
    core_x, core_y = g
    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(core_x, core_y),
        in0_block_w=tuned.in0_block_w,
        out_subblock_h=1,
        out_subblock_w=tuned.out_subblock_w,
        per_core_M=1,
        per_core_N=tuned.per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    out_mc = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )
    if tuned.in0_dram:
        act = ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG)
    elif _is_l1_interleaved(a.memory_config()):
        act = a
    else:
        act = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG)
    out_sharded = ttnn.linear(
        act,
        b,
        program_config=prog_cfg,
        compute_kernel_config=cfg.mlp_compute_kernel_config(),
        memory_config=out_mc,
    )
    if act is not a:
        ttnn.deallocate(act, force=False)
    downstream = memory_config if memory_config is not None else (cfg.decode_act_mc or ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out_sharded, downstream)
    ttnn.deallocate(out_sharded, force=False)
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
    if b_batch > 1:
        k, n = int(b.shape[-2]), int(b.shape[-1])
        mt = max(1, (int(a.shape[-2]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
        if k == 512 and n == 256 and mt == 1:
            kvb2_mc = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
            return _decode_kvb2_per_head_linear(a, b, device=device, memory_config=kvb2_mc)
        kvb_mc = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
        # Decode kv_b1 [1,20,32,192]×[1,20,192,512]: use the full Kt=6 in0 block.
        # _auto_in0_block_w caps at 4 → 3 for Kt=6; sweep (test_decode_batched_matmul_sweep.py)
        # shows full-K block 24.3µs vs 34.4µs at bw=3 (1.41x). Decode only (mt==1); prefill
        # kv_b1 (mt>1) keeps the auto block per its own sweep winner.
        kvb1_bw = k // ttnn.TILE_SIZE if (k == 192 and n == 512 and mt == 1) else None
        return prefill_per_head_linear(
            a,
            b,
            device=device,
            compute_kernel_config=ckc,
            memory_config=kvb_mc,
            in0_block_w=kvb1_bw,
        )
    if b_batch == 1:
        if m_total > ttnn.TILE_SIZE:
            return _prefill_linear_ws_out(a, b, device=device, cfg=cfg, memory_config=memory_config)
        # Swept-optimal decode config for known (K, N) shapes (q_a, q_b, head-parallel w_o).
        tuned = _DECODE_MATMUL_TUNED.get((int(b.shape[-2]), int(b.shape[-1])))
        if tuned is not None:
            tuned_out = _tuned_decode_linear(a, b, device=device, cfg=cfg, memory_config=memory_config, tuned=tuned)
            if tuned_out is not None:
                return tuned_out
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
    act = a
    copied_act = False
    if _is_l1_width_sharded(a):
        act, copied_act = _to_l1_if_needed(a)
    kwargs: dict[str, object] = {}
    mc = memory_config if memory_config is not None else cfg.decode_act_mc
    if mc is not None:
        kwargs["memory_config"] = mc
    kwargs["compute_kernel_config"] = cfg.mlp_compute_kernel_config()
    m_total = 1
    for i in range(len(a.shape) - 1):
        m_total *= int(a.shape[i])
    if m_total > ttnn.TILE_SIZE:
        out = _prefill_linear_ws_out(act, b, device=device, cfg=cfg, memory_config=memory_config)
        if copied_act:
            ttnn.deallocate(act, force=False)
        return out
    kwargs["program_config"] = compute_1d_mlp_down_prog_cfg(device, b, m_total)
    out = ttnn.linear(act, b, **kwargs)
    if copied_act:
        ttnn.deallocate(act, force=False)
    return out


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
    return_sharded: bool = False,
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

    # Prefer per_core_N=4 (swept-optimal for dense gate/up N=2560 → 20 cores, 1.4x vs
    # the 80-core baseline); fall back to 2 for shapes (e.g. shared-expert) where 4
    # doesn't divide n_tiles — preserving the previous behavior there.
    per_core_N = next((p for p in (4, _GATE_UP_PER_CORE_N) if n_tiles % p == 0), None)
    if m_total > ttnn.TILE_SIZE or b_batch != 1 or per_core_N is None:
        return mlp_linear(a, b, device=device, cfg=cfg, memory_config=memory_config)
    out_subblock_w = per_core_N  # out_subblock_h=1 → h*w = per_core_N ≤ 8 (bf16 DST), ≤4 (fp32)

    num_cores = n_tiles // per_core_N
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
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=out_subblock_w,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    # WIDTH_SHARDED output: each core writes its result shard to local L1; to_memory_config gathers downstream.
    out_shard_h = m_tiles * ttnn.TILE_SIZE
    out_shard_w = per_core_N * ttnn.TILE_SIZE
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

    if return_sharded:
        return out_sharded
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
_KVA_IN0_BLOCK_W = 8  # K=2048 → 64 k-tiles; larger K-block halves the K-loop on this
# weight-BW-bound matmul (probe 2048x1344 @ 21 cores: bw=4 15.5µs → bw=8 10.7µs, 1.45x;
# bw>8 no further gain at this grid). _auto_in0_block_w caps at 4 and misses this.
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
    # w_o is always row-parallel under TP (even with ATTN_DP=1).
    if cfg.tp_enabled:
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
