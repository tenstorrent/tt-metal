# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Native TTNNModule vision transformer for dots.ocr.

Full pipeline: PatchEmbed -> 42 VisionBlocks -> post-trunk RMSNorm -> PatchMerger.
All sub-modules are native TTNNModules with proper lifecycle (preprocess_weights,
move_weights_to_device, forward).

Includes: 2D RoPE, vision RMSNorm, SwiGLU MLP, patch embedding, vision attention,
vision block, patch merger, and the top-level vision tower.
"""

from __future__ import annotations


import os

import torch
import torch.nn.functional as F
import ttnn
from ttnn.device import is_wormhole_b0
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

from models.experimental.tt_symbiote.core.module import TTNNModule, TTNNLayerStack
from models.experimental.tt_symbiote.modules.linear import (
    _ccl_num_links,
    _tp_mesh_mapper,
    _tp_requires_ccl,
)
from ttnn.operations.transformer import SDPAProgramConfig

# Tracy (perf): vision Matmul/SDPA show HiFi4; use lower fidelity for ViT matmul/SDPA only.
# Norms (RMS/LayerNorm) keep HiFi4 for stability.
# QKV/O attn matmul lowered to LoFi to match the MLP path -- weights are already
# BFP8 so the LoFi multiplication delta lands inside the existing BFP8 output
# quantization noise. Saves ~50% on the per-layer attn matmul time at the
# 60.9 / 130 TFLOPs peak HiFi2 ceiling we were hitting in tracy.
VISION_MATMUL_MATH_FIDELITY = ttnn.MathFidelity.LoFi
VISION_SDPA_MATH_FIDELITY = ttnn.MathFidelity.LoFi
# All vision matmuls / SDPA / norms run at LoFi -- the residual stream
# carries BFP8 activations (post-attention out, post-MLP out, post-norm
# out) so the higher norm fidelity was being thrown away into a coarser
# downstream representation anyway.
# RMSNorm/LayerNorm at LoFi: when the input residual stream is BFP8 the
# multiplication precision needed for the variance reduce is bounded by
# BFP8's per-tile shared exponent (~7 mantissa bits effective on a tile of
# correlated magnitudes), so dropping from HiFi2 to LoFi is in the noise
# floor of the BFP8 quantization upstream.
VISION_NORM_MATH_FIDELITY = ttnn.MathFidelity.LoFi


def _vision_tower_signpost(header: str) -> None:
    """Emit a Tracy ``TT_SIGNPOST`` marker (visible in device op logs / perf reports)."""
    try:
        from tools.tracy import signpost
    except ImportError:
        return
    signpost(header)


def _tp4_prefill_vision_enabled(device=None) -> bool:
    """True when Wormhole TP4 prefill vision optimizations should be used."""
    body_env = os.environ.get("DOTS_OCR_TEXT_BODY")
    if body_env is not None:
        return body_env.strip().lower() in {"tp4", "tp4_prefill"}
    if device is None or not is_wormhole_b0() or not hasattr(device, "shape"):
        return False
    shape = tuple(int(x) for x in device.shape)
    return len(shape) == 2 and int(shape[-1]) == 4


def _vision_tensor_mem_tag(t: ttnn.Tensor) -> str:
    """Compact memory-config label: buffer (L1/DRAM) + interleaved vs sharded layout."""
    mem = t.memory_config()
    buf = "L1" if mem.buffer_type == ttnn.BufferType.L1 else "DRAM"
    layout = mem.memory_layout
    if layout == ttnn.TensorMemoryLayout.INTERLEAVED or not mem.is_sharded():
        placement = "INTERLEAVED"
    elif layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        placement = "BLOCK_SHARDED"
    elif layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        placement = "WIDTH_SHARDED"
    elif layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        placement = "HEIGHT_SHARDED"
    else:
        placement = str(layout).split(".")[-1] if layout is not None else "UNKNOWN"
    tag = f"{buf}/{placement}"
    shard_spec = mem.shard_spec
    if shard_spec is not None:
        tag += f" grid={shard_spec.grid} shard={tuple(shard_spec.shape)}"
    return tag


def _vision_debug_mem(label: str, t: ttnn.Tensor) -> None:
    """Print tensor shape, dtype, and memory placement between vision-block ops."""
    print(f"[vision_block] {label}: shape={list(t.shape)} dtype={t.dtype} mem={_vision_tensor_mem_tag(t)}")


def _align_vision_sdpa_seq_len(seq_len: int) -> int:
    """Round sequence length up to a tile multiple for vision SDPA.

    Padded keys are masked out via ``attn_mask`` so attention matches the
    unpadded sequence while SDPA sees tile-friendly extents (less implicit
    tilize/untilize work inside the kernel).
    """
    s = int(seq_len)
    return max(32, ((s + 31) // 32) * 32)


def _largest_divisor_le(value: int, limit: int) -> int:
    for c in range(min(value, limit), 0, -1):
        if value % c == 0:
            return c
    return 1


# Cache program_config objects per (grid, m, k, n) so trace capture sees the
# same Python object across calls. Recreating the config in the forward pass
# every iteration breaks ttnn trace dedup and silently drops the entire
# decode/prefill back to eager-mode dispatch (per-op kernel launches), which
# in TP2 makes decode 4x slower than the trace-replay path.
_VISION_MATMUL_PC_CACHE: dict = {}
_VISION_MATMUL_BS_MEM_CACHE: dict = {}
_VISION_O_PROJ_BS_PC_CACHE: dict = {}
_VISION_MLP_DOWN_L1_PC_CACHE: dict = {}
_VISION_MERGER_FC1_BS_PC_CACHE: dict = {}
_VISION_MERGER_FC2_BS_PC_CACHE: dict = {}


def _vision_l1_interleaved_bytes_per_core(tensor: ttnn.Tensor, device) -> int:
    """Per-core L1 bytes of an interleaved activation (norm2 output co-resident with gate)."""
    mem = tensor.memory_config()
    if mem.buffer_type != ttnn.BufferType.L1 or mem.is_sharded():
        return 0
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import _l1_shard_bytes_per_core

    m_dim = 1
    for i in range(len(tensor.shape) - 1):
        m_dim *= int(tensor.shape[i])
    k_dim = int(tensor.shape[-1])
    if m_dim % 32 or k_dim % 32:
        return 0
    return int(_l1_shard_bytes_per_core(device, m_dim, k_dim, tensor.dtype))


def _vision_tp_gate_up_program_config(
    device,
    m_dim: int,
    k_dim: int,
    n_dim: int,
    in0_dtype: ttnn.DataType,
    l1_resident_bytes_per_core: int,
):
    """Gate/up matmul PC for TP vision MLP with optional L1-resident norm2 in0.

    Pin the hand-swept ``wh_tp4_mlp_gate_up_pc`` (obh=11, ibw=8, sub 1x5) for the
    standard prefill shape even when norm2 is L1-resident: silicon sweep shows
    ~0.59 ms vs ~1.3 ms for the conservative ``wh_tp4_matmul_pc`` fallback (obh=4).
    The generic search double-counts the in0 CB on top of the resident shard and
    never considers obh=11.
    """
    from models.experimental.tt_symbiote.modules.vision_tp4_wh import wh_tp4_matmul_pc, wh_tp4_mlp_gate_up_pc

    if m_dim == 11264 and k_dim == 1536 and n_dim == 1056:
        pinned = wh_tp4_mlp_gate_up_pc(device)
        if pinned is not None:
            return pinned

    if _tp4_prefill_vision_enabled(device):
        from models.experimental.tt_symbiote.modules.vision_tp4 import tp4_mlp_gate_up_pc

        pinned = tp4_mlp_gate_up_pc(device)
        if int(l1_resident_bytes_per_core) <= 0:
            return pinned
        pc = wh_tp4_matmul_pc(
            device,
            m_dim,
            k_dim,
            n_dim,
            in0_dtype=in0_dtype,
            out_dtype=ttnn.bfloat8_b,
            l1_resident_bytes_per_core=int(l1_resident_bytes_per_core),
        )
        if pc is not None:
            return pc

    pc = _vision_matmul_program_config(
        device, m_dim, k_dim, n_dim, l1_resident_bytes_per_core=int(l1_resident_bytes_per_core)
    )
    if pc is not None:
        return pc

    if int(l1_resident_bytes_per_core) > 0:
        return wh_tp4_matmul_pc(
            device,
            m_dim,
            k_dim,
            n_dim,
            in0_dtype=in0_dtype,
            out_dtype=ttnn.bfloat8_b,
            l1_resident_bytes_per_core=int(l1_resident_bytes_per_core),
        )
    return None


def _vision_mlp_dram_width_sharded_mem_config(device, k: int, n: int) -> ttnn.MemoryConfig:
    """DRAM WIDTH_SHARDED weight with N split across the matmul compute grid (8 cols).

    12 DRAM-bank sharding makes ``num_blocks_x=12`` in the 2D-mcast kernel, which
    exceeds ``compute_with_storage_grid_size().x`` (8). Use the same ``per_core_N``
    as ``_vision_matmul_program_config`` (17 for N=4224).
    """
    tile = 32
    grid_x = int(device.compute_with_storage_grid_size().x)
    n_tiles = n // tile
    per_core_n = (n_tiles + grid_x - 1) // grid_x
    padded_n = per_core_n * grid_x * tile
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, 0))])
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        (k, padded_n // grid_x),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def _vision_mlp_to_dram_width_sharded_weight(
    device,
    weight_torch: torch.Tensor,
    bias_torch: torch.Tensor | None,
    *,
    weight_dtype: ttnn.DataType,
):
    """Upload ``[out, in]`` torch linear weight as DRAM WIDTH_SHARDED ``[K, N]`` tiles."""
    weight_t = weight_torch.T.contiguous()
    k_dim = int(weight_t.shape[0])
    n_dim = int(weight_t.shape[1])
    tt_weight = ttnn.as_tensor(
        weight_t,
        device=device,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_vision_mlp_dram_width_sharded_mem_config(device, k=k_dim, n=n_dim),
    )
    tt_bias = None
    if bias_torch is not None:
        bias_2d = bias_torch.reshape((1, -1))
        n_bias = int(bias_2d.shape[-1])
        tt_bias = ttnn.as_tensor(
            bias_2d,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_vision_mlp_dram_width_sharded_mem_config(device, k=ttnn.TILE_SIZE, n=n_bias),
        )
    return tt_weight, tt_bias


def _vision_matmul_program_config(device, m_dim: int, k_dim: int, n_dim: int, *, l1_resident_bytes_per_core: int = 0):
    """2D-mcast prefill matmul config for vision-tower DRAM-interleaved matmuls.

    Tracy reports every vision matmul as ``SLOW`` because none of the vision
    ttnn.linear calls passed an explicit program_config -- the auto-config
    falls back to ``in0_block_w=1`` and small subblocks, which is why
    QKV/MLP land at 26-29% TFLOPs and 12-13% DRAM. Setting a 2D-mcast config
    with a larger ``in0_block_w`` reuses the L1-resident weight block across
    more output tiles.

    The 2D-mcast factory hard-asserts
    ``num_blocks_y == m_tiles / per_core_M <= grid_y``, so for vision shapes
    (M=12288 -> m_tiles=384, grid_y=8) we must run with ``per_core_M=48``.
    With BFP8 output the natural ``per_core_M*per_core_N`` output CB is
    ~3.4 MB which doesn't fit in 1.5 MB L1; we use ``out_block_h`` to chunk
    the in-flight output and bound the actual L1 footprint to
    ``out_block_h * per_core_N * tile_bytes`` instead of the full per-core
    output. The kernel iterates ``per_core_M / out_block_h`` blocks per
    core (matmul_multicore_reuse_mcast_2d_program_factory.cpp:111), so the
    total work is unchanged but the L1 fits.

    Returns ``None`` for shapes that don't tile cleanly so callers fall
    back to the auto-config.
    """
    if device is None:
        return None
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    tile = 32

    l1_resident = int(l1_resident_bytes_per_core)
    cache_key = (grid_x, grid_y, m_dim, k_dim, n_dim, l1_resident)
    cached = _VISION_MATMUL_PC_CACHE.get(cache_key)
    if cached is not None or cache_key in _VISION_MATMUL_PC_CACHE:
        return cached
    l1_cb_budget_kb = max(256, 1024 - (l1_resident // 1024))

    if m_dim % tile != 0 or k_dim % tile != 0 or n_dim % tile != 0:
        _VISION_MATMUL_PC_CACHE[cache_key] = None
        return None

    m_tiles = m_dim // tile
    k_tiles = k_dim // tile
    n_tiles = n_dim // tile

    if m_tiles % grid_y != 0:
        _VISION_MATMUL_PC_CACHE[cache_key] = None
        return None

    # ``num_blocks_x = ceil(n_tiles / per_core_n)`` in the 2D mcast factory
    # (matmul_multicore_reuse_mcast_2d_program_factory.cpp:1669), so n_tiles
    # doesn't need to divide grid_x cleanly -- the kernel pads the trailing
    # output tiles internally. This unlocks the MLP gate/up matmuls
    # (12288 x 1536 x 4224, n_tiles=132) which were falling back to the
    # auto-config because 132 % 8 != 0. With per_core_n=ceil(132/8)=17 the
    # 8x8 grid covers 136 tiles (4 padded) and we get the same in0_block_w
    # tuning as the divisible matmuls.
    per_core_m = m_tiles // grid_y
    per_core_n = (n_tiles + grid_x - 1) // grid_x

    if per_core_n > 24 or per_core_m > 64:
        _VISION_MATMUL_PC_CACHE[cache_key] = None
        return None

    # Pick the largest in0_block_w (cap at 8) that divides K_tiles cleanly.
    # Larger in0_block_w => fewer K iterations and better DRAM-to-L1
    # weight-block reuse. For vision QKV (K=48 tiles) this picks 8 (vs the
    # auto-config's 1) -> 6 K iterations instead of 48.
    in0_block_w = _largest_divisor_le(k_tiles, 8)

    # Joint (out_block_h, out_subblock_h, out_subblock_w) selection.
    #
    # Compute is dominated by the number of MMU-style ``matmul_tiles``
    # instructions issued, which equals ``per_core_M * per_core_N * K_tiles
    # / (in0_block_w * dst_area)``. So the *primary* lever for compute
    # throughput is ``dst_area = out_subblock_h * out_subblock_w``: 8/8
    # tiles fills the DST register file (LoFi, fp32_dest_acc_en=False) and
    # is 2x more arithmetic per instruction than 4/8.
    #
    # The trap that the original heuristic fell into: when ``per_core_N``
    # is prime (e.g. 17 for the MLP gate/up at N=4224, n_tiles=132,
    # per_core_n=ceil(132/8)=17) the ``out_subblock_w`` loop can only pick
    # 1, so DST area depends entirely on ``out_subblock_h``. With the
    # default ``out_block_h=12`` the largest h that divides it is 4
    # (12 % 8 != 0), capping the subblock at (4,1) = 4/8 DST. Tracy showed
    # this matmul running at 28.3% FLOPs vs 48% for the same-FLOPs MLP
    # down (which has per_core_n=6, factors {1,2,3,6}, hits area=8 easily).
    #
    # Fix: enumerate candidate ``out_block_h`` values, compute the best
    # achievable ``dst_area`` for each, and pick (dst_area, out_block_h)
    # lexicographically -- max DST utilization first, then largest
    # ``out_block_h`` (= fewest outer-M iters = fewest weight DRAM
    # re-reads, since the kernel re-streams weights per outer-M iter
    # in matmul_multicore_reuse_mcast_2d_program_factory.cpp:111). The
    # per-core L1 budget caps ``out_block_h``: BF16 inputs/intermediates
    # at out_block_h=16 with per_core_n>=17 push interm0 past ~558 KB
    # which combined with the in0/in1/out CBs trips ``program.cpp:934``
    # at ~1.7 MB. Empirically out_block_h=8 fits all four vision shapes
    # at full DST when out_block_h=12 doesn't.
    dst_tiles_budget = 8
    candidate_out_block_h = [16, 12, 8, 6, 4, 3, 2, 1]
    best_area = 0
    best_out_block_h = 1
    best_subblock_h = 1
    best_subblock_w = 1
    for ob_h in candidate_out_block_h:
        if ob_h > per_core_m or per_core_m % ob_h != 0:
            continue

        # L1 sanity cap: when N is large enough that the BF16 interm0 CB
        # (out_block_h * per_core_n * 2048 B) plus the BF16 in0 CB
        # (out_block_h * in0_block_w * 2 * 2048 B) push past ~1.0 MB on
        # their own, skip this candidate. The matmul kernel also allocates
        # in1 + out + bias CBs on top, so leaving headroom here avoids the
        # 1.5 MB-per-core L1 clash that bigger blocks trigger on the
        # large-N (per_core_n>=17) vision matmuls.
        approx_interm_kb = (ob_h * per_core_n * 2048) // 1024
        approx_in0_kb = (ob_h * in0_block_w * 2 * 2048) // 1024
        if approx_interm_kb + approx_in0_kb > l1_cb_budget_kb:
            continue

        # Best DST area achievable with this out_block_h
        cand_area = 0
        cand_h = 1
        cand_w = 1
        for h in range(min(ob_h, dst_tiles_budget), 0, -1):
            if ob_h % h != 0:
                continue
            for w in range(min(per_core_n, dst_tiles_budget // h), 0, -1):
                if per_core_n % w != 0:
                    continue
                area = h * w
                if area > cand_area:
                    cand_area = area
                    cand_h = h
                    cand_w = w
                    break

        # Lexicographic preference: (dst_area, out_block_h) DESC.
        if (cand_area > best_area) or (cand_area == best_area and ob_h > best_out_block_h):
            best_area = cand_area
            best_out_block_h = ob_h
            best_subblock_h = cand_h
            best_subblock_w = cand_w

    if best_area <= 0:
        _VISION_MATMUL_PC_CACHE[cache_key] = None
        return None

    out_block_h = best_out_block_h
    out_block_w = per_core_n
    out_subblock_h = best_subblock_h
    out_subblock_w = best_subblock_w

    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    _VISION_MATMUL_PC_CACHE[cache_key] = pc
    return pc


def _vision_block_sharded_mem(device, m_dim: int, wid_dim: int, *, grid_size: tuple[int, int] | None = None):
    """Cached L1 BLOCK_SHARDED MemoryConfig splitting m_dim across grid_y rows and wid_dim across grid_x columns.

    Returns None if either dimension is not evenly divisible by the grid extents.
    """
    if device is None:
        return None
    grid = device.compute_with_storage_grid_size()
    device_grid_x, device_grid_y = int(grid.x), int(grid.y)
    if grid_size is None:
        grid_x, grid_y = device_grid_x, device_grid_y
    else:
        grid_x, grid_y = int(grid_size[0]), int(grid_size[1])
        if grid_x > device_grid_x or grid_y > device_grid_y:
            return None
    if m_dim % grid_y != 0 or wid_dim % grid_x != 0:
        return None
    cache_key = (grid_x, grid_y, m_dim, wid_dim)
    if cache_key in _VISION_MATMUL_BS_MEM_CACHE:
        return _VISION_MATMUL_BS_MEM_CACHE[cache_key]
    shard_h = m_dim // grid_y
    shard_w = wid_dim // grid_x
    core_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    result = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_set, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )
    _VISION_MATMUL_BS_MEM_CACHE[cache_key] = result
    return result


def _vision_o_proj_bs_program_config(device):
    """Program config for o_proj with L1-interleaved in0 and L1 BLOCK_SHARDED output (12288×1536×1536).

    matmul_device_operation.cpp:841 requires out_subblock_w == per_core_N OR
    out_subblock_h == 1 whenever the output is sharded. The generic
    _vision_matmul_program_config returns out_subblock_h=8 / out_subblock_w=1 for this
    shape (area=8, optimal for DRAM output) which violates the constraint for sharded
    output. Force out_subblock_h=1, out_subblock_w=6 (= per_core_N).
    out_block_h=6 keeps the static CB region below the model's pre-allocated L1 shard
    address (~552 KB). Empirically: ob_h=48→1366 KB, ob_h=16→683 KB, so
    cb_end ≈ 325 KB + 22 KB×ob_h; need cb_end < 552 KB → ob_h ≤ 10 → use ob_h=6.
    Returns None when the device grid is smaller than 8×8.
    """
    if device is None:
        return None
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    if grid_x < 8 or grid_y < 8:
        _VISION_O_PROJ_BS_PC_CACHE[(grid_x, grid_y)] = None
        return None
    cache_key = (grid_x, grid_y)
    if cache_key in _VISION_O_PROJ_BS_PC_CACHE:
        return _VISION_O_PROJ_BS_PC_CACHE[cache_key]
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=6,
        out_block_h=6,
        out_block_w=6,
        per_core_M=48,
        per_core_N=6,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    _VISION_O_PROJ_BS_PC_CACHE[cache_key] = pc
    return pc


def _vision_mlp_down_l1_pc(device):
    """Program config for MLP down-projection when gate_up_mul is L1 interleaved (12288×4224×1536).

    gate_up_mul L1 interleaved occupies 841 KB/core. The generic down_pc picks
    out_block_h=16 → static CBs ~634 KB → total 1474 KB (only 62 KB margin).
    Forcing out_block_h=8 cuts CBs to ~487 KB → total 1328 KB (208 KB margin).
    out_subblock_h=4, out_subblock_w=2 still fills the 8-tile DST register
    (4×2=8) so compute efficiency is unchanged vs ob_h=16.
    Returns None when the device grid is smaller than 8×8.
    """
    if device is None:
        return None
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    if grid_x < 8 or grid_y < 8:
        _VISION_MLP_DOWN_L1_PC_CACHE[(grid_x, grid_y)] = None
        return None
    cache_key = (grid_x, grid_y)
    if cache_key in _VISION_MLP_DOWN_L1_PC_CACHE:
        return _VISION_MLP_DOWN_L1_PC_CACHE[cache_key]
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=6,
        out_subblock_h=4,
        out_subblock_w=2,
        out_block_h=8,
        out_block_w=6,
        per_core_M=48,
        per_core_N=6,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    _VISION_MLP_DOWN_L1_PC_CACHE[cache_key] = pc
    return pc


_VISION_MERGER_FC1_FAST_PC_CACHE: dict = {}


def _vision_merger_fc1_fast_pc(device, m_dim: int):
    """2D-mcast fc1 for patch-merger ``M×6144×6144`` (BFP8 in0 × BFP8 w → BFP4, GELU fused).

    The block-sharded fc1 (``_vision_merger_fc1_bs_program_config``) runs at
    in0_block_w=2 → ~96 K-iterations and ~5.7 ms / 16% TFLOPs (the "SLOW" merger
    matmul in the perf report). This matmul is weight-DRAM-bound (75 MB BFP8
    weight), so the lever is out_block_h = #weight-DRAM passes. With per_core_M=11
    (M=2816 → 88 m-tiles / 8 rows) the only obh>1 divisor is 11 = a single weight
    pass. obh=11 only fits L1 at in0_block_w<=4 (ibw=6 OOMs the CBs); silicon
    sweep 2026-06-11 picks ibw=4 → ~2.1 ms (2.5x). N=6144 → per_core_N=24 over
    8 columns. Returns None off the production WH 8×8 / M=2816 shape so callers
    fall back to the block-sharded path.
    """
    if device is None:
        return None
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    tile = 32
    cache_key = (grid_x, grid_y, m_dim)
    if cache_key in _VISION_MERGER_FC1_FAST_PC_CACHE:
        return _VISION_MERGER_FC1_FAST_PC_CACHE[cache_key]
    m_tiles = m_dim // tile
    # Needs the 8×8 grid, M tiling cleanly over 8 rows with per_core_M==out_block_h
    # (one weight pass), and the production mlp_size=6144 (n_tiles=192 → N=24).
    if grid_x < 8 or grid_y < 8 or m_dim % tile != 0 or m_tiles % 8 != 0:
        _VISION_MERGER_FC1_FAST_PC_CACHE[cache_key] = None
        return None
    per_core_m = m_tiles // 8
    if per_core_m > 64:
        _VISION_MERGER_FC1_FAST_PC_CACHE[cache_key] = None
        return None
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=8,
        out_block_h=per_core_m,
        out_block_w=24,
        per_core_M=per_core_m,
        per_core_N=24,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, False),
        fuse_batch=False,
    )
    _VISION_MERGER_FC1_FAST_PC_CACHE[cache_key] = pc
    return pc


def _vision_merger_fc1_bs_program_config(device, m_dim: int):
    """Program config for patch-merger fc1 (3072×6144×6144) with L1-interleaved
    in0 and L1 BLOCK_SHARDED output (interleaved-in0 / sharded-out pattern).

    The output is block-sharded over an 8×8 core rectangle, even on devices with
    taller compute grids. For M=3072 the shard is [M/8, N/8] = [384, 768]
    (= 12×24 tiles); for smaller buckets (for example M=2816) per_core_M is
    derived from M. The sharded output requires ``out_subblock_h == 1`` since
    ``out_subblock_w (8) != per_core_N (24)``.

    in0 is L1-interleaved BF16 (~590 KB/core; the norm cannot emit BFP8 without a
    typecast op). in0_block_w is held to 2 so the BF16 in0 + sharded-output CBs
    still fit the 1,395 KB bank at out_block_h<=6. GELU is fused. Returns None
    when the device grid is smaller than 8×8 or M does not tile across 8 rows.
    """
    if device is None:
        return None
    grid = device.compute_with_storage_grid_size()
    device_grid_x, device_grid_y = int(grid.x), int(grid.y)
    grid_x, grid_y = 8, 8
    tile = 32
    if device_grid_x < grid_x or device_grid_y < grid_y or m_dim % (tile * grid_y) != 0:
        _VISION_MERGER_FC1_BS_PC_CACHE[(grid_x, grid_y, m_dim)] = None
        return None
    cache_key = (grid_x, grid_y, m_dim)
    if cache_key in _VISION_MERGER_FC1_BS_PC_CACHE:
        return _VISION_MERGER_FC1_BS_PC_CACHE[cache_key]
    per_core_m = (m_dim // tile) // grid_y
    per_core_n = 24
    out_block_h = 1
    for candidate in (6, 4, 3, 2, 1):
        if candidate <= per_core_m and per_core_m % candidate == 0:
            out_block_h = candidate
            break
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=8,
        out_block_h=out_block_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, False),
        fuse_batch=False,
    )
    _VISION_MERGER_FC1_BS_PC_CACHE[cache_key] = pc
    return pc


def _vision_merger_fc2_bs_program_config(device, m_dim: int):
    """Program config for patch-merger fc2 (3072×6144×1536) when its in0 is the
    L1 BLOCK_SHARDED fc1 output and its output stays L1 interleaved.

    in0 shard is [M/8, K/8] on the same 8×8 core rectangle as fc1, so no reshard
    is needed between fc1 and fc2. ``fuse_batch=True`` is mandatory for sharded in0.
    The output is interleaved (not sharded), so choose subblocks from the derived
    per_core_M while keeping as much of the 8-tile DST as possible. Returns None
    when the device grid is smaller than 8×8 or M does not tile across 8 rows.
    """
    if device is None:
        return None
    grid = device.compute_with_storage_grid_size()
    device_grid_x, device_grid_y = int(grid.x), int(grid.y)
    grid_x, grid_y = 8, 8
    tile = 32
    if device_grid_x < grid_x or device_grid_y < grid_y or m_dim % (tile * grid_y) != 0:
        _VISION_MERGER_FC2_BS_PC_CACHE[(grid_x, grid_y, m_dim)] = None
        return None
    cache_key = (grid_x, grid_y, m_dim)
    if cache_key in _VISION_MERGER_FC2_BS_PC_CACHE:
        return _VISION_MERGER_FC2_BS_PC_CACHE[cache_key]
    per_core_m = (m_dim // tile) // grid_y
    per_core_n = 6
    out_block_h = per_core_m
    best_area = 0
    out_subblock_h, out_subblock_w = 1, 1
    for h in range(min(out_block_h, 8), 0, -1):
        if out_block_h % h != 0:
            continue
        for w in range(min(per_core_n, 8 // h), 0, -1):
            if per_core_n % w != 0:
                continue
            area = h * w
            if area > best_area:
                best_area = area
                out_subblock_h, out_subblock_w = h, w
            break
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=8,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=out_block_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    _VISION_MERGER_FC2_BS_PC_CACHE[cache_key] = pc
    return pc


def _vision_matmul_compute_config(device, *, math_fidelity: ttnn.MathFidelity) -> ttnn.DeviceComputeKernelConfig:
    """Compute config for vision linear/matmul ops.

    ``packer_l1_acc=True`` keeps partial output accumulators in L1 across the
    K-loop iterations rather than spilling to DRAM each pass, which directly
    improves throughput on long-K matmuls (K=1536, K=4224 in the dots vision
    MLP). ``math_approx_mode=True`` enables the fast polynomial path for any
    fused transcendental activation (e.g. SILU on the gate matmul).
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _vision_sdpa_compute_config(device, *, math_fidelity: ttnn.MathFidelity) -> ttnn.DeviceComputeKernelConfig:
    """Compute config for vision SDPA and norm ops.

    ``packer_l1_acc=True`` keeps the chunked-SDPA partial accumulators in L1
    instead of round-tripping through DRAM, which directly speeds up the
    inner softmax-reduce loop. ``math_approx_mode=True`` uses the fast
    polynomial path for transcendentals (matches the decode-mode SDPA).
    """
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


# ---------------------------------------------------------------------------
# 2D Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------


def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    """Rotate-half helper for RoPE: [-x2, x1] from [x1, x2]."""
    last = x.shape[-1]
    half = last // 2
    x1 = ttnn.slice(x, (0, 0, 0, 0), (x.shape[0], x.shape[1], x.shape[2], half))
    x2 = ttnn.slice(x, (0, 0, 0, half), (x.shape[0], x.shape[1], x.shape[2], last))
    neg_x2 = ttnn.mul(x2, -1, use_legacy=False)
    return ttnn.concat([neg_x2, x1], dim=-1)


def apply_rotary_tt(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    out_dtype=None,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Apply rotary embedding to Q and K tensors in fp32 then cast back."""
    if out_dtype is None:
        out_dtype = ttnn.bfloat8_b

    f32 = getattr(ttnn, "float32", None)

    if f32 is not None:
        qf = ttnn.typecast(q, dtype=f32)
        kf = ttnn.typecast(k, dtype=f32)
        cos_f = ttnn.typecast(cos, dtype=f32)
        sin_f = ttnn.typecast(sin, dtype=f32)
    else:
        qf, kf, cos_f, sin_f = q, k, cos, sin

    q_embed = ttnn.add(
        ttnn.mul(qf, cos_f, use_legacy=False), ttnn.mul(_rotate_half(qf), sin_f, use_legacy=False), dtype=ttnn.bfloat8_b
    )
    k_embed = ttnn.add(
        ttnn.mul(kf, cos_f, use_legacy=False), ttnn.mul(_rotate_half(kf), sin_f, use_legacy=False), dtype=ttnn.bfloat8_b
    )

    if f32 is not None and out_dtype is not None:
        q_embed = ttnn.typecast(q_embed, dtype=out_dtype)
        k_embed = ttnn.typecast(k_embed, dtype=out_dtype)

    return q_embed, k_embed


class TTNNDotsVision2DRoPE:
    """2D factored RoPE for Dots vision attention.

    Not a TTNNModule -- no learnable weights. Produces cos/sin tensors
    and cu_seqlens given grid_thw and a device reference.
    """

    def __init__(
        self,
        *,
        device,
        head_dim: int = 128,
        spatial_merge_size: int = 2,
        theta: float = 10000.0,
    ):
        self.device = device
        self.head_dim = head_dim
        self.spatial_merge_size = spatial_merge_size
        self.theta = theta

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")

        self.rotary_dim = head_dim // 2
        if self.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {self.rotary_dim}")

        self._inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        self._padded_cache_key = None
        self._padded_cache_rot_mats = None
        self._padded_cache_cu_seqlens = None

    def _compute_cos_sin_torch(
        self,
        grid_thw: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Compute 2D RoPE cos/sin tables on CPU, returning (cos_full, sin_full, cu_seqlens).

        Separates CPU computation from device upload so callers can optionally
        pre-pad (CPU torch.cat) before a single from_torch, avoiding device
        PadDeviceOperation / FillPadDeviceOperation on the critical path.
        """
        g = grid_thw.detach().cpu() if getattr(grid_thw, "is_cuda", False) else grid_thw
        if g.dim() != 2 or g.shape[1] != 3:
            raise ValueError(f"grid_thw must be [N,3], got {g.shape}")

        token_counts = [int(t) * int(h) * int(w) for t, h, w in g.tolist()]
        expected = sum(token_counts)
        if seq_len != expected:
            raise ValueError(f"seq_len={seq_len} != grid_thw total={expected}")

        sms = self.spatial_merge_size
        inv_freq = self._inv_freq

        hpos_segments = []
        wpos_segments = []
        cu = [0]
        running = 0

        for t, h, w in g.tolist():
            t, h, w = int(t), int(h), int(w)
            if h % sms != 0 or w % sms != 0:
                raise ValueError(f"grid {h}x{w} not divisible by spatial_merge_size={sms}")

            h_ids = torch.arange(h, dtype=torch.float32)
            w_ids = torch.arange(w, dtype=torch.float32)

            h_grid = h_ids.unsqueeze(1).expand(h, w)
            w_grid = w_ids.unsqueeze(0).expand(h, w)

            h_grid = h_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)
            w_grid = w_grid.reshape(h // sms, sms, w // sms, sms).permute(0, 2, 1, 3).reshape(-1)

            if t > 1:
                h_grid = h_grid.repeat(t)
                w_grid = w_grid.repeat(t)

            hpos_segments.append(h_grid)
            wpos_segments.append(w_grid)
            running += t * h * w
            cu.append(running)

        hpos_all = torch.cat(hpos_segments) if len(hpos_segments) > 1 else hpos_segments[0]
        wpos_all = torch.cat(wpos_segments) if len(wpos_segments) > 1 else wpos_segments[0]

        freqs_h = hpos_all.unsqueeze(1) * inv_freq.unsqueeze(0)
        freqs_w = wpos_all.unsqueeze(1) * inv_freq.unsqueeze(0)

        cos_half = torch.cat([torch.cos(freqs_h), torch.cos(freqs_w)], dim=-1)
        sin_half = torch.cat([torch.sin(freqs_h), torch.sin(freqs_w)], dim=-1)

        cos_full = torch.cat([cos_half, cos_half], dim=-1).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        sin_full = torch.cat([sin_half, sin_half], dim=-1).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

        return cos_full, sin_full, cu

    def build(
        self,
        grid_thw: torch.Tensor,
        seq_len: int,
    ) -> tuple[tuple[ttnn.Tensor, ttnn.Tensor], list[int]]:
        """Build 2D RoPE cos/sin tables and cu_seqlens for vision attention.

        Returns cu_seqlens as a Python list to avoid repeated device-to-host syncs.
        """
        cos_full, sin_full, cu = self._compute_cos_sin_torch(grid_thw, seq_len)

        mem = ttnn.DRAM_MEMORY_CONFIG
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        cos_tt = ttnn.from_torch(
            cos_full,
            device=self.device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        sin_tt = ttnn.from_torch(
            sin_full,
            device=self.device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        return (cos_tt, sin_tt), cu

    def build_padded(
        self,
        grid_thw: torch.Tensor,
        actual_seq_len: int,
        bucket_size: int,
    ) -> tuple[tuple, list[int]]:
        """Build 2D RoPE cos/sin padded to bucket_size for trace compatibility.

        Pads cos/sin on CPU (torch.cat) before the single from_torch upload,
        eliminating the two device PadDeviceOperation / FillPadDeviceOperation
        that the original ttnn.pad calls generated on the critical path.
        """
        g = grid_thw.detach().cpu() if getattr(grid_thw, "is_cuda", False) else grid_thw
        cache_key = (tuple(int(x) for x in g.reshape(-1).tolist()), int(actual_seq_len), int(bucket_size))
        if cache_key == self._padded_cache_key and self._padded_cache_rot_mats is not None:
            return self._padded_cache_rot_mats, self._padded_cache_cu_seqlens

        cos_full, sin_full, cu_seqlens = self._compute_cos_sin_torch(grid_thw, actual_seq_len)

        if actual_seq_len < bucket_size:
            pad_len = bucket_size - actual_seq_len
            head_dim = cos_full.shape[-1]
            zeros = torch.zeros(1, 1, pad_len, head_dim, dtype=cos_full.dtype)
            cos_full = torch.cat([cos_full, zeros], dim=2)
            sin_full = torch.cat([sin_full, zeros], dim=2)

        mem = ttnn.DRAM_MEMORY_CONFIG
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        cos_tt = ttnn.from_torch(
            cos_full,
            device=self.device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        sin_tt = ttnn.from_torch(
            sin_full,
            device=self.device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        self._padded_cache_key = cache_key
        self._padded_cache_rot_mats = (cos_tt, sin_tt)
        self._padded_cache_cu_seqlens = cu_seqlens
        return self._padded_cache_rot_mats, cu_seqlens


# ---------------------------------------------------------------------------
# Vision RMSNorm
# ---------------------------------------------------------------------------


class TTNNDotsVisionRMSNorm(TTNNModule):
    """RMSNorm for Dots vision blocks.

    When the HF checkpoint contains bias for the norm layer, this falls back
    to LayerNorm (matching HF behavior). Otherwise uses RMSNorm.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        self._use_layer_norm = False
        self._weight_torch = None
        self._bias_torch = None
        self.tt_weight = None
        self.tt_bias = None

    @classmethod
    def from_torch(cls, hf_norm):
        new_norm = cls()
        new_norm._fallback_torch_layer = hf_norm
        new_norm.eps = getattr(hf_norm, "variance_epsilon", getattr(hf_norm, "eps", 1e-5))

        if hasattr(hf_norm, "weight") and hf_norm.weight is not None:
            new_norm._weight_torch = hf_norm.weight.data.clone()
        else:
            raise ValueError("Vision RMSNorm requires a weight parameter")

        if hasattr(hf_norm, "bias") and hf_norm.bias is not None:
            new_norm._bias_torch = hf_norm.bias.data.clone()
            new_norm._use_layer_norm = True

        return new_norm

    def preprocess_weights_impl(self):
        if self._use_layer_norm:
            self.tt_weight = ttnn.from_torch(
                self._weight_torch.unsqueeze(0),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if self._bias_torch is not None:
                self.tt_bias = ttnn.from_torch(
                    self._bias_torch.unsqueeze(0),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        else:
            dim = self._weight_torch.numel()
            tile = 32
            w = self._weight_torch.to(torch.bfloat16)
            w = w.view(1, 1, dim).reshape(1, 1, dim // tile, tile)
            self.tt_weight = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

    def move_weights_to_device_impl(self):
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        self.compute_kernel_config = _vision_sdpa_compute_config(self.device, math_fidelity=VISION_NORM_MATH_FIDELITY)

        if self._use_layer_norm:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if self.tt_bias is not None:
                self.tt_bias = ttnn.to_device(self.tt_bias, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, x: ttnn.Tensor, *, output_l1: bool = False) -> ttnn.Tensor:
        out_mem = ttnn.L1_MEMORY_CONFIG if output_l1 else None
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(
                x,
                ttnn.TILE_LAYOUT,
                memory_config=out_mem or ttnn.DRAM_MEMORY_CONFIG,
            )
        print("post trunk x.shape:", x.shape)
        if self._use_layer_norm:
            print("Using LayerNorm")
            out = ttnn.layer_norm(
                x,
                weight=self.tt_weight,
                bias=self.tt_bias,
                epsilon=self.eps,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=out_mem,
            )
        else:
            print("Using RMSNorm")
            out = ttnn.rms_norm(
                x,
                weight=self.tt_weight,
                epsilon=self.eps,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=out_mem,
            )
        # Default interleaved LayerNorm/RMSNorm keeps DRAM out at S=12288 even when
        # memory_config=L1 is passed; move explicitly so the MLP gate matmul in0 is L1.
        if output_l1:
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        return out


# ---------------------------------------------------------------------------
# Vision SwiGLU MLP
# ---------------------------------------------------------------------------


class TTNNDotsVisionMLP(TTNNModule):
    """SwiGLU MLP for Dots vision blocks: y = fc2(silu(fc1(x)) * fc3(x))."""

    def __init__(self):
        super().__init__()
        self._fc1_weight = None
        self._fc1_bias = None
        self._fc2_weight = None
        self._fc2_bias = None
        self._fc3_weight = None
        self._fc3_bias = None
        self.tt_fc1_weight = None
        self.tt_fc1_bias = None
        self.tt_fc2_weight = None
        self.tt_fc2_bias = None
        self.tt_fc3_weight = None
        self.tt_fc3_bias = None

    @classmethod
    def from_torch(cls, hf_mlp):
        new_mlp = cls()
        new_mlp._fallback_torch_layer = hf_mlp

        if hasattr(hf_mlp, "fc1"):
            new_mlp._fc1_weight = hf_mlp.fc1.weight.data.clone()
            if hf_mlp.fc1.bias is not None:
                new_mlp._fc1_bias = hf_mlp.fc1.bias.data.clone()
        elif hasattr(hf_mlp, "gate_proj"):
            new_mlp._fc1_weight = hf_mlp.gate_proj.weight.data.clone()
            if hf_mlp.gate_proj.bias is not None:
                new_mlp._fc1_bias = hf_mlp.gate_proj.bias.data.clone()

        if hasattr(hf_mlp, "fc2"):
            new_mlp._fc2_weight = hf_mlp.fc2.weight.data.clone()
            if hf_mlp.fc2.bias is not None:
                new_mlp._fc2_bias = hf_mlp.fc2.bias.data.clone()
        elif hasattr(hf_mlp, "down_proj"):
            new_mlp._fc2_weight = hf_mlp.down_proj.weight.data.clone()
            if hf_mlp.down_proj.bias is not None:
                new_mlp._fc2_bias = hf_mlp.down_proj.bias.data.clone()

        if hasattr(hf_mlp, "fc3"):
            new_mlp._fc3_weight = hf_mlp.fc3.weight.data.clone()
            if hf_mlp.fc3.bias is not None:
                new_mlp._fc3_bias = hf_mlp.fc3.bias.data.clone()
        elif hasattr(hf_mlp, "up_proj"):
            new_mlp._fc3_weight = hf_mlp.up_proj.weight.data.clone()
            if hf_mlp.up_proj.bias is not None:
                new_mlp._fc3_bias = hf_mlp.up_proj.bias.data.clone()

        return new_mlp

    def preprocess_weights_impl(self):
        """Linear weights in low precision (PyTorch Linear [out,in]; preprocessing applies TT layout)."""

        def pw(w):
            if w is None:
                return None
            return preprocess_linear_weight(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        def pb(b):
            if b is None:
                return None
            return preprocess_linear_bias(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        # Unfused gate/up projections (two linears). Vision MLP uses DRAM linears without
        # decoder-style CCL, so fusion only saved one matmul launch while forcing two
        # SliceDeviceOperation per block on the fused [gate|up] output; unfused removes those slices.
        self._intermediate_size = None
        self.tt_fused_gate_up_weight = None
        self.tt_fused_gate_up_bias = None
        self.tt_fc1_weight = pw(self._fc1_weight)
        self.tt_fc1_bias = pb(self._fc1_bias)
        self.tt_fc3_weight = pw(self._fc3_weight)
        self.tt_fc3_bias = pb(self._fc3_bias)

        self.tt_fc2_weight = pw(self._fc2_weight)
        self.tt_fc2_bias = pb(self._fc2_bias)

    def _mlp_tp_num_devices(self) -> int:
        """TP-axis device count (cluster_axis=1 = last mesh dim), or 1 if not a TP mesh.

        The vision tower body runs replicated across the TP axis; we shard only the
        MLP weights/FLOPs and re-replicate the output via an all-reduce, so the block
        residual contract (full hidden per device) is preserved.
        """
        dev = self.device
        if dev is None or not hasattr(dev, "get_num_devices") or int(dev.get_num_devices()) <= 1:
            return 1
        if not _tp_requires_ccl(dev):
            return 1
        return int(list(dev.shape)[-1]) if hasattr(dev, "shape") else int(dev.get_num_devices())

    def _mlp_use_tp(self) -> bool:
        """True when the SwiGLU MLP should be tensor-parallel (column gate/up + row down).

        Requires the intermediate dim to divide the TP-axis device count so each device
        holds tile-aligned, equal-width gate/up/down shards.
        """
        num_tp = self._mlp_tp_num_devices()
        if num_tp <= 1 or self._fc1_weight is None or self._fc2_weight is None:
            return False
        intermediate = int(self._fc1_weight.shape[0])
        return intermediate % num_tp == 0

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG

        def _to_dev(t):
            if t is None:
                return None
            return ttnn.to_device(t, self.device, memory_config=mem)

        self.compute_kernel_config = _vision_matmul_compute_config(self.device, math_fidelity=ttnn.MathFidelity.LoFi)

        if self._mlp_use_tp():
            self._move_weights_tp(mem)
            return

        self.tt_fused_gate_up_weight = _to_dev(getattr(self, "tt_fused_gate_up_weight", None))
        self.tt_fused_gate_up_bias = _to_dev(getattr(self, "tt_fused_gate_up_bias", None))
        self.tt_fc1_weight = _to_dev(getattr(self, "tt_fc1_weight", None))
        self.tt_fc1_bias = _to_dev(getattr(self, "tt_fc1_bias", None))
        self.tt_fc2_weight = _to_dev(self.tt_fc2_weight)
        self.tt_fc2_bias = _to_dev(self.tt_fc2_bias)
        self.tt_fc3_weight = _to_dev(getattr(self, "tt_fc3_weight", None))
        self.tt_fc3_bias = _to_dev(getattr(self, "tt_fc3_bias", None))

    def _move_weights_tp(self, mem):
        """Shard the SwiGLU weights for tensor parallelism (Megatron column/row split).

        fc1 (gate) / fc3 (up) are column-parallel: their output (intermediate) dim is
        sharded so each device computes ``intermediate/num_tp`` columns from the full,
        replicated hidden input -- no collective needed. fc2 (down) is row-parallel:
        its contraction (intermediate) dim is sharded so each device produces a partial
        ``[*, hidden]`` sum, which forward() all-reduces back to the replicated stream.
        The fc2 bias is replicated and added once after the all-reduce (fusing it would
        double-count it across the reduce).
        """
        dev = self.device
        # preprocess_linear_weight transposes [out, in] -> [in, out] then shards:
        #   dim=-1 shards the OUT dim (column-parallel), dim=-2 shards the IN/K dim (row-parallel).
        col_mapper = _tp_mesh_mapper(dev, -1)  # gate/up: shard intermediate (output)
        row_mapper = _tp_mesh_mapper(dev, -2)  # down: shard intermediate (contraction)

        def _shard_w(w, mapper):
            if w is None:
                return None
            host = preprocess_linear_weight(
                w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, weights_mesh_mapper=mapper
            )
            return ttnn.to_device(host, dev, memory_config=mem)

        def _shard_b(b, mapper):
            if b is None:
                return None
            host = preprocess_linear_bias(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, weights_mesh_mapper=mapper)
            return ttnn.to_device(host, dev, memory_config=mem)

        def _repl_b(b):
            if b is None:
                return None
            host = preprocess_linear_bias(b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            return ttnn.to_device(host, dev, memory_config=mem)

        self.tt_fused_gate_up_weight = None
        self.tt_fused_gate_up_bias = None
        self.tt_fc1_weight = _shard_w(self._fc1_weight, col_mapper)
        self.tt_fc1_bias = _shard_b(self._fc1_bias, col_mapper)
        self.tt_fc3_weight = _shard_w(self._fc3_weight, col_mapper)
        self.tt_fc3_bias = _shard_b(self._fc3_bias, col_mapper)
        self.tt_fc2_weight = _shard_w(self._fc2_weight, row_mapper)
        self.tt_fc2_bias = _repl_b(self._fc2_bias)

        self._tp_down_pc = None
        self._tp_down_k = None
        if _tp4_prefill_vision_enabled(self.device):
            # tp4_mlp_down_pc -> wh_tp4_mlp_down_pc (swept 2026-06-15: g8x8 obh=11
            # ibw=11 sub1x6, ~351 us in the faithful L1-context bench, -24% vs the
            # prior g6x8 ibw=3 ~464 us). The earlier generic obh=1 search ran
            # ~1449 us / 14% TFLOPs (the "SLOW" down matmul in the perf report).
            from models.experimental.tt_symbiote.modules.vision_tp4 import tp4_mlp_down_pc

            num_tp = self._mlp_tp_num_devices()
            itp = int(self._fc1_weight.shape[0]) // num_tp
            self._tp_down_pc = tp4_mlp_down_pc(self.device, itp=itp)
            self._tp_down_k = itp

    def _tp_mlp_down_program_config(self, m_dim: int, k_dim: int, n_dim: int):
        """Program config for TP row-parallel down matmul (``M×I/TP×H``).

        ``_vision_matmul_program_config`` returns ``None`` on BH 11×10 for
        M=11264 (352 m-tiles does not divide grid_y=10), which leaves the
        auto-config path at ~3.6 ms / 6% TFLOPs.  Prefer the hardware-swept
        TP4 config (``tp4_mlp_down_pc``, arch-dispatched WH/BH) with BFP8 output.
        """
        if not _tp4_prefill_vision_enabled(self.device):
            return _vision_matmul_program_config(self.device, m_dim, k_dim, n_dim)
        cached = getattr(self, "_tp_down_pc", None)
        if cached is not None and m_dim == 11264 and k_dim == int(getattr(self, "_tp_down_k", k_dim)):
            return cached
        from models.experimental.tt_symbiote.modules.vision_tp4 import tp4_mlp_down_pc, tp4_matmul_pc

        pc = tp4_mlp_down_pc(self.device, seq_len=m_dim, itp=k_dim)
        if pc is not None:
            return pc
        return tp4_matmul_pc(
            self.device,
            m_dim,
            k_dim,
            n_dim,
            in0_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat8_b,
        ) or _vision_matmul_program_config(self.device, m_dim, k_dim, n_dim)

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self._mlp_use_tp():
            return self._forward_tp(hidden_states)

        mem = ttnn.DRAM_MEMORY_CONFIG

        # Explicit 2D-mcast program_config: tracy showed gate/up running at
        # 27.7% LoFi TFLOPs / 13% DRAM with the auto-config (in0_block_w=1).
        # Setting a tuned in0_block_w + out_subblock_w increases L1 weight
        # reuse and matches the text-decoder linears' ``SLOW``->faster path.
        m_dim = int(hidden_states.shape[0]) * int(hidden_states.shape[1]) * int(hidden_states.shape[2])
        k_dim = int(self.tt_fc1_weight.shape[-2])
        n_dim = int(self.tt_fc1_weight.shape[-1])

        # Gate/up outputs are BFP8: halves the matmul writeback (38 MB BF16
        # -> 21 MB BFP8) and the matching read in the SILU multiply, with
        # no quality impact since the silu+mul output is already BFP8 in
        # this path. BFP4 weight x BF16 activation -> BFP8 output is a
        # supported matmul dtype combo on Wormhole.
        #
        # Note: an L1 BLOCK_SHARDED activation variant was tried (shard
        # ``hidden_states`` once across the 8x8 grid and reuse for gate/up).
        # It regressed FLOP utilization from 28% to 24% on these matmuls
        # (per-op time 2.42 ms -> 2.83 ms) because the BF16 shard at
        # 576 KB / core forces ``out_block_h`` from 12 down to 6 in the
        # sharded program_config to stay under the 1.5 MB per-core L1 cap.
        # Halving ``out_block_h`` doubles outer-M iterations, doubling weight
        # DRAM re-reads -- the slow vision matmuls are weight-DRAM-bound, so
        # the activation L1 win is more than wiped out by the weight DRAM
        # cost. Stay on the DRAM-interleaved path.
        # in0 is L1 interleaved from norm2 (``output_l1=True`` in the block).
        l1_resident = _vision_l1_interleaved_bytes_per_core(hidden_states, self.device)
        gate_up_pc = _vision_tp_gate_up_program_config(
            self.device, m_dim, k_dim, n_dim, hidden_states.dtype, l1_resident
        ) or _vision_matmul_program_config(self.device, m_dim, k_dim, n_dim, l1_resident_bytes_per_core=l1_resident)
        _vision_debug_mem("gate input", hidden_states)
        gate = ttnn.linear(
            hidden_states,
            self.tt_fc1_weight,
            bias=self.tt_fc1_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=gate_up_pc,
        )
        # ``up`` output to L1 interleaved so the SILU mul below reads it from L1
        # instead of DRAM (saves the ~29 MB bfp4 writeback + ~29 MB read per
        # layer). Same ``gate_up_pc`` -- L1- and DRAM-interleaved output use
        # identical matmul CBs, and this PC is already optimal for the shape
        # (DST area 8, fewest weight re-reads). ``up`` is bfp4 (~456 KB/core)
        # vs ``gate`` bfp8 (~861 KB/core), so ``up`` is the one that can share
        # L1 with the 841 KB ``gate_up_mul``; ``gate`` stays on DRAM.
        # L1 RISK: the binding phase is THIS matmul -- its CBs (per_core_N=17)
        # plus the 456 KB resident L1 output sit right at the ~1536 KB cap.
        # OOM-test this bucket (S=12288) before relying on it; if it overflows,
        # revert to DRAM (do NOT shrink out_block_h -- that doubles weight DRAM
        # re-reads on this weight-DRAM-bound matmul and cancels the win).
        _vision_debug_mem("up input", hidden_states)
        up = ttnn.linear(
            hidden_states,
            self.tt_fc3_weight,
            bias=self.tt_fc3_bias,
            dtype=ttnn.bfloat4_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            program_config=gate_up_pc,
        )
        ttnn.deallocate(hidden_states)
        # SILU dominates this op (the elementwise mul is cheap; the per-tile
        # exp+sigmoid is the bottleneck). ``fast_and_approximate_mode=True``
        # routes the fused SILU through the polynomial exp/sigmoid path,
        # cutting the ~2.2 ms BinaryNg time per vision layer. Output to
        # L1 interleaved so the down-projection reads activation from L1
        # instead of DRAM (52.6 MB saved from DRAM bandwidth per layer).
        # gate_up_mul occupies 841 KB/core; down matmul uses ob_h=8 (487 KB
        # CBs) → 1328 KB peak, 208 KB margin under the 1536 KB L1 cap.
        gate_up_mul = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            fast_and_approximate_mode=True,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Down-projection reads gate_up_mul from L1; DRAM output while gate_up
        # is live (L1 interleaved out would exceed per-core L1 with gate_up).
        down_m = int(gate_up_mul.shape[0]) * int(gate_up_mul.shape[1]) * int(gate_up_mul.shape[2])
        down_k = int(self.tt_fc2_weight.shape[-2])
        down_n = int(self.tt_fc2_weight.shape[-1])
        down_pc = _vision_mlp_down_l1_pc(self.device) or _vision_matmul_program_config(
            self.device, down_m, down_k, down_n
        )
        output = ttnn.linear(
            gate_up_mul,
            self.tt_fc2_weight,
            bias=self.tt_fc2_bias,
            dtype=ttnn.bfloat4_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            program_config=down_pc,
        )
        ttnn.deallocate(gate_up_mul)
        # output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)

        return output

    def _forward_tp(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Tensor-parallel SwiGLU: replicated hidden in -> replicated hidden out.

        The vision tower body runs replicated across the TP axis, so this MLP takes a
        full-width ``[*, hidden]`` input on every device, shards the intermediate-dim
        FLOPs (gate/up column-parallel, no collective), then row-parallel down-projects
        to per-device partial ``[*, hidden]`` sums and all-reduces them so the block
        residual stays replicated and numerically identical to the single-device path
        (up to bf16 reduction order). Weights/FLOPs/activation L1 are all 1/num_tp per
        device, which is the perf win over the previous fully-replicated MLP.
        """
        dev = self.device
        mem = ttnn.DRAM_MEMORY_CONFIG

        m_dim = int(hidden_states.shape[0]) * int(hidden_states.shape[1]) * int(hidden_states.shape[2])
        k_dim = int(self.tt_fc1_weight.shape[-2])  # full hidden (replicated input)
        n_dim = int(self.tt_fc1_weight.shape[-1])  # intermediate / num_tp (column shard)
        l1_resident = _vision_l1_interleaved_bytes_per_core(hidden_states, dev)
        gate_up_pc = _vision_tp_gate_up_program_config(dev, m_dim, k_dim, n_dim, hidden_states.dtype, l1_resident)

        gate = ttnn.linear(
            hidden_states,
            self.tt_fc1_weight,
            bias=self.tt_fc1_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=gate_up_pc,
        )
        up = ttnn.linear(
            hidden_states,
            self.tt_fc3_weight,
            bias=self.tt_fc3_bias,
            dtype=ttnn.bfloat4_b,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=gate_up_pc,
        )
        ttnn.deallocate(hidden_states)

        gate_up_mul = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            fast_and_approximate_mode=True,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # Row-parallel down-projection: each device contracts its intermediate shard
        # into a full-width partial sum. The tp4_prefill experiment uses BFP8 L1
        # out + explicit 2D-mcast PC; default keeps the prior DRAM/BF16 path.
        use_tp4_prefill_vision = _tp4_prefill_vision_enabled(self.device)
        out_mem = ttnn.L1_MEMORY_CONFIG if use_tp4_prefill_vision else mem
        out_dtype = ttnn.bfloat8_b if use_tp4_prefill_vision else ttnn.bfloat16
        down_m = int(gate_up_mul.shape[0]) * int(gate_up_mul.shape[1]) * int(gate_up_mul.shape[2])
        down_k = int(self.tt_fc2_weight.shape[-2])  # intermediate / num_tp (contraction shard)
        down_n = int(self.tt_fc2_weight.shape[-1])  # full hidden
        self._tp_down_k = down_k
        down_pc = self._tp_mlp_down_program_config(down_m, down_k, down_n)
        partial = ttnn.linear(
            gate_up_mul,
            self.tt_fc2_weight,
            bias=None,
            dtype=out_dtype,
            memory_config=out_mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=down_pc,
        )
        ttnn.deallocate(gate_up_mul)

        # all_reduce == reduce_scatter + all_gather (trace-stable; mirrors linear.py).
        # The reduce sums the per-device partials; scatter/gather on dim=3 keeps the
        # full hidden replicated on every device afterwards.
        num_links = _ccl_num_links(dev)
        if len(partial.shape) != 4:
            shp = list(partial.shape)
            while len(shp) < 4:
                shp.insert(0, 1)
            partial = ttnn.reshape(partial, shp)
        reduced = ttnn.reduce_scatter(
            partial,
            dim=3,
            num_links=num_links,
            cluster_axis=1,
            memory_config=out_mem,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(partial)
        output = ttnn.all_gather(
            reduced,
            dim=3,
            num_links=num_links,
            cluster_axis=1,
            memory_config=out_mem,
            topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(reduced)
        # fc2 bias is replicated (full hidden) and added once, after the reduce, so it
        # is not summed num_tp times.
        if self.tt_fc2_bias is not None:
            output = ttnn.add(output, self.tt_fc2_bias)
        return output


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class TTNNDotsVisionPatchEmbed(TTNNModule):
    """Patch embedding for Dots vision (14x14 patches, no CLS token, no pos embed)."""

    def __init__(self):
        super().__init__()
        self.patch_size = 14
        self.in_channels = 3
        self.embed_dim = 1536
        self._proj_weight = None
        self._proj_bias = None
        self._norm_weight = None
        self.tt_proj_weight = None
        self.tt_proj_bias = None
        self.tt_norm_weight = None
        self.vision_matmul_compute_kernel_config = None
        self.vision_norm_compute_kernel_config = None
        self._bypass_tensor_wrapping = True

    @classmethod
    def from_torch(cls, hf_patch_embed, patch_size=14, in_channels=3, embed_dim=1536):
        new_pe = cls()
        new_pe._fallback_torch_layer = hf_patch_embed
        new_pe.patch_size = patch_size
        new_pe.in_channels = in_channels
        new_pe.embed_dim = embed_dim

        proj = None
        if hasattr(hf_patch_embed, "proj"):
            proj = hf_patch_embed.proj
        elif hasattr(hf_patch_embed, "patchifier") and hasattr(hf_patch_embed.patchifier, "proj"):
            proj = hf_patch_embed.patchifier.proj

        if proj is not None:
            w = proj.weight.data.clone()
            if w.dim() == 4:
                w = w.reshape(w.shape[0], -1)
            new_pe._proj_weight = w
            if proj.bias is not None:
                new_pe._proj_bias = proj.bias.data.clone()

        norm = None
        if hasattr(hf_patch_embed, "norm"):
            norm = hf_patch_embed.norm
        elif hasattr(hf_patch_embed, "patchifier") and hasattr(hf_patch_embed.patchifier, "norm"):
            norm = hf_patch_embed.patchifier.norm

        if norm is not None and hasattr(norm, "weight") and norm.weight is not None:
            new_pe._norm_weight = norm.weight.data.clone()

        return new_pe

    def preprocess_weights_impl(self):
        if self._proj_weight is not None:
            # Pad the input-feature (K) dimension of the projection weight to the next
            # tile multiple so that from_torch in forward() can upload x with an already
            # tile-aligned last dimension, turning TilizeWithValPaddingDeviceOperation
            # (which pads 588→608 on device every forward call) into a plain Tilize.
            tile = 32
            k = self._proj_weight.shape[-1]
            k_padded = ((k + tile - 1) // tile) * tile
            w = self._proj_weight.to(torch.bfloat16)
            if k_padded != k:
                w = F.pad(w, (0, k_padded - k))
            self._proj_k_padded = k_padded
            self.tt_proj_weight = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            )
        if self._proj_bias is not None:
            self.tt_proj_bias = ttnn.from_torch(
                self._proj_bias.reshape(1, 1, 1, -1).to(torch.bfloat16),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            )
        if self._norm_weight is not None:
            dim = self._norm_weight.numel()
            tile = 32
            w = self._norm_weight.to(torch.bfloat16)
            w = w.view(1, 1, dim).reshape(1, 1, dim // tile, tile)
            self.tt_norm_weight = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        if self.tt_proj_weight is not None:
            self.tt_proj_weight = ttnn.to_device(self.tt_proj_weight, self.device, memory_config=mem)
        if self.tt_proj_bias is not None:
            self.tt_proj_bias = ttnn.to_device(self.tt_proj_bias, self.device, memory_config=mem)
        if self.tt_norm_weight is not None:
            self.tt_norm_weight = ttnn.to_device(self.tt_norm_weight, self.device, memory_config=mem)

        self.vision_matmul_compute_kernel_config = _vision_matmul_compute_config(
            self.device, math_fidelity=VISION_MATMUL_MATH_FIDELITY
        )
        self.vision_norm_compute_kernel_config = _vision_sdpa_compute_config(
            self.device, math_fidelity=VISION_NORM_MATH_FIDELITY
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor = None,
        activation_memory_config: ttnn.MemoryConfig | None = None,
    ) -> ttnn.Tensor:
        if _tp4_prefill_vision_enabled(self.device):
            _vision_tower_signpost("vision_tower.start")
        mem = activation_memory_config or ttnn.DRAM_MEMORY_CONFIG
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

        if pixel_values.dim() == 2:
            x = pixel_values.to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
            k_padded = getattr(self, "_proj_k_padded", x.shape[-1])
            if k_padded != x.shape[-1]:
                x = F.pad(x, (0, k_padded - x.shape[-1]))
            x_tt = ttnn.from_torch(
                x,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
                mesh_mapper=mapper,
            )

            # Output BFP8 here so the entire residual stream feeding the 42
            # vision blocks is BFP8: the QKV / MLP gate / MLP up matmuls then
            # see BFP8 in0 (instead of BF16) which roughly matches the MLP
            # down matmul's 48% FLOP utilization on the gate/up shapes (BF16
            # in0 was capped at ~28% by the kernel's lower-fidelity inner
            # math path on mixed-precision in0/in1). The ``rms_norm`` below
            # picks up its output dtype from the BFP8 input (the public
            # ``ttnn.layer_norm`` / ``ttnn.rms_norm`` wrapper passes
            # ``dtype=std::nullopt`` to the device prim, which preserves
            # input dtype), so no extra typecast is needed.
            out = ttnn.linear(
                x_tt,
                self.tt_proj_weight,
                bias=self.tt_proj_bias,
                transpose_b=True,
                dtype=ttnn.bfloat8_b,
                memory_config=mem,
                compute_kernel_config=self.vision_matmul_compute_kernel_config,
            )

            if self.tt_norm_weight is not None:
                print("Using RMSNorm")
                print("post patch embed out.shape:", out.shape)
                out = ttnn.rms_norm(
                    out,
                    weight=self.tt_norm_weight,
                    epsilon=1e-5,
                    memory_config=mem,
                    compute_kernel_config=self.vision_norm_compute_kernel_config,
                )

            return out

        B, C, H, W = pixel_values.shape

        if grid_thw is not None:
            g = grid_thw.detach().cpu() if hasattr(grid_thw, "is_cuda") and grid_thw.is_cuda else grid_thw
            if g.dim() == 1:
                g = g.unsqueeze(0)
            temporal = int(g[0, 0].item())
            height_patches = int(g[0, 1].item())
            width_patches = int(g[0, 2].item())
        else:
            temporal = 1
            height_patches = H // self.patch_size
            width_patches = W // self.patch_size

        num_patches = temporal * height_patches * width_patches

        temporal_patch_size = temporal
        x = pixel_values.view(-1, C, temporal_patch_size, self.patch_size, self.patch_size)
        x = x[:, :, 0]
        x = x.reshape(1, 1, num_patches, C * self.patch_size * self.patch_size).to(torch.bfloat16)

        k_padded = getattr(self, "_proj_k_padded", x.shape[-1])
        if k_padded != x.shape[-1]:
            x = F.pad(x, (0, k_padded - x.shape[-1]))

        x_tt = ttnn.from_torch(
            x,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        # See note above on the dim==2 path: BFP8 here keeps the residual
        # stream (and therefore every downstream matmul's in0) BFP8, which
        # is the dominant per-layer compute speedup.
        out = ttnn.linear(
            x_tt,
            self.tt_proj_weight,
            bias=self.tt_proj_bias,
            transpose_b=True,
            dtype=ttnn.bfloat8_b,
            memory_config=mem,
            compute_kernel_config=self.vision_matmul_compute_kernel_config,
        )

        if self.tt_norm_weight is not None:
            print("Using RMSNorm")
            print("post patch embed1 out.shape:", out.shape)
            out = ttnn.rms_norm(
                out,
                weight=self.tt_norm_weight,
                epsilon=1e-5,
                dtype=ttnn.bfloat8_b,
                memory_config=mem,
                compute_kernel_config=self.vision_norm_compute_kernel_config,
            )

        return out


# ---------------------------------------------------------------------------
# Vision Attention
# ---------------------------------------------------------------------------


class TTNNDotsVisionAttention(TTNNModule):
    """Vision attention for Dots OCR with per-segment SDPA and 2D RoPE."""

    def __init__(self):
        super().__init__()
        self.hidden_size = 1536
        self.num_heads = 12
        self.head_dim = 128
        self.num_kv_heads = 12
        self._qkv_weight = None
        self._qkv_bias = None
        self._o_proj_weight = None
        self._o_proj_bias = None

        self.tt_qkv_weight = None
        self.tt_qkv_bias = None
        self.tt_o_proj_weight = None
        self.tt_o_proj_bias = None

    @classmethod
    def from_torch(cls, hf_attn, hidden_size=1536, num_heads=12):
        new_attn = cls()
        new_attn._fallback_torch_layer = hf_attn
        new_attn.hidden_size = hidden_size
        new_attn.num_heads = num_heads
        new_attn.head_dim = hidden_size // num_heads
        new_attn.num_kv_heads = num_heads

        if hasattr(hf_attn, "qkv"):
            new_attn._qkv_weight = hf_attn.qkv.weight.data.clone()
            if hf_attn.qkv.bias is not None:
                new_attn._qkv_bias = hf_attn.qkv.bias.data.clone()
        elif hasattr(hf_attn, "qkv_proj"):
            new_attn._qkv_weight = hf_attn.qkv_proj.weight.data.clone()
            if hf_attn.qkv_proj.bias is not None:
                new_attn._qkv_bias = hf_attn.qkv_proj.bias.data.clone()
        else:
            q_weight = getattr(hf_attn, "q_proj", getattr(hf_attn, "wq", None))
            k_weight = getattr(hf_attn, "k_proj", getattr(hf_attn, "wk", None))
            v_weight = getattr(hf_attn, "v_proj", getattr(hf_attn, "wv", None))
            if q_weight is not None and k_weight is not None and v_weight is not None:
                new_attn._qkv_weight = torch.cat(
                    [q_weight.weight.data, k_weight.weight.data, v_weight.weight.data],
                    dim=0,
                )
                if q_weight.bias is not None and k_weight.bias is not None and v_weight.bias is not None:
                    new_attn._qkv_bias = torch.cat(
                        [q_weight.bias.data, k_weight.bias.data, v_weight.bias.data],
                        dim=0,
                    )

        o_proj = getattr(hf_attn, "proj", getattr(hf_attn, "o_proj", getattr(hf_attn, "out_proj", None)))
        if o_proj is not None:
            new_attn._o_proj_weight = o_proj.weight.data.clone()
            if o_proj.bias is not None:
                new_attn._o_proj_bias = o_proj.bias.data.clone()

        return new_attn

    def _attn_tp_ndev(self) -> int:
        """Number of devices the attention heads are tensor-parallel sharded over.

        Decided purely from the *physical* mesh so it stays independent of the
        decoder's forced-replicate flag (``DOTS_OCR_REST_TP1`` /
        ``linear._REST_REPLICATE``): a ``(1, N)`` or ``(DP, N)`` TP-layout mesh
        with ``N > 1`` head-shards across the TP axis when ``num_heads`` divides
        evenly; the ``(N, 1)`` DP mesh and single-device (P100) keep the
        replicated path.
        """
        dev = getattr(self, "device", None)
        if dev is None or not hasattr(dev, "shape"):
            return 1
        shape = [int(x) for x in dev.shape]
        n = int(shape[-1]) if shape and int(shape[-1]) > 1 else 1
        is_tp_mesh = n > 1
        if n > 1 and is_tp_mesh and self.num_heads % n == 0:
            return n
        return 1

    def _qkv_head_shard_perm(self, ndev: int) -> list[int]:
        """Column permutation that regroups the fused QKV output dim by device.

        The fused QKV weight emits ``[Q(num_heads*hd) | K(...) | V(...)]``. To
        head-shard across ``ndev`` ranks, device ``d`` must own a contiguous
        block ``[Q_heads(d) | K_heads(d) | V_heads(d)]`` so a plain dim=-1 shard
        (``shard_tensor_to_mesh_mapper``) hands each rank exactly its heads, and
        ``nlp_create_qkv_heads(num_heads=heads_per_dev)`` then splits it locally.
        Whole 128-wide heads are permuted, so each head's internal half-half
        RoPE layout is preserved.
        """
        nh = int(self.num_heads)
        hd = int(self.head_dim)
        block = nh * hd  # width of each of the Q/K/V blocks in the fused output
        heads_per_dev = nh // ndev
        perm: list[int] = []
        for d in range(ndev):
            for blk in range(3):  # Q, K, V
                for local_h in range(heads_per_dev):
                    g = d * heads_per_dev + local_h  # global head index
                    start = blk * block + g * hd
                    perm.extend(range(start, start + hd))
        return perm

    def preprocess_weights_impl(self):
        # QKV weights/bias are kept in the native HF "half-half" head_dim
        # layout. Combined with cos/sin built in the same half-half layout
        # (see TTNNDotsVision2DRoPE.build), this lets us use the cheaper
        # ``ttnn.experimental.rotary_embedding`` (non-llama) kernel that
        # preserves input dtype -- so Q/K can stay BFP8 the whole way from
        # the QKV matmul into SDPA, eliminating the 4 typecasts per layer
        # the llama kernel forced (~1.3 ms x 42 layers in vision prefill).
        ndev = self._attn_tp_ndev()
        if ndev > 1:
            # TP head-sharding: regroup the fused QKV columns by device, then
            # keep transposed [in, out] torch weights so move_weights can build
            # the per-device shards with a mesh mapper (mirrors the decoder TP
            # linears). o_proj is kept as the full transposed weight and uploaded
            # replicated: forward gathers ctx to full hidden before a replicated
            # o_proj, so no contraction-dim shard is needed.
            perm = self._qkv_head_shard_perm(ndev)
            qkv_w = self._qkv_weight[perm, :].contiguous()  # [out=3*H, in=H]
            self._qkv_weight_t_tp = qkv_w.t().contiguous()  # [in=H, out=3*H]
            self._qkv_bias_tp = self._qkv_bias[perm].contiguous().reshape(1, -1) if self._qkv_bias is not None else None
            self._o_proj_weight_t_tp = self._o_proj_weight.t().contiguous()  # [in=H, out=H] (full, replicated)
            self._o_proj_bias_tp = self._o_proj_bias  # replicated, fused into the matmul
            return

        self.tt_qkv_weight = preprocess_linear_weight(self._qkv_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        if self._qkv_bias is not None:
            self.tt_qkv_bias = preprocess_linear_bias(self._qkv_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        self.tt_o_proj_weight = preprocess_linear_weight(
            self._o_proj_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        if self._o_proj_bias is not None:
            self.tt_o_proj_bias = preprocess_linear_bias(
                self._o_proj_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG

        def _to_dev(t):
            if t is None:
                return None
            return ttnn.to_device(t, self.device, memory_config=mem)

        self.compute_kernel_config = _vision_matmul_compute_config(
            self.device, math_fidelity=VISION_MATMUL_MATH_FIDELITY
        )

        self._tp_ndev = self._attn_tp_ndev()
        if self._tp_ndev > 1:
            # Column-shard the (head-regrouped) QKV weight/bias on the output dim
            # -> each rank holds its heads' Q/K/V. The o_proj weight is REPLICATED
            # (full [in=H, out=H]): forward all_gathers this rank's head-group ctx
            # to the full hidden and runs the replicated o_proj locally, so only a
            # single all_gather of the (smaller) ctx is needed -- no row-shard +
            # all-reduce. o_proj bias is replicated and fused into the matmul.
            self.tt_qkv_weight = ttnn.as_tensor(
                self._qkv_weight_t_tp,
                device=self.device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=_tp_mesh_mapper(self.device, -1),
                memory_config=mem,
            )
            self.tt_qkv_bias = (
                ttnn.as_tensor(
                    self._qkv_bias_tp,
                    device=self.device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=_tp_mesh_mapper(self.device, -1),
                    memory_config=mem,
                )
                if self._qkv_bias_tp is not None
                else None
            )
            self.tt_o_proj_weight = ttnn.as_tensor(
                self._o_proj_weight_t_tp,
                device=self.device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                memory_config=mem,
            )
            self.tt_o_proj_bias = (
                ttnn.as_tensor(
                    self._o_proj_bias_tp.reshape(1, -1),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                    memory_config=mem,
                )
                if self._o_proj_bias_tp is not None
                else None
            )
            self._tp_o_ctx_dim = None
            self._tp_o_proj_pc = None
            if _tp4_prefill_vision_enabled(self.device):
                from models.experimental.tt_symbiote.modules.vision_tp4 import tp4_o_proj_pc

                # Replicated o_proj contracts the FULL gathered hidden (K=hidden_size),
                # not this rank's head-group width -- so the pc is for the full shape.
                self._tp_o_ctx_dim = self.hidden_size
                self._tp_o_proj_pc = tp4_o_proj_pc(self.device, ctx_dim=self._tp_o_ctx_dim)
        else:
            self.tt_qkv_weight = _to_dev(self.tt_qkv_weight)
            self.tt_qkv_bias = _to_dev(self.tt_qkv_bias)
            self.tt_o_proj_weight = _to_dev(self.tt_o_proj_weight)
            self.tt_o_proj_bias = _to_dev(self.tt_o_proj_bias)

        self.sdpa_compute_kernel_config = _vision_sdpa_compute_config(
            self.device, math_fidelity=VISION_SDPA_MATH_FIDELITY
        )

    def _tp_o_proj_program_config(self, m_dim: int, k_dim: int, n_dim: int):
        """Program config for TP row-parallel o_proj (``M×ctx/TP×H``).

        ``_vision_matmul_program_config`` returns ``None`` on BH 11×10 for
        M=11264; without an explicit PC the auto-config path lands at ~1.3 ms.
        """
        if not _tp4_prefill_vision_enabled(self.device):
            return _vision_matmul_program_config(self.device, m_dim, k_dim, n_dim)
        cached = getattr(self, "_tp_o_proj_pc", None)
        cached_k = getattr(self, "_tp_o_ctx_dim", None)
        if cached is not None and m_dim == 11264 and k_dim == cached_k:
            return cached
        from models.experimental.tt_symbiote.modules.vision_tp4 import tp4_matmul_pc, tp4_o_proj_pc

        pc = tp4_o_proj_pc(self.device, seq_len=m_dim, ctx_dim=k_dim)
        if pc is not None:
            return pc
        return tp4_matmul_pc(
            self.device,
            m_dim,
            k_dim,
            n_dim,
            in0_dtype=ttnn.bfloat8_b,
            out_dtype=ttnn.bfloat8_b,
        ) or _vision_matmul_program_config(self.device, m_dim, k_dim, n_dim)

    def _concat_heads(self, ctx: ttnn.Tensor) -> ttnn.Tensor:
        # Output L1 interleaved so o_proj reads activation from L1 instead of DRAM.
        # nlp_concat_heads supports L1 interleaved output for BFP8/BF16 inputs from
        # either DRAM or L1. True sharding of V is not possible (sdpa_device_operation.cpp:44
        # forbids sharded Q/K/V), so L1 interleaved here is the closest option.
        return ttnn.experimental.nlp_concat_heads(ctx, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _get_sdpa_program_config(self, seq_len: int):
        """Chunked SDPA program config for the vision tower.

        ``exp_approx_mode=True`` uses the fast polynomial exp in the softmax
        kernel — vision SDPA already runs at LoFi math fidelity, so the small
        precision delta from the approximate exp is in the noise.

        Chunk sizes: the per-core attention-scores circular buffer is
        ``q_chunk × k_chunk × sizeof(fp32_accumulator)``. With Wormhole's
        ~1.5 MB usable L1, this hard-caps the product at ~256K elements
        (e.g. 256 × 512 = 128K, leaves headroom for Q/K/V chunk buffers and
        the partial-output accumulator). Going beyond (e.g. 512 × 1024 =
        512K) overflows L1 with a "Statically allocated CBs grow to ... B"
        error.

        Sweep results (test_dots_ocr_vision_sdpa_configs.py candidates, S=12288):

            BFP4 V, q=256, k=512,  L1 out  -> 13,572 us (test baseline)
            BFP4 V, q=256, k=512,  DRAM out -> 13,560 us (noise)
            BFP4 V, q=256, k=1024, DRAM out -> 12,865 us  (-5.2%, kept)
            BFP4 V, q=256, k=1536, DRAM out -> OOM
            BFP8 V, q=256, k=1024, DRAM out -> OOM (+7 KB over L1)

        The win requires both knobs together: BFP4 V (halves V bandwidth and
        frees the 7 KB needed for the k=1024 scores CB) plus DRAM output
        (frees the partial-output CB budget). Either knob alone doesn't move
        the needle. q=128 variants regressed ~17% (too many outer Q passes),
        and q=512 variants OOM or also regressed in earlier ablation.

        NOTE: this op runs WITHOUT an attn_mask on the hot path (images that
        fill their bucket have no key padding -> is_causal=False, mask=None), so
        larger k_chunk = fewer outer K passes = faster. A sweep that forces a
        full [1,1,S,S] mask adds a second q_chunk*k_chunk CB and wrongly favours
        small k -- don't tune from the masked shape.
        """
        grid = self.device.compute_with_storage_grid_size()
        grid_size = ttnn.CoreCoord(grid.x, grid.y)

        if seq_len <= 256:
            q_chunk = k_chunk = max(32, ((seq_len + 31) // 32) * 32)
        elif seq_len <= 1024:
            q_chunk = k_chunk = 128
        elif seq_len <= 2048:
            # Small-S regime (the TP4 per-chip 3-head hot path at the common
            # bucket): q_chunk=128 beats 256 here -- the per-q-chunk softmax/
            # rescale overhead dominates over the extra outer-Q passes when S is
            # small, so the smaller q tile wins. Sweep at S=2048,H=3 (no mask,
            # is_causal=False): q128/k512 172 us vs q256/k512 214 us (~19%);
            # q256/k256 238 us (the config that regressed +13% in-model).
            # q128/k512 scores CB is HALF q256/k512's, so strictly trace-safe.
            q_chunk = 128
            k_chunk = 512
        else:
            # Large-S regime: q_chunk=128, k_chunk=1024. The scores CB is
            # q_chunk*k_chunk*4 B = 512 KB/core -- IDENTICAL to the old q256/k512,
            # so it fits under trace just the same -- but k_chunk=1024 halves the
            # outer-K passes (S/1024 vs S/512). The earlier note only compared
            # q128/k512 vs q256/k512 (same k_chunk) and so picked q256; it never
            # tried q128/k1024, which trades the extra Q passes for half the K
            # passes and wins. Sweep 2026-06-15 (bench_sdpa_tp4.py, S=11264 H=3,
            # K/V=bf4, no mask): q128/k1024 2960 us vs q256/k512 3437 us (-14%);
            # q256/k1024 (1 MB CB) is both slower (3280) and trace-risky; q64/k2048
            # regresses (3763, too many Q passes).
            q_chunk = 128
            k_chunk = 1024
        return SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=True,
        )

    def _sdpa_padded_with_key_mask(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        logical_seq_len: int,
        attn_mask: ttnn.Tensor | None = None,
        slice_to_logical: bool = True,
    ) -> ttnn.Tensor:
        """Run SDPA on ``[1,H,S_pad,D]`` Q/K/V with trailing key padding masked out."""
        logical_seq_len = int(logical_seq_len)
        h = int(q.shape[1])
        d = int(q.shape[3])
        s_pad = int(q.shape[2]) if attn_mask is not None else _align_vision_sdpa_seq_len(logical_seq_len)
        pad = s_pad - logical_seq_len
        if attn_mask is None and pad > 0:
            q_old, k_old, v_old = q, k, v
            q = ttnn.pad(q_old, ((0, 0), (0, 0), (0, pad), (0, 0)), value=0.0)
            k = ttnn.pad(k_old, ((0, 0), (0, 0), (0, pad), (0, 0)), value=0.0)
            v = ttnn.pad(v_old, ((0, 0), (0, 0), (0, pad), (0, 0)), value=0.0)
            ttnn.deallocate(q_old)
            ttnn.deallocate(k_old)
            ttnn.deallocate(v_old)

        owns_attn_mask = False
        if attn_mask is None and pad > 0:
            m = torch.zeros((1, 1, 1, s_pad), dtype=torch.float32)
            m[..., logical_seq_len:] = -1e9
            num_dev = int(self.device.get_num_devices()) if hasattr(self.device, "get_num_devices") else 1
            mapper = ttnn.ReplicateTensorToMesh(self.device) if num_dev > 1 else None
            attn_mask = ttnn.from_torch(
                m,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mapper,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            if attn_mask.dtype != q.dtype:
                attn_mask = ttnn.typecast(attn_mask, q.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            owns_attn_mask = True

        program_config = self._get_sdpa_program_config(s_pad)
        # DRAM output frees the partial-output CB budget so q=256/k=1024 fits.
        # _concat_heads accepts DRAM input; the extra DRAM round-trip on the
        # 22 MB context tensor (~80 us at 250 GB/s) is repaid by the larger
        # k_chunk savings.
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=attn_mask,
            program_config=program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        if owns_attn_mask:
            ttnn.deallocate(attn_mask)
        if slice_to_logical and pad > 0:
            ctx = ttnn.slice(ctx, (0, 0, 0, 0), (1, h, logical_seq_len, d), memory_config=ttnn.L1_MEMORY_CONFIG)
        return ctx

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        cu_seqlens: ttnn.Tensor | torch.Tensor | list | None = None,
        attention_mask: ttnn.Tensor | None = None,
        attention_logical_seq_len: int | None = None,
    ) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem = ttnn.DRAM_MEMORY_CONFIG
        s = int(hidden_states.shape[2])
        # In TP head-sharded mode each rank owns ``num_heads / ndev`` heads; the
        # QKV weight is column-sharded so the matmul + nlp_create_qkv_heads only
        # ever see this rank's local heads.
        ndev = int(getattr(self, "_tp_ndev", 1))
        h = self.num_heads // ndev
        hd = self.head_dim

        # TP4 prefill: qkv matmul output -> L1. It is consumed by
        # nlp_create_qkv_heads and ``hidden_states`` (the norm1 output) is
        # deallocated right after this matmul, so both are freed before SDPA and
        # neither competes with SDPA's per-core scores CB.
        use_tp4_prefill_vision = ndev > 1 and _tp4_prefill_vision_enabled(self.device)
        attn_mem = ttnn.L1_MEMORY_CONFIG if use_tp4_prefill_vision else mem

        # Output the fused QKV in BFP8 directly. At S=12288 this halves bandwidth on:
        #   1. the qkv matmul writeback ([1,1,S,4608]: 113 MB BF16 -> 56 MB BFP8)
        #   2. nlp_create_qkv_heads, which becomes "BFP8 => BFP8" (~221 MB IO -> ~110 MB)
        # Q/K/V all stay BFP8 from the QKV matmul straight into SDPA -- the
        # non-llama ``ttnn.experimental.rotary_embedding`` preserves dtype,
        # so we don't need any typecasts around the rotary like the llama
        # kernel forced before.
        qkv_m = int(hidden_states.shape[0]) * int(hidden_states.shape[1]) * s
        qkv_k = int(self.tt_qkv_weight.shape[-2])
        qkv_n = int(self.tt_qkv_weight.shape[-1])

        # An L1 BLOCK_SHARDED activation variant of this matmul was attempted
        # (shard ``hidden_states`` once across the 8x8 grid, feed the QKV
        # matmul in0 from the resident L1 shard). It regressed the per-op
        # time from 1.9 ms -> 3.0 ms (FLOP utilization 39% -> 25%) because
        # the BF16 shard at 576 KB / core forces ``out_block_h`` from 12
        # down to 6 in the matched program_config to stay under the per-core
        # L1 cap. Halving ``out_block_h`` doubles outer-M iterations, and
        # this matmul is weight-DRAM-bound -- the 2x weight re-reads from
        # DRAM more than wipe out the activation L1 win.
        qkv_pc = _vision_matmul_program_config(self.device, qkv_m, qkv_k, qkv_n)
        qkv = ttnn.linear(
            hidden_states,
            self.tt_qkv_weight,
            bias=self.tt_qkv_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=attn_mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=qkv_pc,
        )
        # norm1 output (this rank's in0) is unused for the rest of attention;
        # free it now so an L1 norm1 output does not survive into SDPA. The block
        # no longer deallocates it.
        ttnn.deallocate(hidden_states)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads // ndev,
            num_kv_heads=self.num_kv_heads // ndev,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        if rot_mats is not None and len(rot_mats) == 2:
            cos, sin = rot_mats

            # ``ttnn.experimental.rotary_embedding`` (non-llama) preserves
            # input dtype, so Q/K stay BFP8 the whole way: no typecasts
            # around the rotary, and SDPA reads BFP8 Q/K/V directly.
            # QKV matmul is BFP8 in0 (post-norm1) x BFP8 weight -> BFP8
            # output now -- the residual stream is BFP8 end-to-end.
            q = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # V downcasts BFP8 -> BFP4 here. Halves V's DRAM bandwidth inside SDPA and
        # frees enough L1 budget for the per-core scores CB to fit k_chunk=1024
        # (q=256/k=1024 OOMs by ~7 KB if V stays BFP8). With k_chunk doubled the
        # number of outer K/V passes halves (12288/1024=12 vs 12288/512=24), and
        # the isolated SDPA sweep measures ~5% per-call savings net of this
        # typecast.
        v = ttnn.typecast(v, ttnn.bfloat4_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # K -> BFP4 too: halves K's SDPA DRAM bandwidth. Sweep 2026-06-15
        # (bench_sdpa_tp4.py, S=11264 H=3): K=bf4 cuts the SDPA op ~3% with PCC
        # unchanged (0.99238 bf4 vs bf8 -- scores are dominated by the q*k tile
        # accumulation, not K's mantissa). K is only consumed by SDPA, so the
        # downcast is local.
        k = ttnn.typecast(k, ttnn.bfloat4_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # SDPA still requires interleaved Q/K/V (sdpa_device_operation.cpp:44 forbids
        # sharded inputs) and at S=12288 the BFP8 Q+K+V (~46 MB) plus SDPA's static
        # circular buffers exceed the per-core L1 budget -- keep these tensors on DRAM.

        # o_proj: in0 is L1 interleaved (nlp_concat_heads writes to L1); output goes to
        # L1 BLOCK_SHARDED so the downstream residual add reads from L1 instead of DRAM.
        # A separate program config is required for BLOCK_SHARDED output because
        # matmul_device_operation.cpp:841 constrains out_subblock_w/h when output is sharded.
        o_k = int(self.tt_o_proj_weight.shape[-2])
        o_n = int(self.tt_o_proj_weight.shape[-1])
        use_tp4_prefill_vision = ndev > 1 and _tp4_prefill_vision_enabled(self.device)
        o_pc = (
            self._tp_o_proj_program_config(qkv_m, o_k, o_n)
            if use_tp4_prefill_vision
            else _vision_matmul_program_config(self.device, qkv_m, o_k, o_n)
        )
        # The block-sharded o_proj fast path is for the single-device replicated
        # case only. In TP the o_proj is row-parallel (in0 holds this rank's
        # heads, weight is contraction-sharded) and the partial sums must be
        # all-reduced. The tp4_prefill experiment uses L1/BFP8; default keeps
        # the prior DRAM/BF16 path.
        out_bs = None if ndev > 1 else _vision_block_sharded_mem(self.device, qkv_m, o_n)
        o_bs_pc = None if ndev > 1 else _vision_o_proj_bs_program_config(self.device)
        tp_out_mem = ttnn.L1_MEMORY_CONFIG if use_tp4_prefill_vision else mem
        tp_out_dtype = ttnn.bfloat8_b if use_tp4_prefill_vision else ttnn.bfloat16

        def _run_o_proj(ctx: ttnn.Tensor) -> ttnn.Tensor:
            ctx = self._concat_heads(ctx)
            if ndev > 1:
                # Head-parallel recombination with a SINGLE collective: instead of
                # a row-parallel o_proj + all-reduce (reduce_scatter + all_gather of
                # the full-width [1,1,S,H] partial), all_gather this rank's narrower
                # head-group context [1,1,S,ctx/TP] up to the full [1,1,S,H], then
                # run a *replicated* (full-weight) o_proj locally. One collective
                # instead of two and ~half the CCL bytes (the gathered ctx is the
                # smaller tensor); the gathered o_proj is replicated so every rank
                # already holds the full output -- no post-matmul gather. The mandatory
                # head exchange is unavoidable (o_proj mixes all heads), but it moves
                # the cheaper tensor. Bias is replicated and fused into the matmul.
                num_links = _ccl_num_links(self.device)
                ctx_full = ttnn.all_gather(
                    ctx,
                    dim=3,
                    num_links=num_links,
                    cluster_axis=1,
                    memory_config=tp_out_mem,
                    topology=ttnn.Topology.Linear,
                )
                ttnn.deallocate(ctx)
                out = ttnn.linear(
                    ctx_full,
                    self.tt_o_proj_weight,
                    bias=self.tt_o_proj_bias,
                    dtype=tp_out_dtype,
                    memory_config=tp_out_mem,
                    compute_kernel_config=self.compute_kernel_config,
                    program_config=o_pc,
                )
                ttnn.deallocate(ctx_full)
                return out
            # BFP8 output keeps the post-attn residual on the BFP8 stream.
            if out_bs is not None and o_bs_pc is not None:
                return ttnn.linear(
                    ctx,
                    self.tt_o_proj_weight,
                    bias=self.tt_o_proj_bias,
                    dtype=ttnn.bfloat8_b,
                    memory_config=out_bs,
                    compute_kernel_config=self.compute_kernel_config,
                    program_config=o_bs_pc,
                )
            return ttnn.linear(
                ctx,
                self.tt_o_proj_weight,
                bias=self.tt_o_proj_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=mem,
                compute_kernel_config=self.compute_kernel_config,
                program_config=o_pc,
            )

        if cu_seqlens is None:
            logical_s = int(attention_logical_seq_len) if attention_mask is not None else s
            ctx = self._sdpa_padded_with_key_mask(
                q,
                k,
                v,
                logical_s,
                attn_mask=attention_mask,
                slice_to_logical=attention_mask is None,
            )
            return _run_o_proj(ctx)

        cu_host = self._cu_seqlens_to_list(cu_seqlens, s)

        if len(cu_host) == 2:
            # Single segment — same path as ``cu_seqlens is None`` (full sequence).
            ctx = self._sdpa_padded_with_key_mask(q, k, v, s)
        else:
            ctx_segments = []

            for seg_start, seg_end in zip(cu_host[:-1], cu_host[1:]):
                seg_start, seg_end = int(seg_start), int(seg_end)
                seg_len = seg_end - seg_start
                if seg_len <= 0:
                    continue

                q_seg = ttnn.slice(q, (0, 0, seg_start, 0), (1, h, seg_end, hd))
                k_seg = ttnn.slice(k, (0, 0, seg_start, 0), (1, h, seg_end, hd))
                v_seg = ttnn.slice(v, (0, 0, seg_start, 0), (1, h, seg_end, hd))

                ctx_seg = self._sdpa_padded_with_key_mask(q_seg, k_seg, v_seg, seg_len)
                ctx_segments.append(ctx_seg)

            ctx = ttnn.concat(ctx_segments, dim=2) if len(ctx_segments) > 1 else ctx_segments[0]

        return _run_o_proj(ctx)

    def _cu_seqlens_to_list(self, cu_seqlens, expected_total: int) -> list[int]:
        if isinstance(cu_seqlens, list):
            cu_host = cu_seqlens
        elif isinstance(cu_seqlens, torch.Tensor):
            cu_host = cu_seqlens.flatten().to(torch.int64).tolist()
        elif isinstance(cu_seqlens, ttnn.Tensor):
            composer = None
            if self.device is not None and self.device.get_num_devices() > 1:
                composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
            out = ttnn.to_torch(cu_seqlens, mesh_composer=composer) if composer else ttnn.to_torch(cu_seqlens)
            try:
                num_dev = self.device.get_num_devices() if self.device is not None else 1
                if num_dev > 1 and out.shape[0] % num_dev == 0:
                    per = out.shape[0] // num_dev
                    out = out[:per]
            except Exception:
                pass
            cu_host = out.flatten().to(torch.int64).tolist()
        else:
            cu_host = list(cu_seqlens)

        if len(cu_host) < 2 or cu_host[0] != 0 or cu_host[-1] != expected_total:
            raise ValueError(f"Invalid cu_seqlens={cu_host} for S={expected_total}")

        return cu_host


# ---------------------------------------------------------------------------
# Vision Block
# ---------------------------------------------------------------------------


class TTNNDotsVisionBlock(TTNNModule):
    """Single vision transformer block with post-norm architecture."""

    def __init__(self):
        super().__init__()
        self.norm1 = None
        self.norm2 = None
        self.attn = None
        self.mlp = None

    @classmethod
    def from_torch(cls, hf_block, hidden_size=1536, num_heads=12):
        new_block = cls()
        new_block._fallback_torch_layer = hf_block

        new_block.norm1 = TTNNDotsVisionRMSNorm.from_torch(hf_block.norm1)
        new_block.norm2 = TTNNDotsVisionRMSNorm.from_torch(hf_block.norm2)

        attn_module = getattr(hf_block, "attn", getattr(hf_block, "attention", getattr(hf_block, "self_attn", None)))
        if attn_module is None:
            raise ValueError("Could not find attention sub-module in HF block")
        new_block.attn = TTNNDotsVisionAttention.from_torch(attn_module, hidden_size=hidden_size, num_heads=num_heads)

        mlp_module = getattr(hf_block, "mlp", getattr(hf_block, "feed_forward", None))
        if mlp_module is None:
            raise ValueError("Could not find MLP sub-module in HF block")
        new_block.mlp = TTNNDotsVisionMLP.from_torch(mlp_module)

        return new_block

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        cu_seqlens=None,
        attention_mask: ttnn.Tensor | None = None,
        attention_logical_seq_len: int | None = None,
    ) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(
                hidden_states, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        _vision_debug_mem("in", hidden_states)

        # TP4 prefill: norm1 output -> L1 (qkv reads in0 from L1). attn() frees
        # ``normed`` right after the qkv matmul, so it is gone before SDPA -- no
        # clash with SDPA's scores CB. (DRAM keeps the prior single-device path.)
        use_tp4 = _tp4_prefill_vision_enabled(self.device)
        residual = hidden_states
        _vision_debug_mem("residual_pre_attn", residual)
        print("norm1 vision before attnhidden_states.shape:", hidden_states.shape)
        normed = self.norm1(hidden_states, output_l1=use_tp4)
        _vision_debug_mem("after norm1", normed)
        # attn() consumes and deallocates ``normed`` internally (after the qkv matmul).
        attn_out = self.attn(
            normed,
            rot_mats=rot_mats,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            attention_logical_seq_len=attention_logical_seq_len,
        )
        _vision_debug_mem("after attn", attn_out)
        # Keep Wormhole's attention residual in BF16 for OCR heading stability;
        # non-Wormhole keeps the faster BFP8 residual stream.
        # residual_dtype = ttnn.bfloat16 if is_wormhole_b0(self.device) else ttnn.bfloat8_b
        hidden_states = ttnn.add(residual, attn_out, dtype=ttnn.bfloat8_b)
        _vision_debug_mem("after attn residual add", hidden_states)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(residual)
        residual = hidden_states
        _vision_debug_mem("residual_pre_mlp", residual)
        print("vision bloock after attnhidden_states.shape:", hidden_states.shape)
        normed = self.norm2(hidden_states, output_l1=True)
        _vision_debug_mem("after norm2", normed)
        mlp_out = self.mlp(normed)
        _vision_debug_mem("after mlp", mlp_out)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(residual, mlp_out, dtype=ttnn.bfloat8_b)
        _vision_debug_mem("out", hidden_states)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        return hidden_states


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------


class TTNNDotsPatchMerger(TTNNModule):
    """Patch merger for Dots vision: spatial merge + LayerNorm/RMSNorm + MLP(GELU)."""

    def __init__(self):
        super().__init__()
        self.hidden_size = 1536
        self.out_hidden_size = 1536
        self.spatial_merge_size = 2
        self.mlp_size = None

        self._use_layer_norm = False
        self._ln_weight = None
        self._ln_bias = None
        self._w1_weight = None
        self._w2_weight = None
        self._w1_bias = None
        self._w2_bias = None

        self.tt_ln_weight = None
        self.tt_ln_bias = None
        self.tt_w1 = None
        self.tt_w2 = None
        self.tt_w1_bias = None
        self.tt_w2_bias = None

    @classmethod
    def from_torch(cls, hf_merger, hidden_size=1536, out_hidden_size=1536, spatial_merge_size=2):
        new_merger = cls()
        new_merger._fallback_torch_layer = hf_merger
        new_merger.hidden_size = hidden_size
        new_merger.out_hidden_size = out_hidden_size
        new_merger.spatial_merge_size = spatial_merge_size
        new_merger.mlp_size = hidden_size * (spatial_merge_size**2)

        ln_q = getattr(hf_merger, "ln_q", getattr(hf_merger, "norm", None))
        if ln_q is not None:
            if hasattr(ln_q, "weight") and ln_q.weight is not None:
                new_merger._ln_weight = ln_q.weight.data.clone()
            if hasattr(ln_q, "bias") and ln_q.bias is not None:
                new_merger._ln_bias = ln_q.bias.data.clone()
                new_merger._use_layer_norm = True

        mlp = getattr(hf_merger, "mlp", getattr(hf_merger, "feed_forward", None))
        if mlp is not None:
            if hasattr(mlp, "0") or (hasattr(mlp, "__getitem__") and len(list(mlp.children())) >= 3):
                try:
                    children = list(mlp.children())
                    fc1 = children[0]
                    fc2 = children[2] if len(children) > 2 else children[1]
                    new_merger._w1_weight = torch.transpose(fc1.weight.data, -2, -1).contiguous()
                    new_merger._w2_weight = torch.transpose(fc2.weight.data, -2, -1).contiguous()
                    if hasattr(fc1, "bias") and fc1.bias is not None:
                        new_merger._w1_bias = fc1.bias.data.clone().reshape(1, 1, 1, -1)
                    if hasattr(fc2, "bias") and fc2.bias is not None:
                        new_merger._w2_bias = fc2.bias.data.clone().reshape(1, 1, 1, -1)
                except (IndexError, AttributeError):
                    pass
            if new_merger._w1_weight is None:
                for name in ("0", "fc1", "linear1"):
                    sub = getattr(mlp, name, None)
                    if sub is not None and hasattr(sub, "weight"):
                        new_merger._w1_weight = torch.transpose(sub.weight.data, -2, -1).contiguous()
                        if hasattr(sub, "bias") and sub.bias is not None:
                            new_merger._w1_bias = sub.bias.data.clone().reshape(1, 1, 1, -1)
                        break
                for name in ("2", "fc2", "linear2"):
                    sub = getattr(mlp, name, None)
                    if sub is not None and hasattr(sub, "weight"):
                        new_merger._w2_weight = torch.transpose(sub.weight.data, -2, -1).contiguous()
                        if hasattr(sub, "bias") and sub.bias is not None:
                            new_merger._w2_bias = sub.bias.data.clone().reshape(1, 1, 1, -1)
                        break

        return new_merger

    def preprocess_weights_impl(self):
        def _to_host(w, layout=ttnn.TILE_LAYOUT):
            if w is None:
                return None
            return ttnn.from_torch(w.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=layout)

        if self._use_layer_norm:
            self.tt_ln_weight = _to_host(self._ln_weight.unsqueeze(0))
            if self._ln_bias is not None:
                self.tt_ln_bias = _to_host(self._ln_bias.unsqueeze(0))
        else:
            if self._ln_weight is not None:
                dim = self._ln_weight.numel()
                tile = 32
                w = self._ln_weight.to(torch.bfloat16)
                w = w.view(1, 1, dim).reshape(1, 1, dim // tile, tile)
                self.tt_ln_weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        self.tt_w1 = (
            ttnn.from_torch(self._w1_weight.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            if self._w1_weight is not None
            else None
        )
        self.tt_w2 = (
            ttnn.from_torch(self._w2_weight.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            if self._w2_weight is not None
            else None
        )
        self.tt_w1_bias = (
            ttnn.from_torch(self._w1_bias.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            if self._w1_bias is not None
            else None
        )
        self.tt_w2_bias = (
            ttnn.from_torch(self._w2_bias.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            if self._w2_bias is not None
            else None
        )

    def move_weights_to_device_impl(self):
        mem = ttnn.DRAM_MEMORY_CONFIG

        def _to_dev(t):
            if t is None:
                return None
            return ttnn.to_device(t, self.device, memory_config=mem)

        self.tt_ln_weight = _to_dev(self.tt_ln_weight)
        self.tt_ln_bias = _to_dev(self.tt_ln_bias)
        self.tt_w1 = _to_dev(self.tt_w1)
        self.tt_w1_bias = _to_dev(self.tt_w1_bias)

        # Col-shard w2 across devices so the patch merger natively produces
        # col-sharded output matching text embeddings.  Weight shape is
        # [intermediate, H] (already transposed); sharding dim=-1 gives each
        # device [intermediate, H/num_devices].
        num_devices = self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1
        if num_devices > 1:
            col_shard_mapper = ttnn.ShardTensor2dMesh(
                self.device,
                dims=(None, -1),
                mesh_shape=list(self.device.shape),
            )
            # Re-create w2 on device with col-shard mapper
            self.tt_w2 = ttnn.from_torch(
                self._w2_weight.to(torch.bfloat16),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=mem,
                mesh_mapper=col_shard_mapper,
            )
            if self._w2_bias is not None:
                self.tt_w2_bias = ttnn.from_torch(
                    self._w2_bias.to(torch.bfloat16),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=mem,
                    mesh_mapper=col_shard_mapper,
                )
            else:
                self.tt_w2_bias = None
        else:
            self.tt_w2 = _to_dev(self.tt_w2)
            self.tt_w2_bias = _to_dev(self.tt_w2_bias)

        self.compute_kernel_config = _vision_matmul_compute_config(
            self.device, math_fidelity=VISION_MATMUL_MATH_FIDELITY
        )

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem = ttnn.DRAM_MEMORY_CONFIG

        # Fold ``[B,1,S,H] -> [B,1,S',mlp_size]``; the shape is known before the
        # norm (which preserves it), so decide the L1 BLOCK_SHARDED path up front.
        b0, b1, r, h = (
            int(hidden_states.shape[0]),
            int(hidden_states.shape[1]),
            int(hidden_states.shape[2]),
            int(hidden_states.shape[3]),
        )
        flat = r * h
        if flat % int(self.mlp_size) != 0:
            raise ValueError(f"PatchMerger reshape: S*H={flat} not divisible by mlp_size={self.mlp_size}")
        new_r = flat // int(self.mlp_size)

        # Block-sharded merger MLP. fc1 in0 is L1-interleaved BF16; fc1 output is L1
        # BLOCK_SHARDED over the 8x8 grid ([M/8, N/8] = [384, 768] shards); fc2
        # consumes the fc1 output shard with no reshard. Requires the production
        # 8x8 grid with M (= folded S/4) divisible by the shard grid.
        # 2D-mcast fc1 (obh=11 ibw=4, GELU fused, ~2.1 ms) beats the block-sharded
        # fc1 (~5.7 ms / 16% TFLOPs) on the production WH 8x8 / M=2816 shape -- this
        # matmul is weight-DRAM-bound and obh=11 is a single weight pass. When the
        # fast PC is available it supersedes the block-sharded path (its BFP4
        # interleaved output feeds the interleaved fc2).
        fast_fc1_pc = _vision_merger_fc1_fast_pc(self.device, new_r)
        fc1_bs_pc = _vision_merger_fc1_bs_program_config(self.device, new_r)
        fc2_bs_pc = _vision_merger_fc2_bs_program_config(self.device, new_r)
        bs_mem = _vision_block_sharded_mem(self.device, new_r, int(self.mlp_size))
        # The block-sharded fast path is tuned for the Wormhole 8x8 compute grid
        # with the production M (= folded S/4 = 3072). On other grids -- e.g. the
        # Blackhole 11x10 grid, or any folded M that doesn't divide the shard
        # extents -- the shard spec / program configs above come back None. Fall
        # back to plain DRAM-interleaved matmuls (auto program config when the
        # generic 2D-mcast config also can't tile the shape), mirroring the
        # o_proj fallback in the vision block. GELU is applied explicitly since
        # it can no longer be fused into the fc1 program config.
        use_bs = fast_fc1_pc is None and fc1_bs_pc is not None and fc2_bs_pc is not None and bs_mem is not None

        if self._use_layer_norm:
            print("Using LayerNorm")
            hidden_states = ttnn.layer_norm(hidden_states, weight=self.tt_ln_weight, bias=self.tt_ln_bias, epsilon=1e-6)
        elif self.tt_ln_weight is not None:
            print("Using RMSNorm")
            hidden_states = ttnn.rms_norm(hidden_states, weight=self.tt_ln_weight, epsilon=1e-6)

        # Fold [B,1,S,H] -> [B,1,S',mlp_size] in TILE (avoids RM untilize/tilize).
        # DRAM (not L1): under trace capture the ~590 KB L1-interleaved in0 gets
        # placed overlapping the fc1 block-sharded CB region (clash). The 2D-mcast
        # matmul streams in0 into CBs from DRAM just the same; frees the L1 edge.
        hidden_states = ttnn.reshape(
            hidden_states, (b0, b1, new_r, int(self.mlp_size)), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        compute_kc = getattr(self, "compute_kernel_config", None)

        if fast_fc1_pc is not None:
            # fc1: 2D-mcast BFP8 in0 (DRAM) -> BFP4 interleaved out, GELU fused.
            hidden_states = ttnn.linear(
                hidden_states,
                self.tt_w1,
                bias=self.tt_w1_bias,
                dtype=ttnn.bfloat4_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=fast_fc1_pc,
                compute_kernel_config=compute_kc,
            )
            # fc2: BFP4 interleaved in0 -> col-sharded BFP8 L1 out (matches text embeds).
            fc2_k = int(self.tt_w2.shape[-2])
            fc2_n = int(self.tt_w2.shape[-1])
            # Tuned TP4 config for the 2816x6144x384 shape (~161 us vs the generic
            # helper's ~563 us); falls back to the generic 2D-mcast PC otherwise.
            from models.experimental.tt_symbiote.modules.vision_tp4 import tp4_merger_fc2_pc

            fc2_pc = tp4_merger_fc2_pc(self.device, seq_len=new_r, k=fc2_k, n=fc2_n) or _vision_matmul_program_config(
                self.device, new_r, fc2_k, fc2_n
            )
            return ttnn.linear(
                hidden_states,
                self.tt_w2,
                bias=self.tt_w2_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=fc2_pc,
                compute_kernel_config=compute_kc,
            )

        if use_bs:
            # fc1: L1-interleaved BF8 in0 -> BLOCK_SHARDED out, GELU fused via program config.
            hidden_states = ttnn.linear(
                hidden_states,
                self.tt_w1,
                bias=self.tt_w1_bias,
                dtype=ttnn.bfloat4_b,
                memory_config=bs_mem,
                program_config=fc1_bs_pc,
                compute_kernel_config=compute_kc,
            )
            # fc2: consumes the fc1 output shard directly -> L1 interleaved out.
            hidden_states = ttnn.linear(
                hidden_states,
                self.tt_w2,
                bias=self.tt_w2_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=fc2_bs_pc,
                compute_kernel_config=compute_kc,
            )
            return hidden_states

        # Generic DRAM-interleaved fallback (e.g. Blackhole 11x10 grid).
        fc1_n = int(self.tt_w1.shape[-1])
        fc1_pc = _vision_matmul_program_config(self.device, new_r, int(self.mlp_size), fc1_n)
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_w1,
            bias=self.tt_w1_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=fc1_pc,
            compute_kernel_config=compute_kc,
        )
        hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        fc2_k = int(self.tt_w2.shape[-2])
        fc2_n = int(self.tt_w2.shape[-1])
        fc2_pc = _vision_matmul_program_config(self.device, new_r, fc2_k, fc2_n)
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_w2,
            bias=self.tt_w2_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=fc2_pc,
            compute_kernel_config=compute_kc,
        )

        return hidden_states


class TTNNDotsVisionBlockStack(TTNNLayerStack):
    """Trace-enabled stack of vision blocks + post-norm + merger.

    Captures the entire vision encoder core as a single trace.
    Different sequence length buckets get different traces automatically
    via the cache key mechanism in TracedRun.
    """

    # Every layer (incl. the O(S^2) SDPA, which is ~52% of vision-tower device
    # time) runs at the bucket the actual sequence is padded up to, so SDPA cost
    # scales with bucket^2. The old 4096->8192->12288 jumps (2x / 1.5x) forced up
    # to ~1064 patches of padding for a typical ~11224-patch image -> ~16% wasted
    # attention FLOPs. The 1024-step fillers below cap padding at <1024 patches in
    # the common large-image range (e.g. 11224 -> 11264 instead of 12288, ~16%
    # less SDPA for that image). All values are multiples of 256 so the per-layer
    # matmul program configs (which need (bucket/32) % grid_y == 0) stay on the
    # tuned 2D-mcast path instead of falling back to auto-config. Each bucket is a
    # separately captured trace, so trim this list to the image sizes you actually
    # serve if trace-capture memory/time becomes a concern.
    SEQ_LEN_BUCKETS = [
        256,
        512,
        768,
        1024,
        1536,
        2048,
        2560,
        3072,
        3584,
        4096,
        5120,
        6144,
        7168,
        8192,
        9216,
        10240,
        11264,
        12288,
        16384,
        20480,
        24576,
    ]

    def __init__(self, blocks, *, post_trunk_norm=None, patch_merger=None):
        super().__init__(blocks)
        self.post_trunk_norm = post_trunk_norm
        self.patch_merger = patch_merger
        self._bypass_tensor_wrapping = True

    @classmethod
    def nearest_bucket(cls, seq_len: int) -> int:
        for bucket in cls.SEQ_LEN_BUCKETS:
            if seq_len <= bucket:
                return bucket
        return -1

    def preprocess_weights_impl(self):
        super().preprocess_weights_impl()
        if self.post_trunk_norm is not None:
            self.post_trunk_norm.preprocess_weights()
        if self.patch_merger is not None:
            self.patch_merger.preprocess_weights()

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        if self.post_trunk_norm is not None:
            self.post_trunk_norm.move_weights_to_device()
        if self.patch_merger is not None:
            self.patch_merger.move_weights_to_device()

    def to_device(self, device):
        super().to_device(device)
        if self.post_trunk_norm is not None:
            self.post_trunk_norm.to_device(device)
        if self.patch_merger is not None:
            self.patch_merger.to_device(device)
        return self

    def forward(self, hidden_states, **kwargs):
        rot_mats = kwargs.get("rot_mats")
        cu_seqlens = kwargs.get("cu_seqlens")
        attention_mask = kwargs.get("attention_mask")
        attention_logical_seq_len = kwargs.get("attention_logical_seq_len")

        for layer in self.layers:
            hidden_states = layer.forward(
                hidden_states,
                rot_mats=rot_mats,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                attention_logical_seq_len=attention_logical_seq_len,
            )

        if self.post_trunk_norm is not None:
            hidden_states = self.post_trunk_norm.forward(hidden_states)

        if self.patch_merger is not None:
            hidden_states = self.patch_merger.forward(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Vision Tower (top-level)
# ---------------------------------------------------------------------------


class TTNNDotsOCRVisionTower(TTNNModule):
    """Native TTNNModule vision tower for dots.ocr.

    Full pipeline: PatchEmbed -> 42 VisionBlocks -> post-trunk RMSNorm -> PatchMerger.
    """

    _patch_embed_cls = TTNNDotsVisionPatchEmbed
    _block_cls = TTNNDotsVisionBlock
    _norm_cls = TTNNDotsVisionRMSNorm
    _merger_cls = TTNNDotsPatchMerger

    def __init__(self):
        super().__init__()
        self._hf_config = None
        self.patch_embed = None
        self.blocks = []
        self.post_trunk_norm = None
        self.patch_merger = None
        self.rope = None
        self.block_stack = None
        self._trace_enabled = True
        self.num_layers = 42
        self.hidden_size = 1536
        self.num_heads = 12
        self.head_dim = 128
        self.spatial_merge_size = 2
        self._bypass_tensor_wrapping = True
        self._attn_mask_cache: dict = {}

    @classmethod
    def from_torch(cls, hf_vision_tower, hf_config=None):
        new_tower = cls()
        new_tower._fallback_torch_layer = hf_vision_tower
        new_tower._hf_config = hf_config or getattr(hf_vision_tower, "config", None)

        vc = None
        if new_tower._hf_config is not None:
            vc = getattr(new_tower._hf_config, "vision_config", new_tower._hf_config)

        if vc is not None:
            new_tower.hidden_size = getattr(vc, "hidden_size", 1536)
            new_tower.num_heads = getattr(vc, "num_attention_heads", 12)
            new_tower.head_dim = new_tower.hidden_size // new_tower.num_heads
            new_tower.num_layers = getattr(vc, "num_hidden_layers", 42)
            new_tower.spatial_merge_size = getattr(vc, "spatial_merge_size", 2)

        patch_embed_module = getattr(hf_vision_tower, "patch_embed", None)
        if patch_embed_module is not None:
            patch_size = getattr(vc, "patch_size", 14) if vc else 14
            in_channels = getattr(vc, "num_channels", 3) if vc else 3
            new_tower.patch_embed = cls._patch_embed_cls.from_torch(
                patch_embed_module,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=new_tower.hidden_size,
            )

        blocks_attr = getattr(hf_vision_tower, "blocks", getattr(hf_vision_tower, "layers", None))
        if blocks_attr is not None:
            new_tower.blocks = []
            for hf_block in blocks_attr:
                block = cls._block_cls.from_torch(
                    hf_block,
                    hidden_size=new_tower.hidden_size,
                    num_heads=new_tower.num_heads,
                )
                new_tower.blocks.append(block)
            new_tower.num_layers = len(new_tower.blocks)

        post_trunk = getattr(
            hf_vision_tower,
            "post_trunk_norm",
            getattr(hf_vision_tower, "norm", None),
        )
        if post_trunk is not None:
            new_tower.post_trunk_norm = cls._norm_cls.from_torch(post_trunk)

        merger = getattr(
            hf_vision_tower,
            "merger",
            getattr(hf_vision_tower, "patch_merger", None),
        )
        if merger is not None:
            out_hidden = new_tower.hidden_size
            new_tower.patch_merger = cls._merger_cls.from_torch(
                merger,
                hidden_size=new_tower.hidden_size,
                out_hidden_size=out_hidden,
                spatial_merge_size=new_tower.spatial_merge_size,
            )

        new_tower.block_stack = TTNNDotsVisionBlockStack(
            new_tower.blocks,
            post_trunk_norm=new_tower.post_trunk_norm,
            patch_merger=new_tower.patch_merger,
        )

        return new_tower

    def preprocess_weights_impl(self):
        if self.patch_embed is not None:
            self.patch_embed.preprocess_weights()
        if self.block_stack is not None:
            self.block_stack.preprocess_weights()
        else:
            for block in self.blocks:
                block.preprocess_weights()
            if self.post_trunk_norm is not None:
                self.post_trunk_norm.preprocess_weights()
            if self.patch_merger is not None:
                self.patch_merger.preprocess_weights()

    def move_weights_to_device_impl(self):
        if self.patch_embed is not None:
            self.patch_embed.move_weights_to_device()
        if self.block_stack is not None:
            self.block_stack.move_weights_to_device()
        else:
            for block in self.blocks:
                block.move_weights_to_device()
            if self.post_trunk_norm is not None:
                self.post_trunk_norm.move_weights_to_device()
            if self.patch_merger is not None:
                self.patch_merger.move_weights_to_device()

        self.rope = TTNNDotsVision2DRoPE(
            device=self.device,
            head_dim=self.head_dim,
            spatial_merge_size=self.spatial_merge_size,
        )

    def to_device(self, device):
        super().to_device(device)
        if self.patch_embed is not None:
            self.patch_embed.to_device(device)
        if self.block_stack is not None:
            self.block_stack.to_device(device)
        else:
            for block in self.blocks:
                block.to_device(device)
            if self.post_trunk_norm is not None:
                self.post_trunk_norm.to_device(device)
            if self.patch_merger is not None:
                self.patch_merger.to_device(device)
        return self

    def merged_vision_sequence_length(self, grid_thw: torch.Tensor, pixel_values: torch.Tensor | None = None) -> int:
        """Merged vision token count on dim=2 after ``patch_merger`` (matches ``forward``).

        Used for scatter gather sizing without running the vision trunk.
        """
        if grid_thw is None:
            raise ValueError("grid_thw is required")
        g = grid_thw.detach().cpu() if hasattr(grid_thw, "is_cuda") and grid_thw.is_cuda else grid_thw
        if g.dim() == 1:
            g = g.unsqueeze(0)
        temporal = int(g[0, 0].item())
        height_patches = int(g[0, 1].item())
        width_patches = int(g[0, 2].item())
        num_patches = temporal * height_patches * width_patches
        if pixel_values is not None and self.patch_embed is not None and pixel_values.dim() == 4:
            patch_size = int(getattr(self.patch_embed, "patch_size", 14))
            _b, _c, h, w = pixel_values.shape
            alt = temporal * (h // patch_size) * (w // patch_size)
            if alt > 0:
                num_patches = min(num_patches, alt)
        # dim != 4 (e.g. HF ``pixel_values`` [num_patches, C*patch*patch]): use grid-derived count only.
        if self.patch_merger is not None:
            return num_patches // (self.spatial_merge_size**2)
        return num_patches

    def build_padded_attention_mask(self, actual_seq_len: int, bucket: int) -> ttnn.Tensor | None:
        """Build the SDPA key mask for bucket-padded vision input outside trace capture.

        Cached per (actual_seq_len, bucket) so the large TilizeDeviceOperation and
        FillPadDeviceOperation only happen once per unique input shape.
        """
        actual_seq_len = int(actual_seq_len)
        bucket = int(bucket)
        if bucket <= actual_seq_len:
            return None

        cache_key = (actual_seq_len, bucket)
        cached = self._attn_mask_cache.get(cache_key)
        if cached is not None:
            return cached

        mask = torch.zeros((1, 1, bucket, bucket), dtype=torch.float32)
        mask[..., actual_seq_len:] = -1e9
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
        result = ttnn.from_torch(
            mask,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=mapper,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self._attn_mask_cache[cache_key] = result
        return result

    def forward_post_patch_embed(
        self,
        x: ttnn.Tensor,
        grid_thw: torch.Tensor,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Vision blocks + post-trunk norm + patch merger (``patch_embed`` already applied)."""
        if _tp4_prefill_vision_enabled(self.device) and isinstance(x, torch.Tensor):
            _vision_tower_signpost("vision_tower.start")
        if grid_thw is None:
            raise ValueError("grid_thw is required for Dots vision")
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)
        if isinstance(x, torch.Tensor):
            mem = ttnn.DRAM_MEMORY_CONFIG
            mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
            x = x.unsqueeze(1) if x.dim() == 3 else x
            x = ttnn.from_torch(
                x.to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (1, 1, x.shape[1], x.shape[2]))

        actual_seq_len = int(x.shape[2])

        bucket = (
            TTNNDotsVisionBlockStack.nearest_bucket(actual_seq_len)
            if self._trace_enabled and self.block_stack is not None
            else -1
        )
        if os.environ.get("DOTS_OCR_PROFILE_SYNC", "").lower() in {"1", "true", "yes", "on"}:
            print(f"[DOTS_OCR_PROFILE_SYNC] vision.actual_seq_len={actual_seq_len} bucket={bucket}")

        if bucket == -1:
            rot_mats, cu_seqlens = self.rope.build(grid_thw, actual_seq_len)

            for block in self.blocks:
                x = block(x, rot_mats=rot_mats, cu_seqlens=cu_seqlens)

            if self.post_trunk_norm is not None:
                x = self.post_trunk_norm(x)

            if self.patch_merger is not None:
                x = self.patch_merger(x)
            _vision_tower_signpost("vision_tower.end")
        else:
            if actual_seq_len < bucket:
                pad_len = bucket - actual_seq_len
                x = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_len), (0, 0)), value=0.0)

            rot_mats, _ = self.rope.build_padded(grid_thw, actual_seq_len, bucket)

            x = self.block_stack(
                x,
                rot_mats=rot_mats,
                cu_seqlens=None,
                attention_mask=attention_mask,
                attention_logical_seq_len=actual_seq_len,
            )

            merged_seq_len = actual_seq_len // (self.spatial_merge_size**2)
            if int(x.shape[2]) > merged_seq_len:
                x = ttnn.slice(x, (0, 0, 0, 0), (int(x.shape[0]), int(x.shape[1]), merged_seq_len, int(x.shape[3])))
            _vision_tower_signpost("vision_tower.end")

        return x

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> ttnn.Tensor:
        """Run the full vision pipeline and return the result as a ttnn.Tensor on device.

        Returns:
            ttnn.Tensor: [1, 1, N_vision, H/num_devices] in TILE_LAYOUT,
                col-sharded across devices (on multi-device), or
                [1, 1, N_vision, H] replicated (on single device).
        """
        if grid_thw is None:
            raise ValueError("grid_thw is required for Dots vision")

        x = self.patch_embed(pixel_values, grid_thw)
        return self.forward_post_patch_embed(x, grid_thw)
