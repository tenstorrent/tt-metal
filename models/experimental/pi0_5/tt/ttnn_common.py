# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common utility functions for TTNN PI0 implementation.

This module provides shared helper functions used across the PI0 model:
    - Sinusoidal positional embeddings for flow matching timesteps
    - Safe tensor operations with dtype handling
    - Device-aware computations
"""

import math
import os
from typing import Optional, Tuple

import torch
import ttnn


# ---------------------------------------------------------------------------
# SDPA precision knobs (env-controllable for A/B testing; defaults are the
# values measured-best on Blackhole on this branch as of the 4-variant sweep
# in /tmp/sdpa_ab_matrix.py).
#
# Defaults:
#   PI0_SDPA_HIFI=2          → MathFidelity.HiFi2.
#                              HiFi4 was tested and *lost* both PCC and perf
#                              (−0.0008 PCC, +1.46 ms). With fp32_dest_acc_en
#                              already enabled, the fp32 accumulator already
#                              captures all the precision the multipliers can
#                              provide — HiFi4 just doubles mathops cost.
#                              ViT-BH-hiRes uses HiFi4 because they don't
#                              also enable fp32 dest; with both on we're
#                              already at fp32-equivalent softmax accuracy.
#   PI0_SDPA_EXP_APPROX=0    → exp_approx_mode=False.
#                              Marginal win on both axes vs True (+0.0003 PCC,
#                              −0.10 ms). Comment in ViT-BH-hiRes line 307:
#                              "False is more correct".
#   PI0_SDPA_PACKER_L1=1     → packer_l1_acc=True.
# (fp32_dest_acc_en is always on — no longer env-gated; the fp32 accumulator is
#  the sole expert/SigLIP SDPA precision path.)
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_sdpa_math_fidelity() -> "ttnn.MathFidelity":
    """Return the SDPA math fidelity, env-controllable for A/B testing.

    Default: HiFi2 (HiFi4 lost both PCC and perf; see comment block above).
    """
    hifi = _env_int("PI0_SDPA_HIFI", 2)
    return ttnn.MathFidelity.HiFi4 if hifi >= 4 else ttnn.MathFidelity.HiFi2


def get_sdpa_exp_approx_mode(seq_len_kv: Optional[int] = None) -> bool:
    """Return SDPA softmax exp_approx_mode.

    Default: False (exact). Per PERF_PLAYBOOKS/04_ATTENTION_SDPA.md §3c the
    short-ratio case should favour True, and the SigLIP isolated sweep
    (Sq=Skv=256, k_chunk=256 single chunk) showed `exp=1` as the wall-clock
    winner. BUT tracy-verified end-to-end (2026-06-04 v7 run): enabling
    exp_approx=True for SigLIP regressed device kernel time by +0.18 µs/call
    (12.90 → 13.08 µs). Same wall-clock-doesn't-translate pattern we saw
    on the VLM band. Per-shape default reverted to False; opt-in via
    `PI0_SDPA_EXP_APPROX_PER_SHAPE=1` if you want to A/B again.

    Global override: `PI0_SDPA_EXP_APPROX={0|1}` forces a single value
    everywhere (the prior global A/B knob).
    """
    explicit = os.environ.get("PI0_SDPA_EXP_APPROX")
    if explicit is not None:
        return explicit.strip().lower() in ("1", "true", "yes", "on")
    per_shape = _env_bool("PI0_SDPA_EXP_APPROX_PER_SHAPE", False)
    if not per_shape or seq_len_kv is None:
        return False
    # Per-shape opt-in path (kept for future A/B): single-chunk K → True.
    return seq_len_kv <= 256


def get_sdpa_compute_kernel_config(seq_len_kv: Optional[int] = None) -> "ttnn.WormholeComputeKernelConfig":
    """Return SDPA compute kernel config matching env knobs.

    Pass `seq_len_kv` to enable per-shape `exp_approx_mode` (04 §3c). When
    omitted, falls back to the previous global default for backwards-compat.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=get_sdpa_math_fidelity(),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=_env_bool("PI0_SDPA_PACKER_L1", True),
    )


def build_dram_width_sharded_memcfg(device, k: int, n: int, tile_size: int = 32, dram_cores: Optional[int] = None):
    """Build a DRAM width-sharded memory config for a weight tensor of shape (K, N).

    Per playbook 05 §3b + 08 §2 the decode-mode matmul recipe is:
      - weights in DRAM width-sharded (one tile-aligned slice per DRAM bank)
      - activations in L1 width-sharded
      - program config = MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig

    `dram_cores` defaults to `device.dram_grid_size().x` (full DRAM grid;
    12 on BH). Caller can override to a smaller power-of-2 to avoid N
    padding when `n` doesn't divide cleanly into 12. For example, pi0.5
    denoise mlp_down has N=1024 — 1024 % 12 ≠ 0 (would pad to 1152), but
    1024 % 8 == 0 (no padding). Using dram_cores=8 constructs a
    sub-range CoreRange((0,0)→(7,0)) of the full DRAM grid.

    Returns (memcfg, padded_n, dram_cores_used). `padded_n` ≥ `n` — the
    caller must pad the host weight tensor along N to `padded_n` before
    upload.
    """
    import math as _math

    dram_size = device.dram_grid_size()
    if dram_cores is None:
        dram_cores = dram_size.x
    assert dram_cores <= dram_size.x, f"dram_cores={dram_cores} exceeds BH limit {dram_size.x}"
    # CoreRange((0,0) → (dram_cores-1, dram_size.y-1)) — first `dram_cores` banks.
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, dram_size.y - 1))}
    )
    padded_n = _math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    return memcfg, padded_n, dram_cores


def get_ln_weight_memory_config() -> "ttnn.MemoryConfig":
    """Memory config for LN/RMSNorm weights (γ/β tensors).

    Default DRAM. `PI0_LN_WEIGHTS_L1=1` opts into L1 placement —
    eliminates the per-LN-call DRAM read of the weight tensor.
    Total budget at full pi0.5 (VLM-2B + expert-300M + SigLIP-27):
    ~14.7 MB of L1 (4.6 VLM + 2.3 expert + 7.8 SigLIP).
    Risk: trace mode may OOM L1; opt-in only.
    Per PERF_PLAYBOOKS/01 §6 + 06 §3 (layout matching).
    """
    return ttnn.L1_MEMORY_CONFIG if _env_bool("PI0_LN_WEIGHTS_L1", False) else ttnn.DRAM_MEMORY_CONFIG


def denoise_loop_fp32() -> bool:
    """Whether to stage the flow-matching Euler integration loop in fp32.

    Default `False` — `x_t ← x_t + dt·v` runs entirely in `bfloat16`,
    matching the dtype-map memory.

    Set `PI0_DENOISE_FP32=1` to keep `x_t` in `float32` across the 10
    Euler steps. The torch reference uses fp32 throughout the integration
    and accumulates ~0 rounding noise. The bf16 version accumulates up to
    ~bf16_eps · ||x_t|| ≈ 0.008 per step (10× per inference), which is
    enough to drift LIBERO trajectories on long-horizon goal/libero_10
    tasks (where small action drift compounds over 220–520 env steps).

    Adds 2 typecasts per step (bf16→fp32 for velocity, fp32→bf16 for
    embed_actions input) but the accumulator stays clean. Measured cost
    is +0.06 ms on the traced perf path (essentially free); accuracy
    lift on `lerobot/pi05_libero_finetuned` is ~+12.5 pp at N=10.
    """
    return _env_bool("PI0_DENOISE_FP32", False)


def get_ttnn_dtype(precision: str) -> ttnn.DataType:
    """
    Convert precision string to TTNN dtype.

    Args:
        precision: "bfloat16", "float32", "bfloat8_b", etc.

    Returns:
        TTNN data type
    """
    dtype_map = {
        "bfloat16": ttnn.bfloat16,
        "float32": ttnn.float32,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": ttnn.bfloat4_b,
    }
    return dtype_map.get(precision, ttnn.bfloat16)


def sdpa_prefill_chunk_sizes(
    seq_len_q: int,
    seq_len_kv: int,
    *,
    tile: int = 32,
) -> Tuple[int, int]:
    """
    q_chunk_size / k_chunk_size for ttnn.transformer.scaled_dot_product_attention.

    Mirrors the baseline in models/tt_transformers/tt/model_config.py
    get_attn_sdpa_prefill_program_config when chunk_start_idx is None: use 256 chunks
    only for long sequences (>= 2048), otherwise 64 — then cap by tile-aligned lengths.
    """
    longest = max(seq_len_q, seq_len_kv)
    # Bands tuned via tests/perf/test_sdpa_all_shapes_sweep.py +
    # tracy device-kernel verification on test_perf_ttnn_full_e2e.py:
    # - longest>=2048: base=256/256 (long context, lots of chunks anyway)
    # - longest>=512:  base=64/256  (VLM-prefill at bs=2 single-pass Sq=Skv=768:
    #                                k=256 beats k=128 by -5.64 µs/call tracy-
    #                                verified (-0.10 ms over 18 calls). Halves
    #                                the K-chunk loop iterations (3 vs 6 at
    #                                S=768). q=64 stays — q=128 regressed by
    #                                +13 µs/call at S=512, kept conservative.
    #                                See [[pi05-sdpa-bs2-sweep-2026-06-05]].)
    # - longest>=128:  base=64/256  (SigLIP Sq=Skv=256 wants single-chunk K.
    #                                At Skv=256, k=256 means 1 chunk vs 4 with
    #                                the prior k=64. Tracy-verified -8 µs/call
    #                                (-39%, -0.217 ms over 27 calls).)
    # - else:          base=64/64   (very short — preserve original behavior)
    # PI0_SDPA_LEGACY_BANDS=1 reverts to pre-2026-06-04 bands for A/B
    # (k=128 at the VLM band — the previous default before this commit).
    import os as _os

    if _os.environ.get("PI0_SDPA_LEGACY_BANDS", "").lower() in ("1", "true", "yes", "on"):
        if longest >= 2048:
            base_q, base_k = 256, 256
        elif longest >= 512:
            base_q, base_k = 64, 128
        else:
            base_q, base_k = 64, 64
    else:
        if longest >= 2048:
            base_q, base_k = 256, 256
        elif longest >= 512:
            base_q, base_k = 64, 256
        elif longest >= 128:
            base_q, base_k = 64, 256
        else:
            base_q, base_k = 64, 64
    q_aligned = ((seq_len_q + tile - 1) // tile) * tile if seq_len_q > 0 else tile
    k_aligned = ((seq_len_kv + tile - 1) // tile) * tile if seq_len_kv > 0 else tile
    q_chunk = min(base_q, q_aligned)
    k_chunk = min(base_k, k_aligned)
    # Two opt-in force-overrides for shape-specific k_chunk tuning. Trade the
    # divisor-aware "no waste" picker for a smaller iteration count when fewer
    # SDPA inner iters beats less masking compute on the trailing iter. At
    # bs=3 chunk=1024 the denoise shape is (Sq=32, Skv=1056); divisor-aware
    # picks k_chunk=32 → 33 K-iters where dispatch overhead dominates. Force
    # =128 → 9 iters (last masked), trades 75% waste on iter 9 for −24 saved
    # iters. Sweep at 5 denoise steps: k=64 / k=128 both win −0.4 ms over
    # the auto pick (k=32) on the bs=3 e2e trace.
    import os as _os

    if seq_len_q <= 64 and seq_len_kv >= 512:
        _force = _os.environ.get("PI0_SDPA_DENOISE_K_FORCE", "").strip()
        if _force.isdigit():
            forced = int(_force)
            if forced >= tile and forced <= base_k:
                return max(q_chunk, tile), forced
    if seq_len_q >= 512 and seq_len_kv >= 512:
        _force = _os.environ.get("PI0_SDPA_PREFILL_K_FORCE", "").strip()
        if _force.isdigit():
            forced = int(_force)
            if forced >= tile:
                return max(q_chunk, tile), forced
    # Divisor-aware k_chunk: SDPA does not short-circuit on masked positions —
    # if k_chunk does not divide k_aligned the last K-iter still processes the
    # full k_chunk columns and masks the trailing ones, wasting compute. Prefer
    # the largest power-of-2 ≤ base_k that divides k_aligned cleanly. For the
    # pi0.5 denoise step (k_aligned=544): 544 % 128 = 32 → drops to 32, the
    # only power-of-2 ≤ 128 that divides 544; cuts denoise SDPA ~1.7×. Power-
    # of-2 candidates only — non-pow2 tile-aligned chunks (e.g. 96) trip the
    # kernel's CB heuristics on some shapes.
    if k_aligned % k_chunk != 0:
        for cand in (256, 128, 64, 32):
            if cand > base_k:
                continue
            if cand < tile:
                break
            if k_aligned % cand == 0:
                k_chunk = cand
                break
        else:
            k_chunk = tile
    return max(q_chunk, tile), max(k_chunk, tile)


def create_sinusoidal_pos_embedding_ttnn(
    time: ttnn.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: Optional[ttnn.Device] = None,
    indices: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    """
    Create sinusoidal positional embeddings for timesteps (pure TTNN version).

    All computations are done on device using TTNN operations.

    Args:
        time: TTNN tensor of shape (batch_size,) with timestep values
        dimension: Embedding dimension (must be divisible by 2)
        min_period: Minimum period for sinusoidal encoding
        max_period: Maximum period for sinusoidal encoding
        device: TTNN device (uses time's device if not specified)

    Returns:
        TTNN tensor of shape (batch_size, dimension) with sinusoidal embeddings
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if device is None:
        device = time.device()

    half_dim = dimension // 2

    # Create fraction [0, 1/(n-1), 2/(n-1), ..., 1] using TTNN
    # ttnn.arange creates [0, 1, 2, ..., n-1], divide by (n-1) to get [0, 1]
    indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
    if half_dim > 1:
        fraction = ttnn.multiply(indices, 1.0 / (half_dim - 1))
    else:
        fraction = indices  # Edge case: half_dim == 1

    # Compute period: min_period * (max_period / min_period) ** fraction
    # = min_period * exp(fraction * log(max_period / min_period))
    log_ratio = math.log(max_period / min_period)
    exponent = ttnn.multiply(fraction, log_ratio)
    period_ratio = ttnn.exp(exponent)
    period = ttnn.multiply(period_ratio, min_period)

    # Compute scaling_factor: (1.0 / period) * 2 * pi
    inv_period = ttnn.reciprocal(period)
    scaling_factor = ttnn.multiply(inv_period, 2 * math.pi)

    # Reshape for broadcasting: scaling_factor [half_dim] -> [1, half_dim]
    scaling_factor = ttnn.reshape(scaling_factor, (1, half_dim))

    # Reshape time for broadcasting: [batch] -> [batch, 1]
    time_reshaped = ttnn.reshape(time, (-1, 1))

    # Compute sin input: time * scaling_factor (broadcasts to [batch, half_dim])
    sin_input = ttnn.matmul(time_reshaped, scaling_factor)

    # Compute sin and cos
    sin_emb = ttnn.sin(sin_input)
    cos_emb = ttnn.cos(sin_input)

    # Concatenate to get [batch, dimension]
    embeddings = ttnn.concat([sin_emb, cos_emb], dim=-1)

    # Clean up
    ttnn.deallocate(indices)
    ttnn.deallocate(fraction)
    ttnn.deallocate(exponent)
    ttnn.deallocate(period_ratio)
    ttnn.deallocate(period)
    ttnn.deallocate(inv_period)
    ttnn.deallocate(scaling_factor)
    ttnn.deallocate(sin_input)

    return embeddings


def safe_cat_ttnn(
    tensors: list,
    dim: int = -1,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    Safely concatenate TTNN tensors.

    Args:
        tensors: List of TTNN tensors to concatenate
        dim: Dimension along which to concatenate
        memory_config: Optional memory config for output

    Returns:
        Concatenated TTNN tensor
    """
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")

    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    return ttnn.concat(tensors, dim=dim, memory_config=memory_config)


def compute_position_ids_ttnn(
    pad_masks: ttnn.Tensor,
    device: Optional[ttnn.Device] = None,
) -> ttnn.Tensor:
    """
    Compute position IDs from padding masks (TTNN version).

    Args:
        pad_masks: Boolean TTNN tensor (batch_size, seq_len)
        device: TTNN device

    Returns:
        Position IDs TTNN tensor (batch_size, seq_len)
    """
    # Use moreh_cumsum for cumulative sum
    cumsum = ttnn.moreh_cumsum(pad_masks, dim=1)

    # Subtract 1 to get 0-indexed positions
    ones = ttnn.ones_like(cumsum)
    position_ids = ttnn.subtract(cumsum, ones)

    return position_ids


def ttnn_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
    """
    Convert TTNN tensor to PyTorch tensor.

    Args:
        tensor: TTNN tensor

    Returns:
        PyTorch tensor
    """
    return ttnn.to_torch(tensor)


def torch_to_ttnn(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: Optional[ttnn.DataType] = None,
    layout: Optional[ttnn.Layout] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    Convert PyTorch tensor to TTNN tensor.

    Args:
        tensor: PyTorch tensor
        device: TTNN device
        dtype: TTNN data type (default: bfloat16)
        layout: TTNN layout (default: TILE_LAYOUT)
        memory_config: Memory configuration (default: DRAM)

    Returns:
        TTNN tensor
    """
    if dtype is None:
        dtype = ttnn.bfloat16
    if layout is None:
        layout = ttnn.TILE_LAYOUT
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def tensor_1d_to_2d_ttnn(
    tensor: "torch.Tensor",
    device: ttnn.Device,
    dtype: Optional[ttnn.DataType] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    Convert 1D PyTorch tensor to 2D TTNN tensor without using torch.unsqueeze().

    Converts [features] -> [1, features] on device using TTNN operations.
    Used for biases, layer norm weights, and other 1D tensors.

    Args:
        tensor: 1D PyTorch tensor of shape [features]
        device: TTNN device
        dtype: TTNN data type (default: bfloat16)
        memory_config: Memory configuration (default: DRAM)

    Returns:
        TTNN tensor of shape [1, features] in TILE_LAYOUT
    """
    if dtype is None:
        dtype = ttnn.bfloat16
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Reshape on host (free) so we can upload directly in TILE_LAYOUT.
    # ttnn.from_torch(layout=TILE_LAYOUT) does the tile padding on host —
    # no device TilizeWithValPadding op emitted, no on-device to_layout call.
    t2d = tensor.reshape(1, tensor.shape[0]).contiguous()
    return ttnn.from_torch(
        t2d,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


# Default exports
create_sinusoidal_pos_embedding = create_sinusoidal_pos_embedding_ttnn
safe_cat = safe_cat_ttnn
compute_position_ids = compute_position_ids_ttnn
