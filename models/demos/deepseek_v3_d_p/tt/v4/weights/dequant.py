# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4-Flash checkpoint dequantization primitives (copied from tt-blaze blaze/weights/deepseek_v4_flash/dequant.py).

The upstream ``deepseek-ai/DeepSeek-V4-Flash`` snapshot is ~160 GB over 48
shards and stores **two different quantizations**, per its ``config.json``
(``quantization_config: {fmt: e4m3, scale_fmt: ue8m0, weight_block_size:
[128, 128]}`` plus ``expert_dtype: fp4``):

* **Routed experts** -- ``layers.L.ffn.experts.E.{w1,w2,w3}.weight`` are
  FP4-E2M1 packed two-per-byte in ``int8``, with E8M0 power-of-two scales over
  32-element blocks along K.
* **Everything else quantized** -- the attention projections
  (``attn.{wq_a,wq_b,wkv,wo_a,wo_b}``) and the shared expert
  (``ffn.shared_experts.{w1,w2,w3}``) are FP8-E4M3 with E8M0 power-of-two
  scales over 128x128 weight blocks.
* **Not quantized** -- norms (``attn_norm``, ``ffn_norm``, ``attn.q_norm``,
  ``attn.kv_norm``), the router (``ffn.gate.weight``), ``attn.attn_sink``, the
  ``hc_*`` mHC scalars, ``embed.weight`` and ``head.weight`` are plain bf16 (or
  fp32 for a bias) and are returned as-is.

Note that the scale suffix here is ``.scale``, NOT DeepSeek-V3's
``_scale_inv`` -- tt-metal's ``dequantize_state_dict`` (keyed on ``_scale_inv``) is
therefore not reusable for this checkpoint, and neither is the shared
``compressed_tensors`` path, which expects reciprocal scales.

"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Canonical FP4-E2M1 value table, low nibble first along K (matches upstream
# ``inference/convert.py``). Index 0 and 8 are both zero -- E2M1 has a signed zero.
FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
FP4_BLOCK_SIZE = 32
FP8_BLOCK_SIZE = (128, 128)

INDEX_FILENAME = "model.safetensors.index.json"


def fp4_e2m1_dequant_to_bf16(packed_i8: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Dequantize one packed FP4-E2M1 routed-expert tile to bfloat16.

    Args:
        packed_i8: ``(out, in // 2)`` int8; each byte holds two FP4 values, low
            nibble first along K.
        scale_e8m0: ``(out, in // 32)`` float8_e8m0fnu; one power-of-two scale
            per 32 consecutive FP4 values on K.

    Returns: ``(out, in)`` bfloat16.
    """
    if packed_i8.dtype != torch.int8:
        raise TypeError(f"packed FP4 must be int8, got {packed_i8.dtype}")
    if packed_i8.ndim != 2 or scale_e8m0.ndim != 2:
        raise ValueError(f"expected 2D tensors, got {packed_i8.ndim}D / {scale_e8m0.ndim}D")
    out_dim, packed_in = packed_i8.shape
    in_dim = packed_in * 2
    want = (out_dim, in_dim // FP4_BLOCK_SIZE)
    if tuple(scale_e8m0.shape) != want:
        raise ValueError(f"fp4 scale {tuple(scale_e8m0.shape)} != {want} for weight {tuple(packed_i8.shape)}")

    x_u8 = packed_i8.view(torch.uint8)
    low = (x_u8 & 0x0F).long()
    high = ((x_u8 >> 4) & 0x0F).long()
    # Interleave low/high so the pair expands along K in storage order.
    unpacked = torch.stack([FP4_E2M1_TABLE[low], FP4_E2M1_TABLE[high]], dim=-1).flatten(-2)
    scales_full = scale_e8m0.float().repeat_interleave(FP4_BLOCK_SIZE, dim=-1)
    return (unpacked * scales_full).to(torch.bfloat16)


def fp8_e4m3_dequant_to_bf16(
    weight_e4m3: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block: tuple[int, int] = FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantize one block-scaled FP8-E4M3 tile (attention proj / shared expert) to bfloat16."""
    if weight_e4m3.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected float8_e4m3fn, got {weight_e4m3.dtype}")
    if scale_e8m0.dtype != torch.float8_e8m0fnu:
        raise TypeError(f"expected float8_e8m0fnu scales, got {scale_e8m0.dtype}")
    out_dim, in_dim = weight_e4m3.shape
    b_out, b_in = block
    want = (out_dim // b_out, in_dim // b_in)
    if tuple(scale_e8m0.shape) != want:
        raise ValueError(f"fp8 scale {tuple(scale_e8m0.shape)} != {want} for weight {tuple(weight_e4m3.shape)}")
    scales_full = scale_e8m0.float().repeat_interleave(b_out, dim=0).repeat_interleave(b_in, dim=1)
    return (weight_e4m3.float() * scales_full).to(torch.bfloat16)


def read_weight_map(model_dir: str | Path) -> dict[str, str]:
    """key -> shard filename, from the safetensors index."""
    index = Path(model_dir) / INDEX_FILENAME
    if not index.is_file():
        raise FileNotFoundError(f"no {INDEX_FILENAME} under {model_dir}")
    with index.open() as fh:
        return json.load(fh)["weight_map"]


def read_tensors(model_dir: str | Path, keys: list[str], *, weight_map=None) -> dict[str, torch.Tensor]:
    """Read ``keys`` out of the shards, opening each shard at most once.

    The checkpoint is far larger than RAM, so callers ask for the slice they
    need (one layer, or one expert) rather than a whole state dict.
    """
    from safetensors import safe_open  # lazy: collection-only hosts lack it

    model_dir = Path(model_dir)
    wm = weight_map if weight_map is not None else read_weight_map(model_dir)
    missing = [k for k in keys if k not in wm]
    if missing:
        raise KeyError(f"checkpoint index has no keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")

    by_shard: dict[str, list[str]] = {}
    for k in keys:
        by_shard.setdefault(wm[k], []).append(k)

    out: dict[str, torch.Tensor] = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as fh:
            for k in shard_keys:
                out[k] = fh.get_tensor(k)
    return out
