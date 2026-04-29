#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Inspect real DeepSeek LM head weights from cache: raw bf16, BFP8 quantized, and folded.

Usage:
    python models/demos/deepseek_v3_b1/scripts/inspect_lm_head_weights.py \
        --model-path /mnt/models/deepseek-ai/DeepSeek-V3 \
        --cache-path /mnt/models/deepseek-ai/cache-2026-03-22
"""

import argparse
from pathlib import Path

import torch

torch.set_printoptions(linewidth=200, precision=6, sci_mode=False)

import ttnn
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict


def _per_shard_quantize(tensor: torch.Tensor, *, num_shards: int, shard_dim: int, dtype) -> torch.Tensor:
    shards = tensor.chunk(num_shards, dim=shard_dim)
    quantized_shards = []
    for shard in shards:
        shard_c = shard.contiguous()
        tt = ttnn.from_torch(shard_c, dtype=dtype, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((32, 32)))
        quantized_shards.append(ttnn.to_torch(tt).to(torch.bfloat16))
    return torch.cat(quantized_shards, dim=shard_dim)


def main():
    parser = argparse.ArgumentParser(description="Inspect DeepSeek LM head weights")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to HF model directory")
    parser.add_argument("--rows", type=int, default=8, help="Number of rows/cols to print from corners")
    args = parser.parse_args()

    model_path = args.model_path
    R = args.rows

    print(f"Loading state dict from {model_path} ...")
    sd = LazyStateDict(model_path)

    lm_key = "lm_head.weight"
    norm_key = "model.norm.weight"

    print(f"\n{'='*80}")
    print("1) RAW LM HEAD WEIGHTS (bf16)")
    print(f"{'='*80}")
    lm_w = sd[lm_key].to(torch.bfloat16)
    print(f"   shape: {lm_w.shape}  dtype: {lm_w.dtype}")
    print(
        f"   min={lm_w.min().item():.6f}  max={lm_w.max().item():.6f}  "
        f"mean={lm_w.float().mean().item():.6f}  std={lm_w.float().std().item():.6f}"
    )
    print(f"\n   Top-left {R}x{R}:")
    print(lm_w[:R, :R])
    print(f"\n   Bottom-right {R}x{R}:")
    print(lm_w[-R:, -R:])

    print(f"\n{'='*80}")
    print("2) GAMMA / FINAL NORM WEIGHTS (bf16)")
    print(f"{'='*80}")
    gamma = sd[norm_key].to(torch.bfloat16)
    print(f"   shape: {gamma.shape}  dtype: {gamma.dtype}")
    print(
        f"   min={gamma.min().item():.6f}  max={gamma.max().item():.6f}  "
        f"mean={gamma.float().mean().item():.6f}  std={gamma.float().std().item():.6f}"
    )
    print(f"\n   First {R*4} values:")
    print(gamma[: R * 4])
    print(f"\n   Last {R*4} values:")
    print(gamma[-R * 4 :])

    print(f"\n{'='*80}")
    print("3) LM HEAD after BFP8 quantization (round-trip: bf16 -> bfp8 -> bf16)")
    print(f"   Quantized per-shard (8 shards along dim=1) to match ShardTensorToMesh")
    print(f"{'='*80}")
    lm_w_T = lm_w.T.contiguous()  # (K=7168, V=129280) — how it's stored on device
    lm_w_bfp8 = _per_shard_quantize(lm_w_T, num_shards=8, shard_dim=1, dtype=ttnn.bfloat8_b)
    quant_err = (lm_w_T.float() - lm_w_bfp8.float()).abs()
    print(f"   shape: {lm_w_bfp8.shape}  (transposed: K x V)")
    print(
        f"   quant error — max={quant_err.max().item():.6f}  mean={quant_err.mean().item():.6f}  "
        f"median={quant_err.median().item():.6f}"
    )
    print(f"\n   Top-left {R}x{R} (bf16 original, transposed):")
    print(lm_w_T[:R, :R])
    print(f"\n   Top-left {R}x{R} (after BFP8 round-trip):")
    print(lm_w_bfp8[:R, :R])
    print(f"\n   Quantization error top-left {R}x{R}:")
    print(quant_err[:R, :R])

    print(f"\n{'='*80}")
    print("4) FOLDED LM HEAD: W_folded = lm_w * gamma, then BFP8 quantize")
    print(f"{'='*80}")
    lm_w_folded = lm_w * gamma  # (V, K) * (K,) broadcast
    lm_w_folded_T = lm_w_folded.T.contiguous()  # (K, V)
    lm_w_folded_bfp8 = _per_shard_quantize(lm_w_folded_T, num_shards=8, shard_dim=1, dtype=ttnn.bfloat8_b)
    fold_quant_err = (lm_w_folded_T.float() - lm_w_folded_bfp8.float()).abs()
    print(f"   shape: {lm_w_folded_bfp8.shape}  (transposed: K x V)")
    print(
        f"   folded min={lm_w_folded.min().item():.6f}  max={lm_w_folded.max().item():.6f}  "
        f"mean={lm_w_folded.float().mean().item():.6f}  std={lm_w_folded.float().std().item():.6f}"
    )
    print(
        f"   quant error — max={fold_quant_err.max().item():.6f}  mean={fold_quant_err.mean().item():.6f}  "
        f"median={fold_quant_err.median().item():.6f}"
    )
    print(f"\n   Top-left {R}x{R} (folded bf16, transposed):")
    print(lm_w_folded_T[:R, :R])
    print(f"\n   Top-left {R}x{R} (folded after BFP8 round-trip):")
    print(lm_w_folded_bfp8[:R, :R])

    print(f"\n{'='*80}")
    print("5) SANITY CHECK: folded vs unfolded logits for a random input")
    print(f"{'='*80}")
    torch.manual_seed(0)
    h = torch.randn(1, 7168, dtype=torch.bfloat16)
    rms = (h.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
    h_normed_no_gamma = (h.float() / rms).to(torch.bfloat16)
    h_normed_with_gamma = (h.float() / rms * gamma.float()).to(torch.bfloat16)

    logits_unfolded = (h_normed_with_gamma.float() @ lm_w_bfp8.float()).squeeze()
    logits_folded = (h_normed_no_gamma.float() @ lm_w_folded_bfp8.float()).squeeze()

    diff = (logits_unfolded - logits_folded).abs()
    print(f"   logits_unfolded argmax={logits_unfolded.argmax().item()}  max={logits_unfolded.max().item():.4f}")
    print(f"   logits_folded   argmax={logits_folded.argmax().item()}  max={logits_folded.max().item():.4f}")
    print(f"   logit diff — max={diff.max().item():.6f}  mean={diff.mean().item():.6f}")
    print(f"   argmax match: {logits_unfolded.argmax().item() == logits_folded.argmax().item()}")
    print(f"\n   First 16 logits (unfolded): {logits_unfolded[:16]}")
    print(f"   First 16 logits (folded):   {logits_folded[:16]}")


if __name__ == "__main__":
    main()
