# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Derive and cross-check a CUDA golden for one Qwen Image 2.1 attention layer.

This uses captured inputs and the pinned upstream RoPE function. The derived
attention projection must match the pipeline's own captured projection before
the intermediate tensors are accepted as TT validation targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = args.cuda_dir / f"step_{args.step:03d}/transformer"
    name = f"transformer_blocks.{args.layer}"
    weights = torch.load(
        args.cuda_dir / f"weights/transformer_block_{args.layer:02d}.pt", weights_only=True, map_location="cpu"
    )

    def read(suffix: str) -> torch.Tensor:
        return torch.load(base / f"{name}.{suffix}.pt", weights_only=True, map_location="cpu").cuda()

    q = read("attn.norm_q")
    k = read("attn.norm_k")
    v = read("attn.to_v").reshape_as(q)
    rope = torch.load(base / name / "input/rotary_emb.pt", weights_only=True, map_location="cpu").cuda()
    segments = json.loads((base / name / "input/segments.json").read_text())
    sequence = q.shape[1]
    prefix_len = segments[-1][1] if segments else 0
    allowed = torch.zeros(sequence, sequence, device="cuda", dtype=torch.bool)
    for start, end, is_text in segments:
        allowed[start:end, :start] = True
        if is_text:
            allowed[start:end, start:end] = torch.ones(end - start, end - start, device="cuda", dtype=torch.bool).tril()
        else:
            allowed[start:end, start:end] = True
    allowed[prefix_len:, :] = True
    key_valid_path = base / name / "input/key_valid.pt"
    if key_valid_path.exists():
        key_valid = torch.load(key_valid_path, weights_only=True, map_location="cpu").cuda().bool()
        allowed &= key_valid[0][None, :]

    with torch.inference_mode():
        q_rope = apply_rotary_emb_qwen(q, rope, use_real=False)
        k_rope = apply_rotary_emb_qwen(k, rope, use_real=False)
        context = (
            F.scaled_dot_product_attention(
                q_rope.transpose(1, 2),
                k_rope.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=allowed[None, None],
                dropout_p=0.0,
                is_causal=False,
            )
            .transpose(1, 2)
            .flatten(2, 3)
        )
        projection = F.linear(context, weights["attn.to_out.0.weight"].cuda())

    upstream = read("attn.to_out.0")
    error = (projection.float() - upstream.float()).abs()
    report = {
        "step": args.step,
        "layer": args.layer,
        "sequence": sequence,
        "prefix_len": prefix_len,
        "projection_max_abs_error": float(error.max()),
        "projection_mean_abs_error": float(error.mean()),
        "projection_exact_fraction": float((projection == upstream).float().mean()),
    }
    for key, tensor in {
        "q_after_rope": q_rope,
        "k_after_rope": k_rope,
        "context": context,
        "projection": projection,
        "attention_allowed": allowed,
    }.items():
        torch.save(tensor.cpu(), args.output_dir / f"{key}.pt")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not torch.allclose(projection, upstream, atol=0.5, rtol=0.01):
        raise AssertionError("derived attention does not reproduce the captured upstream projection")


if __name__ == "__main__":
    main()
