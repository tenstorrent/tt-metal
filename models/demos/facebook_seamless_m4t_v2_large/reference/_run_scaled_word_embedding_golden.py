# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate the golden tensor for `scaled_word_embedding` and verify PCC vs. HF.

Uses the actual HuggingFace `SeamlessM4Tv2ScaledWordEmbedding` class with the
same random weights as the reference, so PCC should be (within fp32 round-off)
exactly 1.0. The golden output is saved to
``models/demos/facebook_seamless_m4t_v2_large/reference/golden/scaled_word_embedding.pt``.
"""

import math
import os
import sys

import torch
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ScaledWordEmbedding

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import scaled_word_embedding_forward  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return float((a @ b) / denom)


def main() -> int:
    torch.manual_seed(0)

    # SeamlessM4T-v2-Large dimensions (from ARCHITECTURE.md / SeamlessM4Tv2Config defaults)
    vocab_size = 256_102
    hidden_size = 1024
    padding_idx = 0
    scale = math.sqrt(hidden_size)  # 32.0

    # Build HF module and grab its random init weight.
    hf_module = (
        SeamlessM4Tv2ScaledWordEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx,
            embed_scale=scale,
        )
        .to(torch.float32)
        .eval()
    )
    weight = hf_module.weight.detach().clone()

    # Deterministic inputs (batch=2, seq_len=16, mix of normal & padding ids)
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(low=1, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long)
    # Sprinkle in a few padding ids so we exercise that path.
    input_ids[0, 0] = padding_idx
    input_ids[1, -1] = padding_idx

    with torch.no_grad():
        ref_out = scaled_word_embedding_forward(input_ids, weight, scale, padding_idx=padding_idx)
        hf_out = hf_module(input_ids)

    pcc_value = pcc(ref_out, hf_out)
    max_abs_diff = (ref_out - hf_out).abs().max().item()
    print(f"shapes: ref={tuple(ref_out.shape)} hf={tuple(hf_out.shape)}")
    print(f"max_abs_diff={max_abs_diff:.3e}  pcc={pcc_value:.6f}")

    # Save golden artifacts. Skip the full 256102x1024 weight (~1 GB) — store a
    # 16-bit checksum so TTNN consumers can verify they're using the matching
    # weight, then reload it via the HF module + same manual_seed if needed.
    golden_dir = os.path.join(os.path.dirname(__file__), "golden")
    os.makedirs(golden_dir, exist_ok=True)
    out_path = os.path.join(golden_dir, "scaled_word_embedding.pt")
    weight_checksum = float(weight.to(torch.float64).sum().item())
    torch.save(
        {
            "input_ids": input_ids,
            "scale": scale,
            "padding_idx": padding_idx,
            "output": ref_out,
            "pcc": pcc_value,
            "max_abs_diff": max_abs_diff,
            "weight_checksum": weight_checksum,
            "weight_seed": 0,  # torch.manual_seed used to init the HF module above
            "config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        },
        out_path,
    )
    print(f"saved golden -> {out_path}")

    if pcc_value < 0.99:
        print(f"FAIL: PCC {pcc_value} < 0.99")
        return 1
    print(f"OK: PCC {pcc_value} >= 0.99")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
