# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verify reference Conformer FFN against HuggingFace and save golden tensor.

Compares
:func:`models.demos.facebook_seamless_m4t_v2_large.reference.functional.conformer_ffn_forward`
against the HuggingFace ``SeamlessM4Tv2ConformerFeedForward`` module with the
same weights, then writes ``golden/conformer_ffn.pt`` and prints final PCC.
"""

from pathlib import Path

import torch
from transformers import SeamlessM4Tv2Config
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import SeamlessM4Tv2ConformerFeedForward

from models.demos.facebook_seamless_m4t_v2_large.reference.functional import conformer_ffn_forward

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return (a @ b / denom).item()


def main() -> float:
    torch.manual_seed(0)

    # SeamlessM4T-v2-Large speech-encoder Conformer FFN dims.
    config = SeamlessM4Tv2Config()
    hidden = config.hidden_size  # 1024
    intermediate = config.speech_encoder_intermediate_size  # 4096
    act_fn = config.speech_encoder_hidden_act  # "swish"
    assert hidden == 1024 and intermediate == 4096, (hidden, intermediate)
    assert act_fn in ("swish", "silu"), act_fn

    hf = SeamlessM4Tv2ConformerFeedForward(config)
    hf.eval()
    # Ensure dropouts are no-ops (eval already disables them; be explicit).
    hf.intermediate_dropout.p = 0.0
    hf.output_dropout.p = 0.0

    state_dict = {
        "intermediate_dense": {
            "weight": hf.intermediate_dense.weight.detach(),
            "bias": hf.intermediate_dense.bias.detach(),
        },
        "output_dense": {
            "weight": hf.output_dense.weight.detach(),
            "bias": hf.output_dense.bias.detach(),
        },
    }

    batch, seq_len = 1, 128
    x = torch.randn(batch, seq_len, hidden, dtype=torch.float32)

    with torch.no_grad():
        hf_out = hf(x)
    ref_out = conformer_ffn_forward(x, state_dict, act_fn=act_fn)

    pcc = _pcc(ref_out, hf_out)
    max_abs = (ref_out - hf_out).abs().max().item()
    print(f"[conformer_ffn] pcc={pcc:.6f} max_abs_diff={max_abs:.3e}")

    assert torch.allclose(ref_out, hf_out, atol=1e-5, rtol=1e-4), f"conformer_ffn diverged from HF: max_abs={max_abs}"
    assert pcc > 0.99, f"PCC {pcc} <= 0.99"

    golden_path = GOLDEN_DIR / "conformer_ffn.pt"
    torch.save(
        {
            "input": x,
            "state_dict": state_dict,
            "output": ref_out,
            "config": {
                "batch": batch,
                "seq_len": seq_len,
                "hidden": hidden,
                "intermediate": intermediate,
                "act_fn": act_fn,
                "dtype": "float32",
                "block": "conformer_ffn",
                "model_id": "facebook/seamless-m4t-v2-large",
                "hf_class": "SeamlessM4Tv2ConformerFeedForward",
            },
        },
        golden_path,
    )
    print(f"[conformer_ffn] saved golden to {golden_path}")

    return pcc


if __name__ == "__main__":
    pcc = main()
    print(f"\nFINAL PCC conformer_ffn: {pcc:.6f}")
