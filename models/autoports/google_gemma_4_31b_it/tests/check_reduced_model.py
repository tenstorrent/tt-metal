"""Numerical control for the two-real-layer wrapper probe."""

import json
from pathlib import Path

import torch

from .generate_full_reference import load_reference


def main():
    model = load_reference((0, 5))
    torch.set_num_threads(4)
    with torch.inference_mode():
        expected = model(torch.tensor([[2] * 33]), use_cache=False).logits.float()
    actual = torch.load("bringup/artifacts/full_model_reduced_logits.pt", weights_only=True)
    report = {}
    for phase, ref, out in [
        ("prefill", expected[:, :32], actual["prefill"].reshape(1, 32, -1)),
        ("decode", expected[:, 32:], actual["decode"].reshape(1, 1, -1)),
    ]:
        flatref, flatout = ref.double().flatten(), out.double().flatten()
        tok = out.argmax(-1)
        report[phase] = {
            "pcc": torch.corrcoef(torch.stack([flatref, flatout]))[0, 1].item(),
            "topk": {
                str(k): (ref.topk(k, dim=-1).indices == tok[..., None]).any(-1).float().mean().item()
                for k in (1, 5, 100)
            },
            "max_abs_diff": (ref - out).abs().max().item(),
            "hf_tokens": ref.argmax(-1).tolist(),
            "tt_tokens": tok.tolist(),
        }
    Path("models/autoports/google_gemma_4_31b_it/doc/full_model/reduced_numerical.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
