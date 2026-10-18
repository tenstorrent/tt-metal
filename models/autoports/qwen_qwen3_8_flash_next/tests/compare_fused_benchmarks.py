# SPDX-License-Identifier: Apache-2.0
"""Compare retained matched-input benchmark tensors using centered FP64 metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidates", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1] / "doc/fused_decoder"
    paths = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/fusion_outputs")
    reference = torch.load(paths / (args.reference + ".pt"), weights_only=True)
    results = {}
    for tag in args.candidates:
        path = paths / (tag + ".pt")
        candidate = torch.load(path, weights_only=True)
        metrics = {mode: pcc_metrics(reference[mode], candidate[mode]) for mode in reference}
        metrics["prefill"]["min_token_pcc"] = min(
            pcc_metrics(reference["prefill"][..., i, :], candidate["prefill"][..., i, :])["pcc"]
            for i in range(reference["prefill"].shape[-2])
        )
        results[tag] = dict(metrics=metrics, outputs_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = dict(
        reference=args.reference,
        reference_outputs_sha256=hashlib.sha256((paths / (args.reference + ".pt")).read_bytes()).hexdigest(),
        candidates=results,
    )
    (root / ("comparison_" + args.reference + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
