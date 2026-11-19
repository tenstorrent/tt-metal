# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only paired KV diagnostics; decoder hidden argmax is never token agreement.

Parent-owned serial commands (use same precision fields except kv_cache_dtype):

  export TT_VISIBLE_DEVICES=0,1,2,3
  export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
  export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
  timeout 900 python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_datatype_decoder_probe \
    --precision-config <BF16-policy.json> --host-evidence <bf16-host.pt> \
    --mode mesh --reference /home/ttuser/qwen38-flash/stage5_real_l3.pt \
    --output <bf16-report.json> --length 33 --capacity 128 --layer 3 \
    --trace --batch 32 --repeats 6 \
    --batch-fixture /home/ttuser/qwen38-flash/stage5_real_prompt_fixture.pt

Repeat serially with BFP8 policy and distinct outputs, then:
  python -m models.autoports.qwen_qwen3_8_flash_next.tests.analyze_datatype_kv_controls \
    --bf16 <bf16-host.pt> --candidate <bfp8-host.pt> --output <paired.json>

Existing stage5 evidence has independent baseline-relative PCCs, no paired raw
mesh outputs or vocabulary top1. Its B32 inputs are 32 distinct checkpoint
embedding rows, matched across runs (not 32 identical users). The B32 branch constructs empty
cache states; its first decode runs at position0. The separate B1 branch covers
33-token prefill followed by three continuation tokens. This analysis requires exact paired input equality.
Optional --bf16-logits/--candidate-logits must be torch-saved dictionaries with
matching input_token_ids and logits [32,...,vocabulary] from a full-model check.
Without those, overall required_gate_passed is false even if decoder PCC passes.
"""

import argparse
import copy
import json
from pathlib import Path


def correlation(left, right):
    import torch

    left, right = left.double().flatten(), right.double().flatten()
    if left.shape != right.shape or not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return float("nan")
    if torch.equal(left, right):
        return 1.0
    left, right = left - left.mean(), right - right.mean()
    denominator = left.norm() * right.norm()
    return float((left @ right) / denominator) if denominator else 0.0


def analyze(control, candidate, control_logits=None, candidate_logits=None):
    import torch

    baseline_policy = copy.deepcopy(control["precision"])
    reduced_policy = copy.deepcopy(candidate["precision"])
    if baseline_policy.pop("kv_cache_dtype") != "bfloat16" or reduced_policy.pop("kv_cache_dtype") != "bfloat8_b":
        raise ValueError("Expected BF16 control and BFP8 KV candidate")
    baseline_policy.pop("config_id")
    reduced_policy.pop("config_id")
    if baseline_policy != reduced_policy:
        raise ValueError("Paired policies must differ only in config_id and KV dtype")
    if control["rows"][0]["input"].shape[-2] != 33:
        raise ValueError("Short-prefix check requires 33 logical prefill rows")
    result = {"scope": "paired decoder outputs; no implicit token-top1 inference", "minimum_per_user_pcc": 0.99}
    for group, key, count in (("rows", "step", 4), ("batch", "user", 32)):
        left, right = control[group], candidate[group]
        if len(left) != count or len(right) != count:
            raise ValueError(f"Expected {count} {group} in each capture")
        metrics = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            if a[key] != index or b[key] != index or not torch.equal(a["input"], b["input"]):
                raise ValueError(f"Mismatched paired {group} input/order at {index}")
            if a["output"].shape != b["output"].shape:
                raise ValueError("Paired output shapes differ")
            scores = [
                correlation(x, y)
                for x, y in zip(
                    a["output"].reshape(-1, a["output"].shape[-1]),
                    b["output"].reshape(-1, b["output"].shape[-1]),
                    strict=True,
                )
            ]
            metrics.append(
                {
                    key: index,
                    "whole_pcc": correlation(a["output"], b["output"]),
                    "min_row_pcc": min(scores),
                    "pcc_passed": all(value >= 0.99 for value in scores),
                }
            )
        result[group] = metrics
    result["decoder_pcc_passed"] = all(row["pcc_passed"] for group in ("rows", "batch") for row in result[group])
    result["token_top1"] = {"status": "missing matched full-model vocabulary logits", "passed": False}
    if (control_logits is None) != (candidate_logits is None):
        raise ValueError("Provide both vocabulary-logit controls")
    if control_logits is not None:
        if not torch.equal(control_logits["input_token_ids"], candidate_logits["input_token_ids"]):
            raise ValueError("Vocabulary controls must have identical input token IDs")
        left, right = control_logits["logits"], candidate_logits["logits"]
        if left.shape != right.shape or left.shape[0] != 32 or left.ndim < 2:
            raise ValueError("Expected paired vocabulary logits with batch dimension 32")
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError("Vocabulary logits must be finite")
        matches = (left.argmax(-1) == right.argmax(-1)).reshape(32, -1).all(-1)
        scores = [correlation(a, b) for a, b in zip(left, right, strict=True)]
        phase_scores = [
            [correlation(x, y) for x, y in zip(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), strict=True)]
            for a, b in zip(left, right, strict=True)
        ]
        result["token_top1"] = {
            "status": "matched full-model vocabulary logits",
            "per_user_agreement": matches.tolist(),
            "per_user_pcc": scores,
            "per_user_phase_pcc": phase_scores,
            "passed": bool(matches.all()) and all(s >= 0.99 for row in phase_scores for s in row),
        }
    result["required_gate_passed"] = result["decoder_pcc_passed"] and result["token_top1"]["passed"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bf16-logits", type=Path)
    parser.add_argument("--candidate-logits", type=Path)
    args = parser.parse_args()
    import torch

    def load(path):
        return None if path is None else torch.load(path, map_location="cpu", weights_only=True)

    result = analyze(load(args.bf16), load(args.candidate), load(args.bf16_logits), load(args.candidate_logits))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("decoder_pcc_passed", "token_top1", "required_gate_passed")}))
    return 0 if result["required_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
