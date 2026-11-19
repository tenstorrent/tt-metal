# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU same-model control: change only attention GR down accumulation arithmetic.

The literal HF BF16 decoder, weights, architecture, token stream, recurrent
state semantics and output dtype are preserved. Separate diagnostic runs
replace its attention hyper-connection down projection accumulation with
FP32 F.linear, FP64 F.linear, or the SFPU-equivalent FP32 addition tree,
then round to BF16 at the unchanged projection boundary. This is a causal
backend-sensitivity experiment, not a replacement reference or relaxed gate.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .probe_batch_intermediates import fixture, metrics
from .probe_gr_rounding import tree_projection


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tt-evidence", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, help="External lossless input/output/route evidence cache")
    args = parser.parse_args()
    torch.set_num_threads(8)
    config, reference, _, requests, hashes = fixture(args.snapshot, (3, 16, 23, 29))
    tt = json.loads(args.tt_evidence.read_text())
    module = reference.attn_hyper_connection.input_mix_weight_down
    original = module.forward
    captured = {}

    def router_hook(_module, _inputs, output):
        captured["indices"] = output[2].clone()

    def attn_hook(_module, _inputs, output):
        captured["mixed"] = output[0].clone()

    hooks = [
        reference.mlp.gate.register_forward_hook(router_hook),
        reference.attn_hyper_connection.register_forward_hook(attn_hook),
    ]
    report = {
        "scope": __doc__,
        "torch_config": torch.__config__.show(),
        "rng_hashes": hashes,
        "tt_evidence": str(args.tt_evidence),
        "tt_evidence_sha256": hashlib.sha256(args.tt_evidence.read_bytes()).hexdigest(),
        "cases": [],
    }
    tensor_cases = []
    for policy in ("literal_bf16", "fp32_down", "fp64_down", "fp32_tree_down"):

        def projection(value):
            if policy == "literal_bf16":
                return original(value)
            if policy == "fp32_tree_down":
                return tree_projection(value, module.weight).to(value.dtype)
            dtype = torch.float64 if policy == "fp64_down" else torch.float32
            return F.linear(value.to(dtype), module.weight.to(dtype)).to(value.dtype)

        with mock.patch.object(module, "forward", projection):
            for request, data in requests.items():
                cache = DynamicCache(config=config)
                for phase, expected in data["phases"].items():
                    value = expected["input"]
                    actual = reference(value, position_embeddings=(None, None), past_key_values=cache)
                    if policy == "literal_bf16":
                        assert torch.equal(actual, expected["output"]), "Literal HF rerun did not reproduce fixture"
                    case = {
                        "request": request,
                        "phase": phase,
                        "policy": policy,
                        "whole": metrics(expected["output"], actual),
                        "contribution": metrics(
                            expected["output"].float() - value.float(), actual.float() - value.float()
                        ),
                        "attention_mixed": metrics(expected["attn_mixed"], captured["mixed"]),
                        "router_indices": captured["indices"].tolist(),
                        "literal_router_indices": expected["router_indices"].tolist(),
                    }
                    reference_tt = tt["requests"].get(str(request), {}).get(phase, {}).get("baseline")
                    if reference_tt:
                        device_indices = reference_tt["router"]["runtime_fp32"]["cpu_topk_on_same_logits"]
                        case["tt_router_indices"] = device_indices
                        case["router_sets_match_tt"] = torch.equal(
                            captured["indices"].sort(-1).values, torch.tensor(device_indices).sort(-1).values
                        )
                    report["cases"].append(case)
                    tensor_cases.append(
                        {
                            "request": request,
                            "phase": phase,
                            "policy": policy,
                            "input": value.clone(),
                            "output": actual.clone(),
                            "attention_mixed": captured["mixed"].clone(),
                            "router_indices": captured["indices"].clone(),
                        }
                    )
    for hook in hooks:
        hook.remove()
    assert "ttnn" not in sys.modules
    if args.tensor_output is not None:
        torch.save({"cases": tensor_cases, "scope": __doc__}, args.tensor_output)
        report["tensor_evidence"] = {
            "path": str(args.tensor_output),
            "bytes": args.tensor_output.stat().st_size,
            "sha256": hashlib.sha256(args.tensor_output.read_bytes()).hexdigest(),
        }
    report["probe_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
