# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only matched HF chunking control for real layer-0 router discontinuities.

Runs the unmodified installed decoder with fresh caches and identical inputs.
No acceptance threshold or runtime implementation is changed by this diagnostic.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf_source

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights


def row_pcc(left, right):
    left = left.float().reshape(-1, left.shape[-1])
    right = right.float().reshape(-1, right.shape[-1])
    left = left - left.mean(-1, keepdim=True)
    right = right - right.mean(-1, keepdim=True)
    return (left * right).sum(-1) / (left.square().sum(-1) * right.square().sum(-1)).sqrt()


def compare(expected, actual, residual):
    whole = row_pcc(expected, actual)
    contribution = row_pcc(expected.float() - residual.float(), actual.float() - residual.float())
    return {
        "whole_output_min_row_pcc": whole.min().item(),
        "contribution_min_row_pcc": contribution.min().item(),
        "whole_output_per_row_pcc": whole.tolist(),
        "contribution_per_row_pcc": contribution.tolist(),
        "whole_output_below_0_995_rows": torch.where(whole < 0.995)[0].tolist(),
        "contribution_below_0_995_rows": torch.where(contribution < 0.995)[0].tolist(),
        "maximum_absolute_error": (expected.float() - actual.float()).abs().max().item(),
        "bitwise_equal": torch.equal(expected, actual),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(932)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    weights = load_weights(args.snapshot, 0, "")
    with torch.device("meta"):
        reference = hf_source.Qwen4ExpTextDecoderLayer(config, 0)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    hidden = (torch.randn(1, 65, 10240) * 0.02).bfloat16()
    captured = []

    def capture(_module, inputs, output):
        captured.append(
            {
                "input": inputs[0].clone(),
                **{name: value.clone() for name, value in zip(("logits", "scores", "indices"), output)},
            }
        )

    hook = reference.mlp.gate.register_forward_hook(capture)
    runs = {}
    elapsed = {}
    for name, chunk in (("full65", 65), ("chunks32", 32), ("tokens1", 1), ("full65_repeat", 65)):
        captured.clear()
        cache = DynamicCache(config=config)
        start = time.monotonic()
        outputs = []
        for begin in range(0, 65, chunk):
            outputs.append(
                reference(
                    hidden[:, begin : begin + chunk].contiguous(),
                    position_embeddings=(None, None),
                    past_key_values=cache,
                )
            )
        runs[name] = {key: torch.cat([call[key] for call in captured], 0) for key in captured[0]}
        runs[name]["output"] = torch.cat(outputs, 1)
        elapsed[name] = time.monotonic() - start
        print(json.dumps({"completed": name, "seconds": elapsed[name]}), flush=True)
    hook.remove()
    baseline = runs["full65"]
    comparisons = {}
    for name, run in runs.items():
        rows = torch.where((baseline["indices"].sort(-1).values != run["indices"].sort(-1).values).any(-1))[0].tolist()
        details = []
        for row in sorted(set(rows + [1])):
            detail = {"row": row}
            for label, values in (("full65", baseline), (name, run)):
                top = values["logits"][row].float().topk(12)
                detail[label] = {
                    "indices": values["indices"][row].tolist(),
                    "scores": values["scores"][row].float().tolist(),
                    "top12_logit_values": top.values.tolist(),
                    "top12_logit_indices": top.indices.tolist(),
                    "top10_boundary_tied": bool(top.values[9] == top.values[10]),
                }
            details.append(detail)
        comparisons[name] = {
            **compare(baseline["output"], run["output"], hidden),
            "router_expert_set_changed_rows": rows,
            "router_logits_max_abs_error": (baseline["logits"].float() - run["logits"].float()).abs().max().item(),
            "router_input_min_row_pcc": row_pcc(baseline["input"], run["input"]).min().item(),
            "router_details": details,
        }
    capture_path = args.output.with_suffix(".pt")
    torch.save({"hidden": hidden, "runs": runs}, capture_path)
    assert "ttnn" not in sys.modules, "CPU-only probe unexpectedly imported TTNN"
    report = {
        "scope": "Unmodified HF layer0 compared against itself under equivalent chunkings; diagnostic only",
        "cpu_only": True,
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "seed": 932,
        "input_shape": list(hidden.shape),
        "input_scale": 0.02,
        "input_dtype": str(hidden.dtype),
        "snapshot": str(args.snapshot),
        "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
        "hf_source": hf_source.__file__,
        "hf_source_sha256": hashlib.sha256(Path(hf_source.__file__).read_bytes()).hexdigest(),
        "capture_artifact": str(capture_path),
        "capture_contents": "Exact inputs, full router logits/scores/indices and outputs for every run",
        "elapsed_seconds": elapsed,
        "comparisons_to_full65": comparisons,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {key: value for key, value in result.items() if "per_row" not in key and key != "router_details"}
                for name, result in comparisons.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
