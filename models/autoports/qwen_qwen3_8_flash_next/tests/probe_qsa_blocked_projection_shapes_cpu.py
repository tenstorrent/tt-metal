# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check one fixed blocked projection against literal HF for every logical Q1..32.

The candidate's arithmetic is row-independent. Compute it on the original
32-row input then slice its result; HF is called independently at every Q.
Four additional chunks start at named critical tokens so Q1 tests those tokens.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from .probe_qsa import load_reference
from .probe_qsa_fp64_indexer_control import digest, save_tensors, write_json
from .probe_qsa_projection_orders_cpu import COORDINATES, blocked_high_low


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--order-cache", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    cache = torch.load(args.order_cache, weights_only=True, mmap=True, map_location="cpu")
    _, model, _, _, provenance = load_reference(args.snapshot)
    assert source["key"] == cache["identity"]["source_key"]
    assert source["key"]["reference_sha256"] == provenance["reference_sha256"]
    assert source["key"]["config_sha256"] == provenance["config_sha256"]
    assert digest(args.source_cache) == cache["identity"]["source_sha256"]
    starts = list(cache["chunks"]) + [position for position, _ in COORDINATES]
    assert len(set(starts)) == len(starts)
    report = {
        "scope": __doc__,
        "accepted": False,
        "device_executed": False,
        "status": "running",
        "identity": {
            "source_key": source["key"],
            "source_sha256": cache["identity"]["source_sha256"],
            "order_cache_sha256": digest(args.order_cache),
            "reference": provenance,
            "harness_sha256": digest(Path(__file__)),
            "policy_source_sha256": digest(Path(__file__).with_name("probe_qsa_projection_orders_cpu.py")),
        },
        "verbose_environment": {name: os.environ.get(name) for name in ["ONEDNN_VERBOSE", "DNNL_VERBOSE"]},
        "input_starts": starts,
        "checks": [],
        "by_Q": {},
        "critical_Q1": [],
    }
    saved = {"identity": report["identity"], "chunks": {}}
    weight = model.indexer.index_qk_proj.weight
    for start in starts:
        hidden = source["hidden"][:, start : start + 32]
        if start in cache["chunks"]:
            candidate_fp32 = cache["chunks"][start]["outputs"]["block1024_high_low_then_partial_add"]
        else:
            products = (hidden[0].T.float()[:, :, None] * weight.T.float()[:, None, :]).contiguous()
            candidate_fp32 = blocked_high_low(products)
            del products
        candidate = candidate_fp32.bfloat16().unsqueeze(0)
        outputs = {}
        for q in range(1, 33):
            print(json.dumps({"projection_shape_begin": [1, q, 2560], "start": start}), flush=True)
            literal = model.indexer.index_qk_proj(hidden[:, :q])
            repeat = model.indexer.index_qk_proj(hidden[:, :q])
            assert torch.equal(literal, repeat)
            difference = literal != candidate[:, :q]
            count = int(difference.sum())
            report["checks"].append(
                {"start": start, "Q": q, "elements": literal.numel(), "mismatches": count, "literal_repeat_exact": True}
            )
            summary = report["by_Q"].setdefault(str(q), {"elements": 0, "mismatches": 0, "failed_input_starts": []})
            summary["elements"] += literal.numel()
            summary["mismatches"] += count
            if count:
                summary["failed_input_starts"].append(start)
            if q == 1:
                for position, column in COORDINATES:
                    if position == start:
                        report["critical_Q1"].append(
                            {
                                "position": position,
                                "column": column,
                                "literal": float(literal[0, 0, column]),
                                "candidate": float(candidate[0, 0, column]),
                                "candidate_fp32": float(candidate_fp32[0, column]),
                            }
                        )
            outputs[q] = literal
        saved["chunks"][start] = {"candidate_fp32": candidate_fp32, "literal_by_Q": outputs}
    report["all_shapes_bitexact"] = all(row["mismatches"] == 0 for row in report["checks"])
    report["status"] = "completed_diagnostic"
    saved["report"] = report
    save_tensors(args.tensor_output, saved)
    report["tensor_output"] = str(args.tensor_output)
    report["tensor_output_sha256"] = digest(args.tensor_output)
    assert "ttnn" not in sys.modules
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "all_shapes_bitexact": report["all_shapes_bitexact"],
                "by_Q": report["by_Q"],
                "critical_Q1": report["critical_Q1"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
