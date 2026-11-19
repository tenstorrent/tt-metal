# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed K-block controls for literal Q1 projection; no oracle or runtime changes."""

import argparse
import json
import sys
from pathlib import Path

import torch

from .probe_qsa import load_reference
from .probe_qsa_fp64_indexer_control import digest, write_json
from .probe_qsa_projection_orders_cpu import blocked_high_low


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--shape-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    shapes = torch.load(args.shape_cache, weights_only=True, mmap=True, map_location="cpu")
    _, model, _, _, provenance = load_reference(args.snapshot)
    assert source["key"] == shapes["identity"]["source_key"]
    assert source["key"]["reference_sha256"] == provenance["reference_sha256"]
    assert source["key"]["config_sha256"] == provenance["config_sha256"]
    assert digest(args.source_cache) == shapes["identity"]["source_sha256"]
    blocks = [32, 64, 128, 256, 320, 512, 640, 1024, 1280, 2560]
    report = {
        "scope": __doc__,
        "accepted": False,
        "device_executed": False,
        "identity": {
            "source_key": source["key"],
            "source_sha256": shapes["identity"]["source_sha256"],
            "shape_cache_sha256": digest(args.shape_cache),
            "reference": provenance,
            "harness_sha256": digest(Path(__file__)),
            "policy_source_sha256": digest(Path(__file__).with_name("probe_qsa_projection_orders_cpu.py")),
        },
        "blocks": blocks,
        "checks": [],
        "aggregate": {str(block): {"mismatches": 0, "elements": 0, "failed_starts": []} for block in blocks},
    }
    for start, saved in shapes["chunks"].items():
        hidden = source["hidden"][:, start : start + 1]
        literal = model.indexer.index_qk_proj(hidden)
        assert torch.equal(literal, saved["literal_by_Q"][1])
        products = (
            hidden[0].T.float()[:, :, None] * model.indexer.index_qk_proj.weight.T.float()[:, None, :]
        ).contiguous()
        item = {"start": start, "policies": {}}
        for block in blocks:
            output = blocked_high_low(products, block).bfloat16().unsqueeze(0)
            difference = output != literal
            count = int(difference.sum())
            item["policies"][str(block)] = {
                "mismatches": count,
                "coordinates": [
                    {
                        "column": column,
                        "literal": float(literal[0, 0, column]),
                        "candidate": float(output[0, 0, column]),
                    }
                    for _, _, column in difference.nonzero().tolist()
                ],
            }
            summary = report["aggregate"][str(block)]
            summary["mismatches"] += count
            summary["elements"] += literal.numel()
            if count:
                summary["failed_starts"].append(start)
        report["checks"].append(item)
    assert "ttnn" not in sys.modules
    report["status"] = "completed_diagnostic"
    write_json(args.output, report)
    print(json.dumps(report["aggregate"]), flush=True)


if __name__ == "__main__":
    main()
