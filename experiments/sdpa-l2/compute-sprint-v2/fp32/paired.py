# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ABBA rounds or independent-KV stress, using the same owned benchmark builder."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import ttnn

import bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", choices=["C", "D"], required=True)
    parser.add_argument("--candidate", choices=bench.CANDIDATES, required=True)
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--seed", type=int, default=1236)
    parser.add_argument("--q-repeats", type=int, default=8)
    parser.add_argument("--k-chunks", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    opts = parser.parse_args()
    assert opts.variant == "C" or not opts.candidate.startswith("c_refine")
    target = bench.HERE / (opts.label + ".json")
    assert not target.exists(), "Use a new label"
    records = []
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        if opts.qualify:
            cases = [("normal", 512)] + [
                (name, 64)
                for name in ("scaled_qk", "outliers", "common_q", "common_k", "common_v", "constant_v", "uniform")
            ]
        else:
            cases = [("normal", opts.k_chunks)]
        for distribution, chunks in cases:
            oracle = None
            rounds = 1 if opts.qualify else opts.rounds
            for round_index in range(rounds):
                order = ["baseline", opts.candidate]
                if round_index % 2:
                    order.reverse()
                for candidate in order:
                    label = f"{opts.label}-{distribution}-r{round_index}-{candidate}"
                    args = argparse.Namespace(
                        label=label,
                        variant=opts.variant,
                        candidate=candidate,
                        mode="accurate" if opts.variant == "D" else "qk4_pv2",
                        distribution=distribution,
                        seed=opts.seed,
                        distinct_kv=opts.qualify,
                        q_repeats=1 if opts.qualify else opts.q_repeats,
                        k_chunks=chunks,
                        warmup=0 if opts.qualify else opts.warmup,
                        iters=1 if opts.qualify else opts.iters,
                        hybrid_block_pack=False,
                    )
                    record, actual = bench.run(device, args)
                    if oracle is None:
                        assert candidate == "baseline"
                        oracle = actual.clone()
                    record["baseline_bitwise_equal"] = bool(
                        torch.equal(actual.view(torch.uint16), oracle.view(torch.uint16))
                    )
                    record["baseline_mismatched_values"] = int(
                        (actual.view(torch.uint16) != oracle.view(torch.uint16)).sum().item()
                    )
                    record["baseline_max_abs_diff"] = float((actual.float() - oracle.float()).abs().max())
                    record["paired_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                    records.append(record)
                    torch.save(actual, bench.HERE / (label + ".pt"))
                    target.write_text(json.dumps(records, indent=2) + "\n")
                    print("RESULT " + json.dumps(record), flush=True)
                    if not record["baseline_bitwise_equal"]:
                        raise AssertionError("Scheduling candidate changed output: " + label)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
