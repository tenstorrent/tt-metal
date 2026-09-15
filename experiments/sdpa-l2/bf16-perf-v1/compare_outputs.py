#!/usr/bin/env python3
"""Compare matched sampled-output hashes from two BF16 validation sweeps."""

import argparse
import json
from pathlib import Path


def read_cases(directory, label):
    cases = {}
    for path in directory.glob(f"{label}-accuracy-*.jsonl"):
        case = int(path.stem.rsplit("-", 1)[1])
        for index, line in enumerate(path.read_text().splitlines()):
            cases[case, index] = json.loads(line)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    before = read_cases(args.directory, args.before)
    after = read_cases(args.directory, args.after)
    assert before and before.keys() == after.keys(), "Missing or unmatched validation cases"
    geometry = (
        "kv_len",
        "q_len",
        "source_q_len",
        "heads",
        "dim",
        "seed",
        "distribution",
        "common_mode",
        "causal",
        "q_chunk",
        "k_chunk",
        "variant",
        "query_sampling",
        "q_round_bits",
        "q_bitceil",
        "q_prescale",
    )
    for case in sorted(before):
        old, new = before[case], after[case]
        assert all(old.get(key) == new.get(key) for key in geometry), f"Input mismatch: {case}"
        assert old["sampled_output_sha256"] == new["sampled_output_sha256"], f"Output mismatch: {case}"
        assert old["l2_pct"] == new["l2_pct"], f"L2 mismatch: {case}"
        print(
            json.dumps(
                {
                    "case": case,
                    "distribution": new["distribution"],
                    "common_mode": new.get("common_mode"),
                    "kv_len": new["kv_len"],
                    "variant": new["variant"],
                    "seed": new["seed"],
                    "l2_pct": new["l2_pct"],
                    "sampled_output_sha256": new["sampled_output_sha256"],
                    "exact_sampled_output_match": True,
                }
            )
        )
    print(json.dumps({"matched_cases": len(before), "all_sampled_outputs_identical": True}))


if __name__ == "__main__":
    main()
