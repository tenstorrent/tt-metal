# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Markdown table of bench_serve.sh results (vllm bench serve --save-result JSON files).

usage: python -m qwen36_resident.serve.summarize_bench RESULT.json [RESULT.json ...]
"""
import json
import sys


def main():
    print(
        "| run | ISL | OSL | requests | TTFT mean (s) | TTFT p50 (s) | TPOT mean (ms) | TPOT p50 (ms) | decode tok/s |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for path in sys.argv[1:]:
        r = json.load(open(path))
        n = r["completed"]
        isl = r["total_input_tokens"] / n
        osl = r["total_output_tokens"] / n
        tpot = r["mean_tpot_ms"]
        name = path.rsplit("/", 1)[-1].removesuffix(".json")
        print(
            f"| {name} | {isl:.0f} | {osl:.0f} | {n} | {r['mean_ttft_ms'] / 1e3:.3f} | {r['median_ttft_ms'] / 1e3:.3f} "
            f"| {tpot:.2f} | {r['median_tpot_ms']:.2f} | {1e3 / tpot:.1f} |"
        )


if __name__ == "__main__":
    main()
