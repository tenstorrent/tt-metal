# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Distinct-KV paired timing; recurring DM, explicitly NOT resident throughput."""

import argparse
import hashlib
import json
from pathlib import Path

import benchmark as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("E", "G"), required=True)
    parser.add_argument("--k-chunks", type=int, default=64)
    parser.add_argument("--q-length", type=int, default=256)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--max-changing", action="store_true")
    parser.add_argument("--paired-maxima", action="store_true")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--iters", type=int, default=11)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite evidence"
    args.mode, args.q_repeats = "distinct", 1
    original_manifest = bench.source_hashes
    wrapper = Path(__file__).resolve()

    def source_hashes(candidate=None):
        pins = original_manifest(candidate)
        pins[str(wrapper.relative_to(bench.ROOT))] = hashlib.sha256(wrapper.read_bytes()).hexdigest()
        return pins

    bench.source_hashes = source_hashes
    bench.torch.set_num_threads(8)
    device = bench.ttnn.open_device(device_id=args.device, trace_region_size=8388608)
    try:
        result = bench.run(device, args)
        result["measurement_scope"] = "Single-core distinct-KV attention with recurring DM; preprocessing excluded"
        result["guard_frequency"] = "Not instrumented; no assumption of all-identity corrections"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps(result, default=str), flush=True)
    finally:
        bench.ttnn.close_device(device)


if __name__ == "__main__":
    main()
