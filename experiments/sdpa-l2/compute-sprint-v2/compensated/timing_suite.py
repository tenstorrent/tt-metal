# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded distinct-input timing controls for an input-dependent specialization."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import benchmark as bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("E", "G"), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--cases", default="normal32k,normal256k,growing,transitions")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument("--iters", type=int, default=11)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    configurations = {
        "normal32k": (256, 64, False, False),
        "normal256k": (256, 512, False, False),
        "growing": (2048, 16, True, False),
        "transitions": (2048, 8, False, True),
    }
    cases = args.cases.split(",")
    assert set(cases) <= configurations.keys() and len(set(cases)) == len(cases)
    paths = [args.output_dir / f"{args.variant.lower()}-{case}.json" for case in cases]
    assert not any(p.exists() for p in paths), "Refusing to overwrite evidence"
    source = Path(__file__).resolve()
    original_manifest = bench.source_hashes

    def source_hashes(candidate=None):
        pins = original_manifest(candidate)
        pins[str(source.relative_to(bench.ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
        return pins

    bench.source_hashes = source_hashes
    bench.torch.set_num_threads(8)
    device = bench.ttnn.open_device(device_id=args.device, trace_region_size=8388608)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, (case, path) in enumerate(zip(cases, paths)):
            q_length, k_chunks, max_changing, paired_maxima = configurations[case]
            run_args = SimpleNamespace(
                variant=args.variant, mode="distinct", q_length=q_length, k_chunks=k_chunks,
                q_repeats=1, distribution="normal", max_changing=max_changing,
                paired_maxima=paired_maxima, candidate=args.candidate, seed=args.seed,
                device=args.device, warmup=args.warmup, iters=args.iters,
                reverse=bool((index + int(args.reverse)) % 2), output=path,
            )
            result = bench.run(device, run_args)
            result["case"] = case
            result["measurement_scope"] = "Single-core distinct-KV attention, recurring input DM; preprocessing excluded"
            result["guard_frequency"] = "Not instrumented. Timing is conditional on input/shape, not universal speedup."
            path.write_text(json.dumps(result, indent=2, default=str) + "\n")
            print(json.dumps({"case": case, "variant": args.variant,
                              "baseline": result["baseline"], "candidate": result["candidate"],
                              "exact": result["baseline_candidate_equal"]}), flush=True)
            gc.collect()
    finally:
        bench.ttnn.close_device(device)


if __name__ == "__main__":
    main()
