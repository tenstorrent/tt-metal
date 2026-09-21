# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded multi-Q, distinct-KV exact-equivalence suite; no timing claims."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import ttnn

import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("E", "G"), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default="normal,common_q,common_k,common_v,constant_v,uniform,outliers")
    parser.add_argument("--q-length", type=int, default=2048)
    parser.add_argument("--k-chunks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    cases = args.cases.split(",")
    allowed = {"normal", "common_q", "common_k", "common_v", "constant_v", "uniform", "outliers"}
    assert set(cases) <= allowed and len(set(cases)) == len(cases)
    paths = [args.output_dir / f"{args.variant.lower()}-{case}.json" for case in cases]
    assert not any(path.exists() for path in paths), "Refusing to overwrite qualification records"
    source = Path(__file__).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    torch.set_num_threads(8)
    device = ttnn.open_device(device_id=args.device, trace_region_size=8388608)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for case, path in zip(cases, paths):
            run_args = SimpleNamespace(
                variant=args.variant, mode="distinct", q_length=args.q_length,
                q_repeats=1, k_chunks=args.k_chunks, distribution=case,
                max_changing=case in ("normal", "outliers"), candidate=args.candidate,
                seed=args.seed, device=args.device, warmup=0, iters=0, reverse=False, output=path,
            )
            result = benchmark.run(device, run_args)
            assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
            result["qualification_runner_sha256"] = {str(source.relative_to(benchmark.ROOT)): source_hash}
            path.write_text(json.dumps(result, indent=2, default=str) + "\n")
            print(json.dumps({"case": case, "variant": args.variant,
                "exact": result["baseline_candidate_equal"], "canonical": result["canonical_adapter_equal"],
                "output_sha256": result["output_sha256"], "accuracy": result["accuracy"]}), flush=True)
            gc.collect()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
