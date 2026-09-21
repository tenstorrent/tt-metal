# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded E/G group2 qualification; numerical rejection is data, invariant failures are not."""
import argparse
import hashlib
import json
from pathlib import Path

import benchmark as bench
import torch
import ttnn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("E", "G"), required=True)
    p.add_argument("--candidate", default="group2_direct")
    p.add_argument("--cases", choices=("short", "stress", "long", "perf"), required=True)
    p.add_argument("--output", type=Path, required=True)
    opt = p.parse_args()
    assert not opt.output.exists()
    source = Path(__file__)
    own_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    torch.set_num_threads(8)
    if opt.cases == "short":
        cases = [("first", 1, 512, "normal", False, False),
                 ("odd_final", 2, 1024, "normal", False, False),
                 ("retained", 3, 1024, "uniform", False, False),
                 ("changing", 3, 1024, "normal", True, False),
                 ("transitions", 8, 1024, "normal", False, True)]
    elif opt.cases == "stress":
        cases = [(d, 16, 1024, d, False, False) for d in
                 ("normal", "outliers", "common_q", "common_k", "common_v", "constant_v",
                  "uniform", "uniform_constant_v", "zero_v")]
    elif opt.cases == "long":
        cases = [(f"{d}-{k}", k, 256, d, False, False) for k in (64, 512)
                 for d in ("normal", "common_v", "constant_v", "uniform", "uniform_constant_v")]
    else:
        cases = [("resident-v1", 512, 256, "normal", False, False),
                 ("resident-v2", 512, 256, "normal", False, False)]
    records = []
    device = ttnn.open_device(device_id=0, trace_region_size=8388608)
    try:
        for label, chunks, qlen, distribution, changing, paired in cases:
            baseline = "v1" if label == "resident-v1" else "v2"
            bench.BASELINE = (bench.V1 / "combined_fence/compute_streaming.hpp" if baseline == "v1"
                              else bench.V2 / "identity_early/compute_streaming.hpp")
            args = argparse.Namespace(variant=opt.variant, baseline=baseline, time_distinct=False,
                mode="resident" if opt.cases == "perf" else "distinct", k_chunks=chunks,
                q_repeats=16 if opt.cases == "perf" else 1, q_length=qlen,
                distribution=distribution, max_changing=changing, paired_maxima=paired,
                candidate=str((bench.HERE / opt.candidate / "compute_streaming.hpp").relative_to(bench.ROOT)),
                seed=20260923, device=0, warmup=10 if opt.cases == "perf" else 0,
                iters=9 if opt.cases == "perf" else 0, reverse=False, output=opt.output)
            result = bench.run(device, args)
            result["case"] = label
            result["suite_source_sha256"] = own_hash
            assert hashlib.sha256(source.read_bytes()).hexdigest() == own_hash
            records.append(result)
            opt.output.write_text(json.dumps(records, indent=2, default=str) + "\n")
            print(json.dumps(dict(case=label, comparison=result["numerical_comparison"],
                                  baseline=result["baseline"], candidate=result["candidate"])), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
