# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Paired distinct-Q/KV recurring-DM timing, separate from resident throughput."""
import argparse
import hashlib
import json
from pathlib import Path

import benchmark as bench
import torch
import ttnn

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--variant", choices=("E", "G"), required=True)
p.add_argument("--output", type=Path, required=True)
opt = p.parse_args()
assert not opt.output.exists()
source = Path(__file__)
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
torch.set_num_threads(8)
cases = [("normal32K", 64, 2048, False, False),
         ("normal256K", 512, 1024, False, False),
         ("changing32K", 64, 2048, True, False),
         ("transitions8K", 16, 2048, False, True)]
records = []
device = ttnn.open_device(device_id=0, trace_region_size=8388608)
try:
    for label, chunks, qlen, changing, paired in cases:
        args = argparse.Namespace(variant=opt.variant, baseline="v2", time_distinct=True,
            mode="distinct", k_chunks=chunks, q_repeats=1, q_length=qlen,
            distribution="normal", max_changing=changing, paired_maxima=paired,
            candidate=str((bench.HERE / "group2_valid/compute_streaming.hpp").relative_to(bench.ROOT)),
            seed=20260924, device=0, warmup=7, iters=15, reverse=True, output=opt.output)
        result = bench.run(device, args)
        result["case"] = label
        result["distinct_driver_sha256"] = source_hash
        result["distinct_driver_source"] = str(source.relative_to(bench.ROOT))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        records.append(result)
        opt.output.write_text(json.dumps(records, indent=2, default=str) + "\n")
        print(json.dumps(dict(case=label, comparison=result["numerical_comparison"],
                              baseline=result["baseline"], candidate=result["candidate"])), flush=True)
finally:
    ttnn.close_device(device)
