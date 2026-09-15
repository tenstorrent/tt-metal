# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only audit: does ideal BF16 rounding itself fail the agreed gates?

No acceptance gates are weakened. This distinguishes precision-limited gate
failures from additional operator error in common-V cases.
"""

import argparse
import json
from pathlib import Path

import torch

from qualify import assess, inputs, make_manifest, positions, repro, tensor_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()
torch.set_num_threads(args.threads)
done = set()
if args.output.exists():
    done = {(r["id"], r["mode"]) for r in map(json.loads, args.output.read_text().splitlines())}
with args.output.open("a") as log:
    for c in make_manifest():
        if c["distribution"] != "common_v":
            continue
        if all((c["id"], mode) in done for mode in ("fast", "accurate")):
            continue
        q, k, v = inputs(c)
        pos = positions(c["q_len"], c["seed"])
        gold = repro.reference(q[..., pos, :], k, v, block=1024)
        ideal = gold.bfloat16()
        for mode in ("fast", "accurate"):
            if (c["id"], mode) in done:
                continue
            per_head, failed = assess(ideal, gold, c, mode, v)
            row = dict(
                c,
                mode=mode,
                per_head=per_head,
                failed_gates=failed,
                status="FAIL" if failed else "PASS",
                oracle="FP64_reference_rounded_to_BF16",
                input_sha256=[tensor_hash(x) for x in (q, k, v)],
                ideal_output_sha256=tensor_hash(ideal),
            )
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
        print("ORACLE_DONE " + c["id"], flush=True)
        del q, k, v, gold, ideal
