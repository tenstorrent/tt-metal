# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify restored retained implementations against frozen full-output hashes."""

import argparse
import json
import os
from pathlib import Path

import torch
import ttnn

from qualify_full import QUAL, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert "TT_SDPA_ACCURACY_DIAG" not in os.environ
    torch.set_num_threads(16)
    records = [
        json.loads(x)
        for x in (ROOT / "experiments/sdpa-l2/qualification-v1/accepted-results.jsonl").read_text().splitlines()
    ]
    selected = [
        r
        for r in records
        if r["group"] == "normal" and r["heads"] == 10 and r["seed"] == 1234 and r["kv_len"] in (32768, 262144)
    ]
    assert len(selected) == 4
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        with args.output.open("x") as log:
            for row in selected:
                q, k, v = QUAL.inputs(row)
                pos = QUAL.positions(row["q_len"], row["seed"])
                actual, digest = QUAL.run_device(device, q, k, v, pos, row["mode"], trace=True)
                assert digest == row["full_output_sha256"], (row["id"], row["mode"])
                result = dict(
                    id=row["id"],
                    mode=row["mode"],
                    status="PASS",
                    full_output_sha256=digest,
                    full_trace_equality=True,
                    full_output_equality_to_retained=True,
                )
                log.write(json.dumps(result) + "\n")
                log.flush()
                print(json.dumps(result), flush=True)
                del q, k, v, actual
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
