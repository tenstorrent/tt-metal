# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual host no-fallback guard, bypassing qualification prechecks."""

import argparse
import json
import os
from pathlib import Path

import torch
import ttnn

from qualify_full import module, verify_diagnostic_source

ANALYZE = module("stress", "experiments/sdpa-l2/stress-analysis/analyze.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["TT_SDPA_ACCURACY_DIAG"] = "3"
    verify_diagnostic_source()
    torch.set_num_threads(8)
    device = ttnn.open_device(device_id=0)
    try:
        with args.output.open("x") as log:
            for length in (2048, 25920, 32769):
                inputs = [torch.zeros((1, 1, length, 128), dtype=torch.bfloat16) for _ in range(3)]
                try:
                    ANALYZE.device_original_q(device, *inputs, torch.tensor([0]))
                except RuntimeError as error:
                    assert "QUALIFICATION_UNSUPPORTED: accuracy diagnostic forbids fallback" in str(error)
                    row = dict(length=length, status="PASS", host_rejected_unsupported=True, fallback_executed=False)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                else:
                    raise AssertionError(f"Unsupported length {length} unexpectedly executed")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
