# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check the first warm B1 decode's cache rows against its device BF16 updates."""

import argparse
import json
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import main as run_decoder


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache-report", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    original = ttnn.experimental.paged_update_cache
    rows = []

    def checked(cache, update, **kwargs):
        result = original(cache, update, **kwargs)
        if len(rows) < 2:
            assert update.dtype == ttnn.bfloat16
            position = int(ttnn.to_torch(kwargs["update_idxs_tensor"]).flatten()[0])
            table = ttnn.to_torch(kwargs["page_table"])
            block = cache.shape[-2]
            page = int(table[0, position // block])
            expected = ttnn.to_torch(update)[0, 0]
            actual = ttnn.to_torch(cache)[page, :, position % block]
            equal = torch.equal(expected, actual)
            rows.append(dict(cache="k" if not rows else "v", position=position, bitwise_equal=equal))
            assert equal, rows
        return result

    ttnn.experimental.paged_update_cache = checked
    try:
        run_decoder()
    finally:
        ttnn.experimental.paged_update_cache = original
        args.cache_report.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
