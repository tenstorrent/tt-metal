# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise valid-row broadcast fusion under unchanged bit-exact ordered tests."""
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tests import probe_ordered_projection
from models.autoports.qwen_qwen3_8_flash_next.tt import ordered_projection
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_ordered_indexer_projection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-row-compute", action="store_true")
    parser.add_argument("--copy-once", action="store_true")
    args, _ = parser.parse_known_args()
    if args.valid_row_compute:
        sys.argv.remove("--valid-row-compute")
    if args.copy_once:
        sys.argv.remove("--copy-once")
    calls = []

    def selected(value, weight, **kwargs):
        calls.append(value.shape[-2])
        return fused_ordered_indexer_projection(
            value, weight, valid_row_compute=args.valid_row_compute, copy_once=args.copy_once, **kwargs
        )

    with patch.object(ordered_projection, "ordered_indexer_projection", selected):
        probe_ordered_projection.main()
    assert calls, "The fused projection was never exercised"
    report = json.loads(args.output.read_text())
    report["fused_projection"] = {
        "source_sha256": hashlib.sha256(
            Path(inspect.getfile(fused_ordered_indexer_projection)).read_bytes()
        ).hexdigest(),
        "logical_call_counts": calls,
        "valid_row_compute": args.valid_row_compute,
        "copy_once": args.copy_once,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
