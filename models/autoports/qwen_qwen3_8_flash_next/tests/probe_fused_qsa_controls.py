# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Export existing QSA probe host metric operands without changing gates or fixtures."""
import argparse
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests import probe_qsa_decoder as probe


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--device-output-cache", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [probe.__file__, *remaining]
    original_case = probe.device_case
    original_metrics = probe.metrics
    rows = []
    context = {}

    def device_case(*values, **options):
        context["length"] = options["length"]
        context["metric_index"] = 0
        return original_case(*values, **options)

    def metrics(expected, actual, residual):
        result = original_metrics(expected, actual, residual)
        rows.append(
            dict(
                length=context["length"],
                metric_index=context["metric_index"],
                expected=expected.clone(),
                actual=actual.clone(),
                residual=residual.clone(),
                metrics=result,
            )
        )
        context["metric_index"] += 1
        return result

    try:
        with patch.object(probe, "device_case", device_case), patch.object(probe, "metrics", metrics):
            probe.main()
    finally:
        args.device_output_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rows=rows,
                probe_sha256=hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest(),
                export_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope="Copies of existing host metric operands after device readback; unchanged original fixture, model calls and acceptance gate",
            ),
            args.device_output_cache,
        )


if __name__ == "__main__":
    main()
