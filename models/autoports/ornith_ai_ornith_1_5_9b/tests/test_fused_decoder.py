# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline synthetic regression using checkpoint-derived stats and real config."""
import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("layer", [0, 3])
def test_fused_real_shape_decoder(layer, tmp_path):
    report = tmp_path / f"layer_{layer}.json"
    subprocess.run(
        [
            "timeout",
            "600",
            sys.executable,
            "-m",
            "models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_fused",
            "--layer",
            str(layer),
            "--synthetic",
            "--lengths",
            "1,32,33",
            "--batch",
            "2",
            "--output",
            str(report),
        ],
        check=True,
    )
    result = json.loads(report.read_text())
    assert result["synthetic_weights"]
    assert result["implementation"] == "FusedDecoder"
    assert all(r["pcc"] >= 0.995 for r in result["records"] if "pcc" in r)
