# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Opt-in trained regression, using the package's existing device fixture."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_trained_generation_envelope(nllb_device_id, tmp_path):
    checkpoint = os.environ.get("NLLB_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set NLLB_TEST_CHECKPOINT for trained-model envelope regression")
    report = tmp_path / "generation-envelope.json"
    timeout = int(os.environ.get("NLLB_TEST_TIMEOUT", "300"))
    assert 30 <= timeout <= 1200, "NLLB_TEST_TIMEOUT must be between 30 and 1200 seconds"
    command = [
        sys.executable,
        str(Path(__file__).with_name("envelope_regression.py")),
        "--checkpoint",
        checkpoint,
        "--device",
        str(nllb_device_id),
        "--output",
        str(report),
        "--precision",
        os.environ.get("NLLB_TEST_PRECISION", "bf16"),
        "--timeout",
        str(timeout - 20),
    ]
    for variable, option in [
        ("NLLB_TEST_CONFIG", "--config"),
        ("NLLB_TEST_TOKENIZER", "--tokenizer-directory"),
        ("NLLB_TEST_FP32_ENVELOPE", "--oracle"),
        ("NLLB_TEST_FP32_ENVELOPE_SHA256", "--oracle-sha256"),
    ]:
        if os.environ.get(variable):
            command.extend([option, os.environ[variable]])
    child = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=timeout)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(report.read_text())
    assert result["passed"] and result["same_tt_passed"] and all(result["checks"].values())
    assert result["tt_calls"] > 0
    print(child.stdout)
