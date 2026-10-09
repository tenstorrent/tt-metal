# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Opt-in trained regression, using the package's existing device fixture."""

import json
import os

import pytest
from models.experimental.nllb.tests.process_runner import run_task


def test_trained_generation_envelope(nllb_device_id, tmp_path):
    checkpoint = os.environ.get("NLLB_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set NLLB_TEST_CHECKPOINT for trained-model envelope regression")
    report = tmp_path / "generation-envelope.json"
    timeout = int(os.environ.get("NLLB_TEST_TIMEOUT", "300"))
    assert 30 <= timeout <= 1200, "NLLB_TEST_TIMEOUT must be between 30 and 1200 seconds"
    options = dict(
        checkpoint=checkpoint,
        device=nllb_device_id,
        output=str(report),
        precision=os.environ.get("NLLB_TEST_PRECISION", "bf16"),
        timeout=timeout - 20,
    )
    for variable, option in [
        ("NLLB_TEST_CONFIG", "config"),
        ("NLLB_TEST_TOKENIZER", "tokenizer_directory"),
        ("NLLB_TEST_FP32_ENVELOPE", "oracle"),
        ("NLLB_TEST_FP32_ENVELOPE_SHA256", "oracle_sha256"),
    ]:
        if os.environ.get(variable):
            options[option] = os.environ[variable]
    child = run_task("envelope", options, timeout=timeout)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(report.read_text())
    assert result["passed"] and result["same_tt_passed"] and all(result["checks"].values())
    assert result["tt_calls"] > 0
    print(child.stdout)
