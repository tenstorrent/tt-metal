# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-pipeline end-to-end PCC correctness gate for the perf optimizer (``tt_hw_planner optimize``).

Single ``main`` pipeline (no per-task ``demo_<task>.py``): discovery pairs the whole model with ONE
numeric PCC gate. The optimizer requires an ``end_to_end`` PCC test with a numeric threshold even for
``--baseline-only`` (it reads the threshold from source; it only RUNS the gate during the
optimization loop). This file supplies that, as the union of the proven per-module PCC checks
(threshold 0.99, real-tensor ``check_with_pcc`` vs the HF reference), each run as its own isolated,
timeout-bounded subprocess so the gate can never hang.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

PCC_THRESHOLD = 0.99

_REPO_ROOT = Path(__file__).resolve().parents[5]
_MODEL_REL = "models/experimental/seamless_m4t_v2_large"

_MODULE_NODES = {
    "text_encoder": "tests/pcc/test_text_encoder.py::test_seamless_m4t_v2_text_encoder_max_seq_pcc",
    "speech_encoder": "tests/pcc/test_speech_encoder.py::test_seamless_m4t_v2_speech_encoder_max_seq_pcc",
    "text_decoder_s2tt": "tests/pcc/test_text_decoder.py::test_seamless_m4t_v2_text_decoder_s2tt_max_enc_seq_pcc",
    "code_hifigan_vocoder": "tests/pcc/test_code_hifigan.py::test_seamless_m4t_v2_code_hifigan_max_unit_seq_pcc",
}

_PCC_RE = re.compile(r"(?i)pcc\D{0,12}(-?\d\.\d+)")
_SKIPPED_RE = re.compile(r"\b[1-9]\d*\s+skipped\b", re.IGNORECASE)
_PASSED_RE = re.compile(r"\b[1-9]\d*\s+passed\b", re.IGNORECASE)


def _run_module_gate(label: str, timeout_s: int) -> float:
    node = f"{_MODEL_REL}/{_MODULE_NODES[label]}"
    logger.info(f"[e2e-pcc] module gate '{label}' (timeout {timeout_s}s): {node}")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-o", "addopts=", node, "-sv"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"[e2e-pcc] module '{label}' TIMED OUT after {timeout_s}s (gate FAIL)")
    out = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join(out.splitlines()[-20:])
    if _SKIPPED_RE.search(out) and not _PASSED_RE.search(out):
        pytest.fail(f"[e2e-pcc] module '{label}' SKIPPED (gate FAIL).\n{tail}")
    if r.returncode != 0 or not _PASSED_RE.search(out):
        pytest.fail(f"[e2e-pcc] module '{label}' did not pass (rc={r.returncode}).\n{tail}")
    m = _PCC_RE.findall(out)
    pcc = float(m[-1]) if m else 1.0
    logger.info(f"[e2e-pcc] '{label}' PASSED at PCC {pcc:.5f}")
    return pcc


def test_main_e2e_pcc():
    """Whole-pipeline PCC gate: every module must pass at PCC >= ``PCC_THRESHOLD``."""
    timeout_s = int(os.environ.get("SEAMLESS_E2E_PCC_TIMEOUT", "600"))
    worst = 1.0
    for label in _MODULE_NODES:
        pcc = _run_module_gate(label, timeout_s)
        worst = min(worst, pcc)
        assert pcc >= PCC_THRESHOLD, f"[e2e-pcc] module '{label}' PCC {pcc:.5f} < {PCC_THRESHOLD}"
    logger.info(f"[e2e-pcc] all modules passed (>= {PCC_THRESHOLD}). PCC: {worst:.5f}")
    assert worst >= PCC_THRESHOLD
