# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in diagnostic; does not enable chunked prefill in model capabilities."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.qwen38_27b_qb2.demo.probe_chunked_prefill import main


@pytest.mark.skipif(os.getenv("QWEN_CHUNKED_STATE_TEST") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_chunked_prefill_state_on_tp4():
    args = SimpleNamespace(
        **{name: Path(os.environ["QWEN_CHUNKED_" + name.upper()]) for name in ("qualification", "precision", "output")}
    )
    main(args)
    report = json.loads((args.output / "progress.json").read_text())
    assert report["state"] == "completed" and report["cleanup_completed"]
    assert len(report["comparisons"]) == 11
    assert report["passed"], report["comparisons"]
