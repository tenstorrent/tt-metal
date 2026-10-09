# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit allocated-hardware diagnostic, without a model-quality pass claim."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.qwen38_27b_qb2.demo.probe_hf_layers import main


@pytest.mark.skipif(os.getenv("QWEN_HF_LAYER_REFERENCE") != "1", reason="explicit allocated-Galaxy diagnostic")
def test_teacher_forced_hf_layer_diagnostic():
    args = SimpleNamespace(
        **{
            name: Path(os.environ["QWEN_HF_" + name.upper()])
            for name in ("weights", "qualification", "precision", "reference", "output")
        }
    )
    control = os.getenv("QWEN_HF_CONTROL_PRECISION")
    args.control_precision = Path(control) if control else None
    main(args)
    report = json.loads((args.output / "progress.json").read_text())
    assert report["state"] == "completed" and report["cleanup_completed"]
    assert len(report["steps"]) == 8
    assert all(len(row["hidden"]) == 65 for row in report["steps"])
    assert report["is_gpqa_score"] is False
