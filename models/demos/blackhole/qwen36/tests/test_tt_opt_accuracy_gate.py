# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-P150 HF accuracy gate for Qwen3.5 checkpoints without a model_targets.yaml entry.

Runs the demo's teacher-forced top-1/top-5 test against the checkpoint's .refpt, with the
floors taken from QWEN35_GATE_TOP1 / QWEN35_GATE_TOP5 instead of the centralized targets.

Run:
  HF_MODEL=Qwen/Qwen3.5-2B MESH_DEVICE=P150 QWEN35_GATE_TOP1=<x> QWEN35_GATE_TOP5=<y> \
  pytest models/demos/blackhole/qwen36/tests/test_tt_opt_accuracy_gate.py -x -q -s
"""
import os

import pytest

import models.tt_transformers.demo.simple_text_demo as simple_text_demo
from models.demos.blackhole.qwen36.demo.text_demo import test_demo_text_accuracy  # noqa: F401


@pytest.fixture(autouse=True)
def _gate_floors(monkeypatch):
    top1 = float(os.environ.get("QWEN35_GATE_TOP1", "0"))
    top5 = float(os.environ.get("QWEN35_GATE_TOP5", "0"))
    monkeypatch.setattr(simple_text_demo, "get_accuracy_thresholds", lambda *a, **k: (top1, top5))
