# SPDX-License-Identifier: Apache-2.0
"""Behavior-preserving oracle for the weight-loading refactor.

First run (baseline, no fixture present) writes fixtures/baseline_logits.pt.
Subsequent runs compare current prefill logits against that baseline at PCC >= 0.9999.
"""
import os
from pathlib import Path

import pytest
import torch

import ttnn  # noqa
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model

pytestmark = run_for_blackhole()

CHECKPOINT_DIR = os.environ.get("HF_MODEL", "/local/ttuser/atupe/Qwen9b")
FIXTURE = Path(__file__).parent / "fixtures" / "baseline_logits.pt"
# Fixed, deterministic prompt — 64 tokens, no padding.
PROMPT_TOKENS = torch.arange(1, 65, dtype=torch.int32).unsqueeze(0)  # [1, 64]


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def test_prefill_logits_snapshot(device):
    # HF_MODEL (hub name or local path) is the single source of truth; the run
    # command exports it, and CHECKPOINT_DIR mirrors it for the fallback default.
    os.environ.setdefault("HF_MODEL", CHECKPOINT_DIR)
    model = Qwen35Model.from_pretrained(device, max_seq_len=2048)
    logits_tt = model.prefill(PROMPT_TOKENS)
    logits = ttnn.to_torch(logits_tt).float().cpu()

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    if not FIXTURE.exists():
        torch.save(logits, FIXTURE)
        pytest.skip(f"Baseline written to {FIXTURE}; re-run to compare.")
    baseline = torch.load(FIXTURE)
    assert logits.shape == baseline.shape, (logits.shape, baseline.shape)
    pcc = _pcc(logits, baseline)
    assert pcc >= 0.9999, f"prefill logits diverged from baseline: PCC={pcc:.6f}"
