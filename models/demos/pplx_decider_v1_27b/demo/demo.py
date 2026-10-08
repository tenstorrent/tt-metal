# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""pplx-decider-v1-27b demo: choice, noul and score decisions on a Blackhole TP mesh."""

import json
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.pplx_decider_v1_27b.tt.model import PplxDecider

SAMPLE_INPUTS = Path(__file__).parent / "sample_inputs.json"


def _label(result):
    if result["type"] == "noul":
        return result["noul"] > 0.5
    return max(result["probabilities"], key=result["probabilities"].get)


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(1800)
@parametrize_mesh_tp()
def test_pplx_decider_demo(mesh_device):
    samples = json.loads(SAMPLE_INPUTS.read_text())

    start = time.perf_counter()
    decider = PplxDecider.from_pretrained(mesh_device)
    logger.info(f"Model load time: {time.perf_counter() - start:.1f} s")

    rows = []
    for sample in samples:
        name, state, question = sample["name"], sample["state"], sample["question"]

        start = time.perf_counter()
        decider.predict(state, question)
        first_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        result = decider.predict(state, question)
        warm_ms = (time.perf_counter() - start) * 1000

        logger.info(f"{name}: {json.dumps(result)}")
        assert _label(result) == sample["expected"], f"{name}: got {_label(result)}, expected {sample['expected']}"
        rows.append((name, _label(result), first_ms, warm_ms))

    logger.info(f"{'name':<10} {'answer':<20} {'first ms':>9} {'warm ms':>9}")
    for name, label, first_ms, warm_ms in rows:
        logger.info(f"{name:<10} {label!s:<20} {first_ms:>9.1f} {warm_ms:>9.1f}")
