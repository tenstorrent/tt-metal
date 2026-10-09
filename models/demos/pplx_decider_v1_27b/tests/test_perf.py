# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 decision latency of PplxDecider, eager vs traced prefill."""

import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.pplx_decider_v1_27b.tests.common import parametrize_mesh_traced
from models.demos.pplx_decider_v1_27b.tt.model import PplxDecider

PROMPT_LENGTHS = (512, 1024, 2048, 3072, 4096)
QUESTION = {"type": "noul", "instructions": "Is this a software license?"}
PCC_THRESHOLD = 0.999
WARMUP_RUNS = 2
TIMED_RUNS = 10


def _build_input_ids(decider, num_tokens):
    tokenizer = decider.tokenizer
    license_text = (Path(decider.args.CKPT_DIR) / "LICENSE").read_text()
    repeats = 9000 // max(1, len(tokenizer(license_text)["input_ids"])) + 2
    body_ids = tokenizer(license_text * repeats, add_special_tokens=False)["input_ids"]
    # The template adds a fixed number of tokens, so the body length is corrected until the total matches.
    body_len = num_tokens
    for _ in range(6):
        input_ids = decider.input_ids(tokenizer.decode(body_ids[:body_len]), QUESTION)
        if len(input_ids) == num_tokens:
            break
        body_len += num_tokens - len(input_ids)
    assert len(input_ids) == num_tokens, f"built {len(input_ids)} tokens, wanted {num_tokens}"
    return input_ids


def _time_ms(decider, input_ids):
    for _ in range(WARMUP_RUNS):
        decider.readout_logits(input_ids)
    times = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        decider.readout_logits(input_ids)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), min(times)


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(1800)
@parametrize_mesh_traced()
def test_pplx_decider_latency(mesh_device):
    decider = PplxDecider.from_pretrained(mesh_device)
    inputs = {t: _build_input_ids(decider, t) for t in PROMPT_LENGTHS}

    eager, eager_logits = {}, {}
    for t, input_ids in inputs.items():
        eager[t] = _time_ms(decider, input_ids)
        eager_logits[t] = decider.readout_logits(input_ids)

    decider.capture_prefill_trace()

    rows, failures = [], []
    for t, input_ids in inputs.items():
        traced_logits = decider.readout_logits(input_ids)
        _, pcc = comp_pcc(eager_logits[t], traced_logits, PCC_THRESHOLD)
        top1_ok = int(torch.argmax(eager_logits[t])) == int(torch.argmax(traced_logits))
        traced = _time_ms(decider, input_ids)
        rows.append((t, *eager[t], *traced, pcc))
        if not (top1_ok and pcc >= PCC_THRESHOLD):
            failures.append(f"T={t}: top1_ok={top1_ok} pcc={pcc}")

    logger.info(f"{'T':>6} {'eager p50':>10} {'eager min':>10} {'traced p50':>11} {'traced min':>11} {'PCC':>8}")
    for t, eager_p50, eager_min, traced_p50, traced_min, pcc in rows:
        logger.info(f"{t:>6} {eager_p50:>10.1f} {eager_min:>10.1f} {traced_p50:>11.1f} {traced_min:>11.1f} {pcc:>8.5f}")
    assert not failures, "; ".join(failures)
