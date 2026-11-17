# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""StrandsDecider latency (batch 1) and batched throughput (eager prefill, one chip) against prompt length.

The batched run repeats one question BATCH times; the call splits it into forwards of at most MAX_BATCH_TOKENS
padded tokens. No time assertions."""

import statistics
import time

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.strands_agent.tests.common import parametrize_mesh
from models.demos.strands_agent.tt.model import StrandsDecider, plan_batches
from models.demos.strands_agent.tt.prompting import encode_request

TARGET_LENGTHS = (128, 256, 512, 1024, 2048, 4000)
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {
            "billing": "payments, invoices, refunds and charges",
            "sales": "new purchases, quotes and pricing",
            "retail": "in-store orders, shipping and returns",
            "support": "product help and troubleshooting",
        },
    }
}
STATE_TEXT = (
    "I was charged twice for my order #48211 this month and nobody has answered my emails for three days. "
    "Please fix this or I will dispute both charges with my bank. "
)
WARMUP_RUNS = 2
TIMED_RUNS = 5
BATCH = 32


def _build_question(decider, target):
    """Repeat the state text word by word; bisect the word count to the first request with >= `target` tokens."""
    words = STATE_TEXT.split()

    def encode(count):
        state = " ".join((words * (count // len(words) + 1))[:count])
        return encode_request(decider.tokenizer, state, QUESTIONS, decider.config.max_length)[0]

    lo, hi = 1, 4 * target
    while lo < hi:
        mid = (lo + hi) // 2
        if len(encode(mid).input_ids) >= target:
            hi = mid
        else:
            lo = mid + 1
    return encode(lo)


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(3600)
@parametrize_mesh()
def test_strands_decider_latency(mesh_device):
    decider = StrandsDecider.from_pretrained(mesh_device)
    rows = []
    for target in TARGET_LENGTHS:
        q = _build_question(decider, target)
        times = {}
        for qs in ([q], [q] * BATCH):
            for _ in range(WARMUP_RUNS):
                decider.question_logits_batch(qs)
            runs = []
            for _ in range(TIMED_RUNS):
                start = time.perf_counter()
                decider.question_logits_batch(qs)
                runs.append((time.perf_counter() - start) * 1000)
            times[len(qs)] = statistics.median(runs)
        forwards = len(plan_batches([len(q.input_ids)] * BATCH, decider.max_batch, decider.max_batch_tokens))
        rows.append((len(q.input_ids), times[1], times[BATCH], forwards))

    logger.info(f"{'tokens':>7} {'B=1 ms':>8} {'B=32 ms':>9} {'forwards':>9} {'questions/s':>12}")
    for tokens, b1, bt, forwards in rows:
        logger.info(f"{tokens:>7} {b1:>8.1f} {bt:>9.1f} {forwards:>9} {1000 * BATCH / bt:>12.1f}")
