# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the on-device XTTS sampler (``tt/xtts_sampler.py``).

Sampling is stochastic, so this checks the deterministic guarantees and the two
mechanisms that keep generation from collapsing (as greedy does): top-k truncation
degenerating to argmax at k=1, valid in-range draws, and the repetition penalty
suppressing an otherwise-dominant token after it is drawn once.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd); export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_sampler.py -s
"""

import torch
from loguru import logger

import ttnn
from models.experimental.xtts.reference.xtts_gpt_model import NUM_AUDIO_TOKENS
from models.experimental.xtts.tt.xtts_sampler import TtSampler


def _logits(device, peak_token=42, peak=12.0):
    x = torch.randn(1, 1, NUM_AUDIO_TOKENS) * 2.0
    x[0, 0, peak_token] = peak  # a dominant token
    return ttnn.from_torch(x.float(), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), int(x.argmax())


def test_top_k_1_is_argmax(device, reset_seeds):
    logits, ref = _logits(device)
    s = TtSampler(device, NUM_AUDIO_TOKENS, temperature=1.0, top_k=1, repetition_penalty=1.0)
    picks = [s.pick(logits) for _ in range(5)]
    logger.info(f"top_k=1 picks {picks} (argmax={ref})")
    assert all(p == ref for p in picks), f"top_k=1 must equal argmax {ref}, got {picks}"


def test_sampling_diverse_in_range_and_rep_penalty(device, reset_seeds):
    logits, ref = _logits(device)
    s = TtSampler(device, NUM_AUDIO_TOKENS, temperature=0.75, top_k=50, repetition_penalty=5.0)
    picks = [s.pick(logits) for _ in range(40)]
    logger.info(f"sampled {len(picks)}: {len(set(picks))} unique; dominant-token count={picks.count(ref)}")

    assert all(0 <= p < NUM_AUDIO_TOKENS for p in picks), "sampled token out of range"
    assert len(set(picks)) > 10, f"sampling not diverse ({len(set(picks))} unique) — sampler may be stuck"
    # rep penalty (5.0) should stop the dominant token from being drawn repeatedly.
    assert picks.count(ref) <= 3, f"repetition penalty ineffective: dominant token drawn {picks.count(ref)}x"
