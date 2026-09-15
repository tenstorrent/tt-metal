# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""STOP suppression must happen BEFORE top-k/top-p, in both samplers.

When ``min_new_tokens`` is active the samplers mask STOP so a short utterance keeps going. If
that mask is applied AFTER candidate filtering and STOP is peaked enough to be the only survivor
of the nucleus (p(STOP) > top_p), the masked token is all that is left:

* the reference sampler ends with every candidate at ``-inf``, so ``softmax`` is NaN and
  ``torch.multinomial`` raises;
* the traced sampler picks STOP itself, because the filtered-out tokens sit at ``bf16(NEG_INF)``
  (-1.000256e30) while the masked STOP sits at ``logit + fp32(-1e30)`` (-1.000000e30) — the
  suppressed token is the strictly largest value, so ``argmax`` returns it.

Neither is reachable at the default ``min_tokens = 0`` (no mask is built at all), so these
construct the peaked-STOP logits directly.
"""

import pytest
import torch

import ttnn
from models.experimental.xtts.config import GENERATION, NEG_INF, NUM_AUDIO_TOKENS, STOP_AUDIO_TOKEN
from models.experimental.xtts.demo.xtts_reference_demo import _sample
from models.experimental.xtts.tt.xtts_sampler import TtSampler

SAMPLER_KW = dict(
    temperature=GENERATION.temperature,
    top_k=GENERATION.top_k,
    top_p=GENERATION.top_p,
    rep=GENERATION.repetition_penalty,
)


def _peaked_stop_logits():
    """Logits where STOP alone survives top-k/top-p (p(STOP) > top_p)."""
    logits = torch.zeros(NUM_AUDIO_TOKENS)
    logits[STOP_AUDIO_TOKEN] = 10.0
    p_stop = torch.softmax(logits / GENERATION.temperature, dim=-1)[STOP_AUDIO_TOKEN]
    assert p_stop > GENERATION.top_p, f"fixture no longer peaks STOP past top_p: {p_stop}"
    return logits


def test_reference_sampler_suppresses_peaked_stop():
    """Suppressed STOP must yield an audio code, not NaN probabilities."""
    token = _sample(_peaked_stop_logits(), set(), suppress=(STOP_AUDIO_TOKEN,), **SAMPLER_KW)
    assert token != STOP_AUDIO_TOKEN, "suppressed STOP was still sampled"
    assert 0 <= token < STOP_AUDIO_TOKEN, f"token {token} is not an audio code"


def test_reference_sampler_still_allows_unsuppressed_stop():
    """The fix must not suppress STOP once the floor is satisfied."""
    token = _sample(_peaked_stop_logits(), set(), suppress=(), **SAMPLER_KW)
    assert token == STOP_AUDIO_TOKEN, "peaked STOP was not sampled with suppression off"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_traced_sampler_suppresses_peaked_stop(device, reset_seeds):
    """pick_dev must not return the very token its bias suppresses."""
    logits = _peaked_stop_logits()
    sampler = TtSampler(
        device,
        NUM_AUDIO_TOKENS,
        GENERATION.temperature,
        GENERATION.top_k,
        GENERATION.repetition_penalty,
        GENERATION.top_p,
    )
    logits_dev = ttnn.from_torch(logits.reshape(1, 1, -1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    bias = torch.zeros(1, NUM_AUDIO_TOKENS)
    bias[0, STOP_AUDIO_TOKEN] = NEG_INF
    bias_dev = ttnn.from_torch(bias, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    # Guard the premise: without the bias, filtering leaves STOP as the sole candidate.
    shaped = ttnn.to_torch(sampler._apply_penalty_temp_topk(logits_dev)).float().flatten()
    survivors = (shaped > NEG_INF / 2).nonzero().flatten().tolist()
    assert survivors == [STOP_AUDIO_TOKEN], f"fixture no longer isolates STOP: {survivors}"

    sampler.reset()
    token = int(ttnn.to_torch(sampler.pick_dev(logits_dev, None, bias_dev)).flatten()[0])
    assert token != STOP_AUDIO_TOKEN, "suppressed STOP was still returned by pick_dev"
    assert 0 <= token < STOP_AUDIO_TOKEN, f"token {token} is not an audio code"
