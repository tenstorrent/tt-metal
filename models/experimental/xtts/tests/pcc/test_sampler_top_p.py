# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A requested top_p must be honoured even when top-k is disabled.

Both samplers computed their cutoffs over a top-k window, so a caller asking for
``top_k=0, top_p=0.85`` got no cutoff at all -- sampling from the whole vocabulary while believing
it had a nucleus. ``top_k >= vocab_size`` collapses to the same disabled state, so it failed the
same way. The window is now the whole vocabulary whenever a nucleus is requested without top-k,
which leaves the kth threshold inert and lets top_p do the cutting.
"""

import pytest
import torch

import ttnn
from models.experimental.xtts.config import NEG_INF, NUM_AUDIO_TOKENS
from models.experimental.xtts.demo.xtts_reference_demo import _sample
from models.experimental.xtts.tt.xtts_sampler import TtSampler

TEMPERATURE, TOP_P = 0.65, 0.85
# Each disables top-k, so a requested top_p used to be dropped: 0 explicitly, the others by
# collapsing through `top_k < vocab_size`.
DISABLED_TOP_K = [0, NUM_AUDIO_TOKENS, NUM_AUDIO_TOKENS + 974]


@pytest.fixture(scope="module")
def logits():
    """Fixed logits with a nucleus far smaller than the vocabulary."""
    torch.manual_seed(0)
    return torch.randn(NUM_AUDIO_TOKENS) * 3.0


@pytest.fixture(scope="module")
def nucleus(logits):
    """The token ids top_p should keep, computed independently of either sampler."""
    probs = torch.softmax(logits / TEMPERATURE, dim=-1)
    order = torch.argsort(probs, descending=True)
    ordered = probs[order]
    keep = (torch.cumsum(ordered, dim=0) - ordered) < TOP_P
    ids = set(order[keep].tolist())
    assert 0 < len(ids) < NUM_AUDIO_TOKENS // 8, f"fixture nucleus is not selective: {len(ids)}"
    return ids


@pytest.mark.parametrize("top_k", DISABLED_TOP_K, ids=["zero", "eq-vocab", "gt-vocab"])
def test_reference_sampler_honours_top_p_without_top_k(top_k, logits, nucleus):
    """Draws must stay inside the nucleus when top-k is disabled."""
    torch.manual_seed(1)
    drawn = {_sample(logits, set(), temperature=TEMPERATURE, top_k=top_k, top_p=TOP_P, rep=1.0) for _ in range(500)}
    assert drawn <= nucleus, f"sampled {sorted(drawn - nucleus)[:8]} outside the top_p nucleus"


def test_reference_sampler_leaves_full_vocab_when_nothing_requested(logits, nucleus):
    """top_p=1.0 with top-k off asks for no cutoff, so the fix must not invent one."""
    torch.manual_seed(1)
    drawn = {_sample(logits, set(), temperature=TEMPERATURE, top_k=0, top_p=1.0, rep=1.0) for _ in range(500)}
    assert not drawn <= nucleus, "unrequested filtering was applied"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("top_k", DISABLED_TOP_K, ids=["zero", "eq-vocab", "gt-vocab"])
def test_traced_sampler_honours_top_p_without_top_k(device, top_k, logits, nucleus, reset_seeds):
    """The device sampler must keep exactly the nucleus when top-k is disabled."""
    sampler = TtSampler(device, NUM_AUDIO_TOKENS, TEMPERATURE, top_k, 1.0, TOP_P)
    try:
        logits_dev = ttnn.from_torch(
            logits.reshape(1, 1, -1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        shaped = ttnn.to_torch(sampler._apply_penalty_temp_topk(logits_dev)).float().flatten()
        survivors = set((shaped > NEG_INF / 2).nonzero().flatten().tolist())
        assert survivors == nucleus, f"kept {len(survivors)} candidates, nucleus is {len(nucleus)}"
    finally:
        sampler.release()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_traced_sampler_matches_reference_window(device, logits, nucleus, reset_seeds):
    """The default top-k path is untouched: both samplers keep the same candidates."""
    sampler = TtSampler(device, NUM_AUDIO_TOKENS, TEMPERATURE, 50, 1.0, TOP_P)
    try:
        logits_dev = ttnn.from_torch(
            logits.reshape(1, 1, -1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        shaped = ttnn.to_torch(sampler._apply_penalty_temp_topk(logits_dev)).float().flatten()
        survivors = set((shaped > NEG_INF / 2).nonzero().flatten().tolist())
        assert survivors == nucleus, "top-k 50 no longer agrees with the independent nucleus"
    finally:
        sampler.release()
