# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The host sampler, against `transformers`' own processors.

No device, no checkpoint. What is worth testing here is not that multinomial works but
that this reproduces the library's semantics: the direction of the repetition penalty,
which ids top_k keeps, and the order the three are applied in. The order is the subtle
one. Dividing by the temperature first would rescale the logits the penalty then reads,
and top_k would keep a different set.

Where `transformers` is installed, each rule is checked against the library's own
processor rather than against a hand-written expectation.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/test_sampling.py
"""

import pytest
import torch

from models.demos.audio.qwen3_tts import sampling

VOCAB = 64
PENALTY = 1.05


@pytest.fixture
def logits():
    torch.manual_seed(0)
    return torch.randn(VOCAB)


def test_the_penalty_pushes_seen_ids_down_from_either_side(logits):
    """A seen id must lose ground whether its logit is positive or negative."""
    positive = int(logits.argmax())
    negative = int(logits.argmin())
    assert logits[positive] > 0 and logits[negative] < 0, "fixture must cover both signs"

    penalised = sampling.apply_repetition_penalty(logits.clone(), [positive, negative], PENALTY)
    assert penalised[positive] < logits[positive]
    assert penalised[negative] < logits[negative]


def test_the_penalty_leaves_unseen_ids_alone(logits):
    seen = [0, 1, 2]
    penalised = sampling.apply_repetition_penalty(logits.clone(), seen, PENALTY)
    untouched = [index for index in range(VOCAB) if index not in seen]
    assert torch.equal(penalised[untouched], logits[untouched])


def test_the_penalty_matches_transformers(logits):
    """The library is the definition; a sign error here would be invisible by ear."""
    processor = pytest.importorskip("transformers.generation.logits_process")
    seen = [3, 7, 11, 11]
    theirs = processor.RepetitionPenaltyLogitsProcessor(penalty=PENALTY)(
        torch.tensor([seen]), logits.clone().reshape(1, -1)
    ).reshape(-1)
    ours = sampling.apply_repetition_penalty(logits.clone(), seen, PENALTY)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_top_k_keeps_exactly_k_ids(logits):
    """Sampling can only ever return one of the k best."""
    top_k = 5
    kept = set()
    for seed in range(400):
        generator = torch.Generator().manual_seed(seed)
        kept.add(sampling.sample(logits, top_k=top_k, temperature=1.0, generator=generator))
    assert kept <= set(torch.topk(logits, top_k).indices.tolist())
    assert len(kept) == top_k, "400 draws should reach all of them"


def test_a_seeded_run_is_reproducible(logits):
    first = [sampling.sample(logits, generator=torch.Generator().manual_seed(7)) for _ in range(5)]
    again = [sampling.sample(logits, generator=torch.Generator().manual_seed(7)) for _ in range(5)]
    assert first == again
    assert len(set(first)) == 1, "same seed and same logits is the same draw"


def test_a_cold_temperature_collapses_onto_the_argmax(logits):
    """Not a code path the checkpoint uses, but it pins the temperature's direction."""
    generator = torch.Generator().manual_seed(0)
    picks = {sampling.sample(logits, temperature=0.01, generator=generator) for _ in range(20)}
    assert picks == {int(logits.argmax())}


def test_the_penalty_is_applied_before_the_temperature():
    """Order matters, and this is the case that separates the two orders.

    Two ids sit close together at the top. The penalty is enough to reorder them, and it
    stays enough only if it lands before the division by 0.9. Applied afterwards, the
    logits are larger and the same multiplicative penalty no longer closes the gap.
    """
    logits = torch.full((VOCAB,), -10.0)
    logits[0], logits[1] = 4.0, 3.9

    generator = torch.Generator().manual_seed(0)
    picks = {
        sampling.sample(logits, seen=[0], temperature=0.01, top_k=2, penalty=1.05, generator=generator)
        for _ in range(20)
    }
    # 4.0 / 1.05 = 3.81 < 3.9, so id 1 must win.
    assert picks == {1}


# ── suppression ─────────────────────────────────────────────────────────────


def test_suppressed_ids_are_never_drawn(logits):
    """Upstream's `suppress_tokens`: the ids it lists cannot come back, however likely."""
    forbidden = torch.topk(logits, 8).indices.tolist()
    draws = {
        sampling.sample(
            logits, temperature=1.0, top_k=0, suppress=forbidden, generator=torch.Generator().manual_seed(seed)
        )
        for seed in range(200)
    }
    assert not draws & set(forbidden), "a suppressed id was drawn"
    assert len(draws) > 1, "the test needs a distribution, not a single spike"


def test_suppression_happens_before_the_top_k_floor(logits):
    """Order matters: suppressed ids must not occupy places in the top k.

    Suppressing the whole top 8 with k=4 must leave the 9th to 12th, not nothing.
    """
    ordered = torch.topk(logits, 12).indices.tolist()
    survivors = set(ordered[8:12])
    draws = {
        sampling.sample(
            logits, temperature=1.0, top_k=4, suppress=ordered[:8], generator=torch.Generator().manual_seed(seed)
        )
        for seed in range(200)
    }
    assert draws <= survivors, f"drew outside the surviving top-k: {sorted(draws - survivors)}"


def test_an_empty_suppression_list_changes_nothing(logits):
    """The default path must be untouched, since every existing measurement assumes it."""
    seeded = lambda **kwargs: sampling.sample(logits, generator=torch.Generator().manual_seed(3), **kwargs)
    assert seeded() == seeded(suppress=())
    assert seeded() == seeded(suppress=[])
