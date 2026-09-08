# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate the captured reference fixtures. No device.

The device PCC test grades the TTNN drafter against these tensors, so a malformed or
mis-conditioned fixture would show up there as a numerics bug in the port. This file pins
the fixture contract instead, so that failure mode is caught here where it is legible.

The acceptance check is the important one. Shapes, dtypes and even PCC can all be perfect
while the *conditioning* is wrong -- wrong tap order, wrong tap offset, wrong anchor, or a
drafted block outside the target's own output distribution -- and the only symptom is that
acceptance collapses. Measured on this checkpoint: a fixture whose context was raw prose
accepted 1/15 candidates, while the same code on the target's own generation accepted
11/15. Both produce identically-shaped tensors.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.blackhole.qwen36.tests.dflash.conftest import FIXTURE_CTX_LENS, load_fixture

# Floor, not a target. The gemma-4-31B DFlash reference reports 7.50 tokens committed per
# iteration and this checkpoint measured 12 at ctx 512; 2 is low enough to tolerate a hard
# block while still failing loudly on a mis-conditioned capture (which scores 0-1).
MIN_GOLDEN_ACCEPTANCE = 2


@pytest.mark.parametrize("ctx_len", FIXTURE_CTX_LENS)
def test_fixture_shapes_and_dtypes(drafter_cfg, ctx_len):
    fx = load_fixture(ctx_len)
    block = drafter_cfg.block_size

    assert fx["ctx_len"] == ctx_len
    assert fx["block_size"] == block
    assert fx["target_hidden"].shape == (1, ctx_len, drafter_cfg.target_feature_size)
    assert fx["noise_embedding"].shape == (1, block, drafter_cfg.hidden_size)
    assert fx["reference_hidden"].shape == (1, block, drafter_cfg.hidden_size)
    assert fx["reference_draft_hidden"].shape == (1, drafter_cfg.num_draft_tokens, drafter_cfg.hidden_size)
    assert fx["reference_candidates"].shape == (drafter_cfg.num_draft_tokens,)
    assert fx["input_ids"].shape == (1, ctx_len)

    # Positions span context AND block: the drafter's q takes the last `block` of these while
    # its k/v take all of them.
    assert fx["position_ids"].shape == (1, ctx_len + block)
    assert torch.equal(fx["position_ids"][0], torch.arange(ctx_len + block))


@pytest.mark.parametrize("ctx_len", FIXTURE_CTX_LENS)
def test_fixture_is_finite_and_nondegenerate(drafter_cfg, ctx_len):
    """Catches an all-zero or NaN capture, which would make PCC meaningless."""
    fx = load_fixture(ctx_len)
    for key in ("target_hidden", "noise_embedding", "reference_hidden"):
        t = fx[key].float()
        assert torch.isfinite(t).all(), f"{key} has non-finite values"
        assert t.abs().max() > 0, f"{key} is all zeros"
        assert t.std() > 0, f"{key} is constant"


@pytest.mark.parametrize("ctx_len", FIXTURE_CTX_LENS)
def test_fixture_draft_hidden_is_the_reference_slice(drafter_cfg, ctx_len):
    """``reference_draft_hidden`` must be the last ``block-1`` rows of ``reference_hidden``.

    Slot 0 is the anchor and is not a proposal; the reference drops it with
    ``hidden[:, 1 - block:, :]``. An off-by-one here would silently shift every candidate.
    """
    fx = load_fixture(ctx_len)
    block = fx["block_size"]
    assert torch.equal(fx["reference_draft_hidden"], fx["reference_hidden"][:, 1 - block :, :])


@pytest.mark.parametrize("ctx_len", FIXTURE_CTX_LENS)
def test_fixture_noise_block_is_anchor_then_masks(drafter_cfg, ctx_len):
    """The noise block must be ``[anchor, MASK, MASK, ...]`` in the target's embedding space.

    Verified by comparing embedding rows rather than token ids, since the fixture stores the
    embeddings: every masked slot must be identical (same mask token), and the anchor slot
    must differ from them.
    """
    fx = load_fixture(ctx_len)
    noise = fx["noise_embedding"][0].float()

    masks = noise[1:]
    assert torch.allclose(masks, masks[0].expand_as(masks)), "masked slots are not all the same token"
    assert not torch.allclose(noise[0], masks[0]), "anchor slot holds the mask token"


@pytest.mark.parametrize("ctx_len", FIXTURE_CTX_LENS)
def test_fixture_golden_acceptance(drafter_cfg, ctx_len):
    """The golden must actually work: the target accepts several leading candidates.

    This is the conditioning check -- see the module docstring for why shapes alone are not
    enough.
    """
    fx = load_fixture(ctx_len)
    acceptance = fx["golden_acceptance"]
    assert acceptance >= MIN_GOLDEN_ACCEPTANCE, (
        f"golden accepts only {acceptance}/{drafter_cfg.num_draft_tokens} candidates at ctx {ctx_len}. "
        "Check tap order/offset, the anchor, and that the drafted block is in the target's own "
        "output distribution (raw-corpus contexts score ~1)."
    )

    # And the recorded acceptance must be consistent with the recorded tensors.
    cand = fx["reference_candidates"]
    recomputed = int((cand == fx["target_argmax"][:-1]).to(torch.int32).cumprod(0).sum())
    assert recomputed == acceptance, f"recorded acceptance {acceptance} != recomputed {recomputed}"


@pytest.mark.parametrize("ctx_len", FIXTURE_CTX_LENS)
def test_fixture_matches_current_checkpoint(drafter_cfg, ctx_len):
    """A stale fixture cannot pass silently: its fingerprint must match the checkpoint.

    Skips when the checkpoint is unreachable, so this stays usable offline.
    """
    from models.demos.blackhole.qwen36.tests.dflash.capture_fixtures import _resolve, _weight_fingerprint

    fx = load_fixture(ctx_len)
    try:
        current = _weight_fingerprint(_resolve(fx["drafter_model"]))
    except Exception as exc:
        pytest.skip(f"drafter checkpoint unavailable: {type(exc).__name__}: {exc}")

    assert fx["drafter_fingerprint"] == current, (
        f"fixture was captured from a different drafter checkpoint "
        f"({fx['drafter_fingerprint']} != {current}); re-run capture_fixtures.py"
    )
