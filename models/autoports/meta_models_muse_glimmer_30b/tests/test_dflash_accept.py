# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DFlash accept/reject rule. No device, no weights."""

from __future__ import annotations

import pytest
import torch

from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_accept import accept_block


def hf_reference(candidates: list[int], selected: list[int]) -> tuple[list[int], int]:
    """Literal transcription of the rule in ``_assisted_decoding``, for cross-checking."""
    cand = torch.tensor([candidates])
    sel = torch.tensor([selected])
    n_matches = int(((~(cand == sel[:, :-1])).cumsum(dim=-1) < 1).sum())
    return sel[:, : n_matches + 1].tolist()[0], n_matches


def test_all_accepted_commits_block_plus_bonus():
    candidates = [10, 11, 12]
    target = [10, 11, 12, 13]
    result = accept_block(candidates, target)
    assert result.tokens == (10, 11, 12, 13)
    assert result.n_matches == 3
    assert result.n_committed == 4, "full acceptance commits block_size tokens from one target forward"


def test_full_rejection_still_commits_the_correction_token():
    """Forward progress must be guaranteed even when every draft is wrong."""
    result = accept_block([10, 11, 12], [99, 98, 97, 96])
    assert result.tokens == (99,)
    assert result.n_matches == 0
    assert result.n_committed == 1


def test_partial_acceptance_stops_at_first_mismatch():
    result = accept_block([10, 11, 12], [10, 11, 77, 78])
    # Position 2 mismatches, so the target's own token 77 is committed and 78 is discarded:
    # 78 was predicted from a sequence containing the rejected token 12.
    assert result.tokens == (10, 11, 77)
    assert result.n_matches == 2


def test_committed_tokens_come_from_target_not_candidates():
    """A later match after a mismatch must not resurrect the candidate."""
    result = accept_block([10, 11, 12], [10, 55, 12, 13])
    assert result.tokens == (10, 55)
    assert 12 not in result.tokens


def test_eos_truncates_the_commit():
    result = accept_block([10, 11, 12], [10, 2, 12, 13], eos_token_ids=(2,))
    assert result.tokens == (10, 2)


def test_max_new_tokens_caps_the_commit():
    result = accept_block([10, 11, 12], [10, 11, 12, 13], max_new_tokens=2)
    assert result.tokens == (10, 11)
    assert result.n_matches <= result.n_committed - 1


def test_length_contract_is_enforced(expect_error):
    with expect_error(ValueError, "expected len"):
        accept_block([10, 11], [10, 11])


@pytest.mark.parametrize("seed", range(64))
def test_matches_hf_rule_on_random_blocks(seed):
    """Randomised equivalence against a literal transcription of the HF rule."""
    generator = torch.Generator().manual_seed(seed)
    block = 15
    candidates = torch.randint(0, 6, (block,), generator=generator).tolist()
    selected = torch.randint(0, 6, (block + 1,), generator=generator).tolist()
    expected_tokens, expected_matches = hf_reference(candidates, selected)
    result = accept_block(candidates, selected)
    assert list(result.tokens) == expected_tokens
    assert result.n_matches == expected_matches
