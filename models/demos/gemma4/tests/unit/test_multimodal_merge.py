# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side check of the multimodal soft-token merge (no device): image, video and
audio soft tokens land on their own placeholder slots, in prompt order, in one list."""
import torch

from models.demos.gemma4.tt.generator_vllm import merge_soft_tokens

IMG, VID, AUD, PAD = 258880, 258884, 258881, 0


def _prompt(spec):
    """spec: list of ('t', n) text ids or (token_id, n) placeholder runs -> ids tensor."""
    ids = []
    for kind, n in spec:
        ids += [1000 + len(ids) + i for i in range(n)] if kind == "t" else [kind] * n
    return torch.tensor(ids, dtype=torch.long)


def _embeds(n, tag):
    # row r of modality `tag` carries (tag, r) so the merged order can be read back
    e = torch.zeros(n, 4)
    e[:, 0] = tag
    e[:, 1] = torch.arange(n)
    return e


def test_merge_orders_modalities_by_prompt_position():
    ids = _prompt([("t", 3), (AUD, 5), ("t", 2), (IMG, 4), ("t", 1), (VID, 6), ("t", 2)])
    llm_ids, positions, merged = merge_soft_tokens(
        ids, [(IMG, _embeds(4, 1)), (VID, _embeds(6, 2)), (AUD, _embeds(5, 3))], PAD
    )
    # every placeholder became pad, text untouched
    assert (
        (llm_ids[ids == IMG] == PAD).all() and (llm_ids[ids == VID] == PAD).all() and (llm_ids[ids == AUD] == PAD).all()
    )
    text = (ids != IMG) & (ids != VID) & (ids != AUD)
    assert torch.equal(llm_ids[text], ids[text])
    # positions ascending and exactly the placeholder slots
    expected = torch.where((ids == IMG) | (ids == VID) | (ids == AUD))[0]
    assert positions.dtype == torch.int32 and torch.equal(positions.long(), expected)
    # the soft tokens follow the prompt: audio (tag 3) first, then image (1), then video (2)
    tags = merged[:, 0].tolist()
    assert tags == [3] * 5 + [1] * 4 + [2] * 6
    # and within a modality the rows keep their own order
    assert merged[:5, 1].tolist() == list(range(5)) and merged[5:9, 1].tolist() == list(range(4))
    assert merged.dtype == torch.bfloat16


def test_merge_single_modality_and_two_images():
    ids = _prompt([("t", 2), (IMG, 3), ("t", 1), (IMG, 3)])
    llm_ids, positions, merged = merge_soft_tokens(ids.view(1, -1), [(IMG, _embeds(6, 1))], PAD)
    assert positions.tolist() == [2, 3, 4, 6, 7, 8]
    assert merged[:, 1].tolist() == list(range(6))  # both images' tokens, in order


def test_merge_rejects_count_mismatch_and_missing_id(expect_error):
    ids = _prompt([("t", 2), (AUD, 4)])
    with expect_error(ValueError, "Placeholder count"):
        merge_soft_tokens(ids, [(AUD, _embeds(5, 3))], PAD)
    with expect_error(ValueError, "no placeholder id"):
        merge_soft_tokens(ids, [(None, _embeds(4, 3))], PAD)
