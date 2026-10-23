# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side check of the multimodal soft-token merge (no device): image, video and
audio soft tokens land on their own placeholder slots, in prompt order, in one list."""
import torch

from models.demos.gemma4.tt.generator_utils import items_within_prefill_window, merge_soft_tokens

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


def test_merge_cached_prefix_fills_trailing_placeholders(expect_error):
    # vLLM prefix caching: the first image sits inside the cached prefix and arrives
    # without its payload, so only the second image's 3 soft tokens are forwarded.
    ids = _prompt([("t", 2), (IMG, 3), ("t", 1), (IMG, 3), ("t", 2)])
    llm_ids, positions, merged = merge_soft_tokens(ids, [(IMG, _embeds(3, 1))], PAD, num_cached=6)
    assert positions.tolist() == [6, 7, 8] and merged[:, 1].tolist() == [0, 1, 2]
    assert (llm_ids[ids == IMG] == PAD).all()  # every placeholder still becomes pad
    # a shortfall that would leave an uncached placeholder empty is an error ...
    with expect_error(ValueError, "Placeholder count"):
        merge_soft_tokens(ids, [(IMG, _embeds(3, 1))], PAD, num_cached=4)
    # ... and so is a shortfall with no cached prefix at all
    with expect_error(ValueError, "Placeholder count"):
        merge_soft_tokens(ids, [(IMG, _embeds(3, 1))], PAD)


def test_items_past_the_chunk_end_wait_for_a_later_chunk():
    # Chunked prefill under load: the first chunk of a 1200-token prompt ends at 300,
    # before its image (offset 420, 270 placeholders); the plugin still forwards the
    # image, so it must be left out of this chunk's merge and taken by the next.
    assert items_within_prefill_window([420], [270], 300) == []
    assert items_within_prefill_window([420], [270], 1200) == [0]
    # Several items: the third lies past the end, the first two (one of them inside a
    # cached prefix) are inside.
    assert items_within_prefill_window([10, 420, 900], [270, 270, 256], 700) == [0, 1]
    # The window never ends inside an item (vLLM disable_chunked_mm_input); an exact end
    # is inside.
    assert items_within_prefill_window([10], [270], 280) == [0]
    # Older plugins without ranges: every item is kept, as before.
    assert items_within_prefill_window(None, None, 300) == []
    assert items_within_prefill_window([None, 420], [None, 270], 300) == [0]
