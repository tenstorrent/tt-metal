# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests of the vllm-tt-plugin batching contract in Molmo2ForConditionalGeneration (no device needed).

Regression tests for the in-process DP=4 path (one engine, four 1x8 replicas): multimodal kwargs arrive as one entry
per batch row, and the runner permutes decode rows with ``slot_remap`` while KV state stays where it was prefilled.
"""

import torch

from models.demos.molmo2.tests.test_generator_vllm_adapter import Molmo2ForConditionalGeneration


def _adapter(dp=4, batch_per_dp=1):
    a = Molmo2ForConditionalGeneration.__new__(Molmo2ForConditionalGeneration)
    a.dp, a.batch_per_dp = dp, batch_per_dp
    a._phys = [(r, l) for r in range(dp) for l in range(batch_per_dp)]
    return a


def test_flatten_is_per_row():
    a = _adapter()
    img, vid = torch.zeros(7, 3), torch.ones(30, 3)
    pixel_values = [[img], None]  # row 0: one image, row 1: video request (no image tensors)
    pixel_values_videos = [None, [vid]]  # the other way around
    assert a._flatten(pixel_values, 0, 2) == [img]
    assert a._flatten(pixel_values, 1, 2) == []
    assert a._flatten(pixel_values_videos, 0, 2) == []  # row 0 must NOT receive row 1's video
    assert a._flatten(pixel_values_videos, 1, 2) == [vid]
    assert a._unwrap(pixel_values_videos, 0, 2) is None
    assert a._unwrap(pixel_values_videos, 1, 2) is vid


def test_text_only_rows_are_empty():
    a = _adapter()
    assert a._flatten([None], 0, 1) == []
    assert a._flatten([[None]], 0, 1) == []
    assert a._unwrap([None, None], 1, 2) is None


def test_legacy_flat_per_image_list_for_single_request():
    a = _adapter()
    p0, p1 = torch.zeros(7, 3), torch.zeros(9, 3)
    assert a._flatten([p0, p1], 0, 1) == [p0, p1]  # 2 images, 1 row -> flat per-image list


def test_slot_remap_moves_indirection_not_data():
    a = _adapter()
    assert a._phys == [(0, 0), (1, 0), (2, 0), (3, 0)]
    a._apply_slot_remap(torch.tensor([1, 0, 2, 3]))  # decode row 0 reads slot 1's state and vice versa
    assert a._phys == [(1, 0), (0, 0), (2, 0), (3, 0)]
    a._apply_slot_remap(None)  # identity: no change
    assert a._phys == [(1, 0), (0, 0), (2, 0), (3, 0)]
    a._apply_slot_remap(torch.tensor([1, 0, 2, 3]))  # a second permutation composes on the current mapping
    assert a._phys == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_slot_remap_must_be_a_permutation():
    a = _adapter()
    try:
        a._apply_slot_remap(torch.tensor([0, 0, 2, 3]))
    except AssertionError:
        return
    raise AssertionError("duplicate slots were accepted")
