# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for prefill trace staging skip logic."""

import torch

from models.demos.gemma4.tt.model import Gemma4Model


def test_page_table_stage_skip_same_object_and_equal_copy():
    cache = {}
    buf_key = 1
    pt = torch.arange(16, dtype=torch.int32).view(1, 16)
    assert not Gemma4Model.should_skip_staged_page_table(cache, buf_key, pt)

    cache[(buf_key, "page_table")] = (pt, pt.detach().clone())
    assert Gemma4Model.should_skip_staged_page_table(cache, buf_key, pt)

    pt_copy = pt.clone()
    assert Gemma4Model.should_skip_staged_page_table(cache, buf_key, pt_copy)

    pt_other = pt.clone()
    pt_other[0, 0] = 99
    assert not Gemma4Model.should_skip_staged_page_table(cache, buf_key, pt_other)


def test_page_table_stage_skip_is_per_buffer():
    cache = {}
    pt = torch.zeros(1, 8, dtype=torch.int32)
    cache[(10, "page_table")] = (pt, pt.clone())
    assert Gemma4Model.should_skip_staged_page_table(cache, 10, pt)
    assert not Gemma4Model.should_skip_staged_page_table(cache, 11, pt)
