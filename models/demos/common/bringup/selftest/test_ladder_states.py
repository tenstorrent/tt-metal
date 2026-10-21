# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F99: integer ladder states (top-k index lists) are compared by the selection overlap, not by the PCC of the ids
(DeepSeek-V4.1 M.1: topk_idx PCC 0.52 for a device whose float states were all >= 0.996)."""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.ladder import index_overlap


def test_index_overlap_tolerates_near_tie_flips_and_catches_a_wrong_selector():
    g = torch.Generator().manual_seed(0)
    want = torch.stack([torch.randperm(1024, generator=g)[:64] for _ in range(256)]).to(torch.int32)
    want[:8, 32:] = -1  # padding of the first rows
    assert index_overlap(want.clone(), want) == 1.0
    flipped = want.clone()
    flipped[::4, 0] = 2000  # a near-tie flip on a quarter of the rows: one id of 64 replaced
    ov = index_overlap(flipped[:, torch.randperm(64, generator=g)], want)  # any order
    assert 0.99 < ov < 1.0
    assert metrics.pcc(flipped.float(), want.float()) < 0.97  # the PCC the old check used fails it
    wrong = torch.stack([torch.randperm(1024, generator=g)[:64] for _ in range(256)]).to(torch.int32)
    assert index_overlap(wrong, want) < 0.2
