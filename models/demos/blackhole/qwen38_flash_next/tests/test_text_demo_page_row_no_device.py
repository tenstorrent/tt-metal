# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The demo's page-table row: k distinct nonzero ids from a seeded shuffle, then zero padding."""

import torch

from models.demos.blackhole.qwen38_flash_next.demo.text_demo import demo_page_row


def test_demo_page_row_layout():
    row = demo_page_row(5, 14, 16, seed=0)
    assert row.dtype == torch.int32 and row.shape == (1, 16)
    ids = row[0, :5].tolist()
    assert len(set(ids)) == 5 and all(1 <= i <= 13 for i in ids)
    assert not row[0, 5:].any()


def test_demo_page_row_seed():
    assert torch.equal(demo_page_row(5, 14, 16, 3), demo_page_row(5, 14, 16, 3))
    assert not torch.equal(demo_page_row(5, 40, 16, 1), demo_page_row(5, 40, 16, 2))
