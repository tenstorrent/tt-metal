# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Scheduler tables retain row ownership across padded prefill and decode."""

from types import SimpleNamespace

import torch

from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM


def adapter():
    model = object.__new__(Gemma4ForCausalLM)
    model.cache = SimpleNamespace(
        batch=4, kv=[None, None], page_tables={i: torch.zeros(4, 8, dtype=torch.int32) for i in range(2)}
    )
    return model


def test_prefill_scatters_live_prefix_and_ignores_padded_rows():
    model = adapter()
    source = torch.tensor([[11, 12], [21, 22], [91, 92], [93, 94]])
    tables = model._tables(model.cache, [source, source], slots=[3, 1])
    expected = torch.zeros(4, 8, dtype=torch.int32)
    expected[3, :2] = torch.tensor([11, 12])
    expected[1, :2] = torch.tensor([21, 22])
    assert torch.equal(tables[0], expected)
    assert tables[0] is tables[1]
    source.fill_(100)
    assert torch.equal(tables[0], expected), "Later scheduler writes must not mutate a submitted table"


def test_decode_preserves_slot_order_and_translates_null_pages():
    model = adapter()
    source = torch.tensor([[11, 12], [21, 22], [-1, -1], [31, 32]])
    tables = model._tables(model.cache, [source, source])
    assert torch.equal(tables[0][:, :2], torch.tensor([[11, 12], [21, 22], [0, 0], [31, 32]]))
    assert torch.count_nonzero(tables[0][:, 2:]) == 0


def test_prefill_rejects_missing_live_rows(expect_error):
    model = adapter()
    source = torch.zeros(1, 8, dtype=torch.int32)
    with expect_error(ValueError, "fewer rows"):
        model._tables(model.cache, [source, source], slots=[0, 1])
