# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device tests of the chain's paging: pool allocation at open and the page-table write."""

from types import SimpleNamespace

from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    Qwen38ChatChainError,
    Qwen38TracedChain,
    _open_pools,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging, paged_table_width

CONTEXT = 8192


class FakeModel:
    allocated_context = CONTEXT

    def __init__(self):
        self.pool_calls = []
        self.table_writes = []

    def allocate_paged_pools(self, paging):
        self.pool_calls.append(paging)
        return {3: object()}

    def write_generic_page_table(self, state, block_ids):
        self.table_writes.append((state, list(block_ids)))


def paging(block_size=128):
    width = paged_table_width(CONTEXT, block_size)
    return Qwen38LanePaging(page_size=block_size, num_pages=width + 9, max_blocks=width)


def _make(target, paged):
    obj = object.__new__(Qwen38TracedChain)
    obj.built_target = target
    obj.state = "state"
    obj.paging_config = paging() if paged else None
    obj.paged_pools = None
    obj.pools_owned = False
    obj._page_table_ids = None
    return obj


def test_pools_allocated_once_when_paging_given():
    model = FakeModel()
    given = paging()
    result = _open_pools(model, given, None)
    assert result[0] is given and result[2] is True
    assert set(result[1]) == {3}
    assert model.pool_calls == [given]


def test_no_paging_no_pools():
    model = FakeModel()
    assert _open_pools(model, None, None) == (None, None, False)
    assert model.pool_calls == []


def test_caller_pools_are_not_reallocated_or_owned():
    model = FakeModel()
    pools = {1: object()}
    got = _open_pools(model, paging(), pools)
    assert got[1] is pools and got[2] is False
    assert model.pool_calls == []


def test_invalid_paging_rejected_before_allocation(expect_error):
    model = FakeModel()
    bad = Qwen38LanePaging(page_size=128, num_pages=20, max_blocks=8)
    with expect_error(ValueError, "must cover the allocated"):
        _open_pools(model, bad, None)
    assert model.pool_calls == []


def test_env_switch_goes_through_same_path(monkeypatch):
    monkeypatch.setenv("QWEN38_PAGED_KV", "1")
    model = FakeModel()
    got = _open_pools(model, None, None, paged_kv_env=True)
    assert got[0].page_size == 128 and got[2] is True and len(model.pool_calls) == 1


def test_write_page_table_writes_then_skips_unchanged():
    model = FakeModel()
    c = _make(SimpleNamespace(model=model), True)
    assert c.write_page_table([5, 2, 7]) is True
    assert c.write_page_table((5, 2, 7)) is False
    assert c.write_page_table([5, 2, 8]) is True
    assert model.table_writes == [("state", [5, 2, 7]), ("state", [5, 2, 8])]


def test_write_page_table_raises_when_unpaged(expect_error):
    model = FakeModel()
    c = _make(SimpleNamespace(model=model), False)
    with expect_error(Qwen38ChatChainError, "needs a paged chain"):
        c.write_page_table([1])
    assert model.table_writes == []
