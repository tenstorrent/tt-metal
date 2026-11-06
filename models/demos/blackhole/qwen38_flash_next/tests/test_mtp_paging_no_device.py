# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device tests of the MTP layer's paging: its pools at open, its verify generic state, the shared page table."""

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging, paged_table_width

CONTEXT = 8192


class FakeMTPAttention:
    def __init__(self):
        self.pool_calls = []

    def allocate_paged_pools(self, paging):
        self.pool_calls.append(paging)
        return SimpleNamespace(kv="mtp-kv", index="mtp-index")


class FakeMTPLayer:
    def __init__(self):
        self.attention = FakeMTPAttention()
        self.state_calls = []

    def allocate_generic_state(self, *args):
        self.state_calls.append(args)
        return "mtp-generic-state"


def paging():
    width = paged_table_width(CONTEXT, 128)
    return Qwen38LanePaging(page_size=128, num_pages=width + 9, max_blocks=width)


def test_mtp_pools_allocated_only_when_paged_and_mtp_on():
    layer = FakeMTPLayer()
    components = SimpleNamespace(decoder_layer=layer)
    given = paging()
    pools = session._open_mtp_pools(components, given)
    assert (pools.kv, pools.index) == ("mtp-kv", "mtp-index")
    assert layer.attention.pool_calls == [given]


def test_no_mtp_pools_without_paging_or_mtp():
    layer = FakeMTPLayer()
    assert session._open_mtp_pools(SimpleNamespace(decoder_layer=layer), None) is None
    assert session._open_mtp_pools(None, paging()) is None
    assert layer.attention.pool_calls == []


def test_paged_verify_state_gets_paging_pools_and_the_chain_table():
    layer = FakeMTPLayer()
    table = object()
    state = SimpleNamespace(paging=paging(), page_table=table)
    pools = object()
    assert mtp_v2._allocate_alignment_generic_state(layer, state, pools) == "mtp-generic-state"
    (args,) = layer.state_calls
    assert args[0] is state.paging and args[1] is pools and args[2] is table  # the table OBJECT: one write serves both


def test_flat_verify_state_is_unchanged():
    layer = FakeMTPLayer()
    state = SimpleNamespace(paging=None, page_table=None)
    mtp_v2._allocate_alignment_generic_state(layer, state, None)
    assert layer.state_calls == [()]


@pytest.mark.parametrize(
    "paged,pools,components", [(True, None, True), (False, object(), True), (True, object(), False)]
)
def test_pools_and_paging_must_agree(monkeypatch, paged, pools, components, expect_error):
    monkeypatch.setattr(mtp_v2, "_validate_mtp_components", lambda model, c: ("layer", "in", "out"))
    owner = object()
    model = SimpleNamespace(_state_owner=owner, rope_table=object(), qsa_position_constants=object(), layers=[])
    state = mtp_v2.Qwen38TTNNTextModelGenericState.__new__(mtp_v2.Qwen38TTNNTextModelGenericState)
    object.__setattr__(state, "_owner", owner)
    object.__setattr__(state, "paging", paging() if paged else None)
    with expect_error(ValueError, "mtp_pools|MTP layer's pools"):
        mtp_v2.allocate_verify_state(
            model,
            state,
            mtp_components=object() if components else None,
            mtp_pools=pools,
            gdn_step_anchor_layers=(),
        )
