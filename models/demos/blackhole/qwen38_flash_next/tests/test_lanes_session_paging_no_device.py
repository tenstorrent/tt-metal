# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the lanes session on shared (paged) pools: page-table mirror and sync, the admission's order, close,
and the capacity admission without the flat KV terms.  Fakes only; no device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_lanes_session as lanes_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes


def _session(lanes: int = 4):
    session = lanes_module.Qwen38LanesSession.__new__(lanes_module.Qwen38LanesSession)
    session.lanes = lanes
    session.model = "model"
    session.verify = SimpleNamespace(rows=5)
    session.draft = "draft"
    session.paged = True
    session.lane_pages = {}
    session.synced_pages = None
    return session


@pytest.fixture
def writes(monkeypatch):
    calls = []

    def fake(model, verify, rows, active, keep=None, *, draft=None):
        calls.append({"rows": [list(r) for r in rows], "active": list(active), "draft": draft})

    monkeypatch.setattr(mtp_lanes, "write_lane_page_table", fake)
    return calls


def test_sync_writes_only_when_the_mirror_changed(writes) -> None:
    session = _session()
    assert session.sync_page_tables() is True  # first sync: the device holds unknown values
    assert writes[-1]["active"] == [] and writes[-1]["draft"] == "draft"
    assert session.sync_page_tables() is False
    session.set_lane_pages(2, [7, 8])
    assert session.sync_page_tables() is True
    assert writes[-1]["active"] == [2] and writes[-1]["rows"][2] == [7, 8]
    assert writes[-1]["rows"][0] == [] and writes[-1]["rows"][1] == []
    assert session.sync_page_tables() is False
    session.set_lane_pages(2, [7, 8])  # same value
    assert session.sync_page_tables() is False
    session.clear_lane_pages(2)  # parked: the rows go to zeros
    assert session.sync_page_tables() is True
    assert writes[-1]["active"] == [] and writes[-1]["rows"][2] == []
    session.clear_lane_pages(2)
    assert session.sync_page_tables() is False
    assert len(writes) == 3


def test_set_lane_pages_refuses_bad_lane_and_empty_pages() -> None:
    session = _session()
    for lane, ids in ((4, [1]), (-1, [1]), (0, [])):
        with pytest.raises(ValueError):  # allow-pytest.raises: host argument checks
            session.set_lane_pages(lane, ids)


def test_paged_admit_writes_the_chain_table_before_the_prefill_and_the_lane_table_after_the_import(
    writes, monkeypatch
) -> None:
    order = []
    imports = []
    monkeypatch.setattr(lanes_module.ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(
        mtp_lanes, "import_lane_state", lambda *args, **kwargs: (order.append("import"), imports.append(kwargs))
    )
    monkeypatch.setattr(mtp_lanes, "write_lane_page_table", lambda *a, **k: order.append("lane_table"))
    ple = (1, 1)
    chain = SimpleNamespace(
        write_page_table=lambda ids: order.append(("chain_table", list(ids))),
        loop_guard=lambda: SimpleNamespace(__enter__=lambda s: None, __exit__=lambda *a: None),
        read_token_row=lambda: 42,
        position=lambda: 3,
    )

    class _Guard:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    chain.loop_guard = lambda: _Guard()

    class _Pager:
        def evict(self, lane, slot):
            order.append("evict")

    fake_session = SimpleNamespace(
        reset=lambda: order.append("reset"),
        prefill_mode="teacher",
        _prefill=lambda suffix, *a: order.append("prefill"),
        ple_context=ple,
    )
    session = _session()
    session.session, session.chain, session.lane_chain = fake_session, chain, object()
    session.mesh = None
    session.model = SimpleNamespace(layers=[])
    session.backbone_pager = session.alignment_pager = _Pager()
    session.backbone_pool = session.alignment_pool = SimpleNamespace(slots=[None])
    ticket = SimpleNamespace(prompt_ids=[5, 6, 7], images=None)
    admitted = session.admit(1, ticket, block_ids=[3, 4])
    assert admitted.pending == 42 and admitted.position == 3
    assert order == [("chain_table", [3, 4]), "reset", "prefill", "evict", "evict", "import", "lane_table"]
    assert imports[0]["shared"] is True
    assert session.lane_pages == {1: [3, 4]}
    with pytest.raises(lanes_module.Qwen38LanesSessionError):  # allow-pytest.raises: paged needs the blocks
        session.admit(2, ticket)


def test_close_keeps_the_chain_pools(monkeypatch) -> None:
    released = []
    monkeypatch.setattr(lanes_module.ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(mtp_lanes, "release_lane_draft_state", lambda *a: released.append("draft"))
    monkeypatch.setattr(mtp_lanes, "release_lane_verify_state", lambda *a: released.append("verify"))
    session = _session()
    session.mesh = "mesh"
    session.width_sets = {}
    session.context_sets = {}
    session.traces = session.output = None
    pools = SimpleNamespace(kv="kv", index="index")
    session.chain = SimpleNamespace(pools=pools, mtp_paged_pools=pools)
    session.backbone_pager = session.alignment_pager = None
    session.closed = False
    session.close()
    assert released == ["draft", "verify"] and session.closed
    assert session.chain.pools is pools and session.chain.mtp_paged_pools is pools


def test_paged_capacity_admission_drops_the_flat_kv_terms() -> None:
    for context in (32768, 131072):
        flat = session_module.lanes_capacity_admission(context, lanes=4, drafts=4)
        paged = session_module.lanes_capacity_admission(context, lanes=4, drafts=4, paged=True)
        assert flat["paged"] is False and paged["paged"] is True
        assert paged["per_lane_bytes_per_device"] < flat["per_lane_bytes_per_device"]
        assert paged["required_free_bytes_per_bank"] < flat["required_free_bytes_per_bank"]
        remainders, paged_remainders = (
            flat["lanes_growth_remainders_bytes_per_bank"],
            paged["lanes_growth_remainders_bytes_per_bank"],
        )
        assert paged_remainders["lane_states"] < remainders["lane_states"]
        assert paged_remainders["pagers"] < remainders["pagers"]
        assert (
            paged["required_largest_contiguous_bytes_per_bank"] == session_module.RESIDENT_MIN_CONTIGUOUS_BYTES_PER_BANK
        )
        # the paged lane cost no longer grows with the context
        assert paged["per_lane_bytes_per_device"] < 400 * 2**20
    small = session_module.lanes_capacity_admission(32768, lanes=4, drafts=4, paged=True)["per_lane_bytes_per_device"]
    large = session_module.lanes_capacity_admission(131072, lanes=4, drafts=4, paged=True)["per_lane_bytes_per_device"]
    assert large - small < 3 * 2**20
    with pytest.raises(ValueError):  # allow-pytest.raises: bool only
        session_module.lanes_capacity_admission(32768, lanes=4, drafts=4, paged=1)
