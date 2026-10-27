# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device) tests of the slot-level prefix cache (``doc/serving/prefix_caching.md``): the
per-slot content chains and multimodal spans (``tt/vllm_config.py::SlotContent``), the hit / miss decision with the
reused-slot downgrade and the KDA-position rule (``SlotPrefixCache``), the snapshot-ring policy and its planner
sizing (``tt/memory_plan.py``), the planner's tail split, and the adapter end to end on the fake mesh model of
``test_generator_vllm_host.py``: a multi-turn hit restores the decode-crossing snapshot and prefills from it, a
miss prefills from 0, a stripped multimodal hit is refused, retained slots park at their content end.

    PYTHONPATH=<worktree>:<tt-metal>/ttnn:<tt-metal> <tt-metal>/python_env/bin/pytest \\
        models/demos/glm53_flash/tests/test_serving_prefix_cache_host.py --noconftest -p no:cacheprovider -q
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.tests.test_generator_vllm_host import (
    PAD,
    _FakeMeshModel,
    _first_token,
    _greedy,
    _next_token,
    expect_error,
)
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.generator_vllm import GLM53FlashForCausalLM
from models.demos.glm53_flash.tt.memory_plan import MemoryPlan, kda_state_bytes_per_user, snapshot_bytes_per_chip
from models.demos.glm53_flash.tt.vllm_config import (
    APC_ENV,
    MMSpan,
    PrefillPlanner,
    ServingGeometry,
    SlotContent,
    SlotPrefixCache,
    SlotTable,
    mm_payload_missing,
    mm_spans_of_row,
)

WORKTREE = Path(__file__).resolve().parents[4]


# ----------------------------------------------------------------------------------------- the gate
def test_capability_gate_reads_the_env_at_import():
    """Default off; ``GLM53_VLLM_PREFIX_CACHING=1`` declares the capability. Checked in a subprocess from INSIDE the
    worktree (Python resolves ``models`` from the cwd first), which also proves the import stays ttnn-free."""
    assert GLM53FlashForCausalLM.model_capabilities["supports_prefix_caching"] is False
    code = (
        "import sys; from models.demos.glm53_flash.tt import generator_vllm as g; "
        "print(g.GLM53FlashForCausalLM.model_capabilities['supports_prefix_caching'], 'ttnn' in sys.modules)"
    )
    env = dict(os.environ, **{APC_ENV: "1"})
    env.pop("PYTHONSAFEPATH", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=WORKTREE, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["True", "False"], out.stdout


# ------------------------------------------------------------------------------------ content chains
def test_slot_content_chain_and_mm_spans():
    c = SlotContent(0, 2, 4096)
    ids = torch.arange(100, dtype=torch.int32) + 7
    c.write(0, ids[:64])
    c.write(64, ids[64:])
    assert c.length == 100 and torch.equal(c.tokens[:100], ids)
    assert c.matches(ids, [], 100) and c.matches(ids, [], 32) and not c.matches(ids, [], 101)
    other = ids.clone()
    other[50] += 1
    assert c.matches(other, [], 50) and not c.matches(other, [], 64)
    # multimodal: an image run inside the compared prefix matches by identifier only
    c.mm = [MMSpan(10, 20, "image", "sha-a")]
    assert c.matches(ids, [MMSpan(10, 20, "image", "sha-a")], 100)
    assert not c.matches(ids, [MMSpan(10, 20, "image", "sha-b")], 100), "a different image, same placeholder ids"
    assert not c.matches(ids, [MMSpan(10, 20, "image", None)], 100), "no identifier forwarded: never a match"
    assert not c.matches(ids, [], 100), "the slot holds an image the request does not"
    assert c.matches(ids, [MMSpan(10, 20, "image", "sha-b")], 10), "the span starts at the compared boundary"
    assert not c.matches(ids, [MMSpan(10, 20, "image", "sha-b")], 11), "a span straddling n counts"
    c.mm = [MMSpan(10, 20, "video", "sha-a")]
    assert not c.matches(ids, [MMSpan(10, 20, "image", "sha-a")], 100), "the kind is part of the content"
    # truncation keeps the prefix, drops later spans and snapshots
    c.mm = [MMSpan(10, 20, "image", "a"), MMSpan(80, 10, "image", "b")]
    c.snapshots = {32: 1, 64: 2, 96: 0}
    c.truncate(64)
    assert c.length == 64 and c.mm == [MMSpan(10, 20, "image", "a")] and c.snapshots == {32: 1, 64: 2}
    assert c.best_snapshot(64) == (64, 2) and c.best_snapshot(63) == (32, 1) and c.best_snapshot(31) is None
    # a decode write past a truncation point rewrites the tail
    c.write(64, torch.tensor([5]))
    assert c.length == 65 and int(c.tokens[64]) == 5


def test_mm_spans_and_payload_from_the_plugin_lists():
    kwargs = {
        "pixel_values": [None, [torch.zeros(4, 1176), None]],
        "image_grid_thw": [None, [torch.tensor([[1, 2, 2]]), None]],
        "pixel_values_videos": [None, [None, None]],
        "mm_placeholder_offsets": [None, [5, 40]],
        "mm_placeholder_lengths": [None, [4, 8]],
    }
    assert mm_spans_of_row(kwargs, 0) == []
    assert mm_spans_of_row(kwargs, 1) == [MMSpan(5, 4, "image", None), MMSpan(40, 8, "image", None)]
    kwargs["mm_identifiers"] = [None, ["h1", "h2"]]
    assert [sp.identifier for sp in mm_spans_of_row(kwargs, 1)] == ["h1", "h2"]
    # item 1 (positions [40, 48)) has no pixels: inside a 64-token cached prefix it cannot be prefilled from 0
    assert mm_payload_missing(kwargs, 1, 64) == [1] and mm_payload_missing(kwargs, 1, 44) == []
    assert mm_payload_missing(kwargs, 0, 64) == []
    video = {
        "pixel_values": [[None]],
        "image_grid_thw": [[None]],
        "pixel_values_videos": [[torch.zeros(8, 1176)]],
        "video_grid_thw": [[torch.tensor([[2, 2, 2]])]],
        "mm_placeholder_offsets": [[3]],
        "mm_placeholder_lengths": [[16]],
    }
    assert mm_spans_of_row(video, 0) == [MMSpan(3, 16, "video", None)]


# ----------------------------------------------------------------------------------- hit decisions
def _geometry(**kw):
    base = dict(rows=2, cols=4, users_per_row=4, max_seq_len=4096, pad_token_id=PAD)
    base.update(kw)
    return ServingGeometry(**base)


def test_hit_miss_reused_slot_downgrade_and_lru():
    g = _geometry()
    apc = SlotPrefixCache(g, entries=2)
    ids_a = torch.arange(300, dtype=torch.int32) + 1
    ids_b = torch.arange(300, dtype=torch.int32) + 1000
    # prompt A in user 0: prefilled to 300, decoded to 320 (a decode crossing snapshot at 320), finished
    apc.begin(0, 0, 0)
    apc.record_prefill(0, 0, 0, ids_a)
    for p in range(300, 320):
        apc.record_decode(0, 0, p, 50 + p)
    assert apc.plan_snapshot(0, 0, 320, "decode") == 0
    apc.release(0, 0)
    content_a = apc.content(0, 0)
    assert content_a.retained and content_a.length == 320 and content_a.snapshots == {320: 0}
    # the multi-turn request: A + its response + a new turn, vLLM hit = 320 (the full blocks of the content)
    turn = torch.cat([content_a.tokens[:320].clone(), torch.full((40,), 9, dtype=torch.int32)])
    hit = apc.lookup(turn, [], 320)
    assert hit is not None and (hit.row, hit.user, hit.position, hit.entry) == (0, 0, 320, 0)
    # a shorter hit (a request sharing only the first 288 tokens): the KDA state is past it -> no slot serves it
    assert apc.lookup(turn, [], 288) is None, "the snapshot is at 320, the hit at 288: a miss"
    # while live the slot is never a candidate
    apc.begin(0, 0, 320)
    assert apc.lookup(turn, [], 320) is None
    apc.release(0, 0)
    assert apc.lookup(turn, [], 320) is not None
    # the slot reused by another prompt: the content is gone, the hit downgrades (lookup None)
    apc.begin(0, 0, 0)
    apc.record_prefill(0, 0, 0, ids_b)
    assert apc.stats["retained_evicted"] == 1
    assert apc.lookup(turn, [], 320) is None and apc.content(0, 0).snapshots == {}
    apc.release(0, 0)
    # LRU rank: a slot without content ranks 0, retained slots by last use (least recent first)
    apc.begin(0, 2, 0)
    apc.record_prefill(0, 2, 0, ids_a)
    apc.release(0, 2)
    assert apc.slot_rank(0, 1) == 0 and apc.slot_rank(0, 0) == 1 and apc.slot_rank(0, 2) == 2
    assert apc.retained_slots() == [(0, 0), (0, 2)]
    # park positions: retained slots at their content end (every replica row is the table's job)
    assert apc.park_positions() == {(0, 0): 300, (0, 2): 300}
    # the slot table consumes slots without content before dropping a resident prefix (balance first)
    t = SlotTable(g)
    users = [t.allocate(i, slot_rank=apc.slot_rank).user for i in range(4)]
    assert users[0] == 1 and set(users) == {0, 1, 2, 3}, users  # column 0 is the least loaded: its unused user 1 first
    t2 = SlotTable(g)
    assert (t2.allocate(0, at=(0, 2)).row, t2.physical(0).user) == (0, 2)
    with expect_error(RuntimeError, "is live"):
        t2.allocate(1, at=(0, 2))


def test_snapshot_ring_policy():
    g = _geometry()
    apc = SlotPrefixCache(g, entries=3)
    apc.begin(0, 1, 0)
    assert apc.plan_snapshot(0, 1, 0, "prefill") is None and apc.plan_snapshot(0, 1, 100, "prefill") is None
    # prefill snapshots take entries 1.. and evict the lowest position once full; entry 0 rolls with the decode
    assert apc.plan_snapshot(0, 1, 16384, "prefill") == 1
    assert apc.plan_snapshot(0, 1, 32768, "prefill") == 2
    assert apc.plan_snapshot(0, 1, 49152, "prefill") == 1, "the lowest prefill position (16384) is evicted"
    assert apc.content(0, 1).snapshots == {32768: 2, 49152: 1}
    assert apc.plan_snapshot(0, 1, 49152, "prefill") is None, "already held"
    assert apc.plan_snapshot(0, 1, 49184, "decode") == 0 and apc.plan_snapshot(0, 1, 49216, "decode") == 0
    assert apc.content(0, 1).snapshots == {32768: 2, 49152: 1, 49216: 0}
    assert apc.content(0, 1).best_snapshot(49215) == (49152, 1) and apc.content(0, 1).best_snapshot(10**6) == (
        49216,
        0,
    )
    one = SlotPrefixCache(g, entries=1)
    one.begin(0, 0, 0)
    assert one.plan_snapshot(0, 0, 32, "prefill") == 0 and one.plan_snapshot(0, 0, 64, "decode") == 0
    assert one.content(0, 0).snapshots == {64: 0}
    none = SlotPrefixCache(g, entries=0)
    assert none.plan_snapshot(0, 0, 32, "decode") is None
    with expect_error(ValueError, "snapshot kind"):
        apc.plan_snapshot(0, 1, 64, "other")


def test_planner_tail_split_and_snapshot_sizing():
    assert PrefillPlanner.walk(0, 300, 16384, split_tail=True) == ((0, 288), (288, 300))
    assert PrefillPlanner.walk(0, 320, 16384, split_tail=True) == ((0, 320),)
    assert PrefillPlanner.walk(0, 32868, 16384, split_tail=True) == (
        (0, 16384),
        (16384, 32768),
        (32768, 32864),
        (32864, 32868),
    )
    assert PrefillPlanner.walk(288, 300, 16384, split_tail=True) == ((288, 300),), "nothing to split off"
    assert PrefillPlanner.walk(0, 300, 16384) == ((0, 300),)
    g = _geometry(max_seq_len=65536, traced_prefill_lengths=(256, 512))
    pp = PrefillPlanner(g)
    assert pp.plan(0, 300, split_tail=True).kind == "traced", "the traced prefill is never split"
    assert pp.plan(32, 300, split_tail=True).chunks == ((32, 288), (288, 300))
    cfg = GLM53FlashConfig()
    per_user = kda_state_bytes_per_user(cfg, 4)
    assert per_user == 34 * (16 * 128 * 128 * 4 + 3 * 6144 * 2), per_user  # 35.2 MiB
    assert snapshot_bytes_per_chip(cfg, 4, 4, 2) == 8 * per_user
    plan = MemoryPlan.estimate(cfg, 8, 4, partition=True, scratch_slot=False)
    fit_1m = plan.max_snapshot_entries(4, 1048576)
    fit_128k = plan.max_snapshot_entries(28, 131072)
    logger.info(
        f"KDA snapshot = {per_user / 2**20:.1f} MiB per user per chip; planner spare 1 M x 4: "
        f"{plan.spare(4, 1048576) / 2**20:.0f} MiB -> {fit_1m} entries per slot; 128 K x 28: "
        f"{plan.spare(28, 131072) / 2**20:.0f} MiB -> {fit_128k} entries per slot\n" + plan.table()
    )
    assert fit_1m == 0, "the 1 M x 4 profile has no DRAM for a ring (host store)"
    assert fit_128k >= 1
    with_ring = MemoryPlan.estimate(cfg, 8, 4, partition=True, scratch_slot=False, snapshot_entries=fit_128k)
    assert with_ring.fits(28, 131072) and not with_ring.fits(28, 131072 + 16384 * 4)
    assert with_ring.bytes_for(28, 131072) - plan.bytes_for(28, 131072) == 28 * fit_128k * per_user


# -------------------------------------------------------------------------------------- adapter e2e
class _FakeApcModel(_FakeMeshModel):
    """The fake mesh model + the snapshot store surface (``snapshot_kda`` / ``restore_kda``)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.kda_snapshots = SimpleNamespace(release=lambda: self.calls.append(("release_snapshots",)))
        self.next_start = [0] * self.max_batch

    def snapshot_kda(self, user, entry):
        self.calls.append(("snapshot_kda", user, entry))

    def restore_kda(self, user, entry, position):
        self.calls.append(("restore_kda", user, entry, position))
        self.next_start[user] = position

    def prefill(self, ids, user=0, start=0):
        if start and start != self.next_start[user]:
            raise AssertionError(f"user {user}: chunk start {start} != next_start {self.next_start[user]}")
        out = super().prefill(ids, user, start)
        self.next_start[user] = start + int(ids.numel())
        return out

    def prefill_traced(self, ids, user=0):
        out = super().prefill_traced(ids, user)
        self.next_start[user] = int(ids.numel())
        return out

    def _step(self):
        super()._step()
        self.next_start = [p + 1 for p in self._fed_pos[: self.max_batch]]


def _apc_adapter(entries=2, users_per_row=4, max_seq_len=65536, tail_split=False, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("GLM53_VLLM_APC_TAIL_SPLIT", "1" if tail_split else "0")
    model = _FakeApcModel(2, 4, users_per_row, max_seq_len)
    g = ServingGeometry(
        rows=2,
        cols=4,
        users_per_row=users_per_row,
        max_seq_len=max_seq_len,
        traced_prefill_lengths=tuple(model.trace_prefill_lengths()),
        pad_token_id=PAD,
    )
    a = GLM53FlashForCausalLM(model, g, prefix_cache_entries=entries)
    n = g.request_capacity
    a.warmup_model_prefill(enable_trace=False, can_sample_on_device=True)
    a.warmup_model_decode(enable_trace=False, max_batch_size=n, num_blocks=1, can_sample_on_device=True)
    a.warmup_model_prefill(enable_trace=True, can_sample_on_device=True)
    a.warmup_model_decode(enable_trace=True, max_batch_size=n, num_blocks=1, can_sample_on_device=True)
    model.calls.clear()
    return a, model


def _decode(a, tokens, positions):
    return a.decode_forward(
        torch.tensor([[t] for t in tokens], dtype=torch.int32),
        torch.tensor(positions),
        sampling_params=_greedy(len(tokens)),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )


def test_multi_turn_hit_restores_the_crossing_snapshot_and_prefills_from_it():
    a, m = _apc_adapter(entries=2)
    L = 300
    prompt = (torch.arange(L) % 60 + 1).to(torch.int32).reshape(1, L)
    a.prefill_forward(prompt, prompt_lens=[L], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    user = a.slots.physical(0).user
    assert not [c for c in m.calls if c[0] == "snapshot_kda"], "300 is not a 32-multiple: no prefill snapshot"
    # 20 decode steps: the slot crosses 320 at the step that consumes position 319 -> one decode snapshot (entry 0)
    tok = 5
    for p in range(L, L + 20):
        out = _decode(a, [tok, 0, 0, 0], [p, -1, -1, -1])
        tok = int(out[0, 0])
    assert [c for c in m.calls if c[0] == "snapshot_kda"] == [("snapshot_kda", user, 0)]
    content = a.apc.content(0, user)
    assert content.length == L + 20 and content.snapshots == {320: 0} and content.live
    a.release_request(0)
    assert content.retained and a.apc.park_positions() == {(0, user): 320}
    # the next turn: the whole content + 40 new tokens, vLLM says 320 cached -> served from the snapshot at 320
    turn = torch.cat([content.tokens[:320].clone(), torch.full((40,), 9, dtype=torch.int32)]).reshape(1, 360)
    m.calls.clear()
    out = a.prefill_forward(
        turn, prompt_lens=[360], start_pos=torch.tensor([320]), empty_slots=[2], sampling_params=_greedy(1)
    )
    kinds = [c for c in m.calls if c[0] in ("restore_kda", "prefill", "prefill_traced")]
    assert kinds == [("restore_kda", user, 0, 320), ("prefill", user, 320, 40)], kinds
    s = a.slots.physical(2)
    assert (s.row, s.user, s.prefilled_to, s.phase) == (0, user, 360, "prefill")
    assert a.apc.stats["hits"] == 1 and a.apc.stats["downgraded"] == 0
    assert content.live and content.length == 360 and torch.equal(content.tokens[:360], turn.reshape(-1))
    first = int(out[0][0, 0])  # device sampling returns (tokens [B, 1], None)
    assert first == _first_token(user, turn.reshape(-1)[320:360].long()), "scored on the resumed device run"
    # the first decode of the new turn continues at 360
    _decode(a, [0, 0, first, 0], [-1, -1, 360, -1])
    assert a.slots.physical(2).position == 361


def test_miss_prefills_from_zero_and_a_shorter_hit_is_not_served(monkeypatch):
    a, m = _apc_adapter(entries=2)
    L = 2048 + 300  # eager: two device chunks? no: one 2348-token chunk (16 K chunk); the tail split lands a snapshot
    prompt = (torch.arange(L) % 60 + 1).to(torch.int32).reshape(1, L)
    # a cached-prefix request with nothing resident: the hit downgrades to a prefill from 0 in a fresh slot
    a.prefill_forward(
        prompt, prompt_lens=[L], start_pos=torch.tensor([512]), empty_slots=[1], sampling_params=_greedy(1)
    )
    calls = [c for c in m.calls if c[0] in ("restore_kda", "prefill", "prefill_traced")]
    assert calls == [("prefill", a.slots.physical(1).user, 0, L)], calls
    assert a.apc.stats["downgraded"] == 1 and a.apc.stats["hits"] == 0
    user = a.slots.physical(1).user
    assert a.apc.content(0, user).length == L
    a.release_request(1)
    # the same prompt again with vLLM's hit 2336 (= the full blocks of the content): the slot has no snapshot <= 2336
    # (2348 is unaligned, nothing crossed in decode) -> a miss again; the retained slot is the one dropped last
    m.calls.clear()
    a.prefill_forward(
        prompt, prompt_lens=[L], start_pos=torch.tensor([2336]), empty_slots=[1], sampling_params=_greedy(1)
    )
    calls = [c for c in m.calls if c[0] in ("restore_kda", "prefill")]
    assert (
        calls[0][0] == "prefill" and calls[0][2] == 0 and calls[0][1] != user
    ), "a slot without content is taken first"
    assert a.apc.stats["downgraded"] == 2
    a.release_request(1)
    # with the tail split, the eager prefill snapshots the prompt's last 32-boundary: the same hit is then served
    # (4 entries: entry 0 rolls with the decode, entries 1-3 hold prefill positions)
    b, mb = _apc_adapter(entries=4, tail_split=True, monkeypatch=monkeypatch)
    b.prefill_forward(prompt, prompt_lens=[L], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    u = b.slots.physical(0).user
    calls = [c for c in mb.calls if c[0] in ("prefill", "snapshot_kda")]
    assert calls == [("prefill", u, 0, 2336), ("snapshot_kda", u, 1), ("prefill", u, 2336, 12)], calls
    b.release_request(0)
    mb.calls.clear()
    b.prefill_forward(
        prompt, prompt_lens=[L], start_pos=torch.tensor([2336]), empty_slots=[3], sampling_params=_greedy(1)
    )
    calls = [c for c in mb.calls if c[0] in ("restore_kda", "prefill")]
    assert calls == [("restore_kda", u, 1, 2336), ("prefill", u, 2336, 12)], calls
    assert b.apc.stats["hits"] == 1
    # a chunked long prompt snapshots every device chunk end (16 K multiples) into the prefill entries
    Lc = 16384 * 3 + 64
    long = (torch.arange(Lc) % 60 + 1).to(torch.int32).reshape(1, Lc)
    mb.calls.clear()
    b.prefill_forward(long, prompt_lens=[Lc], start_pos=torch.tensor([0]), empty_slots=[1], sampling_params=_greedy(1))
    u1 = b.slots.physical(1).user
    snaps = [c for c in mb.calls if c[0] == "snapshot_kda"]
    # 16384 -> 1, 32768 -> 2, 49152 -> 3, the aligned prompt end 49216 -> 1 (the lowest position, 16384, evicted)
    assert snaps == [
        ("snapshot_kda", u1, 1),
        ("snapshot_kda", u1, 2),
        ("snapshot_kda", u1, 3),
        ("snapshot_kda", u1, 1),
    ], snaps
    assert b.apc.content(0, u1).snapshots == {32768: 2, 49152: 3, Lc: 1}
    b.release_request(1)
    # a request sharing the first 40 000 tokens hits at 39 968 (vLLM) and resumes from the 32 768 snapshot
    share = torch.cat([long.reshape(-1)[:40000], torch.full((100,), 7, dtype=torch.int32)]).reshape(1, 40100)
    mb.calls.clear()
    b.prefill_forward(
        share, prompt_lens=[40100], start_pos=torch.tensor([39968]), empty_slots=[2], sampling_params=_greedy(1)
    )
    calls = [c for c in mb.calls if c[0] in ("restore_kda", "prefill", "snapshot_kda")]
    # the hit truncated the record to 32768 (entry 2 stays, entries 1 / 3 free); the tail split ends the resumed chunk
    # on 40096 (a snapshot into the first free entry, 1) and walks the last 4 tokens
    assert calls == [
        ("restore_kda", u1, 2, 32768),
        ("prefill", u1, 32768, 7328),
        ("snapshot_kda", u1, 1),
        ("prefill", u1, 40096, 4),
    ], calls
    c = b.apc.content(0, u1)
    assert (
        c.length == 40100 and c.snapshots == {32768: 2, 40096: 1} and torch.equal(c.tokens[:40100], share.reshape(-1))
    )


def test_stripped_multimodal_hit_is_refused_and_identified_spans_hit():
    a, m = _apc_adapter(entries=2)
    L = 320
    prompt = (torch.arange(L) % 60 + 1).to(torch.int32).reshape(1, L)
    # the plugin's lists for one image item at [100, 116) whose pixels vLLM stripped (fully inside the cached prefix)
    mm = {
        "pixel_values": [[None]],
        "image_grid_thw": [[None]],
        "mm_placeholder_offsets": [[100]],
        "mm_placeholder_lengths": [[16]],
    }
    with expect_error(RuntimeError, "cannot be served"):
        a.prefill_forward(
            prompt, prompt_lens=[L], start_pos=torch.tensor([256]), empty_slots=[0], sampling_params=_greedy(1), **mm
        )
    assert a.apc.stats["refused_mm"] == 1 and a.slots.physical(0) is None
    # a text request whose content later matches an identified image span: the slot records the span at prefill
    # (the mm serving branch supplies the pixels; here the record is what matters) and the hit compares identifiers
    g = a.geometry
    apc = a.apc
    apc.begin(0, 0, 0)
    apc.record_prefill(0, 0, 0, prompt.reshape(-1), [MMSpan(100, 16, "image", "img-1")])
    apc.plan_snapshot(0, 0, 320, "decode")
    apc.release(0, 0)
    with_id = dict(mm, mm_identifiers=[["img-1"]])
    assert apc.lookup(prompt.reshape(-1), mm_spans_of_row(with_id, 0), 320) is not None
    assert apc.lookup(prompt.reshape(-1), mm_spans_of_row(dict(mm, mm_identifiers=[["img-2"]]), 0), 320) is None
    assert apc.lookup(prompt.reshape(-1), mm_spans_of_row(mm, 0), 320) is None, "no identifier: refused, not guessed"
    assert g.max_seq_len == 65536


def test_decode_parks_retained_slots_at_their_content_end():
    a, m = _apc_adapter(entries=1)
    L = 64
    prompt = (torch.arange(L) % 60 + 1).to(torch.int32).reshape(1, L)
    a.prefill_forward(prompt, prompt_lens=[L], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    s0 = a.slots.physical(0)
    assert [c for c in m.calls if c[0] == "snapshot_kda"] == [
        ("snapshot_kda", s0.user, 0)
    ], "64 is aligned: a prefill snapshot"
    a.release_request(0)
    # another request decodes: the retained slot is fed its content end (64) on every replica row, not position 0
    other = torch.full((1, 40), 5, dtype=torch.int32)
    a.prefill_forward(other, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[1], sampling_params=_greedy(1))
    _decode(a, [0, 7, 0, 0], [-1, 40, -1, -1])
    fed = m._fed_pos
    flats = [a.geometry.flat(r, s0.user) for r in range(a.geometry.rows)]
    assert all(fed[f] == 64 for f in flats), fed
    assert all(fed[f] == 0 for f in range(len(fed)) if f not in flats and f not in a.slots.flat_indices(1))
    # the retained slot's content is intact and the hit at 64 restores entry 0 at 64
    turn = torch.cat([prompt.reshape(-1), torch.full((10,), 3, dtype=torch.int32)]).reshape(1, 74)
    m.calls.clear()
    a.prefill_forward(turn, prompt_lens=[74], start_pos=torch.tensor([64]), empty_slots=[0], sampling_params=_greedy(1))
    assert [c for c in m.calls if c[0] in ("restore_kda", "prefill")] == [
        ("restore_kda", s0.user, 0, 64),
        ("prefill", s0.user, 64, 10),
    ]
    # teardown releases the store
    a.release_persistent_capture()
    assert ("release_snapshots",) in m.calls
    assert _next_token(1, 2) == _next_token(1, 2)


def test_apc_off_keeps_the_contiguity_error():
    model = _FakeApcModel(2, 4, 4, 4096)
    g = ServingGeometry(rows=2, cols=4, users_per_row=4, max_seq_len=4096, pad_token_id=PAD)
    a = GLM53FlashForCausalLM(model, g)
    assert a.apc is None
    prompt = torch.full((1, 100), 5, dtype=torch.int32)
    with expect_error(RuntimeError, "not contiguous"):
        a.prefill_forward(prompt, prompt_lens=[100], start_pos=torch.tensor([32]), empty_slots=[0])
    with expect_error(NotImplementedError, "snapshot_kda"):
        GLM53FlashForCausalLM(_FakeMeshModel(2, 4, 4, 4096), g, prefix_cache_entries=1)
    assert GLM53FlashForCausalLM(_FakeMeshModel(2, 4, 4, 4096), g, prefix_cache_entries=0).apc.entries == 0


def test_zz_no_ttnn_was_imported_by_the_suite():
    assert "ttnn" not in sys.modules
    assert pytest is not None


def test_confirm_prefix_cache_hit_returns_the_resume_position():
    """The plugin proposal's hook (doc/serving/prefix_caching.md s.5): the snapshot position a hit resumes from."""
    a, m = _apc_adapter(entries=2)
    ids = torch.arange(300, dtype=torch.int32) + 1
    apc = a.apc
    apc.begin(0, 0, 0)
    apc.record_prefill(0, 0, 0, ids, [MMSpan(10, 20, "image", "img-1")])
    for p in range(300, 320):
        apc.record_decode(0, 0, p, 50 + p)
    apc.plan_snapshot(0, 0, 320, "decode")
    apc.release(0, 0)
    prompt = apc.content(0, 0).tokens[:340].tolist()
    assert a.confirm_prefix_cache_hit(prompt, [("img-1", 10, 20, "image")], 320) == 320
    assert a.confirm_prefix_cache_hit(prompt, [("img-2", 10, 20, "image")], 320) == 0
    assert a.confirm_prefix_cache_hit(prompt, [(None, 10, 20, "image")], 320) == 0
    assert a.confirm_prefix_cache_hit(prompt, [("img-1", 10, 20, "image")], 288) == 0, "no snapshot at or below 288"
    assert a.confirm_prefix_cache_hit(prompt, [], 320) == 0, "the slot holds an image the request does not"
    assert a.confirm_prefix_cache_hit(prompt, [("img-1", 10, 20, "image")], 0) == 0
    b = GLM53FlashForCausalLM(_FakeApcModel(2, 4, 4, 4096), _geometry())
    assert b.confirm_prefix_cache_hit(prompt, [], 320) == 0, "APC off: never a hit"
