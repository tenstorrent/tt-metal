# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device) tests of the slot-level prefix cache (``doc/serving/prefix_caching.md``): the
per-slot content chains and multimodal spans (``tt/vllm_config.py::SlotContent``), the hit / miss decision with the
reused-slot downgrade and the KDA-position rule (``SlotPrefixCache``), the snapshot-ring policy and its planner
sizing (``tt/memory_plan.py``), the planner's tail split, and the adapter end to end on the fake mesh model of
``test_generator_vllm_host.py``: a multi-turn hit restores the decode-crossing snapshot and prefills from it, a
miss prefills from 0, a stripped multimodal hit is refused, retained slots park at their content end. The KDA
snapshot stores (``tt/kda_snapshot.py``) run on a fake ttnn namespace over torch tensors: the scratch-tensor plan,
the save / load data flow of both stores (ring entries and live slots compared row by row) and the rule that no
call allocates after the construction (the fake refuses allocations once frozen). The ``GLM53_VLLM_APC_ALLOC_CHECK``
accounting runs on the fake mesh model's footprint counter: a leaking save / restore / prefill / release is named.

    PYTHONPATH=<worktree>:<tt-metal>/ttnn:<tt-metal> <tt-metal>/python_env/bin/pytest \\
        models/demos/glm53_flash/tests/test_serving_prefix_cache_host.py --noconftest -p no:cacheprovider -q
"""

from __future__ import annotations

import os
import re
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
from models.demos.glm53_flash.tt.kda_snapshot import (
    DeviceKdaSnapshotRing,
    HostKdaSnapshotStore,
    SnapshotDims,
    mask_keep,
    one_hot_row,
)
from models.demos.glm53_flash.tt.memory_plan import MemoryPlan, kda_state_bytes_per_user, snapshot_bytes_per_chip
from models.demos.glm53_flash.tt.packed_inputs import RESIDENT_ENV
from models.demos.glm53_flash.tt.vllm_config import (
    APC_ALLOC_CHECK_ENV,
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
@pytest.fixture(autouse=True)
def _restore_at_any_aligned_snapshot(monkeypatch):
    """The mechanics below restore at 32-aligned snapshot positions (320, 2336, ...). The served default restores only
    at CHUNK-aligned snapshots (``GLM53_VLLM_APC_HIT_ALIGN`` unset = the device chunk: the continued chunk then
    starts where the warm-up's continued-chunk walk started, so its programs exist under the sealed cache); that
    policy is covered by ``test_hit_align_default_restores_only_at_chunk_aligned_snapshots``."""
    monkeypatch.setenv("GLM53_VLLM_APC_HIT_ALIGN", "32")


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
    # under per-row ownership the wave-affine and SP-first rules rank the same way: a fresh prompt takes a slot
    # without retained content before it evicts a resident prefix (served apc_r5 evicted every retained slot by
    # re-taking slot index 0 of rows 0..)
    g2 = _geometry(rows_independent=True)
    t3 = SlotTable(g2)
    first = t3.allocate(0, slot_rank=apc.slot_rank)
    assert (first.row, first.user) not in {(0, 0), (0, 2)}, "the retained slots of row 0 are not the first choice"
    taken = {(t3.allocate(i, slot_rank=apc.slot_rank).row, t3.physical(i).user) for i in range(1, 6)}
    assert (0, 0) not in taken and (0, 2) not in taken, "six prompts fit without evicting a resident prefix"
    # a repeated prompt re-prefills on the free slot that retains it (the same mesh row: its greedy text repeats
    # bit-exactly, rows differ in the last ulp); a live slot or an evicted content is no candidate
    assert apc.slot_holding(ids_b, [], t2) == (0, 0), "slot (0, 0) retains prompt B"
    assert apc.slot_holding(ids_a, [], t2) is None, "prompt A's content was evicted by B"
    assert apc.slot_holding(apc.content(0, 2).tokens[:300], [], t2) is None, "slot (0, 2) is live in t2"


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
        # the DRAM footprint the adapter's GLM53_VLLM_APC_ALLOC_CHECK reads (``GLM53MeshModel.dram_allocated``); a
        # call named in ``leak_on`` grows it by one byte (a buffer allocated under live traces that persists)
        self.dram_bytes = 1 << 20
        self.leak_on: set[str] = set()

    def dram_allocated(self) -> int:
        return self.dram_bytes

    def _leak(self, what: str) -> None:
        if what in self.leak_on:
            self.dram_bytes += 1

    def snapshot_kda(self, user, entry):
        self.calls.append(("snapshot_kda", user, entry))
        self._leak("snapshot_kda")

    def restore_kda(self, user, entry, position):
        self.calls.append(("restore_kda", user, entry, position))
        self.next_start[user] = position
        self._leak("restore_kda")

    def prefill(self, ids, user=0, start=0):
        if start and start != self.next_start[user]:
            raise AssertionError(f"user {user}: chunk start {start} != next_start {self.next_start[user]}")
        out = super().prefill(ids, user, start)
        self.next_start[user] = start + int(ids.numel())
        self._leak("prefill")
        return out

    def prefill_traced(self, ids, user=0):
        out = super().prefill_traced(ids, user)
        self.next_start[user] = int(ids.numel())
        self._leak("prefill")
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


def _decode(a, tokens, positions, reload=True, read=True):
    return a.decode_forward(
        torch.tensor([[t] for t in tokens], dtype=torch.int32),
        torch.tensor(positions),
        sampling_params=_greedy(len(tokens)),
        read_from_device=read,
        reload_inputs=reload,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )


@pytest.mark.parametrize("resident", [False, True])
def test_multi_turn_hit_restores_the_crossing_snapshot_and_prefills_from_it(monkeypatch, resident):
    # resident: the decode steps after the first are the plugin's chain (reload_inputs=False, the fed tokens ignored):
    # the content record and the crossing snapshot come from the ids read after each step
    monkeypatch.setenv(RESIDENT_ENV, "1" if resident else "0")
    a, m = _apc_adapter(entries=2)
    L = 300
    prompt = (torch.arange(L) % 60 + 1).to(torch.int32).reshape(1, L)
    a.prefill_forward(prompt, prompt_lens=[L], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    user = a.slots.physical(0).user
    assert not [c for c in m.calls if c[0] == "snapshot_kda"], "300 is not a 32-multiple: no prefill snapshot"
    # 20 decode steps: the slot crosses 320 at the step that consumes position 319 -> one decode snapshot (entry 0)
    tok = 5
    chain = [tok]
    for k, p in enumerate(range(L, L + 20)):
        out = _decode(a, [tok if not (resident and k) else 0, 0, 0, 0], [p, -1, -1, -1], reload=not (resident and k))
        tok = int(out[0, 0])
        chain.append(tok)
    assert [c for c in m.calls if c[0] == "snapshot_kda"] == [("snapshot_kda", user, 0)]
    content = a.apc.content(0, user)
    assert content.length == L + 20 and content.snapshots == {320: 0} and content.live
    assert content.tokens[L : L + 20].tolist() == chain[:20], "the record holds the tokens the steps consumed"
    assert a._perf_chain == (19 if resident else 0)
    end = L + 20
    if resident:
        # vLLM's async scheduling stages step N + 1 before step N's read lands: a chain step records nothing at
        # staging; the reads, landing in submission order, record the token each step consumed
        h1 = _decode(a, [0, 0, 0, 0], [L + 20, -1, -1, -1], reload=False, read=False)
        h2 = _decode(a, [0, 0, 0, 0], [L + 21, -1, -1, -1], reload=False, read=False)
        assert content.length == L + 20 and a.slots.physical(0).position == L + 22
        t21 = int(a.process_decode_output_host(h1, is_tokens=True)[0, 0])
        t22 = int(a.process_decode_output_host(h2, is_tokens=True)[0, 0])
        assert content.length == L + 22 and content.tokens[L + 20 : L + 22].tolist() == [chain[20], t21]
        assert t22 == _next_token(t21, L + 21) and a._perf_chain == 21
        end = L + 22
    a.release_request(0)
    assert content.retained and a.apc.park_positions() == {(0, user): end}
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
    # (2348 is unaligned, nothing crossed in decode) -> a miss again, re-prefilled from 0 on the slot that retains
    # this very prompt (the same mesh row: a repeated greedy request repeats its text)
    m.calls.clear()
    a.prefill_forward(
        prompt, prompt_lens=[L], start_pos=torch.tensor([2336]), empty_slots=[1], sampling_params=_greedy(1)
    )
    calls = [c for c in m.calls if c[0] in ("restore_kda", "prefill")]
    assert (
        calls[0][0] == "prefill" and calls[0][2] == 0 and calls[0][1] == user
    ), "the slot retaining the prompt is taken (slot affinity)"
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


def test_alloc_check_names_the_call_site_that_changed_the_footprint(monkeypatch):
    """``GLM53_VLLM_APC_ALLOC_CHECK=1``: the footprint after the decode capture is the baseline; a snapshot save or a
    hit's restore must leave the footprint as it found it, a prefill row or a release must return to the baseline;
    the first call site that differs raises with its name. Default off: nothing is read."""
    monkeypatch.setenv(APC_ALLOC_CHECK_ENV, "1")
    a, m = _apc_adapter(entries=2)
    assert a._apc_alloc_checking and a._apc_alloc_baseline == m.dram_bytes
    L = 300
    prompt = (torch.arange(L) % 60 + 1).to(torch.int32).reshape(1, L)

    def run_turn(logical: int, start: int = 0, steps: int = 20):
        ids = prompt if start == 0 else turn
        a.prefill_forward(
            ids,
            prompt_lens=[int(ids.numel())],
            start_pos=torch.tensor([start]),
            empty_slots=[logical],
            sampling_params=_greedy(1),
        )
        tok = 5
        for p in range(int(ids.numel()), int(ids.numel()) + steps):
            feed = [0] * 4
            pos = [-1] * 4
            feed[logical], pos[logical] = tok, p
            tok = int(_decode(a, feed, pos)[logical, 0])
        a.release_request(logical)

    run_turn(0)  # prefill, a decode crossing at 320 (a save), release: every check passes
    content = a.apc.retained_slots()[0]
    turn = torch.cat([a.apc.content(*content).tokens[:320].clone(), torch.full((40,), 9, dtype=torch.int32)]).reshape(
        1, 360
    )
    run_turn(2, start=320, steps=5)  # the hit's restore + its tail prefill pass too
    assert a.apc.stats["hits"] == 1 and m.dram_bytes == a._apc_alloc_baseline
    # a save that leaks: the next decode crossing is refused by name (the delta check: before vs after the save)
    m.leak_on = {"snapshot_kda"}
    with expect_error(RuntimeError, "the decode snapshot of user"):
        run_turn(1)
    assert m.dram_bytes == a._apc_alloc_baseline + 1
    with expect_error(RuntimeError, re.escape("release_request(1)")):  # the leak is still there: the release sees it
        a.release_request(1)
    m.dram_bytes = a._apc_alloc_baseline
    # a restore that leaks
    m.leak_on = {"restore_kda"}
    with expect_error(RuntimeError, "the hit's restore of entry"):
        a.prefill_forward(
            turn, prompt_lens=[360], start_pos=torch.tensor([320]), empty_slots=[3], sampling_params=_greedy(1)
        )
    m.dram_bytes = a._apc_alloc_baseline
    for key in a.apc.retained_slots():
        a.apc.forget(*key)
    a.slots.release(3)
    # a prefill row that leaks (a program compiled after the capture, a constant that was not a chunk transient);
    # the 300-token prompt takes the fake's traced prefill, the leak hook is on both prefill forms
    m.leak_on = {"prefill"}
    with expect_error(RuntimeError, re.escape("the prefill [0, 300) of slot 1")):
        a.prefill_forward(
            prompt, prompt_lens=[L], start_pos=torch.tensor([0]), empty_slots=[1], sampling_params=_greedy(1)
        )
    m.leak_on = set()
    m.dram_bytes = a._apc_alloc_baseline
    # a release after something else grew the footprint
    m.dram_bytes += 1
    with expect_error(RuntimeError, re.escape("release_request(1)")):
        a.release_request(1)
    # default off: the counter is never consulted
    monkeypatch.delenv(APC_ALLOC_CHECK_ENV, raising=False)
    b, mb = _apc_adapter(entries=2)
    assert not b._apc_alloc_checking and b._apc_alloc_baseline is None
    mb.leak_on = {"prefill", "snapshot_kda"}
    b.prefill_forward(prompt, prompt_lens=[L], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    assert mb.dram_bytes == (1 << 20) + 1


# ------------------------------------------------------------------------- KDA snapshot stores on fake ops
class _T:
    """A fake device / host tensor: a torch tensor behind ttnn's attribute surface."""

    def __init__(self, t: torch.Tensor, dtype, layout, device=None):
        self.t, self.dtype, self.layout, self.device = t, dtype, layout, device
        self._alloc = True

    @property
    def shape(self):
        return tuple(self.t.shape)

    def is_allocated(self) -> bool:
        return self._alloc


class _FakeOps:
    """The ttnn surface ``tt/kda_snapshot.py`` uses, on torch tensors. Every op that may allocate counts one
    allocation unless it is given its output (``output_tensor`` / ``optional_output_tensor``); after ``freeze()``
    (the trace capture) an allocation is an error: the allocation discipline the stores must keep. The one-hot
    selection matmul and the mask blend are exact in fp32, as HiFi4 makes them on the device."""

    float32, bfloat16, int32 = "float32", "bfloat16", "int32"
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = "TILE", "ROW_MAJOR"
    DRAM_MEMORY_CONFIG = "dram"
    MathFidelity = SimpleNamespace(HiFi4="HiFi4")
    Shape = tuple

    def __init__(self):
        self.allocations = 0
        self.frozen = False
        self.refreshes = 0
        self.compute_configs: list = []
        self.experimental = SimpleNamespace(view=self._view, paged_fill_cache=self._paged_fill_cache)

    def _allocate(self, what: str) -> None:
        if self.frozen:
            raise AssertionError(f"{what} allocated a device tensor after the trace capture")
        self.allocations += 1

    def freeze(self) -> None:
        self.frozen = True

    def init_device_compute_kernel_config(self, arch, **kw):
        self.compute_configs.append((arch, kw))
        return ("ckc", arch, tuple(sorted(kw.items())))

    def zeros(self, shape, dtype, layout, device, memory_config):
        self._allocate("zeros")
        return _T(
            torch.zeros(tuple(shape), dtype=torch.int32 if dtype == "int32" else torch.float32), dtype, layout, device
        )

    def from_torch(self, t, dtype, layout, device=None, memory_config=None, mesh_mapper=None):
        if device is not None:
            self._allocate("from_torch")
        return _T(t.clone().to(torch.int32 if dtype == "int32" else torch.float32), dtype, layout, device)

    def copy_host_to_device_tensor(self, host, dst) -> None:
        assert host.device is None and dst.device is not None and host.shape == dst.shape, (host.shape, dst.shape)
        dst.t.copy_(host.t)
        self.refreshes += 1

    def deallocate(self, t) -> None:
        t._alloc = False

    def to_torch(self, t) -> torch.Tensor:
        return t.t.clone()

    def slice(self, x, starts, ends, output_tensor=None):
        out = x.t[tuple(slice(a, b) for a, b in zip(starts, ends))]
        if output_tensor is None:
            self._allocate("slice")
            return _T(out.clone(), x.dtype, x.layout, x.device)
        assert output_tensor.shape == tuple(out.shape), ("slice", output_tensor.shape, tuple(out.shape))
        output_tensor.t.copy_(out)
        return output_tensor

    def matmul(self, a, b, compute_kernel_config=None, optional_output_tensor=None):
        assert compute_kernel_config is not None and "HiFi4" in str(compute_kernel_config), compute_kernel_config
        out = torch.matmul(a.t.float(), b.t.float())
        if optional_output_tensor is None:
            self._allocate("matmul")
            return _T(out, a.dtype, a.layout, a.device)
        assert optional_output_tensor.shape == tuple(out.shape), ("matmul", optional_output_tensor.shape, out.shape)
        optional_output_tensor.t.copy_(out)
        return optional_output_tensor

    def _binary(self, name, fn, a, b, output_tensor):
        out = fn(a.t, b.t)  # torch broadcasting = ttnn's
        if output_tensor is None:
            self._allocate(name)
            return _T(out, a.dtype, a.layout, a.device)
        assert output_tensor.shape == tuple(out.shape), (name, output_tensor.shape, tuple(out.shape))
        output_tensor.t.copy_(out)
        return output_tensor

    def multiply(self, a, b, output_tensor=None):
        return self._binary("multiply", torch.mul, a, b, output_tensor)

    def add(self, a, b, output_tensor=None):
        return self._binary("add", torch.add, a, b, output_tensor)

    def _view(self, x, shape):
        return _T(x.t.view(tuple(shape)), x.dtype, x.layout, x.device)  # shares the storage: no allocation

    def _paged_fill_cache(self, cache, inp, page_table, batch_idx_tensor=None):
        """cache [P, 1, 32, V], input [1, 1, S, V]: 32-row block j of the input -> page page_table[batch][j]."""
        b = int(batch_idx_tensor.t.reshape(-1)[0])
        S = inp.t.shape[2]
        assert S % 32 == 0 and cache.t.shape[1:3] == (1, 32), (inp.shape, cache.shape)
        for j in range(S // 32):
            cache.t[int(page_table.t[b, j]), 0] = inp.t[0, 0, j * 32 : (j + 1) * 32]
        return cache


def _fake_kda_model(ops: _FakeOps, B=4, H=2, K=32, V=32, taps=3, W=64, layers=2, tiled=True):
    """Two KDA layers' batched states on the fake ops (``[B, H, K, V]`` fp32, ``[taps, B, W]`` bf16 tiles)."""
    device = SimpleNamespace(arch=lambda: "fake-arch")
    mesh = SimpleNamespace(
        device=device,
        is_single=True,
        replicate=lambda: "replicate",
        shard_2d=lambda r, c: ("shard", r, c),
        device_tensors=lambda t: [t],
    )

    def layer():
        rec = ops.zeros((B, H, K, V), "float32", "TILE", device, "dram")
        conv = ops.zeros((taps, B, W), "bfloat16", "TILE", device, "dram")
        attn = SimpleNamespace(conv_tiled=tiled, kcfg=SimpleNamespace(conv_kernel_size=taps + 1), H=H, K=K, V=V)
        return SimpleNamespace(
            kda_batched=SimpleNamespace(recurrent=rec, convolution=conv),
            _require_weights=lambda: SimpleNamespace(attn=attn),
        )

    return SimpleNamespace(mesh=mesh, max_batch=B, layers=[layer() for _ in range(layers)])


def _randomize(model, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for dec in model.layers:
        st = dec.kda_batched
        st.recurrent.t.copy_(torch.randn(st.recurrent.shape, generator=g))
        st.convolution.t.copy_(torch.randn(st.convolution.shape, generator=g))
        out.append((st.recurrent.t.clone(), st.convolution.t.clone()))
    return out


def test_snapshot_dims_and_scratch_plan():
    ops = _FakeOps()
    d = SnapshotDims.from_model(_fake_kda_model(ops), 2)
    assert (d.users, d.entries, d.E, d.nblk, d.taps, d.W, d.H, d.K, d.V) == (4, 2, 8, 2, 3, 64, 2, 32, 32)
    assert d.entry(1, 0) == 2 and d.entry(3, 1) == 7
    with expect_error(ValueError, "outside"):
        d.entry(4, 0)
    with expect_error(ValueError, "outside"):
        d.entry(0, 2)
    host = d.plan(ring=False)
    ring = d.plan(ring=True)
    assert set(host) == {
        "rec_row",
        "conv_row",
        "conv_live_rows",
        "sel_user",
        "user_mask",
        "user_keep",
        "user_idx",
        "live_pt",
    }
    assert set(ring) - set(host) == {"conv_ring_rows", "sel_entry", "ring_mask", "ring_keep", "entry_idx", "ring_pt"}
    assert ring["rec_row"] == ((1, 2, 32, 32), "float32", "TILE")
    assert ring["conv_row"] == ((3, 1, 64), "bfloat16", "TILE") and ring["conv_live_rows"] == (
        (3, 4, 64),
        "bfloat16",
        "TILE",
    )
    assert ring["sel_user"] == ((3, 1, 4), "bfloat16", "TILE") and ring["sel_entry"] == ((3, 1, 8), "bfloat16", "TILE")
    assert ring["user_mask"] == ((3, 4, 1), "bfloat16", "TILE") and ring["ring_keep"] == ((3, 8, 1), "bfloat16", "TILE")
    assert ring["live_pt"] == ((4, 2), "int32", "ROW_MAJOR") and ring["ring_pt"] == ((8, 2), "int32", "ROW_MAJOR")
    assert ring["user_idx"] == ((1,), "int32", "ROW_MAJOR") and ring["conv_ring_rows"] == (
        (3, 8, 64),
        "bfloat16",
        "TILE",
    )
    assert d.ring_entry_shapes() == ((8, 2, 32, 32), (3, 8, 64))
    sel, (mask, keep) = one_hot_row(3, 4, 2), mask_keep(3, 4, 2)
    assert sel.shape == (3, 1, 4) and sel[:, 0, 2].tolist() == [1.0] * 3 and float(sel.sum()) == 3.0
    assert torch.equal(mask + keep, torch.ones(3, 4, 1)) and mask[:, 2, 0].tolist() == [1.0] * 3
    with expect_error(ValueError, "entries"):
        SnapshotDims.from_model(_fake_kda_model(_FakeOps()), 0)
    with expect_error(NotImplementedError, "block KDA decode"):
        SnapshotDims.from_model(_fake_kda_model(_FakeOps(), tiled=False), 1)
    with expect_error(RuntimeError, "no KDA layer"):
        SnapshotDims.from_model(SimpleNamespace(layers=[], max_batch=4, mesh=None), 1)


def test_device_ring_saves_and_restores_slots_without_allocating():
    ops = _FakeOps()
    m = _fake_kda_model(ops)
    ring = DeviceKdaSnapshotRing(m, 2, ops=ops)
    assert ops.allocations == 2 * 2 + len(ring.plan) + 2 * 2, "the live states, the plan, the ring entries per layer"
    assert ops.compute_configs == [
        ("fake-arch", dict(math_fidelity="HiFi4", math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
    ]
    assert ring.t["live_pt"].t.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]] and ring.t["ring_pt"].t.shape == (8, 2)
    ops.freeze()  # the decode capture: from here on nothing may allocate
    live = _randomize(m, 1)
    ring.save(1, 0)  # flat entry 2
    ring.save(3, 1)  # flat entry 7
    for k in range(2):
        rec, conv = live[k]
        assert torch.equal(ring.rec[k].t[2], rec[1]) and torch.equal(ring.conv[k].t[:, 2, :], conv[:, 1, :])
        assert torch.equal(ring.rec[k].t[7], rec[3]) and torch.equal(ring.conv[k].t[:, 7, :], conv[:, 3, :])
        others = [e for e in range(8) if e not in (2, 7)]
        assert not ring.rec[k].t[others].any() and not ring.conv[k].t[:, others, :].any(), "untouched entries stay zero"
        st = m.layers[k].kda_batched
        assert torch.equal(st.recurrent.t, rec) and torch.equal(
            st.convolution.t, conv
        ), "a save reads the live state only"
    # the decode moved every slot; the load restores slot 1 from entry (1, 0) and leaves the other slots alone
    moved = _randomize(m, 2)
    ring.load(1, 0)
    for k in range(2):
        st = m.layers[k].kda_batched
        assert torch.equal(st.recurrent.t[1], live[k][0][1]) and torch.equal(
            st.convolution.t[:, 1, :], live[k][1][:, 1, :]
        )
        for u in (0, 2, 3):
            assert torch.equal(st.recurrent.t[u], moved[k][0][u]) and torch.equal(
                st.convolution.t[:, u, :], moved[k][1][:, u, :]
            )
    # the selectors are rewritten only when the target changes
    n = ops.refreshes
    ring.save(1, 0)
    assert ops.refreshes == n
    ring.save(2, 1)
    assert ops.refreshes == n + 8, "user + entry selectors (4 tensors each)"
    with expect_error(ValueError, "outside"):
        ring.save(0, 2)
    ring.release()
    assert not ring.rec and not ring.t and ops.frozen


def test_host_store_round_trips_through_the_scratch_without_allocating():
    ops = _FakeOps()
    m = _fake_kda_model(ops)
    store = HostKdaSnapshotStore(m, 1, ops=ops)
    assert "ring_pt" not in store.t and ops.allocations == 2 * 2 + len(store.plan)
    ops.freeze()
    live = _randomize(m, 3)
    store.save(2, 0)
    rec_host, conv_host = store.data[(2, 0)][0]
    assert rec_host.shape == (1, 1, 2, 32, 32) and conv_host.shape == (1, 3, 1, 64), "per chip: [chips, *row shape]"
    assert torch.equal(rec_host[0, 0], live[0][0][2]) and torch.equal(conv_host[0, :, 0, :], live[0][1][:, 2, :])
    moved = _randomize(m, 4)
    store.load(2, 0)
    for k in range(2):
        st = m.layers[k].kda_batched
        assert torch.equal(st.recurrent.t[2], live[k][0][2]) and torch.equal(
            st.convolution.t[:, 2, :], live[k][1][:, 2, :]
        )
        for u in (0, 1, 3):
            assert torch.equal(st.recurrent.t[u], moved[k][0][u]) and torch.equal(
                st.convolution.t[:, u, :], moved[k][1][:, u, :]
            )
    with expect_error(KeyError, "no host snapshot"):
        store.load(1, 0)
    with expect_error(ValueError, "outside"):
        store.save(0, 1)
    store.release()
    assert not store.data and not store.t


def test_hit_align_default_restores_only_at_chunk_aligned_snapshots(monkeypatch):
    """``GLM53_VLLM_APC_HIT_ALIGN`` unset = the device chunk: a multi-turn hit is served from the highest CHUNK-aligned
    snapshot at or below vLLM's hit (the eager prefill's chunk-end snapshot), never from an unaligned decode crossing
    (recorded, kept for a later aligned restore point); ``0`` = any 32-aligned snapshot (the mechanics above)."""
    monkeypatch.delenv("GLM53_VLLM_APC_HIT_ALIGN", raising=False)
    g = _geometry(max_seq_len=65536)
    chunk = g.device_prefill_chunk
    apc = SlotPrefixCache(g, entries=2)
    assert apc.hit_align == chunk
    ids = torch.arange(chunk + 300, dtype=torch.int32) % 1000 + 1
    apc.begin(0, 0, 0)
    apc.record_prefill(0, 0, 0, ids)
    assert apc.plan_snapshot(0, 0, chunk, "prefill") == 1  # the eager prefill's chunk-end snapshot
    for p in range(chunk + 300, chunk + 320):
        apc.record_decode(0, 0, p, 50 + p)
    # a decode crossing unaligned to the chunk is not saved (it could never be restored and, with one ring entry,
    # would evict the chunk-end snapshot: served apc_r2 2026-10-29); a chunk-aligned crossing is entry 0
    assert apc.plan_snapshot(0, 0, chunk + 320, "decode") is None
    assert apc.plan_snapshot(0, 0, 2 * chunk, "decode") == 0 and apc.content(0, 0).snapshots == {chunk: 1, 2 * chunk: 0}
    del apc.content(0, 0).snapshots[2 * chunk]
    apc.release(0, 0)
    turn = torch.cat([apc.content(0, 0).tokens[: chunk + 320].clone(), torch.full((40,), 9, dtype=torch.int32)])
    hit = apc.lookup(turn, [], chunk + 320)
    assert hit is not None and (hit.position, hit.entry) == (chunk, 1), "restored at the chunk end, not at 320 past it"
    assert apc.lookup(turn, [], 320) is None, "a hit below the first chunk-aligned snapshot is a miss"
    monkeypatch.setenv("GLM53_VLLM_APC_HIT_ALIGN", "0")
    any_pos = SlotPrefixCache(g, entries=2)
    assert any_pos.hit_align == 0


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
