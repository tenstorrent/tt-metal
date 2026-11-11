# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only checks of the resident adapter's KV aliasing contract.

g2: vLLM tables -> ttnn prefill tables and resident engine tables, randomized over
vLLM-shaped allocation states. g3 (host half): the generator's serving validation accepts
the aliased geometry. g7 (lint half): no ttnn.deallocate reaches a KV tensor or view.
"""

import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt.generator import CacheState, Gemma4Generator
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import Gemma4ResidentForCausalLM, layer_groups

TT = Path(__file__).resolve().parents[1] / "tt"
KINDS = ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)]
SLIDING_PAGES = 1024 // 128 + 1


def layers():
    return [
        SimpleNamespace(
            kind=k,
            sliding=k == "sliding_attention",
            kv_heads=4 if k == "sliding_attention" else 1,
            head_dim=256 if k == "sliding_attention" else 512,
        )
        for k in KINDS
    ]


def adapter(blocks, batch, capacity):
    model = object.__new__(Gemma4ResidentForCausalLM)
    model.generator = SimpleNamespace(model=SimpleNamespace(layers=layers()))
    model.groups = layer_groups(KINDS)
    model.blocks, model.max_batch_size, model.max_seq_len = blocks, batch, capacity
    model.cache = SimpleNamespace(
        batch=batch,
        kv=[None] * 60,
        page_tables={i: torch.zeros(batch, capacity // 128, dtype=torch.int32) for i in range(60)},
    )
    return model


def test_layer_groups_match_vllm_grouping():
    """The adapter's group map equals vLLM 0.31's grouping of the real Gemma4 31B KV spec."""
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_page_size

    from models.demos.gemma4_31b_qb2.tt.model import checkpoint_path
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(checkpoint_path(), local_files_only=True).text_config
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False, block_size=128),
        max_in_flight_tokens=8192,
        model_config=SimpleNamespace(hf_text_config=config),
    )
    specs = Gemma4ResidentForCausalLM.get_kv_cache_spec(vllm_config)
    groups = _get_kv_cache_groups_uniform_page_size(specs)
    vllm_group = {}
    for g, group in enumerate(groups):
        for name in group.layer_names:
            vllm_group[int(name.split(".")[2])] = g
    assert layer_groups(list(config.layer_types)) == [vllm_group[i] for i in range(60)]
    assert layer_groups(list(config.layer_types)) == [i % 6 for i in range(60)]


class FakeVllm:
    """Shared block pool with vLLM's conventions: block 0 is the null block, groups get
    disjoint ids, sliding blocks wholly outside the window return to the pool as 0 entries,
    and the runner zeroes columns past each row's live block count."""

    def __init__(self, blocks, batch, columns, rng):
        self.free = list(range(1, blocks))
        rng.shuffle(self.free)
        self.batch, self.columns, self.rng = batch, columns, rng
        self.rows = {}

    def admit(self, slot, length):
        self.rows[slot] = {"length": 0, "ids": [[] for _ in range(6)]}
        return self.grow(slot, length)

    def grow(self, slot, length):
        row = self.rows[slot]
        need = min((length + 127) // 128, self.columns)
        for g in range(6):
            while len(row["ids"][g]) < need:
                if not self.free:
                    return False
                row["ids"][g].append(self.free.pop())
        row["length"] = length
        evicted = max(0, (length - 1024) // 128)
        for g in range(5):
            for c in range(min(evicted, len(row["ids"][g]))):
                if row["ids"][g][c]:
                    self.free.append(row["ids"][g][c])
                    row["ids"][g][c] = 0
        return True

    def release(self, slot):
        row = self.rows.pop(slot)
        self.free.extend(b for ids in row["ids"] for b in ids if b)

    def group_tables(self):
        tables = torch.zeros(6, self.batch, self.columns, dtype=torch.int32)
        for slot, row in self.rows.items():
            for g in range(6):
                tables[g, slot, : len(row["ids"][g])] = torch.tensor(row["ids"][g], dtype=torch.int32)
        return tables

    def per_layer(self, groups, *, copies):
        tables = self.group_tables()
        shared = [tables[g].clone() for g in range(6)]
        # Padded runner output gives each layer its own tensor; unpadded output shares one per group.
        return [shared[g].clone() if copies else shared[g] for g in groups]


def check_translation(model, vllm, slots):
    blocks, batch = model.blocks, model.max_batch_size
    group_tables = vllm.group_tables()
    source = vllm.per_layer(model.groups, copies=bool(random.getrandbits(1)))
    tables = model._tables(model.cache, source)
    resident = model._resident_tables(tables)
    assert resident.shape == (batch, 6, model.max_seq_len // 128) and resident.dtype == torch.int32
    for i, group in enumerate(model.groups):
        expected = group_tables[group] * (2 if group == 5 else 1)
        assert torch.equal(tables[i], expected), f"layer {i}"
    full = resident[:, 5]
    assert bool((full % 2 == 0).all()), "a full layer touched an odd page"
    assert bool((full >= 0).all() and (full < 2 * (blocks + batch)).all())
    assert bool((resident[:, :5] >= 0).all() and (resident[:, :5] < blocks + batch).all())
    for slot in range(batch):
        row = vllm.rows.get(slot)
        for g in range(6):
            scratch = (2 if g == 5 else 1) * (blocks + slot)
            ids = row["ids"][g] if row else []
            got = resident[slot, g]
            for c, b in enumerate(ids):
                if g == 5:
                    assert int(got[c]) == 2 * b, (slot, g, c)
                elif b or any(ids[c:]):
                    # Live blocks and evicted (0) blocks before the live tail keep their entries.
                    assert int(got[c]) == b, (slot, g, c)
                else:
                    assert int(got[c]) == scratch, (slot, g, c)
            assert bool((got[len(ids) :] == scratch).all()), "unallocated column must be the slot's scratch page"
            # Every scratch use belongs to this slot and no live block becomes scratch.
            others = [(2 if g == 5 else 1) * (blocks + j) for j in range(batch) if j != slot]
            assert not bool(torch.isin(got, torch.tensor(others, dtype=torch.int32)).any())
    # Prefill: compact rows scattered into their slots.
    if slots:
        rows = torch.stack([s for s in vllm.group_tables()])[:, slots]
        compact = [rows[g].clone() for g in model.groups]
        prefill = model._tables(model.cache, compact, slots=slots)
        for i, group in enumerate(model.groups):
            for r, slot in enumerate(slots):
                assert torch.equal(prefill[i][slot], rows[group][r] * (2 if group == 5 else 1))
            assert bool((prefill[i] % 2 == 0).all()) or group != 5


@pytest.mark.parametrize("seed", range(40))
def test_tables_property(seed):
    rng = random.Random(seed)
    random.seed(seed)
    batch = rng.choice([1, 1, 4, 32])
    capacity = rng.choice([4096, 32768])
    blocks = rng.choice([64, 400, 4320])
    model = adapter(blocks, batch, capacity)
    vllm = FakeVllm(blocks, batch, capacity // 128, rng)
    for _ in range(60):
        live = list(vllm.rows)
        action = rng.random()
        if action < 0.35 and len(live) < batch:
            slot = rng.choice([s for s in range(batch) if s not in vllm.rows])
            if not vllm.admit(slot, rng.randint(1, capacity)):
                vllm.release(slot)
                continue
            check_translation(model, vllm, [slot])
        elif action < 0.8 and live:
            slot = rng.choice(live)
            length = min(capacity, vllm.rows[slot]["length"] + rng.choice([1, 1, 127, 128, 129, 2048]))
            if not vllm.grow(slot, length):
                vllm.release(slot)
            check_translation(model, vllm, [])
        elif live:
            vllm.release(rng.choice(live))
            check_translation(model, vllm, [])


def test_negative_entries_and_empty_rows_go_to_scratch():
    model = adapter(blocks=10, batch=2, capacity=1024)
    source = torch.full((2, 8), -1, dtype=torch.int32)
    source[0, :3] = torch.tensor([4, 5, 6])
    resident = model._resident_tables(model._tables(model.cache, [source] * 60))
    assert resident[0, 0].tolist() == [4, 5, 6] + [10] * 5
    assert resident[0, 5].tolist() == [8, 10, 12] + [20] * 5
    assert resident[1, 0].tolist() == [11] * 8 and resident[1, 5].tolist() == [22] * 8


def test_evicted_sliding_prefix_stays_zero():
    model = adapter(blocks=40, batch=1, capacity=4096)
    sliding = torch.zeros(1, 32, dtype=torch.int32)
    sliding[0, 7:17] = torch.arange(20, 30)
    full = torch.zeros(1, 32, dtype=torch.int32)
    full[0, :17] = torch.arange(1, 18)
    source = [full if i % 6 == 5 else sliding for i in range(60)]
    resident = model._resident_tables(model._tables(model.cache, source))
    assert resident[0, 0, :7].tolist() == [0] * 7
    assert resident[0, 0, 7:17].tolist() == list(range(20, 30))
    assert resident[0, 0, 17:].tolist() == [40] * 15
    assert resident[0, 5, :17].tolist() == [2 * b for b in range(1, 18)]
    assert resident[0, 5, 17:].tolist() == [80] * 15


def test_layers_of_one_group_must_agree(expect_error):
    model = adapter(blocks=10, batch=1, capacity=1024)
    source = [torch.zeros(1, 8, dtype=torch.int32) for _ in range(60)]
    source[12][0, 0] = 3  # layer 12 is group 0, like layers 0, 6, ...
    with expect_error(ValueError, "does not share the page table"):
        model._resident_tables(model._tables(model.cache, source))


def test_get_max_tokens_all_users_budget():
    # c32 at 262K: 4288 blocks + the runner's 32 output pages = 4320; +32 scratch = 4352 pages per pool.
    tokens = Gemma4ResidentForCausalLM.get_max_tokens_all_users(max_num_seqs=32, max_model_len=262144)
    assert tokens // 128 + 32 + 32 == 4352
    assert Gemma4ResidentForCausalLM.get_max_tokens_all_users(max_num_seqs=1, max_model_len=32768) == 121088


class FakeTensor:
    def __init__(self, shape, owner, mesh):
        self.shape, self.owner, self.mesh = shape, owner, mesh
        self.dtype, self.layout = ttnn.bfloat8_b, ttnn.TILE_LAYOUT

    def device(self):
        return self.mesh

    def memory_config(self):
        return ttnn.DRAM_MEMORY_CONFIG

    def buffer_unique_id(self):
        return self.owner


def aliased_state(blocks, batch, capacity):
    mesh = object()
    generator = object.__new__(Gemma4Generator)
    generator.mesh, generator.max_seq_len = mesh, 262144
    generator.model = SimpleNamespace(layers=layers())
    pages = blocks + batch
    kv, scratch = [], {}
    for i, layer in enumerate(generator.model.layers):
        pool = i // 6
        if layer.sliding:
            kv.append((FakeTensor((pages, 4, 128, 256), 2 * pool, mesh), FakeTensor((pages, 4, 128, 256), 2 * pool + 1, mesh)))
            scratch[i] = list(range(blocks, pages))
        else:
            # A view is a second buffer object over the root's address.
            kv.append((FakeTensor((2 * pages, 1, 128, 512), 100 + 2 * pool, mesh), FakeTensor((2 * pages, 1, 128, 512), 101 + 2 * pool, mesh)))
            scratch[i] = [2 * p for p in range(blocks, pages)]
    tables = {i: torch.zeros(batch, capacity // 128, dtype=torch.int32) for i in range(60)}
    return generator, CacheState(kv, tables, batch, capacity, scratch_pages=scratch, vllm_owned=True)


def test_serving_validation_accepts_aliased_pairs(expect_error):
    blocks, batch, capacity = 4320, 32, 262144
    generator, state = aliased_state(blocks, batch, capacity)
    model = adapter(blocks, batch, capacity)
    model.cache = state
    vllm = FakeVllm(blocks, batch, capacity // 128, random.Random(7))
    for slot in range(batch):
        assert vllm.admit(slot, 1 + slot * 100)
    tables = model._tables(state, vllm.per_layer(model.groups, copies=False))
    generator._validate_cache(state, tables)
    # The full layers' highest block maps inside the view and below its scratch pages.
    worst = {i: t.clone() for i, t in tables.items()}
    worst[5][0, 0] = 2 * (blocks - 1)
    generator._validate_cache(state, worst)
    worst[5][0, 0] = 2 * blocks
    with expect_error(ValueError, "private warmup storage"):
        generator._validate_cache(state, worst)


N11 = re.compile(r"ttnn\.deallocate\(\s*([^)]*)\)")


def test_no_forced_deallocate_reaches_kv():
    """N11: ttnn.deallocate defaults to force=True and frees a view's shared root buffer."""
    assert not N11.findall((TT / "generator_resident_vllm.py").read_text())
    for name in ("generator.py", "generator_vllm.py"):
        targets = N11.findall((TT / name).read_text())
        assert all(t.strip() == "saved_seeds" for t in targets), (name, targets)


def test_unaliased_pools_keep_vllm_full_pages():
    """The equivalence tests' unaliased layout (full layers on their own tensors) translates nothing."""
    model = adapter(blocks=10, batch=1, capacity=1024)
    model.cache.full_page_scale = 1
    source = [torch.tensor([[3, 4, 0, 0, 0, 0, 0, 0]], dtype=torch.int32) for _ in range(60)]
    tables = model._tables(model.cache, source)
    assert tables[5].tolist() == [[3, 4, 0, 0, 0, 0, 0, 0]]
    assert model._resident_tables(tables)[0, 5].tolist() == [3, 4] + [10] * 6


class GuardEngine:
    """The engine surface the L1 guard and a decode step touch. Records the call order."""

    def __init__(self, top):
        self.top, self.rebuilt, self.rows, self.status, self.tokens, self.calls = top, [], None, object(), object(), []

    def cb_top(self):
        return self.top

    def rebuild(self, l1_top):
        self.calls.append("rebuild")
        self.rebuilt.append(l1_top)
        self.top = l1_top - 4096

    def read_tokens(self, n):
        return torch.tensor([7, 9][:n])

    def write_rows(self, slots, positions, tokens):
        self.calls.append("write_rows")
        self.rows = (list(slots), list(positions), list(tokens))

    def write_tables(self, tables):
        self.calls.append("write_tables")

    def set_mode(self, feed, write_logits):
        self.calls.append("set_mode")

    def launch(self, rows):
        self.calls.append("launch")


def guard_model(monkeypatch, floor, checked, bad_status=False):
    model = object.__new__(Gemma4ResidentForCausalLM)
    model.engine, model.max_batch_size, model.max_seq_len = GuardEngine(top=1_327_504), 2, 1024
    model.positions, model.mode, model.feed_stale, model.l1_checks = [5, -1], ("external", True), False, []
    model.sampling_trace = "captured"

    def release():
        model.engine.calls.append("release")
        model.sampling_trace = None

    def capture():
        model.engine.calls.append("capture")
        model.sampling_trace = "recaptured"

    monkeypatch.setattr(model, "_release_sampling_trace", release)
    monkeypatch.setattr(model, "_capture_sampling", capture)

    def check_status(status):
        checked.append(status)
        model.engine.calls.append("check_status")
        if bad_status:
            raise RuntimeError("status flags")

    monkeypatch.setattr(model, "_check_status", check_status)
    monkeypatch.setattr(model, "_l1_floor", lambda: floor)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [SimpleNamespace(cpu=lambda: t)])
    return model


@pytest.mark.parametrize("floor", [2_000_000, 1_200_000])
@pytest.mark.parametrize("bad_status", [False, True])
def test_guard_after_lazy_sampling_capture(monkeypatch, floor, bad_status):
    """The lazy capture is guarded like a novel prefill shape. A rebuild (its warmup zeroes the feed and
    status) checks the last step's status first, then writes this step's feed back at its positions."""
    checked = []
    model = guard_model(monkeypatch, floor, checked, bad_status)
    violates = floor < 1_327_504
    if violates and bad_status:
        with pytest.raises(RuntimeError, match="status flags"):
            model._guard_after_capture()
        assert model.engine.rebuilt == [] and model.engine.rows is None
        return
    model._guard_after_capture()
    assert model.l1_checks[0] == ("sampling trace capture", floor, 1_327_504)
    assert len(model.l1_checks) == (2 if violates else 1)  # the re-capture is guarded again
    assert model.engine.rebuilt == ([floor] if violates else [])
    # A rebuild allocates after the capture: the trace is released and captured again before the launch.
    assert model.engine.calls == (["check_status", "rebuild", "release", "write_rows", "capture"] if violates else [])
    assert model.sampling_trace == ("recaptured" if violates else "captured")
    assert model.engine.rows == (([0, 1], [5, -1], [7, 0]) if violates else None)
    assert not model.feed_stale and model.mode == (None if violates else ("external", True))


def test_guard_raises_when_two_rebuilds_release_the_trace(monkeypatch):
    """M1.2's bound: a rebuild that leaves the L1 floor below the CB top again releases the re-captured sampling
    trace a second time. The server then stops with an error instead of serving on a trace a replay could clobber
    (resident-verifier's question Q2: intended, and tested here)."""
    # The reachable sequence (resident-verifier): each rebuild fits under the floor it was given (GuardEngine.rebuild
    # leaves the top 4 KB below it), and each re-capture then lowers the L1 floor below the new top, so the next
    # guard rebuilds again and releases the trace again.
    floor = {"value": 1_200_000}
    model = guard_model(monkeypatch, 0, [])
    monkeypatch.setattr(model, "_l1_floor", lambda: floor["value"])

    def capture_lowering_the_floor():
        model.engine.calls.append("capture")
        model.sampling_trace = "recaptured"
        floor["value"] -= 50_000  # the capture allocated L1 below the rebuilt engine's top

    monkeypatch.setattr(model, "_capture_sampling", capture_lowering_the_floor)
    with pytest.raises(RuntimeError, match="twice in a row"):
        model._guard_after_capture()
    calls = model.engine.calls
    assert calls.count("rebuild") == 2 and calls.count("release") == 2 and calls.count("capture") == 2
    assert len(model.l1_checks) == 2 and model.engine.rebuilt == [1_200_000, 1_150_000]


def test_sampling_trace_is_captured_before_the_launch(monkeypatch):
    """A capture after the launch waits for the step, so the step returns late and the previous step's
    token (a novel prefill's first token) is emitted with this one. Capture, guard, launch, replay."""
    model = guard_model(monkeypatch, 2_000_000, [])
    model.generator = SimpleNamespace(model=SimpleNamespace(layers=layers()), configure_sampling=lambda **k: None)
    model.cache = SimpleNamespace(prefill_history={})
    model.prefill_slot_keys, model.sampling_trace, model.ROWS, model.mode = {}, None, (1,), None
    tables = [torch.zeros(2, 8, dtype=torch.int32) for _ in KINDS]
    monkeypatch.setattr(model, "_tables", lambda cache, source: tables)
    monkeypatch.setattr(model, "_resident_tables", lambda t: t)
    monkeypatch.setattr(model, "_sampling", lambda *a, **k: {})
    calls = model.engine.calls

    def capture():
        calls.append("capture")
        model.sampling_trace = "trace"

    monkeypatch.setattr(model, "_capture_sampling", capture)
    monkeypatch.setattr(model, "_guard_after_capture", lambda: calls.append("guard"))
    monkeypatch.setattr(model, "_replay_sampling", lambda: calls.append("replay"))
    sampled = SimpleNamespace(temperature=[1.0, 1.0])
    for _ in range(2):
        model.decode_forward(torch.tensor([[3], [0]]), torch.tensor([5, -1]), None, model.cache,
                             sampling_params=sampled, read_from_device=False)
    assert calls == ["write_tables", "write_rows", "capture", "guard", "set_mode", "launch", "replay",
                     "write_tables", "write_rows", "launch", "replay"]


# ---------------------------------------------------------------- greedy picks one token on both paths (M2 review)

def device_argmax(raw):
    """The engine's argmax over a bf16 row: the largest order key (key_of, streamer_common.hpp), and on equal keys the
    lower vocab index (the hub's `better`, hub_dm.cpp)."""
    bits = raw.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    keys = torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits | 0x8000)
    return int((keys == keys.max()).nonzero()[0])


def capped_tie():
    """Two distinct bf16 raw logits whose bf16-capped values are equal: the tie the bf16-capped host path had."""
    v = torch.unique(torch.arange(12.0, 60.0, 0.0625).to(torch.bfloat16))
    capped = (30.0 * torch.tanh(v.float() / 30.0)).to(torch.bfloat16)
    i = next(i for i in range(len(v) - 1) if capped[i] == capped[i + 1])
    return v[i], v[i + 1]


def host_path_model(monkeypatch, raw, has_raw=True):
    model = object.__new__(Gemma4ResidentForCausalLM)
    capped_bf16 = (30.0 * torch.tanh(raw.float() / 30.0)).to(torch.bfloat16)
    model.engine = SimpleNamespace(logits="CAPPED", **({"logits_raw": "RAW"} if has_raw else {}))
    rows = {"RAW": raw.float(), "CAPPED": capped_bf16.float()}
    monkeypatch.setattr(model, "_read_logit_rows", lambda logits, n: rows[logits][:n])
    return model, capped_bf16


def test_host_greedy_equals_device_greedy_on_a_capped_tie(monkeypatch):
    """A greedy request picks the same token with and without logprobs. Without, the device argmax picks (raw bf16,
    lower index on ties); with, the plugin's host greedy takes argmax of `_host_logits`. Two raw logits that differ
    but tie once capped in bf16, the larger at the higher index: the bf16-capped rows picked the lower index (the
    M1.1 behaviour, prose 108 / math 90); the fp32 cap of the raw rows picks the device's token."""
    lo, hi = capped_tie()
    raw = torch.full((1, 64), -3.0, dtype=torch.bfloat16)
    raw[0, 10], raw[0, 40] = lo, hi
    model, capped_bf16 = host_path_model(monkeypatch, raw)
    assert device_argmax(raw[0]) == 40
    assert int(torch.argmax(capped_bf16[0].float())) == 10  # the old host path's pick
    assert int(torch.argmax(model._host_logits(1)[0])) == 40
    # an exact raw tie stays a tie: both paths take the lower index
    raw[0, 50] = hi
    model, _ = host_path_model(monkeypatch, raw)
    assert device_argmax(raw[0]) == 40 and int(torch.argmax(model._host_logits(1)[0])) == 40


def test_host_logprobs_are_the_capped_logits(monkeypatch):
    """The reported logprob of the chosen token is log_softmax of the soft-capped logits (fp64 reference)."""
    torch.manual_seed(0)
    raw = (torch.randn(2, 4096) * 8).to(torch.bfloat16)
    model, _ = host_path_model(monkeypatch, raw)
    got = torch.log_softmax(model._host_logits(2).double(), dim=-1)
    want = torch.log_softmax(30.0 * torch.tanh(raw.double() / 30.0), dim=-1)
    assert torch.allclose(got, want, atol=1e-5, rtol=0)


def test_host_logits_without_raw_logits_keep_the_engine_logits(monkeypatch):
    """The stub engine has no raw logits: its (already capped) logits pass through unchanged."""
    raw = (torch.randn(1, 256) * 8).to(torch.bfloat16)
    model, capped_bf16 = host_path_model(monkeypatch, raw, has_raw=False)
    assert torch.equal(model._host_logits(1), capped_bf16.float())


@pytest.mark.parametrize("env, package", [(None, "resident_s3"), ("resident_m3", "resident_m3")])
def test_engine_package_is_selectable(monkeypatch, env, package):
    """M1's adapter builds its engine from resident_s3 unless GEMMA4_RESIDENT_ENGINE_PACKAGE names another package
    (stage 2 of the regression: C at 262K on resident_m3); the config and pools it passes are unchanged."""
    import models.demos.gemma4_31b_qb2.tt.generator_resident_vllm as adapter_module

    if env is None:
        monkeypatch.delenv("GEMMA4_RESIDENT_ENGINE_PACKAGE", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_RESIDENT_ENGINE_PACKAGE", env)
    monkeypatch.setenv("GEMMA4_RESIDENT_ENGINE", "stub")
    imported = []
    fake = SimpleNamespace(
        EngineConfig=lambda **kw: ("cfg", kw), KVPools=lambda **kw: ("pools", kw),
        StubEngine=lambda mesh, cfg, pools: ("stub", cfg, pools), ResidentEngine=None)
    monkeypatch.setattr(adapter_module.importlib, "import_module", lambda name: imported.append(name) or fake)
    model = object.__new__(Gemma4ResidentForCausalLM)
    rope = {("rope", k): {"decode_cos": f"{k}_cos", "decode_sin": f"{k}_sin"} for k in ("sliding_attention", "full_attention")}
    model.generator = SimpleNamespace(model=SimpleNamespace(shared_setup=rope))
    model.mesh, model.max_batch_size, model.max_seq_len = "mesh", 1, 32768
    model.cache, model.groups = SimpleNamespace(kv=["kv"]), [0]
    monkeypatch.setattr(model, "_group_scratch", lambda: ["scratch"])
    kind, cfg, pools = model._build_engine()
    assert imported == [f"models.demos.gemma4_31b_qb2.{package}.engine"] and kind == "stub"
    assert cfg[1]["capacity"] == 32768 and cfg[1]["rows"] == model.ROWS and pools[1]["kv"] == ["kv"]
