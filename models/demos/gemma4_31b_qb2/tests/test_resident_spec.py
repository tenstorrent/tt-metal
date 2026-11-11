# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only checks of the resident speculative adapter (bench/resident/M2_DESIGN_v4.md).

The tap bookkeeping (resident_taps.TapBook, the mirrored window, the selection matrix) is checked against an
independent reference: every tap row is a known function of (layer, position), so a window or a commit can be
compared value for value (torch.equal, V3-7). HostTaps mirrors TapBuffers' device semantics on torch tensors
(where-select staging, zero-initialised ring, selection-matmul flush, mirrored pages), and launch tiles carry NaN
in the rows past R, so a leak shows up. The adapter's propose / verify state machine runs on fake engine and
drafter objects.
"""

import random
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gemma4_31b_qb2.tt import resident_taps as rt
from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm import (
    PLACEHOLDER_TOKEN_ID,
    policy_mode,
    Gemma4ResidentSpecForCausalLM,
    ResidentSpecPolicy,
    Step,
)
from models.demos.gemma4_31b_qb2.tt.generator_resident_vllm import Gemma4ResidentForCausalLM

LAYERS, WIDTH = 5, 4
P = rt.HISTORY_PAGES
TAP_LAYERS = (0, 16, 28, 46, 58)


def tap_value(layer, position):
    return float((layer + 1) * 100000 + position)


def launch_tile(position, rows):
    """A launch's taps: row r is position + r for r < rows; rows past R are NaN (unspecified on device)."""
    tile = torch.full((rt.TILE, LAYERS * WIDTH), float("nan"))
    for r in range(rows):
        for h in range(LAYERS):
            tile[r, h * WIDTH : (h + 1) * WIDTH] = tap_value(h, position + r)
    return tile


def capture(lo, hi, end):
    """A prefill capture of positions [lo, hi): rows past the prompt end are tile padding (NaN here)."""
    return {
        layer: torch.tensor(
            [[tap_value(h, x) if x < end else float("nan")] * WIDTH for x in range(lo, hi)], dtype=torch.float32
        )
        for h, layer in enumerate(TAP_LAYERS)
    }


class HostTaps:
    """TapBuffers on torch tensors, same semantics."""

    def __init__(self):
        self.staging = torch.zeros(rt.TILE, LAYERS * WIDTH)
        self.ring = torch.zeros(rt.RING_SLOTS, LAYERS, rt.TILE, WIDTH)
        self.history = [torch.zeros(2 * P, rt.TILE, WIDTH) for _ in range(LAYERS)]
        self.calls = Counter()

    def stage(self, tile, rows):
        mask = (torch.arange(rt.TILE) < rows).unsqueeze(1)
        self.staging = torch.where(mask, tile, torch.zeros(()))
        self.calls["stage"] += 1

    def ring_insert(self, slot):
        self.ring[slot] = self.staging.reshape(rt.TILE, LAYERS, WIDTH).permute(1, 0, 2)
        self.calls["ring_insert"] += 1

    def commit(self, drafter_taps, rows):
        for h, dest in enumerate(drafter_taps):
            dest[:] = self.staging[: dest.shape[0], h * WIDTH : (h + 1) * WIDTH]

    def _write(self, h, rows, first_page):
        for i in range(rows.shape[0] // rt.TILE):
            for slot in rt.mirror_slots(first_page + i):
                self.history[h][slot] = rows[i * rt.TILE : (i + 1) * rt.TILE]

    def write_prefill(self, parts, first_page):
        for h, part in parts.items():
            self._write(h, part, first_page)

    def flush(self, page, entries):
        sel = rt.selection_matrix(entries, LAYERS).float()
        out = sel @ self.ring.reshape(-1, WIDTH)
        for h in range(LAYERS):
            self._write(h, out[h * rt.TILE : (h + 1) * rt.TILE], page)
        self.calls["flush"] += 1

    def read_window(self, first_page):
        s = rt.window_slot(first_page)
        return [hist[s : s + P].reshape(-1, WIDTH) for hist in self.history]

    def insert_page(self, page, rows, slot):
        heads = torch.stack([hist[rt.mirror_slots(page)[0]] for hist in self.history])
        heads[:, rows:] = 0
        self.ring[slot] = heads


# ---------------------------------------------------------------------- pure arithmetic


@pytest.mark.parametrize("anchor", [1, 31, 32, 33, 2079, 2080, 2081, 2111, 2112, 5000, 32767])
def test_window_covers_the_last_positions(anchor):
    first, start, valid = rt.window(anchor)
    assert start == first * rt.TILE and valid == anchor - start
    assert first == max(0, rt.page_of(anchor - 1) - (P - 1))
    assert 1 <= valid <= rt.WINDOW_ROWS
    if anchor >= rt.WINDOW_ROWS:
        assert valid > rt.WINDOW_ROWS - rt.TILE  # the last 2,049..2,080 positions


def test_mirror_window_is_contiguous():
    """Every page written to both mirrors: any P-page window ending at the newest page is one slot run."""
    slots = [None] * (2 * P)
    for g in range(400):
        for s in rt.mirror_slots(g):
            slots[s] = g
        first = max(0, g - (P - 1))
        s = rt.window_slot(first)
        assert s + P <= 2 * P
        assert slots[s : s + P] == list(range(first, first + P)) or g < P - 1


def test_selection_matrix_is_an_exact_gather():
    ring = torch.randn(rt.RING_SLOTS, LAYERS, rt.TILE, WIDTH).bfloat16()
    entries = [(r, random.Random(r).randrange(rt.RING_SLOTS), random.Random(r + 99).randrange(rt.TILE)) for r in range(0, 32, 3)]
    out = (rt.selection_matrix(entries, LAYERS).float() @ ring.float().reshape(-1, WIDTH)).bfloat16()
    for r, slot, row in entries:
        for h in range(LAYERS):
            assert torch.equal(out[h * rt.TILE + r], ring[slot, h, row])
    unused = [r for r in range(rt.TILE) if r not in {e[0] for e in entries}]
    assert torch.equal(out[unused], torch.zeros(len(unused), WIDTH).bfloat16())


def test_book_continuity_overwrite_rule_and_slot_reuse():
    book = rt.TapBook()
    book.note_prefill(100)
    book.check_write(3)  # page of position 99
    with pytest.raises(ValueError, match="past the last committed"):
        book.check_write(4)
    with pytest.raises(ValueError, match="expected the next position"):
        book.insert(101, 1)
    for i in range(rt.RING_SLOTS):
        book.insert(100 + i, 1)
    with pytest.raises(ValueError, match="still holds unflushed"):
        book.insert(140, 1)  # slot 0 still holds position 100: its page never completed


# ---------------------------------------------------------------------- the adapter on fakes


class FakeEngine:
    def __init__(self, capacity=32768):
        self.cfg = SimpleNamespace(capacity=capacity)
        self.launches, self.rows, self.calls = 0, None, []
        self.buffers = [torch.zeros(rt.TILE, LAYERS * WIDTH)] * 2
        self.status = object()
        self.answers = list(range(500, 532))

    def write_tables(self, tables):
        pass

    def write_rows(self, slots, positions, tokens):
        self.rows = (list(slots), list(positions), list(tokens))

    def set_mode(self, feed, write_logits):
        self.calls.append(("mode", feed, write_logits))

    def launch(self, rows):
        # the engine's launch check (launch_of): every launched row's position, p + R, within the capacity (F_POSITION)
        if self.rows[1][0] + rows > self.cfg.capacity:
            raise AssertionError(f"F_POSITION: R = {rows} launch at {self.rows[1][0]} passes capacity {self.cfg.capacity}")
        self.calls.append(("launch", rows))
        self.buffers = list(self.buffers)
        self.buffers[self.launches % 2] = launch_tile(self.rows[1][0], len(self.rows[1]))
        self.launches += 1

    def taps_buffer(self, k):
        return self.buffers[k % 2]

    def read_tokens(self, n):
        return torch.tensor(self.answers[:n])


class FakeDrafter:
    tap_layers, block, SEED_ROWS = list(TAP_LAYERS), 8, rt.WINDOW_ROWS

    def __init__(self, max_positions=40000):
        self.max_positions = max_positions
        self.taps = [torch.zeros(self.block, WIDTH) for _ in TAP_LAYERS]
        self.calls, self.prepared, self.seeded = [], [], None

    def reset(self):
        self.calls.append("reset")

    def tap_window(self, length):
        end = (length + 31) // 32 * 32
        return max(0, end - self.SEED_ROWS), end

    def prepare(self, anchor, position, committed_start=None, committed=0):
        self.prepared.append((anchor, position, committed_start, committed, [t.clone() for t in self.taps]))

    def seed(self, taps, start, valid):
        self.calls.append("seed")
        self.seeded = ({k: v.clone() for k, v in taps.items()}, start, valid)

    def device_step(self, sample=None):
        self.calls.append("device_step")

    def read_drafts(self, anchor):
        return [anchor + 1 + i for i in range(self.block - 1)]


def tap_capture_range(start, window):
    lo, hi = max(window[0], start // 128 * 128), window[1]
    return (lo, hi) if hi > lo else None


@pytest.fixture
def model(monkeypatch):
    m = object.__new__(Gemma4ResidentSpecForCausalLM)
    m.engine, m.taps, m.book, m.drafter = FakeEngine(), HostTaps(), rt.TapBook(), FakeDrafter()
    m.tap_layers = TAP_LAYERS
    m.spec_policy = ResidentSpecPolicy(Gemma4ResidentSpecForCausalLM.K, mode="force")
    m.counters, m.draft_trace, m.draft_captures = Counter(), None, []
    m.mesh, m.max_seq_len, m.max_batch_size = None, 32768, 1
    m.cache = SimpleNamespace()
    m.mode, m.feed_stale, m.positions, m.sampling_trace, m.l1_checks = None, False, [-1], None, []
    m.generator = SimpleNamespace(tap_capture_range=tap_capture_range)
    m._reset_request()
    forwards = Counter()
    monkeypatch.setattr(m, "_l1_guard", lambda *a, **k: None)
    monkeypatch.setattr(m, "_check_status", lambda status: None)
    monkeypatch.setattr(m, "_tables", lambda cache, source: source)
    monkeypatch.setattr(m, "_resident_tables", lambda tables: tables)
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda *a, **k: "trace")
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "execute_trace", lambda *a, **k: forwards.update(["draft"]))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "release_trace", lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [SimpleNamespace(cpu=lambda: t)])
    m.forwards = forwards
    return m


def prefill(m, end, start=0):
    if start == 0:
        m._reset_request()
    window = m.drafter.tap_window(end)
    lo = tap_capture_range(start, window)
    m._record_prefill_taps({k: v for k, v in capture(*lo, end).items()}, window, start, end)


def ordinary(m, position, host=False):
    """An ordinary R = 1 decode at `position` (what M1's decode_forward does to the engine)."""
    m.engine.write_rows([0], [position], [7])
    launch = m.engine.launches
    m.engine.launch(1)
    m.step = Step("ordinary", launch, position, host=host, dropped=m.offer is not None)
    m.offer = None


def propose(m, position, c=1):
    tokens = torch.full((1, 8), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    positions = torch.full((1, 8), -1, dtype=torch.int32)
    for j in range(c):
        tokens[0, j], positions[0, j] = 900 + j, position + 1 + j
    return m.propose_draft_tokens(7, tokens, positions, torch.tensor([c], dtype=torch.int32))


def verify(m, block, position, n):
    tokens = torch.full((1, 8), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    tokens[0, : n + 1] = torch.tensor(block[: n + 1])
    start = torch.full((1, 8), -1, dtype=torch.int32)
    start[0, : n + 1] = torch.arange(position, position + n + 1)
    return m.decode_forward(tokens, start, None, m.cache, spec_mode="argmax_ids",
                            num_valid_drafts=torch.tensor([n], dtype=torch.int32),
                            accepted_counts=torch.tensor([1], dtype=torch.int32))


def check_seed(m, anchor):
    taps, start, valid = m.drafter.seeded
    first, want_start, want_valid = rt.window(anchor)
    assert (start, valid) == (want_start, want_valid)
    for h, layer in enumerate(TAP_LAYERS):
        rows = taps[layer][:valid]
        want = torch.tensor([[tap_value(h, start + r)] * WIDTH for r in range(valid)])
        assert torch.equal(rows, want), (layer, anchor)


@pytest.mark.parametrize("length", [5, 100, 2079, 2080, 2085, 4000])
def test_first_step_seeds_from_prefill_and_decode_rows(model, length):
    """Lazy seeding at the first propose (V3-4): the window holds prefill rows < L and row L from the decode."""
    prefill(model, length)
    ordinary(model, length)
    out = propose(model, length)
    assert int(out.num_valid[0]) == 7
    check_seed(model, length + 1)
    anchor, position, committed_start, committed, _ = model.drafter.prepared[-1]
    assert (anchor, position, committed_start, committed) == (900, length + 1, length + 1, 0)
    assert model.forwards["draft"] == 1 and model.regime == "hold"


def test_verify_launches_block_and_hold_commits_its_rows(model):
    prefill(model, 300)
    ordinary(model, 300)
    drafts = propose(model, 300).draft_token_ids[0].tolist()
    block = [900] + drafts
    out = verify(model, block, 301, 7)
    assert model.engine.calls[-1] == ("launch", 8)
    assert model.engine.rows == ([0] * 8, list(range(301, 309)), block)
    assert out.argmax_ids.shape == (1, 8) and out.argmax_ids[0].tolist() == list(range(500, 508))
    assert model.feed_stale
    # 3 committed (2 drafts accepted + the correction): rows 301..303 enter drafter.taps, rows >= 3 are 0.
    propose(model, 301, c=3)
    anchor, position, committed_start, committed, taps = model.drafter.prepared[-1]
    assert (position, committed_start, committed) == (301 + 3, 301, 3)
    for h in range(LAYERS):
        want = torch.zeros(8, WIDTH)
        for r in range(3):
            want[r] = tap_value(h, 301 + r)
        assert torch.equal(taps[h], want)  # the NaN rows past R=8's committed count never reach the drafter


def test_verify_rejects_a_block_that_is_not_the_proposal(model):
    prefill(model, 300)
    ordinary(model, 300)
    drafts = propose(model, 300).draft_token_ids[0].tolist()
    with pytest.raises(ValueError, match="not the retained proposal"):
        verify(model, [900] + [d + 1 for d in drafts], 301, 7)


def test_host_sampled_and_dropped_offers_never_draft(model):
    """V3-1: no seed and no forward for a host-sampled step; a dropped offer suppresses retries for good."""
    prefill(model, 300)
    for i in range(40):
        ordinary(model, 300 + i, host=True)
        assert int(propose(model, 300 + i).num_valid[0]) == 0
    assert "seed" not in model.drafter.calls and model.forwards["draft"] == 0
    assert model.counters["spec_nonspeculable_steps"] == 40
    ordinary(model, 340)  # device-sampled again: the forced policy enters
    assert int(propose(model, 340).num_valid[0]) == 7
    ordinary(model, 341)  # the offer is dropped: the runner sent an ordinary decode
    assert int(propose(model, 341).num_valid[0]) == 0
    for i in range(20):
        ordinary(model, 342 + i)
        assert int(propose(model, 342 + i).num_valid[0]) == 0
    assert model.counters["spec_offers_dropped"] == 1 and model.forwards["draft"] == 1


def test_long_plain_stretch_wraps_the_mirror_and_reentry_seeds_exactly(model, monkeypatch):
    """An exit stretch past 65 pages (the history wraps), then a re-entry: the window is the last positions."""
    prefill(model, 1000)
    decide = iter([False] * 2500 + [True])
    monkeypatch.setattr(model.spec_policy, "decide", lambda position, entering: next(decide))
    for i in range(2500):
        ordinary(model, 1000 + i)
        assert int(propose(model, 1000 + i).num_valid[0]) == 0
    ordinary(model, 3500)
    assert int(propose(model, 3500).num_valid[0]) == 7
    check_seed(model, 3501)


def test_k0_in_hold_exits_and_the_next_entry_reseeds(model, monkeypatch):
    prefill(model, 300)
    ordinary(model, 300)
    drafts = propose(model, 300).draft_token_ids[0].tolist()
    verify(model, [900] + drafts, 301, 7)
    monkeypatch.setattr(model.spec_policy, "decide", lambda position, entering: False)
    assert int(propose(model, 301, c=2).num_valid[0]) == 0 and model.regime == "plain"
    ordinary(model, 303)
    monkeypatch.setattr(model.spec_policy, "decide", lambda position, entering: entering)
    seeds = model.drafter.calls.count("seed")
    assert int(propose(model, 303).num_valid[0]) == 7
    assert model.drafter.calls.count("seed") == seeds + 1
    check_seed(model, 304)


def test_near_capacity_declines_without_prepare(model):
    model.drafter.max_positions = 310
    prefill(model, 300)
    ordinary(model, 300)
    drafts = propose(model, 300).draft_token_ids[0].tolist()  # anchor 301: 301 + 8 <= 310
    verify(model, [900] + drafts, 301, 7)
    prepared = len(model.drafter.prepared)
    assert int(propose(model, 301, c=3).num_valid[0]) == 0 and model.exhausted  # anchor 304: 312 > 310
    assert len(model.drafter.prepared) == prepared


def test_anchor_position_must_follow_the_step(model):
    prefill(model, 300)
    ordinary(model, 300)
    with pytest.raises(ValueError, match="Anchor position"):
        propose(model, 301)


def test_spec_plan():
    plan = Gemma4ResidentSpecForCausalLM.spec_plan(None, 1, 7)
    assert plan.effective_k == 7 and plan.supports_narrow_decode and plan.accept_modes == ("argmax_ids",)
    assert Gemma4ResidentSpecForCausalLM.spec_plan(None, 1, 5).supported_k == (7,)
    assert Gemma4ResidentSpecForCausalLM.spec_plan(None, 2, 7).supported_k == ()
    caps = Gemma4ResidentSpecForCausalLM.model_capabilities
    assert caps["supports_spec_decode"] and caps["spec_ids_are_sampled"] and not caps["supports_async_spec_decode"]
    assert Gemma4ResidentForCausalLM.model_capabilities.get("supports_spec_decode") is None


@pytest.mark.parametrize("floor", [1_494_528, 1_350_000, 1_300_000])
def test_r8_drop_is_observable_and_declines(model, monkeypatch, floor):
    """F5: the guard compares the floor with every R's CB top (resident_m2 cb_tops). Below the R = 8 top (the
    attention leader's, which no l1_top lowers) the engine drops to rows (1,), logs, counts and never drafts; below
    the R = 1 top too, R = 1 is rebuilt at the floor as in M1."""
    built = []

    class Engine(FakeEngine):
        def __init__(self, tops):
            super().__init__()
            self.tops, self.rebuilt = dict(tops), []

        def close(self):
            pass

        def warmup(self):
            pass

        def cb_tops(self):
            return self.tops

        def cb_top(self):
            return max(self.tops.values())

        def rebuild(self, l1_top):
            self.rebuilt.append(l1_top)
            self.tops = {r: min(t, l1_top) for r, t in self.tops.items()}

    model.engine = Engine({1: 1_327_504, 8: 1_382_800})
    model.cache = SimpleNamespace(page_tables={0: torch.zeros(1, 8, dtype=torch.int32)})
    monkeypatch.setattr(model, "_l1_floor", lambda: floor)
    monkeypatch.setattr(model, "_build_engine", lambda: built.append(Engine({1: 1_327_504})) or built[-1])
    del model.__dict__["_l1_guard"]  # the fixture's no-op: exercise the class's guard
    model._l1_guard("draft trace capture")
    if floor >= 1_382_800:
        assert model.ROWS == (1, 8) and not built and model.counters["spec_r8_dropped"] == 0
        return
    assert model.ROWS == (1,) and model.counters["spec_r8_dropped"] == 1 and model.engine is built[0]
    assert built[0].rebuilt == ([floor] if floor < 1_327_504 else [])
    prefill(model, 300)
    ordinary(model, 300)
    assert int(propose(model, 300).num_valid[0]) == 0 and model.counters["spec_r8_unavailable"] == 1
    assert "seed" not in model.drafter.calls and model.forwards["draft"] == 0


def test_chunked_prefill_continuation_then_entry(model):
    """Two prefill calls (an unaligned continuation): the first call's partial page is re-written by the second,
    and the first entry's window is exact."""
    prefill(model, 8150)
    prefill(model, 9000, start=8150)
    ordinary(model, 9000)
    assert int(propose(model, 9000).num_valid[0]) == 7
    check_seed(model, 9001)


def test_sampled_verify_seeds_rows_with_the_plain_seeds_and_captures_first(model, monkeypatch):
    """§6: row j of a sampled verify at p uses the seed plain decoding uses at p + j (seeded requests); the lazy
    sampling capture runs before the block's rows are written (its guard may rebuild and restore M1's feed)."""
    configured, order = [], []
    model.generator.max_seq_len = 32768
    model.generator.configure_sampling = lambda **k: configured.append(k)
    monkeypatch.setattr(model, "_capture_sampling", lambda: order.append("capture") or setattr(model, "sampling_trace", 1))
    monkeypatch.setattr(model, "_guard_after_capture", lambda: order.append("guard"))
    monkeypatch.setattr(model, "_replay_sampling", lambda: order.append("replay"))
    rows = model.engine.write_rows
    monkeypatch.setattr(model.engine, "write_rows", lambda *a: order.append("rows") or rows(*a))
    prefill(model, 300)
    ordinary(model, 300)
    drafts = propose(model, 300).draft_token_ids[0].tolist()
    params = SimpleNamespace(temperature=[0.8], top_k=[20], top_p=[0.95], seed=[1234])
    order.clear()
    tokens = torch.tensor([[900] + drafts], dtype=torch.int32)
    start = torch.arange(301, 309, dtype=torch.int32).reshape(1, 8)
    model.decode_forward(tokens, start, None, model.cache, spec_mode="argmax_ids",
                         num_valid_drafts=torch.tensor([7], dtype=torch.int32),
                         accepted_counts=torch.tensor([1], dtype=torch.int32), sampling_params=params)
    assert order[:4] == ["capture", "guard", "rows", "replay"]
    cfg = configured[-1]
    assert cfg["batch"] == 8 and cfg["seed"] == [1234 + 301 + j for j in range(8)]
    assert cfg["top_k"] == [20] * 8 and cfg["temperature"] == [0.8] * 8
    assert model.engine.calls[-2:] == [("mode", "none", True), ("launch", 8)]


def test_spec_prefill_passes_the_window_and_records_it(model, monkeypatch):
    """The speculative prefill hands the generator one absolute window per row and records what it captured."""
    calls = []

    def generator_prefill(tokens, **k):
        calls.append(k)
        lo, hi = tap_capture_range(k["start_positions"][0], k["tap_windows"][0])
        k["taps"][0].update(capture(lo, hi, k["prompt_lens"][0]))
        return torch.zeros(1, 1, 8)

    model.generator = SimpleNamespace(
        tap_capture_range=tap_capture_range,
        model=SimpleNamespace(layers=[SimpleNamespace(sliding=i % 6 != 5) for i in range(60)]),
        prefill_forward=generator_prefill,
        prefill_shapes=set(),
    )
    model.prefill_slot_keys = {}
    released = Counter()
    monkeypatch.setattr(model, "_tables", lambda cache, source, slots=None: [torch.zeros(1, 8, dtype=torch.int32)] * 60)
    monkeypatch.setattr(model, "_release_sampling_trace", lambda: released.update(["sampling"]))
    monkeypatch.setattr(model, "_release_draft_trace", lambda: released.update(["draft"]))
    cache = SimpleNamespace(batch=1, capacity=32768)
    model.prefill_forward(torch.zeros(1, 3000, dtype=torch.int64), None, cache, [3000], start_pos=[0])
    assert calls[0]["tap_windows"] == [(3008 - 2080, 3008)] and calls[0]["taps"][0].keys() == set(TAP_LAYERS)
    assert released == {"sampling": 1, "draft": 1}  # a novel prefill shape (F3)
    ordinary(model, 3000)
    assert int(propose(model, 3000).num_valid[0]) == 7
    check_seed(model, 3001)


def test_guard_rebuild_releases_and_recaptures_both_traces(model, monkeypatch):
    """resident-verifier's M1.1 finding: a guard rebuild allocates after the traces were captured. The rebuild
    releases both; the sampling trace is captured again before the launch, the draft trace at the next draft."""
    order, built = [], []

    class Engine(FakeEngine):
        def __init__(self, tops):
            super().__init__()
            self.tops = dict(tops)

        def close(self):
            pass

        def warmup(self):
            pass

        def cb_tops(self):
            return self.tops

        def cb_top(self):
            return max(self.tops.values())

    floors = iter([1_494_528] * 10)  # until the verify: then its sampling capture's guard finds the floor low
    Engine.read_tokens = lambda self, n: torch.tensor([600 + self.rows[1][0]] * n)
    model.engine = Engine({1: 1_327_504, 8: 1_382_800})
    model.cache = SimpleNamespace(page_tables={0: torch.zeros(1, 8, dtype=torch.int32)})
    model.generator.max_seq_len = 32768
    seeds = []
    model.generator.configure_sampling = lambda **k: seeds.append(k["seed"]) if k.get("batch") == 1 else None
    model.max_batch_size = 1
    monkeypatch.setattr(model, "_l1_floor", lambda: next(floors))
    monkeypatch.setattr(model, "_build_engine", lambda: built.append(Engine({1: 1_327_504})) or built[-1])
    del model.__dict__["_l1_guard"]  # the fixture's no-op: exercise the class's guard

    def capture():
        order.append("capture")
        model.sampling_trace = f"sampling{order.count('capture')}"

    def release():
        if model.sampling_trace is not None:
            order.append("release sampling")
        model.sampling_trace = None

    monkeypatch.setattr(model, "_capture_sampling", capture)
    monkeypatch.setattr(model, "_release_sampling_trace", release)
    monkeypatch.setattr(model, "_replay_sampling", lambda: order.append(f"replay {model.sampling_trace}"))
    prefill(model, 300)
    ordinary(model, 300)
    drafts = propose(model, 300).draft_token_ids[0].tolist()
    assert model.draft_trace == "trace"
    params = SimpleNamespace(temperature=[1.0], top_k=[20], top_p=[0.95], seed=[7])
    tokens = torch.tensor([[900] + drafts], dtype=torch.int32)
    start = torch.arange(301, 309, dtype=torch.int32).reshape(1, 8)
    order.clear()
    floors = iter([1_350_000] + [1_494_528] * 10)
    out = model.decode_forward(tokens, start, None, model.cache, spec_mode="argmax_ids",
                               num_valid_drafts=torch.tensor([7], dtype=torch.int32),
                               accepted_counts=torch.tensor([1], dtype=torch.int32), sampling_params=params)
    # The capture's guard dropped R = 8 (the floor below its top): both traces were released, the sampling trace
    # captured again before the launch, and the block ran as 8 causal R = 1 launches, each row with its own seed.
    assert order[:3] == ["capture", "release sampling", "capture"] and order.count("replay sampling2") == 8
    assert model.ROWS == (1,) and model.counters["spec_r8_dropped"] == 1 and model.draft_trace is None
    assert model.engine is built[0] and model.engine.calls.count(("launch", 1)) == 8
    assert ("launch", 8) not in model.engine.calls
    assert seeds == [[7 + 301 + j] for j in range(8)]
    assert out.argmax_ids[0].tolist() == [600 + 301 + j for j in range(8)]


def test_r8_drop_uses_drop_rows_or_frees_the_old_engine_first(model, monkeypatch):
    """resident-analyst's review: close() keeps the weights alive, so a new engine built while the old one is
    referenced runs its weight loop on a fragmented heap. drop_rows (resident_m2) keeps the weights and R = 1 and
    builds nothing; without it, the old engine is unreachable before the new one is constructed."""
    import weakref

    class Old(FakeEngine):
        def close(self):
            pass

        def cb_tops(self):
            return {1: 1_327_504, 8: 1_382_800}

        def cb_top(self):
            return 1_382_800

    model.cache = SimpleNamespace(page_tables={0: torch.zeros(1, 8, dtype=torch.int32)})
    monkeypatch.setattr(model, "_l1_floor", lambda: 1_350_000)
    del model.__dict__["_l1_guard"]

    class Droppable(Old):
        def drop_rows(self, rows):
            self.dropped = tuple(rows)

        def cb_top(self):
            return 1_327_504 if getattr(self, "dropped", None) else 1_382_800

    model.engine = Droppable()
    monkeypatch.setattr(model, "_build_engine", lambda: pytest.fail("drop_rows must not construct an engine"))
    model._l1_guard("draft trace capture")
    assert model.engine.dropped == (8,) and model.ROWS == (1,) and model.counters["spec_r8_dropped"] == 1

    model.ROWS = Gemma4ResidentSpecForCausalLM.ROWS
    model.engine = Old()
    old = weakref.ref(model.engine)

    class New(Old):
        def warmup(self):
            pass

        def cb_top(self):
            return 1_327_504

    def build():
        assert old() is None, "the old engine (and its weights) must be gone before a new weight loop"
        return New()

    monkeypatch.setattr(model, "_build_engine", build)
    model._l1_guard("draft trace capture")
    assert isinstance(model.engine, New) and model.counters["spec_r8_dropped"] == 2


def test_policy_never_loads_but_never_speculates(model, monkeypatch):
    """The regression's F2 run (GEMMA4_RESIDENT_SPEC_POLICY=never): the speculative adapter serves with no seed and
    no draft forward, but still records its taps, so its cost is what F2 measures against plain resident decode."""
    model.spec_policy = ResidentSpecPolicy(Gemma4ResidentSpecForCausalLM.K, mode="never")
    prefill(model, 300)
    for i in range(70):
        ordinary(model, 300 + i)
        assert int(propose(model, 300 + i).num_valid[0]) == 0
    assert "seed" not in model.drafter.calls and model.forwards["draft"] == 0 and model.drafter.prepared == []
    assert model.taps.calls["ring_insert"] == 70 and model.counters["declines"] == 70


@pytest.mark.parametrize("env, mode", [({}, "adaptive"), ({"GEMMA4_RESIDENT_SPEC_FORCE": "1"}, "force"),
                                       ({"GEMMA4_RESIDENT_SPEC_POLICY": "never"}, "never")])
def test_policy_mode_env(monkeypatch, env, mode):
    for key in ("GEMMA4_RESIDENT_SPEC_FORCE", "GEMMA4_RESIDENT_SPEC_POLICY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert policy_mode() == mode
    monkeypatch.setenv("GEMMA4_RESIDENT_SPEC_POLICY", "sometimes")
    with pytest.raises(ValueError, match="expected adaptive"):
        policy_mode()


# ---------------------------------------------------------------- per-request policy (lead's ruling on M2 item 3)

K_DRAFT = Gemma4ResidentSpecForCausalLM.K
COMMITS = {  # per-verify commits (accepted + 1): prose is e1e's streamed prose; code / math carry e1c's means
    "prose": [8, 8, 3, 8, 8, 2, 3, 3, 1, 1, 2, 8, 1, 2, 3, 1, 1, 3, 3, 1, 8, 2, 1, 2, 1, 1, 1, 8, 2, 2, 3, 3, 1, 1, 4,
              3, 8, 1, 1, 2, 3, 1, 2, 3, 1, 2, 5, 6, 1, 1, 1, 1, 2, 8, 4, 5, 3, 8, 2, 2, 4, 2, 4, 7, 2, 3, 1, 2, 2, 8,
              4, 3, 6, 1, 3, 1, 2, 8, 5],
    "code": [4 if i % 3 else 3 for i in range(65)],
    "math": [7 if i % 7 else 1 for i in range(42)],
    "zero": [1] * 256,
}
PROMPT = {"code": 43, "prose": 39, "math": 73, "zero": 128}


def drive(policy, name, total=255, start=None):
    """A request through the adapter's decision path (`_propose`'s regimes): plain -> decide(entering=True); an
    entry drafts, the verify commits the next COMMITS value, observe, decide(entering=False); an exit is plain().
    `start` overrides the prompt length (the request's first position)."""
    policy.new_request()  # the prefill
    commits, regime, offer, committed, out = iter(COMMITS[name]), "plain", False, 0, []
    while committed < total:
        c = min(next(commits, 1), total - committed) if offer else 1
        if offer:
            policy.observe(c - 1, K_DRAFT)
        committed += c
        position = (PROMPT[name] if start is None else start) + committed
        if regime == "hold":
            ok = policy.decide(position, entering=False)
            if not ok:
                regime = "plain"
                policy.plain()
        else:
            ok = policy.decide(position, entering=True)
            regime = "hold" if ok else "plain"
        offer = ok
        out.append((position, regime, ok, policy.policy.last_reason))
    policy.new_request()  # release_request
    return out


def test_policy_decisions_are_per_request():
    """No state crosses requests: a request decides the same on a fresh policy, after other requests in any order,
    and after a long run of declines (the e1e case: streamed code, the first request after start, declined all)."""
    alone = {name: drive(ResidentSpecPolicy(K_DRAFT, mode="adaptive"), name) for name in COMMITS}
    for order in (["code", "prose", "math"], ["math", "zero", "prose", "code"], ["zero", "zero", "code", "math"]):
        policy = ResidentSpecPolicy(K_DRAFT, mode="adaptive")
        for name in order:
            assert drive(policy, name) == alone[name], (order, name)
    assert all(ok for _, _, ok, _ in alone["math"][:1]) and any(ok for _, _, ok, _ in alone["code"])


def test_policy_first_proposal_trials():
    """Every request's first proposal enters (a trial from the uninformed prior), on a fresh policy and after a
    request that declined most of its steps: before, the bucket was 0 at server start ("budget" on every step)."""
    policy = ResidentSpecPolicy(K_DRAFT, mode="adaptive")
    for name in ("code", "zero", "prose", "zero", "math"):
        first = drive(policy, name)[0]
        assert first[2] and first[3] == "trial", (name, first)


def test_policy_zero_acceptance_exits_after_trial():
    """A request whose drafts are never accepted holds its trial for trial_limit steps, then exits; the plain
    stretch after the exit is governed by the retry rules (no re-entry within the next 32 proposals)."""
    policy = ResidentSpecPolicy(K_DRAFT, mode="adaptive")
    out = drive(policy, "zero")
    limit = policy.config.trial_steps
    held = next(i for i, (_, regime, _, _) in enumerate(out) if regime == "plain")
    assert 1 <= held <= limit + 1, (held, limit, out[: limit + 3])
    assert not any(ok for _, _, ok, _ in out[held: held + 32])


def test_policy_decisions_never_read_time(monkeypatch):
    """Decisions use the fixed priors and the request's own acceptance only: replayed under different simulated
    step timings (every clock the process can read, fast and steady vs slow and jittery), the decisions are equal.
    Calibration stays unfed (no note_step), so no measured time can enter a decision either."""
    import time as time_module

    def run(step_s, jitter):
        clock = {"t": 1000.0, "n": 0}

        def now(*_):
            clock["n"] += 1
            clock["t"] += step_s * (1.0 + (jitter if clock["n"] % 2 else -jitter))
            return clock["t"]

        for fn in ("time", "perf_counter", "monotonic"):
            monkeypatch.setattr(time_module, fn, now)
        monkeypatch.setattr(time_module, "perf_counter_ns", lambda: int(now() * 1e9))
        policy = ResidentSpecPolicy(K_DRAFT, mode="adaptive")
        out = [drive(policy, name) for name in ("prose", "code", "zero", "math")]
        assert policy.policy.calibration.served_intervals == 0
        return out

    assert run(0.011, 0.0) == run(0.025, 0.4)


@pytest.mark.parametrize("mode, expected", [("force", True), ("never", False)])
def test_policy_force_and_never_unchanged(mode, expected):
    """force drafts on every speculable step and never on none, request after request, as before."""
    policy = ResidentSpecPolicy(K_DRAFT, mode=mode)
    for _ in range(3):
        policy.new_request()
        assert all(policy.decide(100 + i, entering=bool(i % 2)) is expected for i in range(40))


@pytest.mark.parametrize("capacity", [32768, 262144])  # the dev-32k and 262K speculative launches
@pytest.mark.parametrize("left", range(1, 10))
def test_no_r8_verify_in_the_last_block(model, capacity, left):
    """m1l-verifier (2026-11-11): an R = 8 launch is flagged F_POSITION when p + 8 passes the engine's capacity, even
    with fewer valid rows. An anchor `left` positions before the capacity offers drafts only when anchor + 8 fits; a
    verify there launches inside the capacity (FakeEngine raises like the engine otherwise); closer, the step is a
    plain R = 1 step and the request stays exhausted. The bound is the engine's capacity, not max_seq_len."""
    model.engine.cfg.capacity, model.max_seq_len = capacity, capacity + 64  # the engine's capacity is the bound
    model.drafter.max_positions = capacity + 8 + 32  # build_drafter's RoPE range
    anchor = capacity - left
    prefill(model, anchor - 1)
    ordinary(model, anchor - 1)
    out = propose(model, anchor - 1)
    if left >= 8:
        assert int(out.num_valid[0]) == 7 and not model.exhausted
        verify(model, [900] + out.draft_token_ids[0].tolist(), anchor, 7)  # R = 8 rows anchor .. anchor + 7 < capacity
        assert model.engine.calls[-1] == ("launch", 8)
    else:
        assert int(out.num_valid[0]) == 0 and model.exhausted
        ordinary(model, anchor)  # the plain step the decline leaves: R = 1, inside the capacity
        assert int(propose(model, anchor).num_valid[0]) == 0


def test_verify_refuses_an_r8_block_past_capacity(model):
    """The verify's own check: an R = 8 block whose 8 rows pass the engine's capacity is refused before the launch."""
    model.engine.cfg.capacity = 1024
    prefill(model, 1000)
    ordinary(model, 1000)
    drafts = propose(model, 1000).draft_token_ids[0].tolist()
    model.engine.cfg.capacity = 1008  # anchor 1001 + 8 > 1008
    with pytest.raises(ValueError, match="R = 8 limit"):
        verify(model, [900] + drafts, 1001, 7)


def test_plain_pages_use_resident_selection_matrices():
    """M2 item 1: TapBuffers keeps the selection matrix of every plain-regime page on device (one committed row per
    launch, 32 consecutive ring slots from any first slot). Every page a run of single-row launches completes, after
    any mix of earlier multi-row commits, is one of those 40 patterns; a page holding a multi-row commit is not (it
    uploads its own matrix, as before)."""
    plain = {rt.plain_page_entries(first) for first in range(rt.RING_SLOTS)}
    assert len(plain) == rt.RING_SLOTS
    book = rt.TapBook()
    book.note_prefill(128)
    position, seen_plain, seen_other = 128, 0, 0
    commits = [1] * 70 + [3, 8, 1, 5, 2, 8, 8, 4] + [1] * 130 + [6, 1, 1, 7] + [1] * 96
    for c in commits:
        book.insert(position, c)
        for page in book.complete_pages():
            entries = tuple(book.entries(page))
            if all(row == 0 for _, _, row in entries):
                assert entries in plain, (page, entries[:3])
                seen_plain += 1
            else:
                assert entries not in plain
                seen_other += 1
            book.flushed(page)
        position += c
    assert seen_plain >= 6 and seen_other >= 2  # this sequence completes 7 plain pages and 3 mixed ones


class _FakeT:
    """A stand-in device tensor: shape only."""

    def __init__(self, shape, tag="t"):
        self.shape = self.padded_shape = tuple(shape)
        self.tag, self.dtype, self.layout = tag, None, None

    def __getitem__(self, idx):
        dims = []
        for d, s in zip(self.shape, idx if isinstance(idx, tuple) else (idx,)):
            dims.append(len(range(*s.indices(d))) if isinstance(s, slice) else 1)
        return _FakeT(dims + list(self.shape[len(dims):]), "slice")


class _FakeTtnn:
    """Just enough ttnn for TapBuffers on the host: counts device allocations and records each fill's table."""

    bfloat16, int32, uint32, TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = "bf16", "i32", "u32", "tile", "rm", "dram"
    MathFidelity = SimpleNamespace(HiFi4="hifi4")
    BufferType = SimpleNamespace(DRAM="dram", L1="l1")

    def __init__(self):
        self.device_allocs, self.fills, self.uploads, self.matmul_sel = 0, [], 0, []
        self.experimental = SimpleNamespace(view=lambda t, shape: _FakeT(shape, "view"),
                                            paged_fill_cache=lambda cache, part, table, batch_idx: self.fills.append(table))

    def ReplicateTensorToMesh(self, mesh):
        return None

    def from_torch(self, host, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        if device is not None:
            self.device_allocs += 1
        return _FakeT(host.shape, "const" if device is not None else "host")

    def init_device_compute_kernel_config(self, *a, **k):
        return None

    def copy_host_to_device_tensor(self, host, dest):
        self.uploads += 1

    def where(self, *a, **k):
        return None

    def matmul(self, sel, rows, **k):
        self.matmul_sel.append(sel)
        return _FakeT((1, 1, sel.shape[2], rows.shape[3]), "out")

    def get_memory_view(self, mesh, kind):
        return SimpleNamespace(num_banks=8, total_bytes_free_per_bank=1 << 30)

    def slice(self, src, start, end, **k):
        return None

    def concat(self, parts, dim=0, **k):
        return _FakeT((1, len(parts), 32, rt.TAP_WIDTH), "concat")


def test_tap_buffers_allocate_nothing_after_seal(monkeypatch):
    """M2 item 1 / the TTSampling-trace clobber: every persistent TapBuffers buffer is built in __init__, before any
    trace capture; once sealed, the per-step work (stage, ring insert, plain and mixed flushes, a prefill write)
    allocates no device buffer, ring inserts use the constant slot tables, and plain flushes use the resident
    selection matrices (no upload)."""
    fake = _FakeTtnn()
    monkeypatch.setattr(rt, "ttnn", fake)
    mesh = SimpleNamespace(arch=lambda: "blackhole")
    taps = rt.TapBuffers(mesh, LAYERS)
    built = fake.device_allocs
    assert len(taps.plain_sel) == rt.RING_SLOTS and len(taps.slot_tables) == rt.RING_SLOTS
    taps.seal()
    book = rt.TapBook()
    book.note_prefill(128)
    position, uploads_before = 128, fake.uploads
    for c in [1] * 64:  # two plain pages
        taps.stage(_FakeT((1, 1, 32, LAYERS * rt.TAP_WIDTH)), c)
        slot = book.insert(position, c)
        taps.ring_insert(slot)
        assert fake.fills[-1] is taps.slot_tables[slot]
        for page in book.complete_pages():
            taps.flush(page, book.entries(page))
            assert fake.matmul_sel[-1] is taps.plain_sel[tuple(book.entries(page))]
            book.flushed(page)
        position += c
    assert fake.uploads == uploads_before  # no table, no matrix uploaded on the plain path
    for c in [3, 8, 8, 8, 5]:  # a mixed page: its matrix is uploaded into `sel`, still no allocation
        taps.stage(_FakeT((1, 1, 32, LAYERS * rt.TAP_WIDTH)), c)
        taps.ring_insert(book.insert(position, c))
        for page in book.complete_pages():
            taps.flush(page, book.entries(page))
            assert fake.matmul_sel[-1] is taps.sel
            book.flushed(page)
        position += c
    taps.write_prefill({0: _FakeT((1, 1, 3 * rt.TILE, rt.TAP_WIDTH))}, 0)
    taps.read_window(0)  # the seed's window read and a page read, after seal: built in __init__, nothing allocated
    taps.read_page(4)
    assert taps.sealed and fake.device_allocs == built


def test_tap_flush_shares_the_replays_command_queue():
    """The flush's temporaries (matmul output, slices) are safe only if the flush runs in order with every trace
    replay on one command queue (otherwise the TTSampling clobber pattern applies). Pinned structurally, since the
    mesh exposes no command-queue count to Python: the plugin opens the mesh with the default single queue (it never
    passes num_command_queues); every trace capture and replay in the serving adapters and the resident engines names
    cq_id=0; resident_taps names no queue at all (its ops run on queue 0)."""
    import pathlib
    import re as _re

    root = pathlib.Path(rt.__file__).resolve().parents[1]  # the package dir of the (staged or live) resident_taps
    live = pathlib.Path("/home/ttuser/experiments/gemma4/tt-metal/models/demos/gemma4_31b_qb2")
    plugin = pathlib.Path("/home/ttuser/experiments/gemma4/src/vllm-tt-plugin/src/vllm_tt_plugin")
    assert "num_command_queues" not in (plugin / "worker.py").read_text()
    assert not _re.search(r"cq_id|queue_id", pathlib.Path(rt.__file__).read_text())
    sources = [root / "tt" / name for name in ("generator_resident_vllm.py", "generator_resident_spec_vllm.py")]
    sources = [p if p.exists() else live / "tt" / p.name for p in sources]
    sources += [live / pkg / "engine.py" for pkg in ("resident_s3", "resident_m2", "resident_m3")]
    sources += [live / "tt" / name for name in ("generator.py", "dflash.py")]  # ttnn's model / sampling, the drafter
    calls = 0
    for path in sources:
        for call in _re.findall(r"(?:execute_trace|begin_trace_capture|end_trace_capture)\([^)]*\)", path.read_text()):
            assert "cq_id=0" in call, (path.name, call)
            calls += 1
    assert calls >= 20


def test_cost_priors_are_monotone_and_match_their_knots():
    """The position-keyed cost priors (lead's ruling on Q3): fixed constants, piecewise-linear between the knots,
    clamped at both ends; both tables rise with position; the knots are the measured engine times plus the 0.87 ms
    served host time (dev-32k M1 engine below 32K, resident_m3_1 at 33K / 128K / 262K, VERIFY_M3_1 §4)."""
    import models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm as spec

    want = {spec.PRIOR_PLAIN_MS: [10.21, 12.11, 12.49, 13.03, 15.34, 18.68],
            spec.PRIOR_VERIFY_MS: [14.91, 15.49, 15.87, 16.21, 18.50, 21.90]}
    for table, engine_ms in want.items():
        assert [x for x, _ in table] == [128, 1024, 8192, 33000, 128000, 262000]
        for (x, ms), e in zip(table, engine_ms):
            assert spec.interpolate(table, x) == pytest.approx(e + spec.SERVED_HOST_MS)
        grid = [spec.interpolate(table, p) for p in range(0, 270_000, 500)]
        assert all(b >= a for a, b in zip(grid, grid[1:]))  # monotone in position
        assert spec.interpolate(table, 1) == table[0][1] and spec.interpolate(table, 10**6) == table[-1][1]
    # a verify costs more than a plain step at every position, by a ratio that shrinks with context
    ratios = [spec.interpolate(spec.PRIOR_VERIFY_MS, p) / spec.interpolate(spec.PRIOR_PLAIN_MS, p) for p in (1024, 64_000, 262_000)]
    assert all(r > 1 for r in ratios) and ratios == sorted(ratios, reverse=True)


def test_policy_deterministic_at_1k_64k_262k():
    """decide() is a function of the request alone at short, middle and long context: the same requests at 1K, 64K and
    262K positions give identical decision sequences on fresh policies and in any order of serving."""
    cases = [("prose", 1_000), ("code", 64_000), ("math", 261_800)]
    alone = {c: drive(ResidentSpecPolicy(K_DRAFT, mode="adaptive"), c[0], start=c[1]) for c in cases}
    for order in (cases, cases[::-1], [cases[1], cases[0], cases[2]]):
        policy = ResidentSpecPolicy(K_DRAFT, mode="adaptive")
        for c in order:
            assert drive(policy, c[0], start=c[1]) == alone[c], c
    assert drive(ResidentSpecPolicy(K_DRAFT, mode="adaptive"), "math", start=261_800) == alone[("math", 261_800)]
