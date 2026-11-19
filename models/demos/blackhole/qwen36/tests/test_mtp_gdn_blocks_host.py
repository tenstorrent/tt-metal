# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): Qwen36MTPContractForCausalLM's speculative verify on the GDN state in vLLM KV cache blocks.

* GdnBlockLayout.verify_tables, the one rule that turns mamba_state (start_blocks, windows) into gdn_spec_step's
  ctrl page and page index: which pages each row reads and writes, padding rows, a start block among the row's own
  windows, row moves.
* The rule against the plugin's block selection (tests/spec_block_helpers.py) on a ledger of what every block holds:
  over random acceptance, row moves, padding rows, block crossings (mode "align") and block reuse, every verify reads
  the state of exactly the tokens its request committed.
* A chained verify at a block crossing under asynchronous scheduling: the ctrl start words of the device count
  (GdnBlockState.ctrl_candidates) and the window equal those of the drained verify of the same step.
* The device tests' asynchronous runner (test_mtp_gdn_blocks._async_run) against a stand-in of the model class that
  checks the plugin contract on every call: the chained and the drained run of the same requests agree, also while
  requests leave middle rows and the runner sends slot_remap, and while requests join with a prefill behind the
  unread verify.
* GdnBlockState.set_verify_blocks: one ctrl page and one index per MambaSpec group, bound per group at allocation.
* MTPBlockEngine's bookkeeping: the drafter starts only after a prompt's last chunk, a row's first verify writes the
  drafter KV of the prompt's last position before the verify, and a row move moves no GDN state.
* TPGatedDeltaNet's verify call: gdn_spec_step's speed options at the geometry the operation's tests cover only.
"""

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.gdn_block_helpers import PluginLayout
from models.demos.blackhole.qwen36.tests.spec_block_helpers import SpecBlocks
from models.demos.blackhole.qwen36.tt import gdn_block_state as gbs
from models.demos.blackhole.qwen36.tt.kv_cache_spec import SPEC_STEP_HOLD, GdnDims, plan_gdn_block_layout

TP8 = GdnDims(nk=2, nv=6, dk=128, dv=128, kconv=4)
ATTENTION = [i % 4 == 3 for i in range(16)]  # 4 attention layers, 12 GDN layers in 3 groups
NUM_BLOCKS = 200


def _layout(kv_bf8=True, block_size=1024):
    """The TP=8 Qwen3.8 layout of a bfloat8_b 1024-token block: heads 0..3 in K, 4..5 and the history in V."""
    return plan_gdn_block_layout(block_size=block_size, kv_heads=1, head_dim=256, kv_bf8=kv_bf8, dims=TP8)


def _pages(lay, block):
    """(state page per head, history page per head) of ``block``, as index_host names them."""
    idx = lay.index_host([block], NUM_BLOCKS)[:, 0].tolist()
    return idx[: TP8.nv], idx[TP8.nv :]


# ------------------------------------------------------------------------------------------- verify_tables


@pytest.mark.parametrize("K", [1, 3])
def test_verify_tables_name_the_start_and_window_pages(K):
    T, rows, nv = K + 1, 4, TP8.nv
    lay = _layout()
    starts = [5, 0, 9, 12]  # row 1 is a padding row
    windows = torch.tensor([[6 + t for t in range(T)], [0] * T, [20 + 2 * t for t in range(T)], [12] + [30] * K])
    ctrl, idx = lay.verify_tables(starts, windows, NUM_BLOCKS, rows)
    assert ctrl.shape[0] == 1 and ctrl.shape[1] % 16 == 0 and ctrl.shape[1] >= 1 + rows + 2 * rows * nv
    assert idx.shape == (rows, 2 * T * nv) and ctrl.dtype == idx.dtype == torch.int32
    assert not ctrl[0, : 1 + rows].any(), "window parity and mi words must be 0"
    for r in range(rows):
        rec, hist = _pages(lay, starts[r])
        if r == 1:  # padding rows are held: they write no GDN state into the null block
            rec = hist = [SPEC_STEP_HOLD] * nv
        assert ctrl[0, 1 + rows + r * nv : 1 + rows + (r + 1) * nv].tolist() == rec
        assert ctrl[0, 1 + rows + rows * nv + r * nv : 1 + rows + rows * nv + (r + 1) * nv].tolist() == hist
        for t in range(T):
            rec, hist = _pages(lay, int(windows[r, t]))
            assert idx[r, t * nv : (t + 1) * nv].tolist() == rec, f"row {r} token {t}: state pages"
            assert idx[r, T * nv + t * nv : T * nv + (t + 1) * nv].tolist() == hist, f"row {r} token {t}: history"
    # heads in V are addressed past the K view's pages (gdn_spec_step's ring_pool2), history in the bf16 view
    rec5, _ = _pages(lay, 5)
    assert lay.heads_k == 4 and rec5[4] >= lay.fp32_pages(NUM_BLOCKS) > rec5[3]


def test_verify_tables_pad_missing_rows_and_follow_row_moves(expect_error):
    lay, T = _layout(), 4
    starts, windows = [7, 8], torch.tensor([[7, 40, 41, 42], [50, 51, 52, 53]])
    ctrl, idx = lay.verify_tables(starts, windows, NUM_BLOCKS, 4)
    null_ctrl, null_idx = lay.verify_tables([], torch.zeros(0, T), NUM_BLOCKS, 4)
    assert torch.equal(idx[2:], null_idx[2:]), "rows past the windows use the null block"
    nv = TP8.nv
    assert (ctrl[0, 5 + 2 * nv : 5 + 4 * nv] == SPEC_STEP_HOLD).all(), "rows past the windows are held"
    assert not null_ctrl[0, :5].any() and set(null_idx.unique().tolist()) == set(
        lay.index_host([0], NUM_BLOCKS)[:, 0].tolist()
    )
    moved_ctrl, moved_idx = lay.verify_tables(starts[::-1], windows.flip(0), NUM_BLOCKS, 4)
    assert torch.equal(moved_idx[:2], idx[:2].flip(0)), "a row move permutes the index rows only"
    for part in (slice(5, 5 + 2 * nv), slice(5 + 4 * nv, 5 + 6 * nv)):
        words = ctrl[0, part].reshape(2, nv)
        assert torch.equal(moved_ctrl[0, part].reshape(2, nv), words.flip(0))
    with expect_error(ValueError, ".*"):
        lay.verify_tables([1, 2, 3], windows, NUM_BLOCKS, 4)


# --------------------------------------------------------------------- the rule against the plugin's selection


class Ledger:
    """What every block holds, as the device would after running gdn_spec_step on verify_tables' pages: block ->
    (request, tokens covered). A verify's row reads the state at its start pages and writes the state after token t
    (start + t + 1 tokens, garbage past the valid drafts) at its window pages."""

    def __init__(self, lay):
        self.lay = lay
        self.blocks = {}
        self.page_block = {}

    def block_of(self, page_list):
        key = tuple(page_list)
        if key not in self.page_block:
            for b in range(NUM_BLOCKS):
                self.page_block[tuple(_pages(self.lay, b)[0] + _pages(self.lay, b)[1])] = b
        return self.page_block[key]

    def verify(self, ctrl, idx, rows, T, live):
        """live: row -> (request, tokens before the step, valid drafts). Returns row -> what its start held."""
        nv, read = TP8.nv, {}
        for r in live:
            rec = ctrl[0, 1 + rows + r * nv : 1 + rows + (r + 1) * nv].tolist()
            hist = ctrl[0, 1 + rows + rows * nv + r * nv : 1 + rows + rows * nv + (r + 1) * nv].tolist()
            read[r] = self.blocks.get(self.block_of(rec + hist))
        for r, (req, n, valid) in live.items():
            for t in range(T):
                rec = idx[r, t * nv : (t + 1) * nv].tolist()
                hist = idx[r, T * nv + t * nv : T * nv + (t + 1) * nv].tolist()
                self.blocks[self.block_of(rec + hist)] = (req, n + t + 1) if t <= valid else ("garbage", req)
        return read


@pytest.mark.parametrize("mode", ["none", "align"])
@pytest.mark.parametrize("K", [1, 3])
def test_every_verify_reads_the_state_its_request_committed(mode, K):
    """Random drafts, acceptance, row moves, padding rows and request turnover: each verify of each request starts
    from the state after exactly its committed tokens, and no live request's state is overwritten by another row."""
    T, rows = K + 1, 4
    lay = _layout()
    # the plugin's columns at 64 tokens per block in mode "align": many block crossings (the pages stay the layout's)
    plugin = PluginLayout(
        ATTENTION, 1, 256, TP8, True, num_blocks=NUM_BLOCKS, block_size=64 if mode == "align" else 1024
    )
    blocks = SpecBlocks(plugin, K, mode)
    ledger = Ledger(lay)
    rng = random.Random(7 + K)
    reqs = {}  # request -> [row, committed tokens]
    next_req = 0
    for step in range(60):
        # admit new requests into free rows: a prefill writes the state after its prompt into windows[:, 0]
        free = [r for r in range(rows) if r not in [v[0] for v in reqs.values()]]
        while free and len(reqs) < 3:
            row, n = free.pop(0), rng.randint(5, 300)
            blocks.admit(row)
            dest, _ = blocks.prefill(row, 0, n)
            ledger.blocks[dest[0]] = (next_req, n)
            reqs[next_req] = [row, n]
            next_req += 1
        # move rows: the plugin may give a request another row between steps
        if rng.random() < 0.3 and len(reqs) >= 2:
            a, b = rng.sample(list(reqs), 2)
            ra, rb = reqs[a][0], reqs[b][0]
            for d in (blocks.cols, blocks.last, blocks.count):
                d[ra], d[rb] = d[rb], d[ra]
            reqs[a][0], reqs[b][0] = rb, ra
        live, sel = {}, {}
        for req, (row, n) in reqs.items():
            valid = rng.randint(0, K)
            win, start = blocks.decode(row, n, 1 + valid)
            sel[row] = (win, start)
            live[row] = (req, n, valid)
        state = blocks.state(sel, rows)
        g0 = state.groups[0]
        ctrl, idx = lay.verify_tables(g0.start_blocks, g0.windows[:, :T], NUM_BLOCKS, rows)
        read = ledger.verify(ctrl, idx, rows, T, live)
        for row, (req, n, valid) in live.items():
            assert read[row] == (req, n), f"step {step}: request {req} in row {row} read {read[row]}, not ({req}, {n})"
            a = rng.randint(1, 1 + valid)
            blocks.commit(row, a)
            reqs[req][1] = n + a
        for req in [q for q in reqs if rng.random() < 0.1]:  # finish: the blocks go back to the pool
            blocks.release(reqs.pop(req)[0])


# ----------------------------------------------------------------------------- GdnBlockState verify tables


@pytest.fixture
def copies(monkeypatch):
    calls = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **kw: SimpleNamespace(host=t.clone()))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: None)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, dev: calls.append((host.host, dev)))
    monkeypatch.setattr(ttnn, "zeros", lambda *a, **kw: SimpleNamespace(window=True))
    return calls


def _block_state(T=4, bmax=4):
    plugin = PluginLayout(ATTENTION, 1, 256, TP8, True, num_blocks=NUM_BLOCKS, block_size=1024)
    st = object.__new__(gbs.GdnBlockState)
    st.mesh, st.bmax, st.num_blocks, st.dims, st.layout = None, bmax, NUM_BLOCKS, TP8, _layout()
    st.layers = {i: SimpleNamespace(state_blocks=SimpleNamespace()) for i, a in enumerate(ATTENTION) if not a}
    st._group_names = st._group_layers = None
    st._decode_blocks, st._index_refs = {}, {}
    st.verify_tokens = T
    st._bind_verify(plugin.per_layer_specs)
    return st, plugin


def test_verify_tables_are_bound_per_mambaspec_group(copies):
    """Every GDN layer reads its group's ctrl page and index: the k-th GDN layer of each layer slot is in group k, as
    vLLM groups them. set_verify_blocks writes each group's tables once per step, from that group's blocks."""
    st, plugin = _block_state()
    for k, layers in enumerate(plugin.gdn_groups):
        for i in layers:
            vb = st.layers[i].state_blocks.verify
            assert vb.ctrl is st._verify_ctrl[k] and vb.idx is st._verify_idx[k] and vb.T == 4
    blocks = SpecBlocks(plugin, 3, "none")
    sel = {}
    for row in (0, 2):
        blocks.admit(row)
        blocks.prefill(row, 0, 100)
        sel[row] = blocks.decode(row, 100, 4)
    state = blocks.state(sel, 4)
    copies.clear()
    st.set_verify_blocks(state)
    assert len(copies) == 2 * plugin.num_groups
    for g in range(plugin.num_groups):
        want = st.layout.verify_tables(state.groups[g].start_blocks, state.groups[g].windows, NUM_BLOCKS, 4)
        (ctrl, ctrl_dev), (idx, idx_dev) = copies[2 * g], copies[2 * g + 1]
        assert ctrl_dev is st._verify_ctrl[g] and idx_dev is st._verify_idx[g]
        assert torch.equal(ctrl, want[0]) and torch.equal(idx, want[1])


def test_unchanged_verify_tables_are_not_copied_again(copies):
    """A table the device already holds is not copied: a second verify of the same blocks copies nothing, and a row
    whose start block moves (a new accepted count) copies only the ctrl pages. Clearing the tables restages all."""
    st, plugin = _block_state()
    blocks = SpecBlocks(plugin, 3, "none")
    blocks.admit(0)
    blocks.prefill(0, 0, 100)
    sel = {0: blocks.decode(0, 100, 4)}
    st.set_verify_blocks(blocks.state(sel, 4))
    copies.clear()
    st.set_verify_blocks(blocks.state(sel, 4))
    assert copies == []
    blocks.commit(0, 2)
    sel = {0: blocks.decode(0, 102, 4)}
    st.set_verify_blocks(blocks.state(sel, 4))
    assert {id(dev) for _, dev in copies} <= {id(t) for t in st._verify_ctrl} and len(copies) == plugin.num_groups
    copies.clear()
    st.clear_verify_blocks()
    assert len(copies) == 2 * plugin.num_groups


def test_verify_tables_refuse_other_groupings_and_short_windows(copies, expect_error):
    st, plugin = _block_state()
    blocks = SpecBlocks(plugin, 3, "none")
    blocks.admit(0)
    blocks.prefill(0, 0, 10)
    state = blocks.state({0: blocks.decode(0, 10, 2)}, 4)
    swapped = list(state.groups)
    a, b = swapped[0], swapped[1]
    names_a = list(a.layer_names)
    names_a[0], names_b0 = b.layer_names[0], a.layer_names[0]
    swapped[0] = a.__class__(layer_names=tuple(names_a), windows=a.windows, start_blocks=a.start_blocks)
    swapped[1] = b.__class__(
        layer_names=(names_b0,) + b.layer_names[1:], windows=b.windows, start_blocks=b.start_blocks
    )
    with expect_error(ValueError, "k-th GDN layer"):
        st.set_verify_blocks(state.__class__(groups=tuple(swapped)))
    narrow = [
        g.__class__(layer_names=g.layer_names, windows=g.windows[:, :2], start_blocks=g.start_blocks)
        for g in state.groups
    ]
    st2, _ = _block_state()
    with expect_error(ValueError, "state blocks per row"):
        st2.set_verify_blocks(state.__class__(groups=tuple(narrow)))


@pytest.mark.parametrize("K", [1, 3])
def test_a_chained_verify_at_a_block_crossing_names_the_drained_verifys_blocks(copies, K):
    """Mode "align" under asynchronous scheduling, 64-token blocks (many crossings): vLLM schedules each verify while
    the one before it is unread and counts that one's drafts as accepted, and the window follows that count, so next to
    a crossing it begins one column past the column of the last query token at a low count. For the count the device
    computes, the ctrl start words that the chain program picks from GdnBlockState.ctrl_candidates equal those of the
    drained verify of the same step, which reads the start block of that count, and both write the same window; every
    verify reads the state its request committed. Such moved windows follow every count 1 to K."""
    T, rows, nv = K + 1, 4, TP8.nv
    st, _ = _block_state(T=T, bmax=rows)
    small = PluginLayout(ATTENTION, 1, 256, TP8, True, num_blocks=NUM_BLOCKS, block_size=64)
    chained, drained = SpecBlocks(small, K, "align"), SpecBlocks(small, K, "align")
    ledger = Ledger(st.layout)
    rng = random.Random(11 + K)
    p0 = {}  # row -> column-0 position of its current verify (its request is the row)
    for row in range(rows):
        p0[row] = rng.randint(5, 300)
        for blocks in (chained, drained):
            blocks.admit(row)
            dest, _ = blocks.prefill(row, 0, p0[row])
        ledger.blocks[dest[0]] = (row, p0[row])
    # The first verify after the prefills: nothing is in flight when vLLM schedules it.
    sel = {r: chained.decode(r, p0[r], T) for r in range(rows)}
    for r in range(rows):
        drained.decode(r, p0[r], T)
    g0 = chained.state(sel, rows).groups[0]
    ctrl, idx = st.layout.verify_tables(g0.start_blocks, g0.windows, NUM_BLOCKS, rows)
    moved = set()
    for step in range(150):
        read = ledger.verify(ctrl, idx, rows, T, {r: (r, p0[r], K) for r in range(rows)})
        assert read == {r: (r, p0[r]) for r in range(rows)}, f"step {step}: {read}"
        counts = {r: rng.randint(1, T) for r in range(rows)}
        sel_c, by_count, sel_d = {}, {}, {}
        for r in range(rows):
            scheduled = p0[r] + T
            win, by_count[r] = chained.chained(r, scheduled)
            sel_c[r] = (win, None)
            drained.commit(r, counts[r])
            sel_d[r] = drained.decode(r, p0[r] + counts[r], T, scheduled=scheduled)
            assert win == sel_d[r][0], f"step {step} row {r}: chained window {win}, drained {sel_d[r][0]}"
            if (scheduled + K) // small.block_size != (p0[r] + counts[r] + K) // small.block_size:
                moved.add(counts[r])
            p0[r] += counts[r]
        state_c = chained.state(sel_c, rows, by_count=by_count)
        st.set_verify_blocks(state_c, stage_ctrl=False)
        cand = st.ctrl_candidates(state_c).to(torch.int64)
        for g, group in enumerate(drained.state(sel_d, rows).groups):
            k = st._verify_order[1][g]
            want, want_idx = st.layout.verify_tables(group.start_blocks, group.windows, NUM_BLOCKS, rows)
            picked = torch.zeros_like(want)
            for r in range(rows):
                words = cand[(counts[r] - 1) * rows + r, 2 * nv * k : 2 * nv * (k + 1)].to(torch.int32)
                picked[0, 1 + rows + r * nv : 1 + rows + (r + 1) * nv] = words[:nv]
                picked[0, 1 + rows + rows * nv + r * nv : 1 + rows + rows * nv + (r + 1) * nv] = words[nv:]
            assert torch.equal(picked, want), f"step {step} group {g}: the chained ctrl words differ from the drained"
            if g == 0:
                ctrl, idx = picked, want_idx
    assert moved == set(range(1, T)), f"moved windows after counts {sorted(moved)}"


# ----------------------------------------------------------------------------------------- MTPBlockEngine


@pytest.fixture
def engine(monkeypatch):
    from models.demos.blackhole.qwen36.tt import spec_engine

    B, K = 4, 3
    log = []
    gdn = SimpleNamespace(remap_spec_slots=lambda idx: log.append(("gdn_remap", idx)))
    model = SimpleNamespace(
        mtp=SimpleNamespace(attention=SimpleNamespace()),
        num_devices=8,
        mesh_device=MagicMock(),
        args=SimpleNamespace(vocab_size=32, max_batch_size=B, dim=64),
        layers=[SimpleNamespace(is_full_attention=False, attention=gdn)],
        gdn_blocks=SimpleNamespace(verify_tokens=K + 1, set_verify_blocks=lambda s: log.append(("tables", s))),
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt).to(torch.int32),
        spec_feed_rows=lambda h: h,
        rope=SimpleNamespace(rope_delta=0),
    )

    def verify_traced(
        tokens, positions, mi, read_logits=False, page_tables=None, inactive=None, rope_deltas=None, after_enqueue=None
    ):
        log.append(("verify", tokens, positions, inactive, rope_deltas))
        if after_enqueue is not None:
            after_enqueue()
        return [0] * (B * (K + 1)), "feed", None

    model.verify_traced = verify_traced
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 128, 64, 10**6
    eng._last_hidden = [f"h{u}" for u in range(B)]
    eng._mtp_pt_rows = [f"pt{u}" for u in range(B)]
    monkeypatch.setattr(eng, "_stage", lambda host, buf, *a: log.append(("stage", buf, host)))
    monkeypatch.setattr(
        eng, "_warm_mtp_last", lambda h, tok, pos, pt, delta=None: log.append(("mtp_last", h, tok, pos, pt, delta))
    )
    monkeypatch.setattr(
        eng, "_warm_mtp_chunk", lambda f, cs, n, prompt, pt: log.append(("mtp_chunk", cs, n, len(prompt)))
    )
    monkeypatch.setattr(eng, "_capture_last_row", lambda f, i, slot: log.append(("last_row", i, slot)))
    return eng, log


def _verify(eng, starts, active, state="S"):
    B, T = eng.B, eng.T
    tokens = torch.arange(B * T).reshape(B, T)
    pt = torch.ones(B, eng.nb, dtype=torch.int32)
    return eng.verify(tokens, torch.tensor(starts), None, pt, active=active, mamba_state=state)


def test_drafter_starts_after_the_prompts_last_chunk(engine, expect_error):
    """An intermediate chunk keeps no last row and leaves the row out of decode; the last chunk keeps the feed row of
    the prompt's last token; the row's first verify writes that position's drafter KV with its first token, over the
    row's own drafter table and no other, after the verify tables and the replay are enqueued, and only once."""
    eng, log = engine
    hook = eng.prefill_hook(2, torch.arange(3000), torch.ones(1, 8), final=False)
    hook("h", 0, 2048)
    hook("h", 2048, 952)
    eng.finish_prefill(2, 3000, final=False)
    assert ("last_row", 2999 - 2048, 2) not in log and 2 not in eng._fresh and 2 not in eng._active
    with expect_error(ValueError, "never prefilled"):
        _verify(eng, [0, 0, 3000, 0], [False, False, True, False])
    log.clear()
    hook = eng.prefill_hook(2, torch.arange(5000), torch.ones(1, 8), final=True)
    hook("h", 3000, 2000)
    eng.finish_prefill(2, 5000, final=True)
    assert log == [("mtp_chunk", 3000, 2000, 5000), ("last_row", 1999, 2)]
    log.clear()
    _verify(eng, [0, 0, 5000, 0], [False, False, True, False])
    names = [e[0] for e in log]
    assert names == ["tables", "verify", "stage", "mtp_last"]
    assert log[2][1] == "pt2" and torch.equal(log[2][2], torch.ones(1, eng.nb, dtype=torch.int32))
    assert log[3][1:] == ("h2", 2 * eng.T, 4999, "pt2", 0)
    log.clear()
    _verify(eng, [0, 0, 5002, 0], [False, False, True, False])
    assert [e[0] for e in log] == ["tables", "verify"]
    with expect_error(ValueError, "needs mamba_state"):
        _verify(eng, [0, 0, 5003, 0], [False, False, True, False], state=None)


def test_first_verify_at_the_wrong_position_raises(engine, expect_error):
    eng, log = engine
    eng.finish_prefill(0, 100, final=True)
    with expect_error(ValueError, "prompt has 100 tokens"):
        _verify(eng, [101, 0, 0, 0], [True, False, False, False])


def test_row_moves_move_the_drafter_rows_only(engine):
    """A slot_remap moves the drafter's per-slot rows and bookkeeping; the GDN state follows mamba_state."""
    eng, log = engine
    eng.finish_prefill(0, 100, final=True)
    eng.finish_prefill(3, 50, final=True)
    eng.remap_slots([3, 1, 2, 0])
    assert not [e for e in log if e[0] == "gdn_remap"]
    assert eng._fresh == {0: 50, 3: 100} and eng._last_hidden[0] == "h3" and eng._active == {0, 3}
    eng.release(3)
    assert 3 not in eng._fresh and 3 not in eng._active


def test_slot_keeps_its_requests_mrope_delta(engine):
    """The prompt's last chunk keeps the M-RoPE delta of the row's prefill (rope.rope_delta) for the slot. The slot's
    first verify writes the drafter KV of the prompt's last position at that delta, and every verify gives each row its
    own slot's delta. The delta moves with a row move and resets when the slot is released."""
    eng, log = engine
    eng.model.rope.rope_delta = -12
    eng.prefill_hook(2, torch.arange(300), torch.ones(1, 8), final=False)("h", 0, 300)
    assert eng._rope_deltas == [0, 0, 0, 0], "an intermediate chunk keeps no delta"
    eng.prefill_hook(2, torch.arange(500), torch.ones(1, 8), final=True)("h", 300, 200)
    eng.finish_prefill(2, 500, final=True)
    eng.model.rope.rope_delta = 0  # a text prompt prefilled into slot 0 after it
    eng.prefill_hook(0, torch.arange(40), torch.ones(1, 8), final=True)("h", 0, 40)
    eng.finish_prefill(0, 40, final=True)
    assert eng._rope_deltas == [0, 0, -12, 0]
    log.clear()
    _verify(eng, [40, 0, 500, 0], [True, False, True, False])
    warms = {e[3]: e[5] for e in log if e[0] == "mtp_last"}
    assert warms == {39: 0, 499: -12}
    assert [e[4] for e in log if e[0] == "verify"] == [[0, 0, -12, 0]]
    eng.remap_slots([2, 1, 0, 3])
    assert eng._rope_deltas == [-12, 0, 0, 0]
    eng.release(0)
    assert eng._rope_deltas == [0, 0, 0, 0]


def test_drafter_rows_take_their_slots_delta(engine, monkeypatch):
    """Every drafter RoPE row takes each row's own delta: the recorded drafter's reseed rows (user-major, T per user)
    and draft steps (the pack's table rows), and the eager draft steps and reseeds."""
    from models.demos.blackhole.qwen36.tt import spec_engine
    from models.demos.blackhole.qwen36.tt.spec_step import draft_reseed_rope, draft_step_rope

    eng, _ = engine
    B, T, K = eng.B, eng.T, eng.K
    eng._rope_deltas = [0, -12, 5, 0]
    tables, steps, staged = [], [], {}
    eng.model._rope_tp_cos_sin_decode_torch = lambda pos, deltas=None: tables.append(list(deltas)) or (None, None)
    eng.model.ttnn_mtp_decode_forward = lambda *a, rope_deltas=None, **kw: steps.append(list(rope_deltas)) or (
        "logits",
        "h",
    )
    eng.model.rope_table_rows = lambda: 4096
    eng._stage = lambda host, buf, *a: staged.__setitem__(buf, host)
    eng._dr_pack, eng._dr_rs_pt, eng._dr_rs_staged = "pack", "rs_pt", None
    eng._read_ids = lambda ts: [[0] * B for _ in ts]
    eng._dr_outs = [None] * K
    eng.mtp_extra_steps, eng.mesh = 0, None
    monkeypatch.setattr(ttnn, "execute_trace", lambda *a, **kw: None)
    monkeypatch.setattr(spec_engine, "draft_argmax", lambda mtp, logits: "ids")
    ctok = torch.zeros(B, T, dtype=torch.int64)
    cpos = torch.arange(1, T + 1).repeat(B, 1)
    pt = torch.ones(B, eng.nb, dtype=torch.int32)
    m, p0, no = [1, 2, 0, 0], [100] * B, [False] * B
    eng._propose_recorded(ctok, m, p0, no, no, pt)
    pack = staged["pack"]
    reseed = pack[draft_reseed_rope(K), : B * T].tolist()
    assert reseed == [100 + d + j for d in eng._rope_deltas for j in range(T)]
    for k in range(K):
        assert pack[draft_step_rope(K, k), :B].tolist() == [100 + mu + k + d for mu, d in zip(m, eng._rope_deltas)]
    assert tables == [], "the recorded drafter stages table rows, no host cos / sin"
    eng._vfeed = SimpleNamespace(shape=(1, 1, B * T, 64))
    eng.mtp = SimpleNamespace(forward_decode=lambda *a, **kw: (None, "h"))
    eng._reseed_mtp_batched(ctok, cpos, m, pt)
    assert tables == [[d for d in eng._rope_deltas for _ in range(T)]]
    eng._draft([0] * B, "anchor", [7] * B)
    assert steps == [eng._rope_deltas] * K
    steps.clear()
    eng._eager_row = lambda feed, r: "row"
    eng._reseed_mtp(ctok, cpos, m)
    assert steps == [[0], [-12], [-12]]


TP8_GEOMETRY = (6, 2, 128, 128, 4)  # Nv, Nk, Dk, Dv, conv kernel per device at TP=8


@pytest.mark.parametrize(
    "geometry, T, fast",
    [(TP8_GEOMETRY, 4, True), (TP8_GEOMETRY, 12, True), (TP8_GEOMETRY, 1, False), ((12, 4, 128, 128, 4), 4, False)],
    ids=["tp8-T4", "tp8-T12", "tp8-T1", "tp4-T4"],
)
def test_a_verify_passes_the_fast_gdn_spec_step_options_at_the_tested_geometry(monkeypatch, geometry, T, fast):
    """TPGatedDeltaNet._forward_verify_blocks hands gdn_spec_step value_blocks=4 and kernel_opts=1274 on a verify
    (T > 1) at the TP=8 geometry the operation's tests cover, and no option at any other geometry (TP=4: Nv = 12)."""
    from models.demos.blackhole.qwen36.tt.gdn import tp

    calls = []
    monkeypatch.setattr(ttnn.experimental.gdn, "gdn_spec_step", lambda *a, **k: calls.append(k) or "gated")
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: SimpleNamespace(shape=shape))
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(tp.tpc, "matmul_1d_decode", lambda *a, **k: SimpleNamespace(shape=(1, 1, 32, 64)))
    monkeypatch.setattr(tp, "tt_all_reduce", lambda partial, *a, **k: partial)
    nv, nk, dk, dv, kernel = geometry
    layer = SimpleNamespace(
        Nv=nv, Nk=nk, Dk=dk, Dv=dv, K=kernel, qkvz_dim_tp=64, scale=1.0, mesh=None, tt_ccl=None, cfg=None,
        tw=dict.fromkeys(["qkvz", "conv_taps_rows", "dt_bias_fp32", "neg_exp_A_fp32", "norm_w_1d", "out"]),
        args=SimpleNamespace(gdn_qkvz_decode_1d_progcfg=None, ccl_topology=lambda: None),
        state_blocks=SimpleNamespace(verify=SimpleNamespace(T=T, ctrl="ctrl", idx="idx"), rec_k="rk", rec_v="rv", hist="h"),
        _row_proj=lambda gated, w: SimpleNamespace(shape=(1, 32, 64)),
    )  # fmt: skip
    layer._spec_step_options = lambda t: tp.TPGatedDeltaNet._spec_step_options(layer, t)
    tp.TPGatedDeltaNet._forward_verify_blocks(layer, SimpleNamespace(shape=(1, 32, 64)), 2 * T, 2)
    (kw,) = calls
    assert kw["conv_hist"] == "h" and kw["ring_pool2"] == "rv" and kw["ring_page_idx"] == "idx"
    for key, value in tp.SPEC_STEP_FAST_OPTIONS.items():
        assert kw.get(key) == (value if fast else None), f"{key} at geometry {geometry}, T = {T}"


def _next_token(seq, vocab):
    """The stand-in model's greedy token after the token sequence ``seq``."""
    h = len(seq)
    for t in seq[-6:]:
        h = (h * 1000003 + int(t)) % 2147483647
    return h % vocab


def _stand_in_drafts(seq, K, vocab):
    """The stand-in drafter's K drafts after ``seq``: the model's tokens, about one in three of them wrong."""
    out = []
    for k in range(K):
        t = _next_token(seq + out, vocab)
        out.append((t + 1) % vocab if _next_token(seq + out + [k + 7], 97) < 30 else t)
    return out


def _accept(drafts, ids):
    c = 0
    while c < len(drafts) and drafts[c] == ids[c]:
        c += 1
    return c + 1


class _ContractModel:
    """Stand-in of Qwen36MTPContractForCausalLM for _async_run, on the host: a deterministic language model and drafter
    over B state slots. Each slot holds the inputs its request consumed; a verify of a slot first consumes the inputs of
    the slot's last verify up to the count of the greedy walk, then answers the model's token after each input. A
    chained verify (spec_predecessor) builds each live row from the predecessor row that slot_remap names: the token at
    the walk's count, then the drafter's drafts. Every call checks what the plugin promises: a slot_remap is a
    permutation applied before the step; a drained verify's tokens, positions and counts continue the slot's last
    verify; a chained row continues a live row of the predecessor whose slot was not released, at the placeholder
    positions of a full acceptance; a propose names the committed tokens of the oldest verify still to propose, in its
    rows; a prefill takes a free slot, and no verify chains over a verify a prefill went behind."""

    def __init__(self, B, K, vocab):
        self.B, self.K, self.T, self.vocab = B, K, K + 1, vocab
        self.slots = [None] * B  # per slot: (inputs consumed, its last verify's row)
        self.to_propose = []  # per verify still to propose: its rows {row: (inputs before it, block, ids, drafts)}
        self.engine = SimpleNamespace(chained_steps=0, materialized_steps=0, prefills_over_pending=0)
        self.last = None
        self.prefill_behind_last = False

    def _engine(self):
        return self.engine

    def zero_kv_cache_blocks(self, kv, block_ids):
        pass

    def release_request(self, slot):
        assert self.slots[slot] is not None, f"slot {slot} holds no request"
        self.slots[slot] = None

    def prefill_forward(self, tokens, page_table, kv, prompt_lens, start_pos, empty_slots, mamba_state):
        (slot,) = empty_slots
        assert self.slots[slot] is None, f"a prefill into slot {slot}, which holds a request"
        if self.to_propose:
            self.engine.prefills_over_pending += 1
            self.prefill_behind_last = True
        prompt = tokens[0, : prompt_lens[0]].tolist()
        first = _next_token(prompt, self.vocab)
        self.slots[slot] = (prompt, (prompt, [], [first], 0))  # the prompt consumed, its first token next
        logits = torch.zeros(1, 1, self.vocab)
        logits[0, 0, first] = 1
        return logits, None

    def _consumed(self, slot):
        """The slot's inputs after its last verify's commit, and the next input (the token at the walk's count)."""
        base, block, ids, nv = self.slots[slot][1]
        c = _accept(block[1 : nv + 1], ids)
        return base + block[:c], ids[c - 1]

    def decode_forward(self, tokens, start_pos, page_table, slot_remap=None, spec_predecessor=None, **kw):
        B, T, K = self.B, self.T, self.K
        if slot_remap is not None:
            remap = [int(x) for x in slot_remap]
            assert sorted(remap) == list(range(B)), remap
            self.slots = [self.slots[s] for s in remap]
        if "spec_mode" not in kw:  # an ordinary [B, 1] decode: a verify of no draft, as the model runs it
            out = torch.zeros(B, 1, self.vocab)
            rows = {}
            for r in range(B):
                if int(start_pos[r]) < 0:
                    continue
                seq, x = self._consumed(r)
                tok = int(tokens[r, 0])
                assert tok == x and int(start_pos[r]) == len(seq), f"row {r}: the token after the prefill"
                rows[r] = (seq, [tok], [_next_token(seq + [tok], self.vocab)], 0)
                self.slots[r] = (seq, rows[r])
                out[r, 0, rows[r][2][0]] = 1
            self.to_propose.append(rows)
            return out
        nv = torch.as_tensor(kw["num_valid_drafts"]).tolist()
        rows = {}
        if spec_predecessor is not None:
            assert spec_predecessor is self.last, "the predecessor is the last verify"
            assert not self.prefill_behind_last, "a chained verify over a verify a prefill went behind"
            pred_rows = self.to_propose[-1]
            self.engine.chained_steps += 1
        for r in range(B):
            if int(start_pos[r, 0]) < 0:
                continue
            assert self.slots[r] is not None, f"row {r} reads a released slot"
            seq, x = self._consumed(r)
            if spec_predecessor is None:
                n = nv[r]
                block = [int(t) for t in tokens[r, : n + 1]]
                assert block[0] == x, f"row {r}: block {block} does not continue with {x}"
                assert block[1:] == _stand_in_drafts(seq + [block[0]], K, self.vocab)[:n], f"row {r}: drafts"
                assert int(start_pos[r, 0]) == len(seq), f"row {r}: position {int(start_pos[r, 0])} != {len(seq)}"
                last_row = self.slots[r][1]
                c = _accept(last_row[1][1 : last_row[3] + 1], last_row[2])
                assert int(kw["accepted_counts"][r]) == c, f"row {r}: accepted_counts"
            else:
                last_row = self.slots[r][1]
                assert any(last_row is v for v in pred_rows.values()), f"row {r} continues no row of the predecessor"
                assert int(start_pos[r, 0]) == len(last_row[0]) + T, f"row {r}: placeholder positions"
                n = min(nv[r], K)
                block = [x] + _stand_in_drafts(seq + [x], K, self.vocab)[:n]
            ids = [_next_token(seq + block[: j + 1], self.vocab) for j in range(n + 1)]
            rows[r] = (seq, block, ids, n)
            self.slots[r] = (seq, rows[r])
        self.to_propose.append(rows)
        self.prefill_behind_last = False
        ids = torch.full((B, T), -1, dtype=torch.int64)
        for r, (_, _, i, _) in rows.items():
            ids[r, : len(i)] = torch.tensor(i)
        self.last = SimpleNamespace(argmax_ids=ids)
        return self.last

    def read_decode_output(self, ids):
        return ids

    def propose_draft_tokens(self, K, committed, positions, counts):
        rows = self.to_propose.pop(0)
        drafts = torch.zeros(self.B, K, dtype=torch.int32)
        num_valid = torch.zeros(self.B, dtype=torch.int32)
        for r, (base, block, ids, n) in rows.items():
            c = _accept(block[1 : n + 1], ids)
            assert int(counts[r]) == c and committed[r, :c].tolist() == ids[:c], f"row {r}: the commit of another row"
            assert int(positions[r, 0]) == len(base) + 1, f"row {r}: committed positions"
            drafts[r] = torch.tensor(_stand_in_drafts(base + block[:c] + [ids[c - 1]], K, self.vocab))
            num_valid[r] = K
        return SimpleNamespace(draft_token_ids=drafts, num_valid=num_valid)


def _contract_pair(B=4, K=3, block_size=64):
    layout = SimpleNamespace(block_size=block_size, num_groups=2, group_names=[("a",), ("b",)], num_blocks=4096)
    bpu = 8
    return SimpleNamespace(
        K=K, bmax=B, bs=block_size, vocab=1000, layout=layout, bpu=bpu, kv_spec=None, spec_model=None,
        spec=_ContractModel(B, K, 1000),
        pages=lambda q: torch.arange(1 + q * bpu, 1 + (q + 1) * bpu, dtype=torch.int32),
        state=lambda model, blocks: {0: [(torch.tensor(blocks), torch.tensor(blocks))]},
    )  # fmt: skip


@pytest.mark.parametrize("kind", ["finish", "skip"])
def test_the_async_runner_moves_rows_as_the_plugin_does(kind):
    """test_mtp_gdn_blocks._async_run on the contract stand-in: requests leave middle rows (finished and released, or
    left out with their slots held and released one verify later), the runner condenses the rows and sends slot_remap;
    every call keeps the plugin contract, and the chained run equals the drained run (committed tokens, counts and
    the blocks of each remaining request's last committed count)."""
    from models.demos.blackhole.qwen36.tests.test_mtp_gdn_blocks import _async_run, _condense

    assert _condense([0, 1, 2, 3], {1}) == [0, 3, 2] and _condense([0, 3, 2], {0}) == [2, 3]
    assert _condense([0, 1, 2, 3], {3}) == [0, 1, 2] and _condense([0, 1, 2, 3], {0, 2}) == [3, 1]
    K = 3
    prompts = {q: list(range(10 + q, 60 + 7 * q)) for q in range(4)}
    hint = lambda q, p0: K if p0 % 5 else 1
    leave = {9: {1: kind}, 19: {0: kind}}
    runs = {}
    for chained in (False, True):
        pair = _contract_pair()
        runs[chained] = _async_run(pair, prompts, chained, hint, 30, leave)
        assert all(s is None for s in pair.spec.slots), "every slot released at the end"
    (drained, drained_states, _, _), (chained, chained_states, moved, remapped) = runs[False], runs[True]
    assert drained == chained and len(drained) == 30
    assert {q for q in drained[8]} == {0, 1, 2, 3} and {q for q in drained[9]} == {0, 2, 3}
    assert {q for q in drained[19]} == {2, 3}
    assert remapped == 2 and set(drained_states) == {2, 3}
    for q in drained_states:
        assert torch.equal(drained_states[q][0][0][0], chained_states[q][0][0][0])
    accepted = {c for record in drained for _, (p0, commits) in record.items() for c in [len(commits)]}
    assert accepted == set(range(1, K + 2)), "every count"


def test_the_async_runner_without_moves_keeps_its_rows():
    """Without requests that leave, _async_run sends no slot_remap and both runs agree (the block crossing test's
    runs)."""
    from models.demos.blackhole.qwen36.tests.test_mtp_gdn_blocks import _async_run

    K = 3
    prompts = {q: list(range(5 + q, 40 + 9 * q)) for q in range(4)}
    hint = lambda q, p0: K
    runs = [_async_run(_contract_pair(), prompts, chained, hint, 20) for chained in (False, True)]
    assert runs[0][0] == runs[1][0] and runs[1][3] == 0


def test_the_async_runner_prefills_joining_requests_as_the_plugin_does():
    """test_mtp_gdn_blocks._async_run with joins on the contract stand-in: a request joins into the free slot 3, and
    later one into the slot and row of a middle-row request that finished; the chained run prefills while the verify
    before the join is unread (each prefill behind it) and builds the joining request's first verify from the host. It
    equals the drained run, and the drained run prefills behind nothing."""
    from models.demos.blackhole.qwen36.tests.test_mtp_gdn_blocks import _async_run, _condense

    assert _condense([0, 1, 2], {1}, [5]) == [0, 5, 2] and _condense([0, 1, 2], set(), [4]) == [0, 1, 2, 4]
    assert _condense([0, 1, 2, 4], {0, 1}, [5]) == [5, 4, 2]
    K = 3
    prompts = {q: list(range(10 + q, 60 + 7 * q)) for q in range(3)}
    joins = {6: {4: list(range(3, 40))}, 16: {5: list(range(8, 70))}}
    leave = {16: {1: "finish"}}
    hint = lambda q, p0: K if p0 % 5 else 1
    runs = {}
    for chained in (False, True):
        pair = _contract_pair()
        runs[chained] = _async_run(pair, prompts, chained, hint, 30, leave, joins)
        assert pair.spec.engine.prefills_over_pending == (2 if chained else 0)
        assert all(s is None for s in pair.spec.slots)
    drained, chained = runs[False][0], runs[True][0]
    assert drained == chained
    assert set(drained[6]) == {0, 1, 2, 4} and set(drained[16]) == {0, 5, 2, 4} and 1 in drained[15]
    assert set(runs[False][1]) == set(runs[True][1]) == {0, 2, 4, 5}
    for q in runs[False][1]:
        assert torch.equal(runs[False][1][q][0][0][0], runs[True][1][q][0][0][0])
