# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): Qwen36MTPContractForCausalLM's speculative verify on the GDN state in vLLM KV cache blocks.

* GdnBlockLayout.verify_tables, the one rule that turns mamba_state (start_blocks, windows) into gdn_spec_step's
  ctrl page and page index: which pages each row reads and writes, padding rows, a start block among the row's own
  windows, row moves.
* The rule against the plugin's block selection (tests/spec_block_helpers.py) on a ledger of what every block holds:
  over random acceptance, row moves, padding rows, block crossings (mode "align") and block reuse, every verify reads
  the state of exactly the tokens its request committed.
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
    )

    def verify_traced(tokens, positions, mi, read_logits=False, page_tables=None, inactive=None):
        log.append(("verify", tokens, positions, inactive))
        return [0] * (B * (K + 1)), "feed", None

    model.verify_traced = verify_traced
    eng = spec_engine.MTPBlockEngine(model, B, K)
    eng.captured, eng.nb, eng.block_size, eng._scratch_block = True, 128, 64, 10**6
    eng._last_hidden = [f"h{u}" for u in range(B)]
    eng._mtp_pt_rows = [f"pt{u}" for u in range(B)]
    monkeypatch.setattr(eng, "_stage_mtp_tables", lambda pt: log.append(("stage_mtp",)))
    monkeypatch.setattr(eng, "_warm_mtp_last", lambda h, tok, pos, pt: log.append(("mtp_last", h, tok, pos, pt)))
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
    the prompt's last token; the row's first verify writes that position's drafter KV with its first token, before
    the verify tables and the replay, and only once."""
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
    assert names == ["stage_mtp", "mtp_last", "tables", "verify"]
    assert log[1][1:] == ("h2", 2 * eng.T, 4999, "pt2")
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
    monkeypatch.setattr(ttnn.experimental.kda, "gdn_spec_step", lambda *a, **k: calls.append(k) or "gated")
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
