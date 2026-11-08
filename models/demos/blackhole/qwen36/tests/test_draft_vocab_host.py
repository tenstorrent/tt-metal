# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for QWEN36_DRAFT_VOCAB_TOPK (tt/draft_vocab.py): flag parsing, K rounding, the id ranking (v2 rank
files and v1 counts files), the shipped rank file, the full-head fallback guard (QWEN36_DRAFT_VOCAB_FALLBACK) and, with
the ttnn package (no device), the drafter / engine wiring.

Run: pytest models/demos/blackhole/qwen36/tests/test_draft_vocab_host.py -q
"""

import json

import numpy as np
import pytest
import torch

from models.demos.blackhole.qwen36.tt import draft_vocab as dv

VOCAB = 248320
SPECIALS = list(range(248044, 248077))  # the Qwen3.8 tokenizer's added tokens (<|im_end|> = 248046)


@pytest.mark.parametrize("val,expect", [(None, 0), ("", 0), ("0", 0), ("-5", 0), ("65536", 65536), (" 98304 ", 98304)])
def test_flag(monkeypatch, val, expect):
    if val is None:
        monkeypatch.delenv(dv.DRAFT_VOCAB_TOPK_ENV, raising=False)
    else:
        monkeypatch.setenv(dv.DRAFT_VOCAB_TOPK_ENV, val)
    assert dv.DRAFT_VOCAB_TOPK_ENV == "QWEN36_DRAFT_VOCAB_TOPK"
    assert dv.draft_vocab_topk() == expect


@pytest.mark.parametrize(
    "k,expect",
    [
        (0, 0),
        (1, 256),
        (256, 256),
        (32768, 32768),
        (32769, 33024),
        (98304, 98304),
        (248064, 248064),
        (248320, 0),
        (10**6, 0),
    ],
)
def test_round_topk(k, expect):
    assert dv.round_topk(k, VOCAB, 8) == expect


def test_order_ids():
    # always (ascending) first; then the ranked ids in order without repeating always ids; then the rest ascending
    assert dv.order_ids(8, np.array([6, 1]), np.array([5, 6, 2, 9, -1])).tolist() == [1, 6, 5, 2, 0, 3, 4, 7]


def test_rank_order():
    ids = np.array([3, 5, 7, 9], dtype=np.int64)
    score = np.array([1, 5, 5, 2], dtype=np.int64)
    always = np.array([9, 2], dtype=np.int64)
    # always (ascending) first; then counted ids by count, lower id on a tie, without repeating 9; then the rest
    assert dv.rank_ids(10, ids, score, always).tolist() == [2, 9, 5, 7, 3, 0, 1, 4, 6, 8]
    # ids at or above the vocab and zero counts are ignored
    ids = np.array([1, 12], dtype=np.int64)
    assert dv.rank_ids(4, ids, np.array([0, 9]), np.zeros(0, dtype=np.int64)).tolist() == [0, 1, 2, 3]


def _shipped():
    with np.load(dv.DEFAULT_RANK_FILE, allow_pickle=False) as d:
        return d["rank"].astype(np.int64), d["always"].astype(np.int64), json.loads(str(d["meta"]))


def test_shipped_rank_file():
    assert dv.DEFAULT_RANK_FILE.exists() and dv.draft_vocab_file() == dv.DEFAULT_RANK_FILE
    rank, always, meta = _shipped()
    assert rank.ndim == 1 and len(rank) > 130000 and np.unique(rank).shape == rank.shape
    assert rank.min() >= 0 and rank.max() < VOCAB
    assert always.tolist() == SPECIALS
    assert meta["format"] == "draft_vocab_rank_v2" and meta["vocab_size"] == VOCAB
    assert meta["groups"] == {
        "qwen": 1.0,
        "en": 1.0,
        "es": 0.25,
        "pt": 0.25,
        "fr": 0.25,
        "de": 0.25,
        "it": 0.25,
        "ru": 0.25,
    }
    assert 248046 in rank[:5000].tolist()  # <|im_end|> ends every stopped Qwen completion


@pytest.mark.parametrize("k", [65536, 98304, 131072])
def test_select(k):
    sel = dv.select_draft_vocab(k, VOCAB)
    assert sel.dtype == torch.int64 and sel.shape == (k,)
    assert torch.all(sel[1:] > sel[:-1]), "sorted, unique"
    assert set(SPECIALS) <= set(sel.tolist())
    rank, _, _ = _shipped()
    assert set(rank[: k - len(SPECIALS)].tolist()) <= set(sel.tolist()), "the top of the ranking is kept"
    assert torch.equal(sel, dv.select_draft_vocab(k, VOCAB)) and dv.ids_digest(sel) == dv.ids_digest(sel.clone())
    smaller = dv.select_draft_vocab(k // 2, VOCAB)
    assert set(smaller.tolist()) <= set(sel.tolist()), "a smaller K is a subset (one ranking)"


def test_counts_file_override(monkeypatch, tmp_path):
    f = tmp_path / "counts.npz"  # a v1 counts file: ranked by count
    np.savez(f, ids=np.array([10, 20], dtype=np.uint32), completion=np.array([1, 3], dtype=np.uint32))
    monkeypatch.setenv(dv.DRAFT_VOCAB_FILE_ENV, str(f))
    assert dv.select_draft_vocab(4, 64).tolist() == [0, 1, 10, 20]
    assert dv.selection_order(64)[:3].tolist() == [20, 10, 0]


def test_rank_file_override(monkeypatch, tmp_path):
    f = tmp_path / "rank.npz"  # a v2 rank file: the stored order
    np.savez(f, rank=np.array([30, 7, 50], dtype=np.uint32), always=np.array([63], dtype=np.uint32))
    monkeypatch.setenv(dv.DRAFT_VOCAB_FILE_ENV, str(f))
    assert dv.selection_order(64)[:5].tolist() == [63, 30, 7, 50, 0]
    assert dv.select_draft_vocab(5, 64).tolist() == [0, 7, 30, 50, 63]


# ---- the full-head fallback guard (QWEN36_DRAFT_VOCAB_FALLBACK) --------------------------------------------------- #
@pytest.mark.parametrize("val,expect", [(None, True), ("1", True), ("0", False), (" 0 ", False)])
def test_fallback_flag(monkeypatch, val, expect):
    if val is None:
        monkeypatch.delenv(dv.DRAFT_VOCAB_FALLBACK_ENV, raising=False)
    else:
        monkeypatch.setenv(dv.DRAFT_VOCAB_FALLBACK_ENV, val)
    assert dv.DRAFT_VOCAB_FALLBACK_ENV == "QWEN36_DRAFT_VOCAB_FALLBACK"
    assert dv.draft_vocab_fallback() is expect


def _guard(window=8, trigger=2, rows=3):
    return dv.DraftVocabGuard(np.arange(0, 100), 200, rows, window=window, trigger=trigger)  # kept: ids 0..99


def test_guard_window():
    g = _guard()
    assert dv.GUARD_WINDOW == 128 and dv.GUARD_TRIGGER == 2
    g.observe(0, [1, 150, 2, 3])  # one out-of-set id
    assert not g.off_vocab(0) and g.misses(0) == 1
    g.observe(0, [-1, 160, 248400])  # padding / out-of-range ids are skipped; 160 is the second miss
    assert g.off_vocab(0) and g.misses(0) == 2 and g.any_off({0, 1}) and not g.any_off({1, 2}) and not g.any_off(())
    g.observe(0, [5] * 5)  # 150 leaves the 8-token window (1 150 2 3 160 5 5 5 | 5 5)
    assert g.misses(0) == 1 and not g.off_vocab(0)
    g.observe(0, [5] * 3)  # and so does 160
    assert g.misses(0) == 0
    assert [g.misses(r) for r in (1, 2)] == [0, 0]


def test_guard_long_input_and_reset():
    g = _guard()
    g.observe(1, [150, 151] + [1] * 6 + [170, 2, 3, 4, 5, 6, 7, 180])  # longer than the window: its last 8 count
    assert g.misses(1) == 2 and g.off_vocab(1)
    g.observe(1, [1])
    assert g.misses(1) == 1  # 170 left the window
    g.reset(1)
    assert g.misses(1) == 0 and g.seen[1] == 0 and not g.off_vocab(1)


def test_guard_matches_ring():
    """Same decisions as a plain ring of the last ``window`` flags, on a random stream fed in blocks of 0..4."""
    rng = np.random.default_rng(3)
    g = _guard(window=16, trigger=3, rows=2)
    ring = []
    for _ in range(400):
        n = int(rng.integers(0, 5))
        ids = rng.choice([1, 2, 3, 150], size=n, p=[0.3, 0.3, 0.3, 0.1]).tolist()
        blk = torch.full((2, 4), -1, dtype=torch.int64)
        blk[1, :n] = torch.tensor(ids, dtype=torch.int64)
        g.observe_block(blk, [0, n])
        ring = (ring + [i >= 100 for i in ids])[-16:]
        assert g.off_vocab(1) == (sum(ring) >= 3) and g.misses(1) == min(3, sum(ring)) and not g.off_vocab(0)


def test_guard_block_rows_remap():
    g = _guard()
    tok = torch.tensor([[150, 151, -1, -1], [1, 2, 3, -1], [160, 161, 162, 163]])
    g.observe_block(tok, torch.tensor([2, 3, 1]), rows={0, 1})  # row 2 not live: skipped
    assert [g.misses(r) for r in range(3)] == [2, 0, 0] and g.seen.tolist() == [2, 3, 0]
    g.observe_block(tok, [0, 0, 4])  # count 0: nothing; row 2: all four
    assert [g.misses(r) for r in range(3)] == [2, 0, 2] and g.seen.tolist() == [2, 3, 4]
    g.remap([2, 0, 1])  # row i takes row idx[i]
    assert g.seen.tolist() == [4, 2, 3] and g.off_vocab(0) and g.off_vocab(1) and not g.off_vocab(2)


def test_guard_for(monkeypatch):
    import types

    ids = torch.arange(0, 256, dtype=torch.int64)
    vp = types.SimpleNamespace(ids=ids, k=256, vocab_size=1000)
    assert dv.guard_for(types.SimpleNamespace(_vocab_prune=None), 8) is None  # no pruned head: no guard
    monkeypatch.setenv(dv.DRAFT_VOCAB_FALLBACK_ENV, "0")
    assert dv.guard_for(types.SimpleNamespace(_vocab_prune=vp), 8) is None
    monkeypatch.delenv(dv.DRAFT_VOCAB_FALLBACK_ENV)
    g = dv.guard_for(types.SimpleNamespace(_vocab_prune=vp), 8)
    assert g.rows == 8 and g.keep.shape == (1000,) and g.keep[:256].all() and not g.keep[256:].any()


# ---- wiring (ttnn ops monkeypatched with recorders; needs the ttnn package, no device) ----------------------------- #
class _FakePrune:
    """Stand-in for DraftVocabPrune: owns logits of width ``shard``; records the ids_of calls."""

    def __init__(self, shard):
        self.shard, self.weight, self.calls = shard, "PRUNED_W", []

    def owns(self, logits):
        return int(logits.shape[-1]) == self.shard

    def ids_of(self, logits, mode):
        self.calls.append(mode)
        return "pruned_ids"


def _fake_mtp(monkeypatch, vp):
    ttnn = pytest.importorskip("ttnn")
    import types

    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

    seen = []
    monkeypatch.setattr(ttnn, "linear", lambda x, w, **kw: seen.append((w, kw)) or "logits")
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: t)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(ttnn, "mesh_partition", lambda t, **kw: t)
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices = 8
    m.args = types.SimpleNamespace(get_norm_config=lambda *a: {})
    m.embd = lambda ids: types.SimpleNamespace(shape=(1, 1, 640))
    m._fuse = lambda a, b, mode: "fused"
    m.decoder = types.SimpleNamespace(forward=lambda *a, **kw: "hidden")
    m.head_norm = lambda x, mode, norm_config: "normed"
    m.draft_lm_head_weight = "DRAFT_W"
    m._sharded_argmax = True
    m._fp32_cfg = "fp32cfg"
    m._lm_head_linear = lambda x, weight=None, **kw: ttnn.linear(x, weight, **kw)
    m._vocab_prune = vp
    return m, seen


@pytest.mark.parametrize("pruned", [False, True])
def test_forward_decode_head_weight(monkeypatch, pruned):
    vp = _FakePrune(12288) if pruned else None
    m, seen = _fake_mtp(monkeypatch, vp)
    m.forward_decode("h", "tok", "pos", "cos", "sin", "pt")
    assert len(seen) == 1
    assert seen[0][0] == ("PRUNED_W" if pruned else "DRAFT_W")
    assert seen[0][1]["compute_kernel_config"] == "fp32cfg"
    with m.full_draft_head():  # QWEN36_DRAFT_VOCAB_FALLBACK: the full head even with a pruned one built
        m.forward_decode("h", "tok", "pos", "cos", "sin", "pt")
    m.forward_decode("h", "tok", "pos", "cos", "sin", "pt")  # the switch is back off after the block
    assert [w for w, _ in seen[1:]] == ["DRAFT_W", "PRUNED_W" if pruned else "DRAFT_W"]


def test_engine_head_choice(monkeypatch):
    """_dv_decide (QWEN36_DRAFT_VOCAB_FALLBACK) sets the step's head from the active rows; _dr_trace_for then picks the
    full-head trace set (bucket, else its B-row trace) and otherwise the unchanged pruned / bucket lookup."""
    import types

    pytest.importorskip("ttnn")
    from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e._dr_trace, e._dr_traces = 8, "D8", {4: "D4", 2: "D2"}
    assert e._dv_guard is None and not e._dv_full_now  # class defaults: the flag-off engine
    assert [e._dr_trace_for(b) for b in (None, 8, 4, 1)] == ["D8", "D8", "D4", "D8"]
    e._dv_guard, e._active = _guard(rows=8), {0, 3}
    e.mtp, e.dv_head_steps, e._dr_traces_full = types.SimpleNamespace(_draft_full=False), {"pruned": 0, "full": 0}, {}
    e._dv_guard.observe(5, [150, 151])  # row 5 is off-vocab but not active
    e._dv_decide()
    assert not e._dv_full_now and not e.mtp._draft_full and e._dr_trace_for(4) == "D4"
    e._dv_guard.observe(3, [150, 151])
    e._dv_decide()
    assert e._dv_full_now and e.mtp._draft_full
    assert e._dr_trace_for(4) == "D4", "no full-head traces captured: the pruned set"
    e._dr_traces_full = {8: "F8", 4: "F4", 2: "F2"}
    assert [e._dr_trace_for(b) for b in (None, 8, 4, 1)] == ["F8", "F8", "F4", "F8"]
    assert e.dv_head_steps == {"pruned": 1, "full": 1}


def test_draft_argmax_routes_pruned_logits(monkeypatch):
    import types

    pytest.importorskip("ttnn")
    from models.demos.blackhole.qwen36.tt import mtp as mtp_mod
    from models.demos.blackhole.qwen36.tt.spec_argmax_fused import ROWS

    vp = _FakePrune(12288)
    fused_calls = []
    m = types.SimpleNamespace(
        _vocab_prune=vp,
        _sharded_argmax=True,
        _fused_argmax=types.SimpleNamespace(ids=lambda lg, mode: fused_calls.append(mode) or "full_ids"),
    )
    assert mtp_mod.draft_argmax(m, types.SimpleNamespace(shape=(1, 1, 8, 12288))) == "pruned_ids"
    assert mtp_mod.draft_argmax(m, types.SimpleNamespace(shape=(1, 1, 8, 31040))) == "full_ids"
    assert vp.calls == [ROWS] and fused_calls == [ROWS]
    m._vocab_prune = None  # flag off: the shipped routing
    assert mtp_mod.draft_argmax(m, types.SimpleNamespace(shape=(1, 1, 8, 31040))) == "full_ids"


def test_build_lm_head_weight_rows(monkeypatch):
    ttnn = pytest.importorskip("ttnn")
    import pathlib
    import types

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    calls = []
    monkeypatch.setattr(ttnn, "as_tensor", lambda t, **kw: calls.append((t, kw)) or "dev")
    monkeypatch.setattr(ttnn, "ShardTensorToMesh", lambda dev, dim: ("shard", dim))
    mesh = types.SimpleNamespace(shape=(1, 8))  # a 1xN mesh: tp_shard_mapper reads its shape
    fake = types.SimpleNamespace(_lmhead_vocab_sharded=True, mesh_device=mesh, num_devices=8)
    rows = torch.tensor([1, 4, 6, 7, 9, 10, 12, 15], dtype=torch.int64)
    w = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)  # [vocab, dim]
    Qwen36Model._build_lm_head_weight(fake, w, pathlib.Path("/c"), ttnn.bfloat8_b, cache_tag=".draft_top8_x", rows=rows)
    ((t, kw),) = calls
    assert t is w and kw["cache_file_name"] == pathlib.Path("/c/output.weight.vshard.draft_top8_x")
    assert kw["mesh_mapper"] == ("shard", -1) and kw["dtype"] == ttnn.bfloat8_b
    assert torch.equal(kw["preprocess"](w), w[rows].T)  # [dim, len(rows)], rows in order
    with pytest.raises(AssertionError):  # a row subset must not reuse the base head's cache name
        Qwen36Model._build_lm_head_weight(fake, w, pathlib.Path("/c"), ttnn.bfloat8_b, rows=rows)


def test_init_vocab_prune_needs_fused_argmax(monkeypatch):
    import types

    pytest.importorskip("ttnn")
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

    m = Qwen36MTP.__new__(Qwen36MTP)
    m.num_devices, m.args = 8, types.SimpleNamespace(vocab_size=VOCAB)
    m._vocab_prune = None
    built = []
    parent = types.SimpleNamespace(_build_lm_head_weight=lambda *a, **kw: built.append(kw))
    for sharded, fused, k in ((True, None, 65536), (False, object(), 65536), (True, object(), VOCAB)):
        m._sharded_argmax, m._fused_argmax = sharded, fused
        m._init_vocab_prune(parent, {"output.weight": None}, None, k)
        assert m._vocab_prune is None
    assert built == []
