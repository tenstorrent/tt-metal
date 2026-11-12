# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the prefill matmul on the fused decode stream copies (tt/stream_prefill_linear.py).

Random [K, N] weights and biases are packed with the real fused_decode/stream.py stream_linear_layout (and
qkv_column_order); reference_linear runs exactly what the cores compute (the work split of plan_work, each core's
reader byte offsets into its bank, the x block with the ones tile per row, the M block / weight group / column / DST
block loops, the writer's column order and zero columns) and must equal x @ W + bias, the bias being the sum of the
stream's NBIAS bias-tile rows. The kernels' loops are mirrored by compute_order / writer_pages, checked against each
other and for covering every output tile once. The LM head is packed per device as fused_decode/terminal.py
DecodeTerminal does (replicated here: that module needs the sampling stack) and moved to the prefill vocab split with
vocab_resplit_pieces.

Without ttnn (e.g. a Mac) fused_decode/config.py and stream.py are loaded against a small ttnn stand-in, and the bias
tiles' BFP4 rounding (ttnn.from_torch) is replaced by a rounding to 4 significant bits, so the NBIAS residual terms are
all in play.

    pytest --noconftest models/demos/gpt_oss/tests/unit/test_stream_prefill_linear.py
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_TT = Path(__file__).resolve().parents[2] / "tt"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ttnn_stand_in():
    mod = types.ModuleType("ttnn")
    mod.TILE_SIZE = 32
    names = {}

    class Placeholder:
        def __init__(self, name):
            self.name = name

    def __getattr__(name):  # dtypes and the like: distinct hashable placeholders
        if name.startswith("__"):
            raise AttributeError(name)
        return names.setdefault(name, Placeholder(name))

    mod.__getattr__ = __getattr__
    return mod


def _round4(t):
    """Round to 4 significant bits (a stand-in for the device's BFP4 rounding of the bias terms)."""
    m, e = torch.frexp(t.float())
    return torch.ldexp(torch.round(m * 16) / 16, e)


def _load_stream():
    """fused_decode/stream.py as the submodule of a package that skips fused_decode/__init__.py."""
    stub = None
    try:
        import ttnn

        real = hasattr(ttnn, "from_torch")  # (not the source tree's namespace package)
    except ImportError:
        real = False
    if not real:
        saved = sys.modules.get("ttnn")
        stub = sys.modules["ttnn"] = _ttnn_stand_in()
    pkg = types.ModuleType("_gpt_oss_fused_decode")
    pkg.__path__ = [str(_TT / "fused_decode")]
    sys.modules[pkg.__name__] = pkg
    stream = importlib.import_module(f"{pkg.__name__}.stream")
    if stub is not None:
        stream._bfp4_round = _round4
        if saved is None:
            del sys.modules["ttnn"]
        else:
            sys.modules["ttnn"] = saved
    return stream


st = _load_stream()
sp = _load("gpt_oss_stream_prefill_linear", _TT / "stream_prefill_linear.py")
TILE = 32
BANKS = 8
BF16, BF8 = 2048, 1088  # tile bytes
CORES = 11 * 10  # a harvested Blackhole compute grid
BUDGET = sp.L1_CAP


def _bias_of(layout, spec):
    """[N]: the bias the stream's bias tiles add (the sum of their first NBIAS rows), in output column order."""
    b = torch.zeros(spec.n, dtype=torch.float64)
    for bank in range(BANKS):
        shard = layout[:, bank * TILE : (bank + 1) * TILE].double()
        for j, col in enumerate(spec.columns(bank, 0, spec.real[bank])):
            t = (j * spec.kt + spec.kx) * TILE
            b[col * TILE : (col + 1) * TILE] = shard[t : t + sp.NBIAS].sum(0)
    return b


def _check(k, n, m_tiles, order=None, x_tile=BF16, budget=BUDGET, out_width=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(k, n, generator=gen)
    bias = torch.randn(n, generator=gen)
    layout = st.stream_linear_layout(w, bias, BANKS, order)
    spec = sp.StreamSpec(k, n, BANKS, order)
    assert layout.shape == (spec.rows, BANKS * TILE)
    out_tiles = (out_width or n) // TILE
    plan = sp.plan_work(spec, m_tiles, CORES, x_tile, BF8, BF16, budget, out_tiles)
    x = torch.randn(m_tiles * TILE, k, generator=gen)
    got = sp.reference_linear(x, layout, spec, plan, out_tiles)
    b = _bias_of(layout, spec)
    assert torch.allclose(b, bias.double(), rtol=0, atol=2e-3 * bias.abs().max().item()), "bias terms do not sum to b"
    want = x.double() @ w.double() + b
    assert not got.isnan().any(), "an output tile was not written"
    assert torch.allclose(got[:, :n], want, rtol=1e-9, atol=1e-9), "wrong values or column order"
    assert not got[:, n:].any(), "zero columns not zero"
    return spec, plan


def _coverage(spec, plan, m_tiles, out_tiles):
    """Every output tile written once; the writer's pages follow the compute kernel's tile order."""
    seen = []
    for it in plan.items:
        cols = spec.columns(it.bank, it.j0, it.cols)
        order = sp.compute_order(it, plan)
        pages = sp.writer_pages(it, plan, cols, out_tiles)
        zeros = it.m_cnt * (it.z1 - it.z0)
        assert len(pages) == zeros + len(order)
        assert pages[zeros:] == [(it.m0 + r) * out_tiles + cols[c] for r, c in order]
        seen += pages
    assert sorted(seen) == list(range(m_tiles * out_tiles)), "an output tile is missing or written twice"


@pytest.mark.parametrize("m_tiles", [1, 4, 13])
def test_qkv_kv_first_order(m_tiles):
    """gpt-oss-20b QKV at TP 4: K = 2880, N = (16 + 2 + 2) * 64 = 1280 (40 tiles, C = 5), K / V-first slot order:
    the output comes back in Q, K, V column order."""
    k, n = 2880, 1280
    order = st.qkv_column_order(n // TILE, 16 * 64 // TILE, BANKS)
    assert order != list(range(len(order)))
    spec, plan = _check(k, n, m_tiles, order, seed=m_tiles)
    assert spec.C == 5 and spec.real == [5] * BANKS
    assert plan.resident, plan  # the core's columns are read once
    _coverage(spec, plan, m_tiles, n // TILE)


def test_o_proj_bf8_input():
    """o_proj at TP 4: K = 1024, N = 2880 (90 tiles, C = 12; bank 7 holds 6 real slots + 6 padding), bf8 input."""
    spec, plan = _check(1024, 2880, 9, x_tile=BF8, seed=3)
    assert spec.C == 12 and spec.real == [12] * 7 + [6]
    assert all(it.j0 + it.cols <= spec.real[it.bank] for it in plan.items), "a core reads padding slots"
    _coverage(spec, plan, 9, 90)


def test_router_one_bank():
    """Router: N = 32 experts, one real slot on bank 0 (C = 1): every core reads bank 0 and splits the rows."""
    spec, plan = _check(2880, 32, 6, seed=4)
    assert spec.real == [1] + [0] * 7
    assert {it.bank for it in plan.items} == {0}
    _coverage(spec, plan, 6, 1)


@pytest.mark.parametrize("budget", [BUDGET, 120 * 1024, 60 * 1024])
def test_m_blocks_and_tails(budget):
    """Smaller L1 budgets: several M blocks per core, DST blocks with tails (rows not a multiple of 4)."""
    k, n, m_tiles = 256, 37 * TILE, 23  # C = 5, 3 padding slots on the last bank
    spec, plan = _check(k, n, m_tiles, budget=budget, seed=5)
    _coverage(spec, plan, m_tiles, n // TILE)
    if budget < BUDGET:
        assert any(it.m_cnt > plan.MB for it in plan.items), plan


@pytest.mark.parametrize("G, MB", [(2, 3), (4, 8), (1, 1)])
def test_streamed_weight_groups(G, MB):
    """Weights streamed in groups of G columns per M block (the LM head's mode), partial last group and M block."""
    gen = torch.Generator().manual_seed(9)
    k, n, m_tiles = 96, 21 * TILE, 11  # C = 3
    w, bias = torch.randn(k, n, generator=gen), torch.randn(n, generator=gen)
    layout = st.stream_linear_layout(w, bias, BANKS)
    spec = sp.StreamSpec(k, n, BANKS)
    items = [sp.WorkItem(b, 0, spec.real[b], m0, min(5, m_tiles - m0)) for b in range(BANKS) for m0 in (0, 5, 10)]
    plan = sp.WorkPlan(items, MB, G, False, 2, 0.0)
    x = torch.randn(m_tiles * TILE, k, generator=gen)
    got = sp.reference_linear(x, layout, spec, plan)
    assert torch.allclose(got, x.double() @ w.double() + _bias_of(layout, spec), rtol=1e-9, atol=1e-9)
    _coverage(spec, plan, m_tiles, n // TILE)


def test_zero_columns():
    """out_width > N: the extra columns are zeros, written once each."""
    k, n = 128, 20 * TILE
    spec, plan = _check(k, n, 3, out_width=64 * TILE, seed=6)
    _coverage(spec, plan, 3, 64)


def test_plan_fits_budget_and_cores():
    """Plans for the model's shapes: within the cores and the L1 budget, x / weight circular buffers as planned."""
    for k, n, x_tile in ((2880, 1280, BF16), (1024, 2880, BF8), (2880, 32, BF16), (2880, 50272, BF16)):
        spec = sp.StreamSpec(k, n, BANKS)
        for m_tiles in (1, 4, 32, 256, 1024, 4096):
            for budget in (BUDGET, 500 * 1024):
                plan = sp.plan_work(spec, m_tiles, CORES, x_tile, BF8, BF8, budget)
                cb = plan.x_bufs * plan.MB * spec.kt * x_tile + plan.w_pages * spec.kt * BF8 + 2 * sp.MAX_RT * BF8
                assert cb <= budget and len(plan.items) <= CORES, (k, n, m_tiles, plan)
                assert sum(it.m_cnt * it.cols for it in plan.items) == m_tiles * spec.n_tiles
                if n == 50272:  # the LM head: ~19 MB of columns per bank stream through the ring
                    assert not plan.resident
                elif m_tiles >= 4 and budget == BUDGET:  # QKV / o_proj / router: each core reads its columns once
                    assert plan.resident, (k, n, m_tiles, plan)


def test_lm_head_vocab_split():
    """LM head at TP 4: decode stream device d = vocab [d * vd, (d + 1) * vd) (vd = 50272), logits written zero-padded
    to the prefill width 65536 (model.py compute_per_device_vocab), moved to the prefill split. K is reduced to 64 (the
    columns are what is checked); one row tile (the last token's)."""
    gen = torch.Generator().manual_seed(7)
    vocab, tp, hidden = 201088, 4, 64
    vd = -(-vocab // tp // TILE) * TILE
    per_device = 1 << (vd - 1).bit_length()
    assert (vd, per_device) == (50272, 65536)
    w = torch.randn(hidden, vocab, generator=gen)  # [hidden, vocab], what DecodeTerminal transposes the HF weight to
    spec = sp.StreamSpec(hidden, vd, BANKS)
    assert spec.C == 197
    x = torch.randn(TILE, hidden, generator=gen)
    logits = []
    for d in range(tp):  # fused_decode/terminal.py DecodeTerminal.__init__
        lo, hi = d * vd, min((d + 1) * vd, vocab)
        wd = torch.zeros(hidden, vd)
        wd[:, : hi - lo] = w[:, lo:hi]
        bias = torch.zeros(vd)
        bias[hi - lo :] = -1.0e30
        layout = st.stream_linear_layout(wd, bias, BANKS)
        plan = sp.plan_work(spec, 1, CORES, BF16, BF8, BF8, BUDGET, per_device // TILE)
        logits.append(sp.reference_linear(x, layout, spec, plan, per_device // TILE))
    gathered = torch.cat(logits, dim=-1)  # all_gather
    full = torch.cat([gathered[:, a:b] for a, b in sp.vocab_resplit_pieces(tp, vd, per_device)], dim=-1)
    # the stored prefill LM head was the weight zero-padded to tp * per_device and column-sharded (mesh_partition)
    ref = x.double() @ torch.nn.functional.pad(w, (0, tp * per_device - vocab)).double()
    assert torch.allclose(full, ref, rtol=1e-9, atol=1e-9)


def test_vocab_resplit_pieces_cover_every_column_once():
    pieces = sp.vocab_resplit_pieces(4, 50272, 65536)
    cols = torch.cat([torch.arange(a, b) for a, b in pieces])
    assert torch.equal(cols.sort().values, torch.arange(4 * 65536))
    assert all(a % TILE == 0 and b % TILE == 0 for a, b in pieces)


@pytest.mark.parametrize("kind", ["qkv", "router"])
def test_unpack_stream_weight(kind):
    """unpack_stream_weight (TopKRouter.with_expert_order's host copy) recovers W exactly."""
    gen = torch.Generator().manual_seed(8)
    k, n = (2880, 1280) if kind == "qkv" else (2880, 32)
    order = st.qkv_column_order(n // TILE, 32, BANKS) if kind == "qkv" else None
    w = torch.randn(k, n, generator=gen)
    layout = st.stream_linear_layout(w, torch.randn(n, generator=gen), BANKS, order)
    assert torch.equal(sp.unpack_stream_weight(layout, sp.StreamSpec(k, n, BANKS, order)), w)


def test_padding_before_real_slot_rejected():
    order = list(range(16))
    order[0], order[15] = order[15], order[0]  # bank 0 slot 0 holds padding column 15 (N = 15 tiles, C = 2)
    with pytest.raises(AssertionError):
        sp.StreamSpec(32, 15 * TILE, BANKS, order)
