# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the prefill re-layout of the fused decode stream copies (tt/stream_relayout.py).

Random [K, N] weights are packed with the real fused_decode/stream.py stream_linear_layout (and qkv_column_order) and
recovered exactly by stream_relayout.relayout, run through a torch backend that performs the same op sequence and
shapes as the device backend (view / transpose / permute / slice / concat / pad). The LM head is packed per device as
fused_decode/terminal.py DecodeTerminal does (replicated here: that module needs the sampling stack); its logits are
moved to the prefill vocab split with vocab_resplit_pieces.

Without ttnn (e.g. a Mac) fused_decode/config.py and stream.py are loaded against a small ttnn stand-in, and the bias
tiles' BFP4 rounding (ttnn.from_torch) is replaced by the identity: the bias tile is not part of the rebuilt weight.
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
        stream._bfp4_round = lambda t: t.float()
        if saved is None:
            del sys.modules["ttnn"]
        else:
            sys.modules["ttnn"] = saved
    return stream


st = _load_stream()
rl = _load("gpt_oss_stream_relayout", _TT / "stream_relayout.py")
TILE = 32
BANKS = 8


class TorchOps:
    """relayout's torch backend: [1, 1, rows, cols] float tensors, the device backend's shapes and steps."""

    def __init__(self):
        self.calls = []

    def load(self, stream, work):
        self.calls.append("load")
        return stream.reshape(1, 1, *stream.shape[-2:]).clone()

    def cast(self, x, dtype):
        self.calls.append("cast")
        return x.clone()

    def reshape(self, x, shape):
        assert shape[-1] == x.shape[-1] and shape[-2] % TILE == 0 and x.shape[-2] % TILE == 0, "not a tile view"
        return x.reshape(shape)

    def transpose(self, x):
        self.calls.append("transpose")
        return x.transpose(-2, -1).contiguous()

    def swap01(self, x):
        self.calls.append("permute")
        return x.permute(1, 0, 2, 3).contiguous()

    def slice(self, x, starts, ends):
        assert all(v % TILE == 0 for v in list(starts[-2:]) + list(ends[-2:])), "tile-aligned slices only"
        if list(starts) == [0] * 4 and list(ends) == list(x.shape):
            return x
        self.calls.append("slice")
        return x[tuple(slice(a, b) for a, b in zip(starts, ends))].clone()

    def concat(self, pieces):
        self.calls.append("concat")
        return torch.cat(pieces, dim=3)

    def pad_cols(self, x, cols):
        self.calls.append("pad")
        return torch.nn.functional.pad(x, (0, cols))

    def free(self, x):
        pass


def _rand(k, n, gen):
    return torch.randn(k, n, generator=gen)


def _check(w, layout, plan, **kw):
    assert layout.shape == (plan.rows, BANKS * TILE)
    assert plan.rows == st.columns_per_bank(plan.n, BANKS) * (plan.k // TILE + 1) * TILE  # linear_stream_rows
    ops = TorchOps()
    out = rl.relayout(layout, plan, ops, None, **kw)
    width = kw.get("pad_n") or plan.n
    assert out.shape == (1, 1, plan.k, width)
    assert torch.equal(out[0, 0, :, : plan.n], w), "rebuilt weight differs"
    assert not out[0, 0, :, plan.n :].any(), "padding columns are not zero"
    return ops


def test_column_runs():
    assert rl.column_runs([0, 1, 2, 5, 6, 3]) == [(0, 3), (5, 2), (3, 1)]
    assert rl.column_runs([]) == []


@pytest.mark.parametrize("tp", [4])
def test_qkv_kv_first_order(tp):
    """gpt-oss-20b QKV at TP 4: K = 2880, N = (16 + 2 + 2) * 64 = 1280 (40 tiles, C = 5), K / V-first slot order."""
    gen = torch.Generator().manual_seed(1)
    k, n, q_cols = 2880, 1280, 16 * 64 // TILE
    w = _rand(k, n, gen)
    order = st.qkv_column_order(n // TILE, q_cols, BANKS)
    plan = rl.LinearStreamPlan(k, n, BANKS, order)
    assert plan.C == 5 and len(plan.runs) == 16  # 8 K / V columns (slot 0 of each bank) + 8 runs of 4 Q columns
    ops = _check(w, st.stream_linear_layout(w, torch.randn(n, generator=gen), BANKS, order), plan)
    assert ops.calls.count("transpose") == 2 and ops.calls.count("permute") == 1


def test_qkv_column_order_identity():
    """GPT_OSS_QKV_FUSED_ROPE_KV=0: the QKV stream in column order."""
    gen = torch.Generator().manual_seed(2)
    w = _rand(2880, 1280, gen)
    plan = rl.LinearStreamPlan(2880, 1280, BANKS)
    assert plan.runs == [(0, 40)]
    _check(w, st.stream_linear_layout(w, torch.randn(1280, generator=gen), BANKS), plan)


def test_o_proj():
    """o_proj at TP 4: K = 1024, N = 2880 (90 tiles, C = 12, 6 zero padding slots)."""
    gen = torch.Generator().manual_seed(3)
    w = _rand(1024, 2880, gen)
    plan = rl.LinearStreamPlan(1024, 2880, BANKS)
    assert plan.C == 12 and plan.rows == 12 * 33 * TILE
    _check(w, st.stream_linear_layout(w, torch.randn(2880, generator=gen), BANKS), plan)


def test_router_one_slot_per_bank():
    """Router: N = 32 experts, one column on bank 0 (C = 1: no transposes)."""
    gen = torch.Generator().manual_seed(4)
    w = _rand(2880, 32, gen)
    plan = rl.LinearStreamPlan(2880, 32, BANKS)
    assert plan.C == 1 and plan.rows == 91 * TILE
    ops = _check(w, st.stream_linear_layout(w, torch.randn(32, generator=gen), BANKS), plan)
    assert "transpose" not in ops.calls and "permute" not in ops.calls


@pytest.mark.parametrize("groups", [1, 3, 4])
def test_bank_groups_and_padding(groups):
    """Bank groups (identity order) give the same weight; pad_n appends zero columns."""
    gen = torch.Generator().manual_seed(5)
    k, n = 96, 37 * TILE  # C = 5, 3 padding slots on the last bank
    w = _rand(k, n, gen)
    plan = rl.LinearStreamPlan(k, n, BANKS)
    _check(w, st.stream_linear_layout(w, torch.randn(n, generator=gen), BANKS), plan, bank_groups=groups, pad_n=2048)


def test_bank_groups_need_identity_order():
    plan = rl.LinearStreamPlan(64, 1280, BANKS, st.qkv_column_order(40, 32, BANKS))
    with pytest.raises(AssertionError):
        plan.bank_groups(4)


def test_lm_head_vocab_split():
    """LM head at TP 4: decode stream device d = vocab [d * vd, (d + 1) * vd) (vd = 50272, padded ids masked by the bias
    only), rebuilt per device padded to the prefill width 65536 (model.py compute_per_device_vocab), logits moved to the
    prefill split. K is reduced to 64 (the columns are what is checked)."""
    gen = torch.Generator().manual_seed(6)
    vocab, tp, hidden = 201088, 4, 64
    vd = -(-vocab // tp // TILE) * TILE
    per_device = 1 << (vd - 1).bit_length()
    assert (vd, per_device) == (50272, 65536)
    w = torch.randn(hidden, vocab, generator=gen)  # [hidden, vocab], what DecodeTerminal transposes the HF weight to
    plan = rl.LinearStreamPlan(hidden, vd, BANKS)
    assert plan.C == 197
    logits = []  # of the identity input: each device's logits are its rebuilt weight's columns
    for d in range(tp):  # fused_decode/terminal.py DecodeTerminal.__init__
        lo, hi = d * vd, min((d + 1) * vd, vocab)
        wd = torch.zeros(hidden, vd)
        wd[:, : hi - lo] = w[:, lo:hi]
        bias = torch.zeros(vd)
        bias[hi - lo :] = -1.0e30
        layout = st.stream_linear_layout(wd, bias, BANKS)
        rebuilt = rl.relayout(layout, plan, TorchOps(), None, bank_groups=4, pad_n=per_device)[0, 0]
        assert torch.equal(rebuilt[:, :vd], wd) and not rebuilt[:, vd:].any()
        logits.append(rebuilt)
    gathered = torch.cat(logits, dim=-1)  # all_gather
    full = torch.cat([gathered[:, a:b] for a, b in rl.vocab_resplit_pieces(tp, vd, per_device)], dim=-1)
    # model.py: the prefill LM head weight zero-padded to tp * per_device and column-sharded (mesh_partition)
    ref = torch.nn.functional.pad(w, (0, tp * per_device - vocab))
    for d in range(tp):
        assert torch.equal(torch.chunk(full, tp, dim=-1)[d], torch.chunk(ref, tp, dim=-1)[d])


def test_vocab_resplit_pieces_cover_every_column_once():
    pieces = rl.vocab_resplit_pieces(4, 50272, 65536)
    cols = torch.cat([torch.arange(a, b) for a, b in pieces])
    assert torch.equal(cols.sort().values, torch.arange(4 * 65536))
    assert all(a % TILE == 0 and b % TILE == 0 for a, b in pieces)
