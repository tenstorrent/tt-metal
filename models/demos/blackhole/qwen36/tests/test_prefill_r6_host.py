# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of the round-6 prefill / join host levers:

* QWEN36_PREFILL_HOST_TOLIST: MTPSpecEngine._prefill_row converts the prompt with one tolist() (spec_engine._prompt_ids_fast)
  and hashes the chunk-resume digests from the int32 row; the ids, the row, the digests and every call the prefill makes
  are the same as with the per-token conversion.
* QWEN36_JOIN_LOGITS_ONE_COPY: the replicated final logits are read from device 0 only; same values.
* QWEN36_PREFILL_GDN_GLUE: the one-hot selector that replaces the mid-tile b slice picks b exactly (torch model of it).

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_prefill_r6_host.py -q
"""
import types

import numpy as np
import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_chunked_prefill_host import NB, SLOT, _setup, fake_pt
from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import VOCAB, _row, fake_ttnn
from models.demos.blackhole.qwen36.tests.test_spec_prefix_fast_host import _cache, _prompt
from models.demos.blackhole.qwen36.tt import spec_engine as se
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

__all__ = ["fake_ttnn", "fake_pt"]  # fixture re-exports

R6_FLAGS = (
    "QWEN36_PREFILL_HOST_TOLIST",
    "QWEN36_JOIN_LOGITS_ONE_COPY",
    "QWEN36_JOIN_TOPP_CANDIDATES",
    "QWEN36_PREFILL_GDN_GLUE",
)


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    for f in R6_FLAGS:
        monkeypatch.delenv(f, raising=False)
    monkeypatch.setattr(tpc, "_R6_FLAGS", {})
    yield


def _set(monkeypatch, name, on):
    monkeypatch.setattr(tpc, "_R6_FLAGS", {})
    if on:
        monkeypatch.setenv(name, "1")
    else:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# flags
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", R6_FLAGS)
def test_flags_default_off_and_read_once(monkeypatch, name):
    fn = {
        "QWEN36_PREFILL_HOST_TOLIST": tpc.prefill_host_tolist_enabled,
        "QWEN36_JOIN_LOGITS_ONE_COPY": tpc.join_logits_one_copy_enabled,
        "QWEN36_JOIN_TOPP_CANDIDATES": tpc.join_topp_candidates_enabled,
        "QWEN36_PREFILL_GDN_GLUE": tpc.prefill_gdn_glue_enabled,
    }[name]
    assert fn() is False
    monkeypatch.setenv(name, "1")
    assert fn() is False  # read once
    monkeypatch.setattr(tpc, "_R6_FLAGS", {})
    assert fn() is True
    monkeypatch.setenv(name, "0")
    assert fn() is True  # read once
    for raw in ("0", "", "true", "yes"):
        monkeypatch.setattr(tpc, "_R6_FLAGS", {})
        monkeypatch.setenv(name, raw)
        assert fn() is False


# --------------------------------------------------------------------------- #
# QWEN36_PREFILL_HOST_TOLIST
# --------------------------------------------------------------------------- #
def _reference(prompt_ids):
    lst = [int(t) for t in torch.as_tensor(prompt_ids).reshape(-1)]
    return lst, torch.tensor([lst], dtype=torch.int32)


@pytest.mark.parametrize(
    "ids",
    [
        torch.randint(0, 248320, (1, 9000), dtype=torch.int64),
        torch.randint(0, 248320, (9000,), dtype=torch.int32),
        torch.randint(0, 248320, (4, 300), dtype=torch.int64)[2, :257],  # non-contiguous-able view of a batch row
        list(range(5, 777)),
        np.arange(3, 1000, dtype=np.int64),
        torch.zeros(0, dtype=torch.int64),
    ],
)
def test_prompt_ids_fast_matches_per_token(ids):
    lst, row = se._prompt_ids_fast(ids)
    ref_lst, ref_row = _reference(ids)
    assert lst == ref_lst and all(type(x) is int for x in lst)
    assert row.dtype == torch.int32 and tuple(row.shape) == tuple(ref_row.shape) and torch.equal(row, ref_row)
    assert row.is_contiguous()


def test_prompt_ids_fast_copies():
    ids = torch.arange(100, dtype=torch.int32)
    _, row = se._prompt_ids_fast(ids)
    ids[0] = 12345
    assert int(row[0, 0]) == 0  # the prompt row never aliases the caller's tokens


@pytest.mark.parametrize("ids", [torch.tensor([1.0, 2.7]), torch.tensor([True, False]), [1.5, 2.0]])
def test_prompt_ids_fast_declines_non_integer(ids):
    assert se._prompt_ids_fast(ids) == (None, None)


@pytest.mark.parametrize("e", [0, 1, 128, 4095, 4096, 9000, 12000])
def test_chunk_digest_row_equals_list(e):
    ids = _prompt(9000, seed=3)
    _, row = se._prompt_ids_fast(ids)
    assert MTPSpecEngine._chunk_digest(row[0], e) == MTPSpecEngine._chunk_digest(ids, e)


def _norm(x):
    """Comparable form of a recorded call argument: tensors (at any depth) as (dtype, shape, values)."""
    if isinstance(x, torch.Tensor):
        return ("tensor", str(x.dtype), tuple(x.shape), x.tolist())
    if isinstance(x, (list, tuple)):
        return type(x).__name__, tuple(_norm(y) for y in x)
    if isinstance(x, dict):
        return tuple(sorted((str(k), _norm(v)) for k, v in x.items()))
    return x


def _chunked_run(monkeypatch, on, tokens):
    """Option B chunked prefill of one prompt: non-final [0, 4096), non-final [4096, 8192), final [8192, 9000), all calls
    through the real _prefill_row / _prefill_fast / plan_spec_prefix / GdnPrefixStateCache over fakes."""
    _set(monkeypatch, "QWEN36_PREFILL_HOST_TOLIST", on)
    cache, calls = _cache(), []
    e = _setup(calls, cache)
    outs, progs = [], []
    for start, end, final in ((0, 4096, False), (4096, 8192, False), (8192, 9000, True)):
        ids = torch.tensor(tokens[:end], dtype=torch.int64).reshape(1, -1)[0]  # the plugin's tokens[i, :L] view
        kw = {} if final else dict(final=False)
        outs.append(MTPSpecEngine._prefill_row(e, SLOT, ids, _row(NB), None, start_pos=start, **kw))
        progs.append(dict(e._chunk_prog()))
    return dict(
        outs=outs,
        progs=progs,
        calls=[_norm(c) for c in calls],
        positions=list(cache._positions),
        lru=list(cache._lru),
        prompts={k: _norm(v) for k, v in e._prompts.items()},
        fresh=dict(e._fresh),
        exact=getattr(e, "chunk_exact_resumes", 0),
        misses=getattr(e, "chunk_exact_misses", 0),
    )


def test_prefill_row_host_tolist_identical(monkeypatch, fake_ttnn, fake_pt):
    tokens = _prompt(9000, seed=5)
    off = _chunked_run(monkeypatch, False, tokens)
    on = _chunked_run(monkeypatch, True, tokens)
    for k in off:
        if k == "outs":
            assert all(torch.equal(a, b) for a, b in zip(off[k], on[k]))
        else:
            assert off[k] == on[k], k
    assert off["exact"] == 2 and off["misses"] == 0  # both continuations resumed exactly (digests matched)
    assert on["prompts"][SLOT][1][0][0] == "list" and all(type(x) is int for x in on["prompts"][SLOT][1][0][1])


def test_prefill_row_host_tolist_digest_mismatch_still_refused(monkeypatch, fake_ttnn, fake_pt):
    """A different prompt on the row (same length) is refused exactly as today: the tensor digest differs too."""
    _set(monkeypatch, "QWEN36_PREFILL_HOST_TOLIST", True)
    cache, calls = _cache(), []
    e = _setup(calls, cache)
    a, b = _prompt(9000, seed=1), _prompt(9000, seed=2)
    MTPSpecEngine._prefill_row(e, SLOT, torch.tensor(a[:4096]), _row(NB), None, start_pos=0, final=False)
    MTPSpecEngine._prefill_row(e, SLOT, torch.tensor(b[:8192]), _row(NB), None, start_pos=4096, final=False)
    assert getattr(e, "chunk_exact_misses", 0) == 1 and getattr(e, "chunk_exact_resumes", 0) == 0


# --------------------------------------------------------------------------- #
# QWEN36_JOIN_LOGITS_ONE_COPY
# --------------------------------------------------------------------------- #
class _Replicated:
    """A fake replicated mesh tensor: n identical device copies of a [1, 1, 32, V] tile row (logical [1, 1, V])."""

    def __init__(self, n=8, v=VOCAB + 8):
        torch.manual_seed(0)
        self.copy = torch.randn(1, 1, v).to(torch.bfloat16)
        self.n = n


def test_logits_one_copy_same_values(monkeypatch):
    import ttnn

    seen = []

    def to_torch(t, mesh_composer=None, **k):
        seen.append("concat" if mesh_composer is not None else "one")
        if isinstance(t, _Replicated):
            return torch.cat([t.copy] * t.n, dim=0)
        return t  # a device-0 shard

    monkeypatch.setattr(ttnn, "to_torch", to_torch)
    monkeypatch.setattr(ttnn, "ConcatMeshToTensor", lambda mesh, dim: ("comp", dim))
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda t: [t.copy.clone() for _ in range(t.n)])
    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    e = types.SimpleNamespace(mesh="mesh", vocab=VOCAB, _skip_logits_read=False)
    e._replicated_to_torch = lambda t: MTPSpecEngine._replicated_to_torch(e, t)
    off = MTPSpecEngine._read_replicated_logits(e, _Replicated())
    _set(monkeypatch, "QWEN36_JOIN_LOGITS_ONE_COPY", True)
    on = MTPSpecEngine._read_replicated_logits(e, _Replicated())
    assert seen == ["concat", "one"]
    assert off.dtype == on.dtype == torch.float32 and off.shape == on.shape == (VOCAB,)
    assert torch.equal(off, on)


# --------------------------------------------------------------------------- #
# QWEN36_PREFILL_GDN_GLUE: the b selector (torch model of TPGatedDeltaNet._build_b_sel / _beta_select)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Nv", [1, 6, 8, 16])
def test_b_selector_picks_b_exactly(Nv):
    sel = torch.zeros(1, 2 * Nv, Nv, dtype=torch.bfloat16)
    sel[0, Nv + torch.arange(Nv), torch.arange(Nv)] = 1.0
    torch.manual_seed(Nv)
    ab = (torch.randn(1, 4096, 2 * Nv) * torch.logspace(-30, 30, 4096).reshape(1, -1, 1)).to(torch.bfloat16)
    ab[0, :8] = torch.tensor([float("-0.0")] * (2 * Nv), dtype=torch.bfloat16)
    got = (ab.float() @ sel.float()).to(torch.bfloat16)  # fp32 accumulation of one nonzero term
    want = ab[..., Nv:]
    nz = want != 0
    assert torch.equal(got[nz], want[nz])  # every value bit-exact
    assert bool((got[~nz] == 0).all())  # zeros stay zero (a -0.0 may come back +0.0)
    assert torch.equal(torch.sigmoid(got.float()), torch.sigmoid(want.float()))


# --------------------------------------------------------------------------- #
# QWEN36_PREFILL_GDN_GLUE: kda_conv_prefill vs kda_conv_prefill_tile op sequences (fake ttnn ops)
# --------------------------------------------------------------------------- #
class _FT:
    """Fake device tensor: a name, a shape, a layout and a buffer type."""

    def __init__(self, name, shape, layout, buf):
        self.name, self.shape, self.layout, self._buf = name, tuple(shape), layout, buf

    def memory_config(self):
        return types.SimpleNamespace(buffer_type=self._buf)

    def __repr__(self):
        return self.name


@pytest.fixture
def fake_kda(monkeypatch):
    import ttnn

    ops = []

    def to_layout(t, layout, memory_config=None, **k):
        ops.append(("to_layout", t.name, "RM" if layout == ttnn.ROW_MAJOR_LAYOUT else "TILE"))
        return _FT(f"{t.name}.{'rm' if layout == ttnn.ROW_MAJOR_LAYOUT else 'tile'}", t.shape, layout, ttnn.BufferType.DRAM)

    def slc(t, start, end, memory_config=None, **k):
        ops.append(("slice", t.name, tuple(start), tuple(end)))
        return _FT(f"{t.name}[{start[1]}:{end[1]},{start[2]}:{end[2]}]", [e - s for s, e in zip(start, end)], t.layout,
                   ttnn.BufferType.DRAM)

    def matmul(a, b, **k):
        ops.append(("matmul", a.name, b.name))
        return _FT(f"mm({a.name},{b.name})", (1, a.shape[1], b.shape[-1]), ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM)

    def conv(xin, hist, *a, **k):
        ops.append(("conv", xin.name, xin.layout == ttnn.TILE_LAYOUT))
        return tuple(_FT(n, (1, xin.shape[1], 32), ttnn.TILE_LAYOUT, ttnn.BufferType.DRAM) for n in "qkv")

    monkeypatch.setattr(ttnn, "to_layout", to_layout)
    monkeypatch.setattr(ttnn, "slice", slc)
    monkeypatch.setattr(ttnn, "matmul", matmul)
    monkeypatch.setattr(ttnn, "to_memory_config", lambda t, mc, *a, **k: ops.append(("to_dram", t.name)) or
                        _FT(t.name + ".dram", t.shape, t.layout, ttnn.BufferType.DRAM))
    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a, **k: ops.append(("free", t.name)))
    monkeypatch.setattr(ttnn.experimental.kda, "qkv_causal_conv1d_silu", conv)
    monkeypatch.setattr(ttnn, "QkvCausalConv1dSiluProgramConfig", lambda **k: k)
    return ops


def _kda(tile_input, conv_sel, width=1280, buf=None, T=2048):
    import ttnn

    from models.demos.blackhole.qwen36.tt.gdn.tp import kda_conv_prefill, kda_conv_prefill_tile

    qkv = _FT("qkv", (1, T, width), ttnn.TILE_LAYOUT, buf or ttnn.BufferType.L1)
    hist = _FT("hist", (1, 3, 1280), ttnn.ROW_MAJOR_LAYOUT, ttnn.BufferType.DRAM)
    sel = None if not conv_sel else _FT("conv_sel", (1, 3, T), ttnn.TILE_LAYOUT, ttnn.BufferType.L1)
    fn = kda_conv_prefill_tile if tile_input else kda_conv_prefill
    return fn(qkv, T, hist, ("t0", "t1", "t2", "t3"), (256, 256, 768), "start", conv_sel=sel)


def test_kda_glue_off_is_todays_sequence(fake_kda):
    _, _, _, st = _kda(False, False)
    assert fake_kda[0] == ("to_layout", "qkv", "RM") and fake_kda[1] == ("conv", "qkv.rm", False)
    assert ("slice", "qkv.rm", (0, 2045, 0), (1, 2048, 1280)) in fake_kda and st.name.endswith(".tile")


def test_kda_glue_tile_input_carry_from_last_tile_row(fake_kda):
    _, _, _, st = _kda(True, False)
    assert fake_kda[0] == ("conv", "qkv", True)  # the TILE projection itself, no untilize
    assert not any(o[0] == "to_layout" and o[1] == "qkv" for o in fake_kda)
    assert ("slice", "qkv", (0, 2016, 0), (1, 2048, 1280)) in fake_kda  # its last tile row only
    rm = next(o for o in fake_kda if o[0] == "slice" and o[1].endswith(".rm"))
    assert rm[2:] == ((0, 29, 0), (1, 32, 1280))  # rows T-3..T-1
    assert ("free", "qkv") not in fake_kda  # the caller owns qkv


def test_kda_glue_in_place_projection_reads_only_conv_columns(fake_kda):
    import ttnn

    _kda(True, False, width=2060, buf=ttnn.BufferType.DRAM, T=8192)
    assert fake_kda[0] == ("conv", "qkv", True)
    assert ("slice", "qkv", (0, 8160, 0), (1, 8192, 1280)) in fake_kda  # the conv columns of the last tile row


def test_kda_glue_masked_bucket_keeps_conv_sel_carry(fake_kda):
    """A masked bucket's carry is the conv_sel rows (its last REAL rows), never the bucket's last tile row (padding)."""
    _, _, _, st = _kda(True, True)
    assert fake_kda[0] == ("conv", "qkv", True)
    assert ("matmul", "conv_sel", "qkv.dram") in fake_kda and st.name == "mm(conv_sel,qkv.dram)"
    assert not any(o[0] == "slice" and o[1] == "qkv" for o in fake_kda)


def _gdn_stub(glue=True, kda=True, fuse_ab=True, fuse_agmm=True):
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet

    g = types.SimpleNamespace(_prefill_glue=glue, _gdn_kda_conv=kda, _fuse_ab=fuse_ab, _fuse_agmm=fuse_agmm)
    return lambda *a: TPGatedDeltaNet._prefill_glue_plan(g, *a)


def test_glue_plan_gates(monkeypatch):
    monkeypatch.setattr(tpc, "prefill_l1_rows_ok", lambda rows: rows <= 2048)
    plan = _gdn_stub()
    assert plan(8192, None, None) == (True, True)  # big chunk: conv reads qkvzab in place
    assert plan(2048, None, None) == (True, False)  # L1 chunk: keep the q|k|v slice, conv reads it as TILE
    assert plan(8192, None, {"conv_sel": "sel"}) == (True, False)  # masked bucket: conv_sel needs the q|k|v slice
    assert plan(8192, None, {"conv_sel": None}) == (True, True)
    assert plan(8192, 7000, None) == (False, False)  # eager valid_len path: today's ops
    assert plan(32, None, None) == (False, False)  # one tile (decode widths): today's ops
    assert plan(8200, None, None) == (False, False)  # not a multiple of 32
    assert _gdn_stub(glue=False)(8192, None, None) == (False, False)  # flag off: today's ops
    assert _gdn_stub(kda=False)(8192, None, None) == (False, False)  # no KDA conv: the MAC FIR path, untouched
    assert _gdn_stub(fuse_agmm=False)(8192, None, None) == (True, False)
