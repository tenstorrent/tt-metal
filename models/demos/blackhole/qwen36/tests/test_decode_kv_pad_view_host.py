# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the decode KV write of TPAttention.forward_decode keeps K and V alive until it reads them.

forward_decode pads the decode K / V ([1, B, NKV, HD] TILE, NKV < 32) to 32 rows before the paged KV write. Their tiles
already hold 32 rows, so ttnn.pad adds no tile: it fills the tile padding in place and returns a view of the same
buffer. ttnn.deallocate frees a buffer even while a view of it is live, so freeing K / V right after the pad left the KV
write reading a freed buffer that its own allocations could overwrite. These tests model both pad results: a view and a
copy.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention import tp as tp_mod

NH, NKV, HD = 3, 1, 256


class Buf:
    """A fake device tensor with a buffer that a view shares and that records whether it was freed."""

    def __init__(self, name, shape, view_of=None):
        self.name, self.shape = name, tuple(shape)
        self.buf = view_of.buf if view_of is not None else SimpleNamespace(freed=False)

    def buffer_address(self):
        return id(self.buf)

    def __repr__(self):
        return f"Buf({self.name})"


@pytest.fixture
def run(monkeypatch):
    """Run a paged kv_only forward_decode of B rows with ttnn stand-ins. Returns the pad inputs, the pad outputs and the
    reads of the pad outputs, each (consumer, pad output, its buffer still allocated)."""

    def go(B, pad_is_view, **kw):
        srcs, pads, reads = [], [], []

        def pad(t, *a, **k):
            out = Buf(f"pad({t.name})", (1, B, 32, HD), view_of=t if pad_is_view else None)
            srcs.append(t)
            pads.append(out)
            return out

        def read(consumer, t):
            if t in pads:
                reads.append((consumer, t.name, not t.buf.freed))

        def deallocate(t, *a, **k):
            if isinstance(t, Buf):
                t.buf.freed = True

        def to_memory_config(t, cfg, *a, **k):
            read("to_memory_config", t)
            return Buf(f"mc({t.name})", t.shape)

        def update(cache, x, update_idxs_tensor=None, page_table=None):
            read("paged_update_cache", x)

        def new(op):
            return lambda t, *a, **k: Buf(f"{op}({t.name})", t.shape)

        monkeypatch.setattr(ttnn, "L1_MEMORY_CONFIG", "L1")
        monkeypatch.setattr(ttnn, "pad", pad)
        monkeypatch.setattr(ttnn, "deallocate", deallocate)
        monkeypatch.setattr(ttnn, "to_memory_config", to_memory_config)
        monkeypatch.setattr(ttnn, "rms_norm", new("rms_norm"))
        monkeypatch.setattr(ttnn, "multiply", new("mul"))
        monkeypatch.setattr(ttnn, "SDPAProgramConfig", lambda **k: "sdpa_cfg")
        monkeypatch.setattr(ttnn.experimental, "paged_update_cache", update)
        monkeypatch.setattr(tp_mod, "apply_partial_rope_decode", lambda t, *a: Buf(f"rope({t.name})", t.shape))

        a = object.__new__(tp_mod.TPAttention)
        a.mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        a.args = SimpleNamespace(kv_update_shard_cfg="kv_cfg")
        a.tw = {"q_norm": "q_norm", "k_norm": "k_norm"}
        a.NH, a.NKV, a.HD, a.rope_dim, a.B = NH, NKV, HD, 64, B
        a.decode_sdpa_max_cores = None
        a.use_paged, a.paged_k, a.paged_v, a.k_caches = True, "k_cache", "v_cache", None
        a._use_nlp_decode_heads = True
        a._qkv = lambda x: ("qg", "kp", "vp")
        a._make_heads_decode = lambda qg, kp, vp, b: (
            Buf("q", (1, b, NH, HD)),
            Buf("gate", (1, b, NH, HD)),
            Buf("k", (1, b, NKV, HD)),
            Buf("v", (1, b, NKV, HD)),
        )

        def aliased(keys, values, k_p, v_p, *args, **k):
            read("_write_kv_aliased", k_p)
            read("_write_kv_aliased", v_p)

        a._write_kv_aliased = aliased
        x = SimpleNamespace(shape=(1, 1, B, 640))
        out = a.forward_decode(x, "pos", "cos", "sin", page_table=SimpleNamespace(shape=(B, 4)), kv_only=True, **kw)
        assert out is None
        return SimpleNamespace(srcs=srcs, pads=pads, reads=reads)

    return go


@pytest.mark.parametrize("pad_is_view", [True, False], ids=["pad_view", "pad_copy"])
@pytest.mark.parametrize(
    "B, kw",
    [
        (32, dict(alias_kv_write=True, spec_n_users=8)),
        (4, dict(alias_kv_write=True)),
        (8, {}),
    ],
    ids=["grouped_aliased_write", "per_row_aliased_write", "batched_write"],
)
def test_the_kv_write_reads_k_and_v_before_their_buffers_are_freed(run, B, kw, pad_is_view):
    r = run(B, pad_is_view, **kw)
    assert len(r.pads) == 2 and len(r.reads) == 2, r.reads
    assert all(alive for _, _, alive in r.reads), f"the KV write read a freed buffer: {r.reads}"


@pytest.mark.parametrize("pad_is_view", [True, False], ids=["pad_view", "pad_copy"])
def test_k_and_v_are_freed_once_after_the_kv_write(run, pad_is_view):
    """The batched write frees the padded K / V after its shard copy, and forward_decode frees the roped K and the
    head-split V when the pad copied them: no buffer is left allocated."""
    r = run(8, pad_is_view)
    assert all(t.buf.freed for t in r.srcs + r.pads)


@pytest.mark.parametrize("is_view", [True, False], ids=["view", "copy"])
def test_free_unless_view_frees_the_source_only_of_a_copy(monkeypatch, is_view):
    freed = []
    monkeypatch.setattr(ttnn, "deallocate", lambda t: freed.append(t))
    src = Buf("src", (1, 1, 8, 32))
    out = Buf("out", (1, 1, 8, 32), view_of=src if is_view else None)
    tpc.free_unless_view(src, out)
    assert freed == ([] if is_view else [src])
