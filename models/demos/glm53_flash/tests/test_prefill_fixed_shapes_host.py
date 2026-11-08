# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of ``GLM53_PREFILL_FIXED_SHAPES`` (tt-blaze G53F-0313): the index maintenance of a prefill chunk with
every shape a function of the padded chunk must leave the SAME cache bytes as the per-length path, and draw its programs
from one fixed set per (start, padded length).

``ttnn`` is replaced inside ``tt/indexer.py`` by a small torch emulation (``_Emu``) of the ops the maintenance uses
(slices, reshapes, the pooling, pad, typecast, where / compares, matmul, the paged cache writes): the emulation checks the
COMPOSITION (which rows are written, zero-padded, kept, which ring slots), the device check checks the ops themselves.
Every emulated op call is also recorded as a program signature (op + input specs + the attributes a program hash keeps;
the compare scalars are runtime args), so the tests can say which calls would compile a new program."""

from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("ttnn")
from models.demos.glm53_flash.tt import indexer as IX  # noqa: E402

P, D, TILE = 4, 128, 32


class _T:
    """An emulated device tensor: a torch tensor + the layout tag."""

    def __init__(self, t: torch.Tensor, layout="TILE"):
        self.t, self.layout = t, layout

    @property
    def shape(self):
        return tuple(self.t.shape)

    @property
    def dtype(self):
        return self.t.dtype

    def is_allocated(self):
        return True


def _spec(x):
    return (x.shape, str(x.dtype)) if isinstance(x, _T) else None


class _Emu(types.SimpleNamespace):
    """The ops of ``maintain_prefill`` / ``write_pools`` / ``_write_tail_rows`` on torch, recording program signatures."""

    bfloat16, float32, int32, uint32 = torch.bfloat16, torch.float32, torch.int32, torch.int64
    bfloat8_b = "bfp8"
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = "TILE", "RM", "DRAM"
    Tensor = _T
    MeshDevice = type("MeshDevice", (), {})

    def __init__(self):
        super().__init__()
        self.sigs: list = []
        self.uploads = 0
        emu = self

        class _Exp:
            @staticmethod
            def paged_fill_cache(cache, x, page_table, batch_idx=0, batch_idx_tensor=None):
                emu.sigs.append(("paged_fill_cache", _spec(cache), _spec(x), _spec(page_table), batch_idx))
                blocks = page_table.t[batch_idx].tolist()
                n = x.shape[-2] // TILE
                for i in range(n):
                    cache.t[blocks[i], 0] = x.t[0, 0, i * TILE : (i + 1) * TILE].to(cache.t.dtype)

            @staticmethod
            def paged_update_cache(cache, rows, update_idxs_tensor=None, page_table=None):
                emu.sigs.append(("paged_update_cache", _spec(cache), _spec(rows)))
                for b in range(rows.shape[1]):
                    blk = int(page_table.t[b, 0])
                    cache.t[blk, 0, int(update_idxs_tensor.t[b])] = rows.t[0, b, 0].to(cache.t.dtype)

        self.experimental = _Exp()

    # --- ops (each records its program signature)
    def _rec(self, *sig):
        self.sigs.append(sig)

    def slice(self, x, starts, ends):
        self._rec("slice", _spec(x), tuple(starts), tuple(ends))
        idx = tuple(slice(a, b) for a, b in zip(starts, ends))
        return _T(x.t[idx].clone(), x.layout)

    def reshape(self, x, shape):
        self._rec("reshape", _spec(x), tuple(shape))
        return _T(x.t.reshape(shape).clone(), x.layout)

    def typecast(self, x, dtype):
        self._rec("typecast", _spec(x), str(dtype))
        return _T(x.t.to(dtype), x.layout)

    def add(self, a, b):
        self._rec("add", _spec(a), _spec(b))
        return _T(a.t + b.t)

    def multiply(self, a, b):
        self._rec("multiply", _spec(a), _spec(b))
        return _T(a.t * b.t)

    def softmax(self, x, dim, compute_kernel_config=None):
        self._rec("softmax", _spec(x), dim)
        return _T(torch.softmax(x.t, dim=dim))

    def sum(self, x, dim, keepdim=False):
        self._rec("sum", _spec(x), dim, keepdim)
        return _T(x.t.sum(dim=dim, keepdim=keepdim))

    def concat(self, xs, dim):
        self._rec("concat", tuple(_spec(x) for x in xs), dim)
        return _T(torch.cat([x.t for x in xs], dim=dim))

    def pad(self, x, padding, value):
        self._rec("pad", _spec(x), tuple(map(tuple, padding)), value)
        flat = [v for p in reversed(padding) for v in p]
        return _T(torch.nn.functional.pad(x.t, flat, value=value))

    def lt(self, x, s):  # the scalar is a runtime arg (binary_ng): not in the signature
        self._rec("lt", _spec(x))
        return _T((x.t < s).to(x.t.dtype))

    def ge(self, x, s):
        self._rec("ge", _spec(x))
        return _T((x.t >= s).to(x.t.dtype))

    def where(self, c, a, b):
        self._rec("where", _spec(c), _spec(a), _spec(b) if isinstance(b, _T) else "scalar")
        bt = b.t if isinstance(b, _T) else torch.tensor(b, dtype=a.t.dtype)
        return _T(torch.where(c.t != 0, a.t, bt))

    def matmul(self, a, b, dtype=None, compute_kernel_config=None):
        self._rec("matmul", _spec(a), _spec(b), str(dtype))
        return _T((a.t.double() @ b.t.double()).to(dtype))

    def to_memory_config(self, x, cfg):
        return x

    def create_sharded_memory_config(self, **_):
        return None

    def num_cores_to_corerangeset(self, *a, **k):
        return None

    def deallocate(self, x):
        pass

    def zeros(self, shape, dtype=None, layout=None, device=None, memory_config=None):
        return _T(torch.zeros(shape, dtype=dtype), layout)

    def from_torch(self, t, dtype=None, layout=None, device=None, memory_config=None):
        self.uploads += 1
        return _T(t.to(dtype).clone(), layout)

    ShardStrategy = types.SimpleNamespace(HEIGHT="H")


class _Dev:
    def compute_with_storage_grid_size(self):
        return None


def _setup(monkeypatch, max_seq_len: int, seed: int = 0):
    emu = _Emu()
    monkeypatch.setattr(IX, "ttnn", emu)
    monkeypatch.setattr(IX, "POOL_DTYPE", torch.bfloat16)
    monkeypatch.setattr(IX.TtDSAIndexer, "_tail_sel_shared", {})
    cfg = types.SimpleNamespace(index_head_dim=D, index_kpool=P, index_select_pools=512)
    dev = _Dev()
    g = torch.Generator().manual_seed(seed)

    def cache():
        c = IX.TtIndexCache(dev, cfg, max_batch=2, max_seq_len=max_seq_len)
        # stale content everywhere (a previous prompt's pools and ring rows): the paths must agree on what they keep
        c.pool_cache.t.copy_(torch.randn(c.pool_cache.t.shape, generator=g).to(torch.bfloat16))
        c.ring_k.t.copy_(torch.randn(c.ring_k.t.shape, generator=g).to(torch.bfloat16))
        c.ring_g.t.copy_(torch.randn(c.ring_g.t.shape, generator=g).to(torch.bfloat16))
        return c

    idx = IX.TtDSAIndexer.__new__(IX.TtDSAIndexer)
    idx.device, idx.mesh, idx.cfg, idx.P, idx.D = dev, None, cfg, P, D
    idx.ape = _T(torch.randn(1, 1, P, D, generator=g))
    idx.fp32 = None
    idx._const_cache = {}
    return emu, idx, cache, g


def _run(monkeypatch, emu, idx, cache, k, g, *, start, seq_len, fixed: bool):
    monkeypatch.setenv("GLM53_PREFILL_FIXED_SHAPES", "1" if fixed else "0")
    emu.sigs.clear()
    idx.maintain_prefill(None, cache, 1, seq_len, start=start, kg=(_T(k.clone()), _T(g.clone())))
    return set(emu.sigs)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


@pytest.mark.parametrize("start, T", [(0, 256), (128, 512), (36864, 5120), (12288, 12288)])
def test_fixed_shapes_same_cache_bytes(monkeypatch, start, T):
    """Per (start, T_pad): every seq_len class (tail 0..3, n = 0, n a tile multiple, the incomplete pool opening a new
    tile, a full chunk) leaves bit-identical pool caches (both users) and rings under the two paths."""
    emu, idx, mk, g = _setup(monkeypatch, max_seq_len=start + T + 4096)
    k = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    gg = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    lens = sorted({1, 2, 3, 4, 5, 127, 128, 129, 130, 131, 132, T - 1, T, T // 2 + 3, T - 100, max(1, T - 129)})
    for n in [x for x in lens if 0 < x <= T]:
        torch.manual_seed(n)
        a, b = mk(), mk()
        b.pool_cache.t.copy_(a.pool_cache.t)
        b.ring_k.t.copy_(a.ring_k.t)
        b.ring_g.t.copy_(a.ring_g.t)
        _run(monkeypatch, emu, idx, a, k, gg, start=start, seq_len=n, fixed=False)
        _run(monkeypatch, emu, idx, b, k, gg, start=start, seq_len=n, fixed=True)
        assert torch.equal(_bits(a.pool_cache.t), _bits(b.pool_cache.t)), f"pool cache differs at seq_len {n}"
        assert torch.equal(_bits(a.ring_k.t), _bits(b.ring_k.t)), f"ring k differs at seq_len {n}"
        assert torch.equal(_bits(a.ring_g.t), _bits(b.ring_g.t)), f"ring g differs at seq_len {n}"


def test_fixed_shapes_programs_per_geometry(monkeypatch):
    """Ends 100 tokens apart in one (start 36864, T_pad 5120) chunk: the fixed path's program signatures all come from
    one set (the tail-3 call's), the per-length path adds new ones for every end."""
    start, T = 36864, 5120
    emu, idx, mk, g = _setup(monkeypatch, max_seq_len=start + T)
    k = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    gg = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    ref = _run(monkeypatch, emu, idx, mk(), k, gg, start=start, seq_len=4999, fixed=True)  # tail 3
    seen_on, seen_off, new_off = set(ref), set(), []
    for n in range(4150, 5120, 100):
        on = _run(monkeypatch, emu, idx, mk(), k, gg, start=start, seq_len=n, fixed=True)
        assert on <= ref, f"seq_len {n}: new fixed-path programs {on - ref}"
        seen_on |= on
        off = _run(monkeypatch, emu, idx, mk(), k, gg, start=start, seq_len=n, fixed=False)
        new_off.append(len(off - seen_off))
        seen_off |= off
    assert seen_on == ref
    assert min(new_off[1:]) >= 8, f"the per-length path should compile new programs per end: {new_off}"


def test_fixed_shapes_fallbacks(monkeypatch):
    """Geometries outside the fixed path (start not a 128 multiple, pool tiles past the user's allocation) fall back
    with nothing consumed and stay identical; the knob unset keeps the per-length path."""
    emu, idx, mk, g = _setup(monkeypatch, max_seq_len=2048)  # 768 pools per user = 24 pool tiles
    T = 1024
    k = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    gg = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    for start, n in ((32, 200), (64, 3), (2176, 101)):
        assert not idx._maintain_fixed(mk(), 1, _T(k), _T(gg), start=start, seq_len=n), (start, n)
    assert idx._maintain_fixed(mk(), 1, _T(k), _T(gg), start=2048, seq_len=101)  # tiles 16..23: the last fits
    for start, n in ((32, 200), (64, 3), (2176, 101)):
        a, b = mk(), mk()
        b.pool_cache.t.copy_(a.pool_cache.t)
        b.ring_k.t.copy_(a.ring_k.t)
        _run(monkeypatch, emu, idx, a, k, gg, start=start, seq_len=n, fixed=False)
        _run(monkeypatch, emu, idx, b, k, gg, start=start, seq_len=n, fixed=True)
        assert torch.equal(_bits(a.pool_cache.t), _bits(b.pool_cache.t))
        assert torch.equal(_bits(a.ring_k.t), _bits(b.ring_k.t))
    monkeypatch.delenv("GLM53_PREFILL_FIXED_SHAPES", raising=False)
    assert not IX.fixed_shapes_enabled()


def test_tail_selector_single_entry(monkeypatch):
    """The one-hot tail selector is one shared entry per device (replaced per chunk), not a constant per prompt end."""
    emu, idx, mk, g = _setup(monkeypatch, max_seq_len=8192)
    T = 512
    k = torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16)
    for n in (101, 205, 301, 401, 501):
        _run(monkeypatch, emu, idx, mk(), k, k, start=0, seq_len=n, fixed=True)
    assert len(IX.TtDSAIndexer._tail_sel_shared) == 1
    assert [key for key in idx._const_cache] == [("pool_rows", T // P)]


def test_slice_end_of_the_export(monkeypatch):
    """The runner export's device slices run to the padded chunk end under the knob (the model's T_pad: 256-row units of
    the 8-row mesh, the runner's bucket), to the prompt end without it."""
    from models.demos.glm53_flash.tt.runners import runtime as RT

    rt = RT.GLM53FlashPrefillRuntime.__new__(RT.GLM53FlashPrefillRuntime)
    rt.model = types.SimpleNamespace(mesh=types.SimpleNamespace(rows=8))
    monkeypatch.setenv("GLM53F_RUNNER_BUCKETS", "coarse")
    monkeypatch.setenv("GLM53_PREFILL_FIXED_SHAPES", "0")
    assert rt._slice_end(36864, 36864 + 4999) == 36864 + 4999
    monkeypatch.setenv("GLM53_PREFILL_FIXED_SHAPES", "1")
    assert rt._slice_end(36864, 36864 + 4999) == 36864 + 5120  # coarse: 512-token steps to 8192
    assert rt._slice_end(0, 100) == 256
    assert rt._slice_end(12288, 12288 + 9000) == 12288 + 9216  # 1024-token steps above 8192
    monkeypatch.setenv("GLM53F_RUNNER_BUCKETS", "")
    assert rt._slice_end(0, 300) == 512  # the model's own 256-row units
