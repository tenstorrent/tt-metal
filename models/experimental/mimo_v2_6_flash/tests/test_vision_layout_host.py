# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the TTNN ViT's layout (tt/vision/vit.py), no device:

* the blocked window attention (64-query blocks against 192 keys, the pass's band mask and sink indicator) equals
  the reference's per-sequence masked attention (HF semantics), with padding, several frames and sinks;
* the column-order permutation of the patch rows equals the reference's merge-unit ``apply_index``;
* passes keep whole frames, cover every row once and respect ``max_rows`` and the window count;
* passes run at one of the row buckets with a fixed-length ``cu_window_seqlens``.
"""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference import vision as ref
from models.experimental.mimo_v2_6_flash.tt.vision.vit import (
    BLOCK,
    CU_ENTRIES,
    MAX_SEQS_PER_PASS,
    ROW_BUCKETS,
    PassGeometry,
    band_masks,
    padded_rows,
    plan_passes,
)

WINDOW = 64


def reference_attention(q, k, v, grids, window, sinks):
    """HF semantics on the real rows: q [H, n, D] (pre-scaled), k / v [n, D] (one KV head), sinks [H] or None."""
    cu = ref.cu_seqlens(grids).tolist()
    out = []
    for a, b in zip(cu[:-1], cu[1:]):
        n = b - a
        mask = ref.window_bias(torch.arange(n), n, window, torch.float64)
        bias = mask.unsqueeze(0).repeat(q.shape[0], 1, 1)
        if sinks is not None:
            bias[:, :, 0] += sinks.view(-1, 1)
        s = q[:, a:b] @ k[a:b].T + bias
        out.append(torch.softmax(s, dim=-1) @ v[a:b])
    return torch.cat(out, dim=1)


def blocked_attention(q, k, v, geo, window, sinks):
    """What TtVisionBlock.band_attention computes, in float64 torch. q [H, rows, D], k / v [rows, D]."""
    heads, rows, d = q.shape
    nb = rows // BLOCK
    halo = window // BLOCK
    mask, ind = band_masks(geo, window)
    qb = q.reshape(heads, nb, BLOCK, d).permute(1, 0, 2, 3).reshape(nb, heads * BLOCK, d)

    def windows(t):  # as TtVisionBlock._windows: key rows outside the pass read the nearest row (masked anyway)
        keys = (torch.arange(nb) * BLOCK - window).view(nb, 1) + torch.arange((2 * halo + 1) * BLOCK).view(1, -1)
        return t[keys.clamp(0, rows - 1)]

    s = (qb @ windows(k).transpose(1, 2)).reshape(nb, heads, BLOCK, -1) + mask.double()
    if sinks is not None:
        s = s + ind.double() * sinks.view(1, heads, 1, 1)
    s = s.reshape(nb, heads * BLOCK, -1)
    o = torch.softmax(s, dim=-1) @ windows(v)
    return o.reshape(nb, heads, BLOCK, d).permute(1, 0, 2, 3).reshape(heads, rows, d)


@pytest.mark.parametrize(
    "grids",
    [
        [(1, 14, 14)],  # one image shorter than the window span, padded
        [(1, 28, 28)],  # 784 rows, padded to 896 or the 1024 bucket
        [(3, 10, 12), (1, 16, 8)],  # several frames, then an image
    ],
)
@pytest.mark.parametrize("with_sinks", [False, True])
@pytest.mark.parametrize("buckets", [ROW_BUCKETS, None])
def test_blocked_window_attention_matches_reference(grids, with_sinks, buckets):
    torch.manual_seed(0)
    geo = PassGeometry.build(grids, buckets)
    heads, d = 4, 64
    q = torch.randn(heads, geo.rows, d, dtype=torch.float64)
    k = torch.randn(geo.rows, d, dtype=torch.float64)
    v = torch.randn(geo.rows, d, dtype=torch.float64)
    sinks = torch.randn(heads, dtype=torch.float64) * 3 if with_sinks else None
    got = blocked_attention(q, k, v, geo, WINDOW, sinks)[:, : geo.n]
    expected = reference_attention(q[:, : geo.n], k[: geo.n], v[: geo.n], grids, WINDOW, sinks)
    torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)


def test_column_order_rows_match_apply_index():
    grids = [(2, 6, 10), (1, 8, 4)]
    n = sum(t * h * w for t, h, w in grids)
    x = torch.randn(n, 3)
    index = ref.col_window_index(grids, 2)
    rows = ref.patch_row_index(index, 4)
    assert torch.equal(x[rows], ref.apply_index(x, index, 4))
    assert torch.equal(x[rows][torch.argsort(rows)], x)


@pytest.mark.parametrize("max_rows", [100, 880, 4096, 16384])
def test_passes_keep_whole_frames(max_rows):
    grid = [(1, 28, 28), (16, 22, 40), (1, 64, 64)]
    passes = plan_passes(grid, max_rows)
    frames = [(h, w) for t, h, w in grid for _ in range(t)]
    covered = []
    row = 0
    for start, end, sub in passes:
        assert start == row
        rows = sum(t * h * w for t, h, w in sub)
        assert end - start == rows
        assert rows <= max_rows or (len(sub) == 1 and sub[0][0] == 1), (rows, sub)
        covered += [(h, w) for t, h, w in sub for _ in range(t)]
        row = end
    assert covered == frames and row == sum(t * h * w for t, h, w in grid)


def test_buckets_and_fixed_window_count():
    assert len(ROW_BUCKETS) == 21 and ROW_BUCKETS[0] == 1024 and ROW_BUCKETS[-1] == 32768
    assert all(b / a <= 1.25 for a, b in zip(ROW_BUCKETS, ROW_BUCKETS[1:]))
    assert padded_rows(784) == 1024 and padded_rows(1200) == 1280 and padded_rows(4096) == 4096
    assert padded_rows(14080) == 14336 and padded_rows(32768) == 32768
    assert padded_rows(32769) % 128 == 0 and padded_rows(784, None) == 896  # past the buckets / no buckets
    for grids in ([(1, 28, 28)], [(16, 22, 40)], [(1, 64, 64)], [(3, 10, 12), (1, 16, 8)]):
        geo = PassGeometry.build(grids)
        assert geo.rows in ROW_BUCKETS
        bounds, cu = geo.seq_bounds(), geo.cu_window_seqlens()
        assert len(cu) == CU_ENTRIES and cu[: len(bounds)] == bounds
        assert all(c == geo.rows for c in cu[len(bounds) :])


def test_passes_cap_the_number_of_frames():
    grid = [(600, 2, 16)]  # 600 frames of 32 patches: the window count, not the rows, splits them
    passes = plan_passes(grid, 16384)
    frames = [sum(t for t, _, _ in sub) for _, _, sub in passes]
    assert frames == [MAX_SEQS_PER_PASS, MAX_SEQS_PER_PASS, 600 - 2 * MAX_SEQS_PER_PASS]
    for _, _, sub in passes:
        assert len(PassGeometry.build(sub).cu_window_seqlens()) == CU_ENTRIES


def test_host_input_cache_is_bounded_by_bytes():
    """The cached host inputs of the pass geometries stay under a byte budget, not only an entry count: 64 large
    geometries (many image sizes / video lengths) were 0.7-1.3 GB of host RAM per row engine."""
    import ttnn
    from models.experimental.mimo_v2_6_flash.tt.vision import vit

    class Host:
        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype

    inputs = {"cos": Host((1, 1, 1024, 80), ttnn.bfloat16), "to_col": Host((1, 1024), ttnn.uint32)}
    assert vit.host_input_bytes(inputs) == 1024 * 80 * 2 + 1024 * 4
    made = []

    def make(size):
        def build():
            made.append(size)
            return size

        return build

    cache = vit.HostInputCache(max_entries=4, max_bytes=100, size_of=lambda host: host)
    for key in "abc":
        assert cache.get(key, make(30)) == 30
    assert len(cache) == 3 and cache.bytes == 90
    cache.get("a", make(30))  # a hit: nothing made, "a" is now the most recent
    assert made == [30, 30, 30]
    cache.get("d", make(30))  # 120 bytes: the least recently used ("b") goes
    assert list(cache.entries) == ["c", "a", "d"] and cache.bytes == 90
    cache.get("big", make(500))  # larger than the budget: it stays alone (the pass that asked for it needs it)
    assert list(cache.entries) == ["big"] and cache.bytes == 500
    cache.get("e", make(10))
    assert list(cache.entries) == ["e"] and cache.bytes == 10
    small = vit.HostInputCache(max_entries=2, max_bytes=10**9, size_of=lambda host: host)
    for key in "abc":
        small.get(key, make(1))
    assert list(small.entries) == ["b", "c"]  # the entry bound still holds
    assert vit.HostInputCache().max_bytes == vit.HOST_INPUT_CACHE_BYTES == 128 << 20


def test_capture_traces_needs_the_buffers_warm_allocates(expect_error):
    """capture_traces() runs after the LLM captured its traces; the persistent buffers of the traced buckets must
    exist by then (warm() allocates them before), so it refuses to allocate them itself."""
    from models.experimental.mimo_v2_6_flash.tt.vision.encoder import MiMoVisionEncoder

    class Vit:
        def __init__(self, prepared):
            self.prepared, self.calls = prepared, []

        def prepared_traces(self):
            return self.prepared

        def prepare_traces(self, buckets):
            self.calls.append(("prepare", buckets))

        def capture_traces(self):
            self.calls.append("capture")
            return {rows: 0 for rows in self.prepared}

    enc = object.__new__(MiMoVisionEncoder)
    enc.trace_buckets = (1024, 2048)
    enc.vit = Vit(prepared=(1024,))
    with expect_error(RuntimeError, r"before warm\(\): the traced buckets \[2048\]"):
        enc.capture_traces()
    assert enc.vit.calls == []  # nothing allocated, nothing captured
    enc.vit = Vit(prepared=(1024, 2048))
    assert enc.capture_traces() == {1024: 0, 2048: 0} and enc.vit.calls == ["capture"]
