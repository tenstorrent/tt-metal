# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.73 (HISTORY_GATHER_OVERLAP): the history SP all_gather on its own sub-device (tt/history_overlap.py) next to the
indexer's top-k, standalone at the model's per-chip shapes (no model load; device, run_safe_pytest.sh, the spec's mesh).

Per position (the steady chunk at ``start``; P73_POSITIONS) and split (P73_SPLITS: default = the model's,
tt/history_overlap.default_split; row0 / row0:a-b / col the P.73 probes): the KV source's local compressed rows
R = nb * hs a chip (hs = stripe / 2 = 80 on the 8x4) as tt/sparse_attention._compressed takes them: tiled, TP gather
(serial, as the model), then the SP gather (a) serial (tt/ccl.all_gather, the off path) and (b) on the gather
sub-device inside a region next to topk_large_indices of a [Sq, kv_len] bf16 score block on the region's compute cores.
Checks: (b)'s rows bit-identical to (a)'s (every chip at a short position, chips 0 and the last at a long one) and to
the host's chip-linear order (short); the region twice: the same bytes; the top-k on the region's cores bit-identical
to the capped grid's (asserted last, after the timings). Prints the host-timed (synchronize around) median of 5 runs
and what the region hides = serial gather + top-k - region.

P.73, 8x4, ms a source (serial gather + top-k -> region): default (row0:0-5) 51200: 0.58 -> 0.48, 1024000: 7.28 ->
5.45 (-1.83); row0 (99 cores) 1M 7.16 -> 5.30 but the top-k differs (ties); row0:3-8 / 6-11 and col slower than serial.

Skipped where tt/history_overlap.build finds no free column (no core cap: every box but the Galaxy)."""

import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize
from models.demos.deepseek_v41_flash_d_p.tt import ccl, cores, history_overlap
from models.demos.deepseek_v41_flash_d_p.tt.indexer import topk_large_indices
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

DRAM = ttnn.DRAM_MEMORY_CONFIG
SP_AXIS, TP_AXIS = 0, 1
CHUNK, RATIO, WIDTH, TOPK = 5120, 2, 512, 512
POSITIONS = [int(p) for p in os.environ.get("P73_POSITIONS", "51200,1024000").split(",")]
REPS = 5


def _sync(mesh):
    ttnn.synchronize_device(mesh)


def _timed(mesh, fn, free):
    """median ms of REPS runs of fn() (2 warm-ups), synchronized around each; ``free(out)`` after each."""
    ts = []
    for k in range(2 + REPS):
        _sync(mesh)
        t0 = time.perf_counter()
        out = fn()
        _sync(mesh)
        if k >= 2:
            ts.append((time.perf_counter() - t0) * 1e3)
        free(out)
    return statistics.median(ts), ts


def _chip(t, i):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[i].cpu())


SPLITS = os.environ.get("P73_SPLITS", "default").split(",")  # (tt/history_overlap.build: row0 / row0:a-b / col)
_BUILT: dict = {}


@mesh_parametrize(timeout_s=0)
@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("start", POSITIONS, ids=[f"p{p}" for p in POSITIONS])
def test_history_gather_overlap(mesh_device, start, split):
    if mesh_device is None:
        pytest.skip("device test")
    mesh = mesh_device
    cores.init_caps(mesh)
    ccl.setup(mesh)
    key = (id(mesh), split)
    if key not in _BUILT:  # (one sub-device manager a mesh and split)
        _BUILT[key] = history_overlap.build(mesh, None if split == "default" else split)
    ov = _BUILT[key]
    if ov is None:
        pytest.skip("no free grid column under the core cap (not a Galaxy): the serial path only")
    sp, tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
    n = sp * tp
    hs = CHUNK // sp // tp // RATIO
    nb = -(-(start + CHUNK) // CHUNK)
    r = nb * hs
    sq = CHUNK // sp // tp
    kv_len = min(-(-((start + CHUNK) // RATIO) // 32) * 32, 1 << 30)
    g = mesh.compute_with_storage_grid_size()
    print(
        f"\n[P.73] {split}: mesh {sp}x{tp}, grid {g.x}x{g.y}, compute {ov.compute} ({ov.compute.num_cores()} cores), "
        f"gather {ov.gather}, links {ov.links} (rs_links {[ccl.rs_links(mesh, a) for a in (0, 1)]}); "
        f"start {start}: nb {nb}, R {r} rows a chip, SP gather out {n * r} rows ({n * r * WIDTH * 2 / 2**20:.0f} MiB a "
        f"chip); top-k [{sq}, {kv_len}]"
    )
    torch.manual_seed(start)
    src = torch.randn(sp, tp, r, WIDTH).to(torch.bfloat16)
    sl = ttnn.from_torch(
        src,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, 1)),
    )
    logits = ttnn.from_torch(
        torch.randn(1, 1, sq, kv_len).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    # (as _compressed: tiled, TP gather on the full queue)
    tl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=cores.sub_cores(sl))
    tpg = ccl.all_gather(tl, 2, TP_AXIS)
    topk = lambda: topk_large_indices(logits, k=TOPK, valid_length=kv_len)  # noqa: E731  (the capped grid)
    op = (
        ttnn.bringup.topk_large_indices
        if settings.get("INDEX_TOPK_OP") == "fork"
        else ttnn.experimental.topk_large_indices
    )
    topk_c = lambda: op(logits, sub_core_grids=ov.compute, k=TOPK, valid_length=kv_len)  # noqa: E731

    # correctness: serial vs region, bit for bit; region twice; top-k inside vs outside
    ser = ccl.all_gather(tpg, 2, SP_AXIS)
    k_ser = topk()
    reg, k_reg = ov.region(lambda: ov.all_gather(tpg, 2, SP_AXIS), topk_c)
    reg2, k_reg2 = ov.region(lambda: ov.all_gather(tpg, 2, SP_AXIS), topk_c)
    _sync(mesh)
    topk_diff = []
    chips = range(n) if start <= 2 * CHUNK * 10 else (0, n - 1)
    for i in chips:
        a = _chip(ser, i)
        b, b2 = _chip(reg, i), _chip(reg2, i)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"chip {i}: region gather != serial gather"
        assert torch.equal(b.view(torch.int16), b2.view(torch.int16)), f"chip {i}: region gather not deterministic"
        if start <= 2 * CHUNK * 10:
            want = src.reshape(n * r, WIDTH)
            assert torch.equal(a.reshape(n * r, WIDTH).view(torch.int16), want.view(torch.int16)), f"chip {i} order"
        ka, kb, kb2 = _chip(k_ser, i), _chip(k_reg, i), _chip(k_reg2, i)
        if not (torch.equal(ka, kb) and torch.equal(kb, kb2)):
            topk_diff.append(i)
    for t in (ser, reg, reg2, k_ser, k_reg, k_reg2):
        ttnn.deallocate(t)
    print(
        f"[P.73] {split} start {start}: region gather == serial gather on chips {list(chips)}; top-k on the region's "
        f"grid differs from the capped grid's on chips {topk_diff}"
    )

    def free(out):
        for t in out if isinstance(out, tuple) else (out,):
            if isinstance(t, ttnn.Tensor) and t.is_allocated():
                ttnn.deallocate(t)

    m_ser, t_ser = _timed(mesh, lambda: ccl.all_gather(tpg, 2, SP_AXIS), free)
    m_top, t_top = _timed(mesh, topk, free)
    m_topc, t_topc = _timed(mesh, topk_c, free)
    m_gcol, t_gcol = _timed(mesh, lambda: ov.region(lambda: ov.all_gather(tpg, 2, SP_AXIS), lambda: None), free)
    m_reg, t_reg = _timed(mesh, lambda: ov.region(lambda: ov.all_gather(tpg, 2, SP_AXIS), topk_c), free)
    m_both, t_both = _timed(mesh, lambda: (ccl.all_gather(tpg, 2, SP_AXIS), topk()), free)
    hidden = m_both - m_reg
    print(
        f"[P.73] {split} start {start} (median of {REPS}, ms, host-timed with a sync around): SP gather serial {m_ser:.2f}, "
        f"top-k {m_top:.2f} (on the region's compute grid {m_topc:.2f}), serial gather + top-k {m_both:.2f}, gather alone on the gather column (region) "
        f"{m_gcol:.2f}, region gather || top-k {m_reg:.2f}; hidden {hidden:.2f} ms a source"
    )
    print(f"[P.73] runs: ser {t_ser} top {t_top} gcol {t_gcol} reg {t_reg} both {t_both}")
    for t in (tl, tpg, sl, logits):
        ttnn.deallocate(t)
    assert not topk_diff, f"top-k on the region's compute grid is not the capped grid's on chips {topk_diff}"


@mesh_parametrize(timeout_s=0)
@pytest.mark.parametrize("start", POSITIONS, ids=[f"p{p}" for p in POSITIONS])
def test_history_gather_rows(mesh_device, start):
    """P.73 (overseer route): the history gathered as the slab's rows (ROW_MAJOR, TP then SP; tt/ccl.all_gather takes
    high_bw_all_gather for any row count) vs the off path's tilize -> TP -> SP -> untilize: bit-identical, timed."""
    if mesh_device is None:
        pytest.skip("device test")
    mesh = mesh_device
    cores.init_caps(mesh)
    ccl.setup(mesh)
    sp, tp = mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
    n = sp * tp
    hs = CHUNK // sp // tp // RATIO
    r = -(-(start + CHUNK) // CHUNK) * hs
    torch.manual_seed(start)
    src = torch.randn(sp, tp, r, WIDTH).to(torch.bfloat16)
    sl = ttnn.from_torch(
        src,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=(0, 1)),
    )

    def tiled():
        tl = ttnn.to_layout(sl, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=cores.sub_cores(sl))
        a = ccl.all_gather(tl, 2, TP_AXIS)
        b = ccl.all_gather(a, 2, SP_AXIS)
        out = cores.to_rm(b, memory_config=DRAM)
        for t in (tl, a, b):
            ttnn.deallocate(t)
        return out

    def rows():
        a = ccl.all_gather(sl, 2, TP_AXIS)
        b = ccl.all_gather(a, 2, SP_AXIS)
        ttnn.deallocate(a)
        return b

    x, y = tiled(), rows()
    _sync(mesh)
    chips = range(n) if start <= 2 * CHUNK * 10 else (0, n - 1)
    for i in chips:
        assert torch.equal(_chip(x, i).view(torch.int16), _chip(y, i).view(torch.int16)), f"chip {i}"
    ttnn.deallocate(x)
    ttnn.deallocate(y)
    m_t, _ = _timed(mesh, tiled, ttnn.deallocate)
    m_r, _ = _timed(mesh, rows, ttnn.deallocate)
    print(
        f"\n[P.73] rows start {start}: bit-identical on chips {list(chips)}; tiled path {m_t:.2f} ms, rows {m_r:.2f} ms"
    )
    ttnn.deallocate(sl)
