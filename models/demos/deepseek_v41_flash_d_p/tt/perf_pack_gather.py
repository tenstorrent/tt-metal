# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.11 (CKV_PACK_BEFORE_GATHER): the packed gathers on the run's mesh, old vs new, on random rows. Not a gate:
byte checks and timings for BREADCRUMBS.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_pack_gather.py -s

For the window blocks (bfloat8_b, W rows an SP row) and the compressed-KV records (bfloat4_b, W / 2 and W rows):
  old = the SP row's bf16 block all_gathered over SP, then every chip packs every row (ckv_bfp4 into the table);
  new = each chip packs its own block into the load-time local buffer, the records all_gathered over SP, slice_written
        into the table (tt/attention.WindowTables.packed / gather_packed, the model's calls).
Checks on every chip: the table bytes equal (window blocks on the bytes the format defines, DS41-144), the rounded
values of the new path equal the old path's rows of the chip's block, a partial chunk's pad mask (mask_local vs the
iota mask) gives the same bytes. Timings: the median of N warm iterations, synchronised (device-bound: ~ the device
time of the sequence) and host-only (dispatch).
"""

from __future__ import annotations

import importlib
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v41_flash_d_p.tt import ccl
from models.demos.deepseek_v41_flash_d_p.tt.attention import WINDOW, WINDOW_BLOCK_BYTES, WindowTables
from models.demos.deepseek_v41_flash_d_p.tt.common import SP_AXIS

ck = importlib.import_module("ttnn.bringup.ckv_bfp4_ttnn.ckv_bfp4")
_spec = importlib.util.spec_from_file_location(
    "ckv_bfp4_reference", ck.KERNELS.parent / "tests" / "reference.py"
)  # (the fork's host rule: the block mask)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)
DRAM = ttnn.DRAM_MEMORY_CONFIG
WARM, N = 3, 20
CHUNK = 5120


def _time(mesh, fn, sync_each: bool):
    for _ in range(WARM):
        fn()
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        if sync_each:
            ttnn.synchronize_device(mesh)
        ts.append((time.perf_counter() - t0) * 1e3)
    ttnn.synchronize_device(mesh)
    return statistics.median(ts)


def _chips(t):
    return [ttnn.to_torch(d).view(torch.uint8) for d in ttnn.get_device_tensors(t)]


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_pack_gather(mesh_device):
    mesh = mesh_device
    ccl.configure()
    ccl.setup(mesh)
    tables = WindowTables(mesh, CHUNK)
    geo = tables.geo
    sp, w = geo.sp, geo.w
    g = torch.Generator().manual_seed(0)
    res = {}

    def block_input(n):
        """natural rows [SP n, 512] and the SP row's block on each chip [1, 1, n, 512] bf16 TILE."""
        x = torch.randn(sp * n, 512, generator=g) * torch.pow(
            2.0, torch.randint(-8, 4, (sp * n, 1), generator=g).float()
        )
        x = x.bfloat16()
        t = ttnn.from_torch(
            x.view(sp, 1, n, 512),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, None)),
        )
        return x, t

    def old_mask(gat, valid):
        if valid >= gat.shape[-2]:
            return gat
        full = gat.shape[-2] == tables.iota.shape[-2]  # (K-1: a full-range slice shares the iota's buffer)
        iota = tables.iota if full else ttnn.slice(tables.iota, [0, 0, 0, 0], [1, 1, gat.shape[-2], 1])
        keep = ttnn.lt(iota, float(valid), dtype=ttnn.bfloat16, memory_config=DRAM)
        out = ttnn.multiply(gat, keep, dtype=ttnn.bfloat16, memory_config=DRAM)
        for t in ([] if full else [iota]) + [keep, gat]:
            ttnn.deallocate(t)
        return out

    # ---------------------------------------------------------------- window blocks (bfloat8_b), W rows a SP row
    x, local = block_input(w)
    pages = (WINDOW + CHUNK) // 32
    tab_old, tab_new = (
        ttnn.zeros([1, 1, pages, WINDOW_BLOCK_BYTES], dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        for _ in range(2)
    )
    kw = dict(dtype=ttnn.bfloat8_b, blocks=True, memory_config=DRAM, prepared=True)

    def win_old(valid=CHUNK):
        gat = old_mask(ccl.all_gather(local, 2, SP_AXIS), valid)
        ttnn.bringup.ckv_bfp4(gat, tab_old, WINDOW, compact=True, **kw)
        ttnn.deallocate(gat)

    def win_new(valid=CHUNK):
        m = tables.mask_local(local, valid)
        ttnn.bringup.ckv_bfp4(m, tables.packed("blocks", w), 0, compact=True, **kw)
        if m is not local:
            ttnn.deallocate(m)
        gat = tables.gather_packed("blocks", w)
        c = WINDOW // 32
        ttnn.experimental.slice_write(
            gat, tab_new, [0, 0, c, 0], [1, 1, c + gat.shape[-2], WINDOW_BLOCK_BYTES], [1, 1, 1, 1]
        )
        if not tables.is_kept(gat):
            ttnn.deallocate(gat)

    for valid in (CHUNK, 2000):
        win_old(valid)
        win_new(valid)
        xm = x.clone()
        xm[valid:] = 0
        exp, mask = ref.encode_blocks(xm)
        for i, (a, b) in enumerate(zip(_chips(tab_old), _chips(tab_new))):
            a, b = (t.reshape(pages, WINDOW_BLOCK_BYTES)[WINDOW // 32 :] for t in (a, b))
            bad = ((a != b) & mask).any(-1).sum().item()
            badr = ((b != exp) & mask).any(-1).sum().item()
            assert bad == 0 and badr == 0, f"window valid {valid} chip {i}: {bad} blocks != old, {badr} != host rule"
        print(f"PACK_GATHER window valid {valid}: blocks equal on {mesh.get_num_devices()} chips", flush=True)
    res["window old (bf16 gather + pack all), synced"] = _time(mesh, win_old, True)
    res["window new (pack own + gather blocks + slice_write), synced"] = _time(mesh, win_new, True)
    res["window old, host"] = _time(mesh, win_old, False)
    res["window new, host"] = _time(mesh, win_new, False)

    # ---------------------------------------------------------------- compressed KV records (bfloat4_b)
    for n in sorted({w // 2, w}):
        x, local = block_input(n)
        hist = 4 * sp * n
        g0 = 2 * sp * n
        r_old, r_new = (
            ttnn.zeros([1, 1, hist, 320], dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh) for _ in range(2)
        )
        outs = {}

        def rows_old(valid=sp * n, keep=False):
            gat = old_mask(ccl.all_gather(local, 2, SP_AXIS), valid)
            rd = ttnn.bringup.ckv_bfp4(gat, r_old, g0, rounded=True, memory_config=DRAM, prepared=True)
            ttnn.deallocate(gat)
            if keep:
                outs["old"] = rd
            else:
                ttnn.deallocate(rd)

        def rows_new(valid=sp * n, keep=False):
            m = tables.mask_local(local, valid)
            rd = ttnn.bringup.ckv_bfp4(
                m, tables.packed("rows", n), 0, rounded=True, memory_config=DRAM, prepared=True, compact=True
            )
            if m is not local:
                ttnn.deallocate(m)
            gat = tables.gather_packed("rows", n)
            ttnn.experimental.slice_write(gat, r_new, [0, 0, g0, 0], [1, 1, g0 + gat.shape[-2], 320], [1, 1, 1, 1])
            if not tables.is_kept(gat):
                ttnn.deallocate(gat)
            if keep:
                outs["new"] = rd
            else:
                ttnn.deallocate(rd)

        for valid in (sp * n, sp * n // 3):
            rows_old(valid, keep=True)
            rows_new(valid, keep=True)
            xm = x.clone()
            xm[valid:] = 0
            exp = ref.encode_rows(xm)
            olds = [ttnn.to_torch(d).reshape(sp * n, 512) for d in ttnn.get_device_tensors(outs["old"])]
            news = [ttnn.to_torch(d).reshape(n, 512) for d in ttnn.get_device_tensors(outs["new"])]
            for i, (a, b) in enumerate(zip(_chips(r_old), _chips(r_new))):
                assert torch.equal(a, b), f"rows n {n} valid {valid} chip {i}: tables differ"
                assert torch.equal(b.reshape(hist, 320)[g0 : g0 + sp * n], exp), f"chip {i}: != host rule"
                r = i // mesh.shape[1]  # the chip's SP row
                assert torch.equal(news[i], olds[i][r * n : (r + 1) * n]), f"chip {i}: rounded rows differ"
            for t in outs.values():
                ttnn.deallocate(t)
            print(f"PACK_GATHER rows n {n} valid {valid}: tables and rounded rows equal", flush=True)
        res[f"rows n {n} old (bf16 gather + pack all), synced"] = _time(mesh, rows_old, True)
        res[f"rows n {n} new (pack own + gather records + slice_write), synced"] = _time(mesh, rows_new, True)
        res[f"rows n {n} old, host"] = _time(mesh, rows_old, False)
        res[f"rows n {n} new, host"] = _time(mesh, rows_new, False)

    for k, v in res.items():
        print(f"PACK_GATHER {k}: {v:.3f} ms", flush=True)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_window_layer(mesh_device):
    """A window layer's whole write sequence as the model runs it (TtWindowAttention._window_rows(blocks=True) +
    _release_window vs _window_packed, on a TtWindowCache), over consecutive chunks (the carry from the previous chunk)
    and at one start again and again (the carry read from the served cache): host time a call (no sync) and synced
    wall a call, median of N."""
    from models.demos.deepseek_v41_flash_d_p.tt.attention import TtWindowAttention, TtWindowCache

    mesh = mesh_device
    ccl.configure()
    ccl.setup(mesh)
    tables = WindowTables(mesh, CHUNK)
    geo = tables.geo
    att = TtWindowAttention.__new__(TtWindowAttention)
    att.geo, att.tables, att.fold, att.win8 = geo, tables, True, True
    g = torch.Generator().manual_seed(1)
    kv = ttnn.from_torch(
        torch.randn(geo.sp, geo.tp, geo.q_rows, 512, generator=g).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1)),
    )
    kv = ttnn.reshape(kv, [1, 1, geo.q_rows, 512])
    calls = 2 * (WARM + N) + 4
    cache = TtWindowCache(mesh, tables, (calls + 2) * CHUNK)
    res = {}

    def old(start):
        carry, kv_rm = att._window_rows(kv, start, start + CHUNK, cache, blocks=True)
        att._release_window(carry, kv_rm, cache, start, start + CHUNK)

    def new(start):
        att._window_packed(kv, start, start + CHUNK, cache)

    for name, fn in (("old", old), ("new", new)):
        for sync in (False, True):
            pos = iter(range(CHUNK, calls * CHUNK, CHUNK))
            res[f"window layer {name}, consecutive chunks, {'synced' if sync else 'host'}"] = _time(
                mesh, lambda fn=fn, pos=pos: fn(next(pos)), sync
            )
            res[f"window layer {name}, one start (carry from the cache), {'synced' if sync else 'host'}"] = _time(
                mesh, lambda fn=fn: fn(2 * CHUNK), sync
            )
    for k, v in res.items():
        print(f"PACK_GATHER {k}: {v:.3f} ms", flush=True)
