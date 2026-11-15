# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.32 (DS41-276): SnapshotSend's prepared program must follow the slab's geometry, not only the block pattern.

The P.67 gate failure (test_decoder_skip_decode after test_determinism in one process: layer 0's served snapshot
off != decode at chunk-aligned stops) came from here: the program of a pattern was built once, at its first call, and
reused for any slab. Its TensorAccessor compile-time args hold the slab's tensor shape, and an ND-sharded accessor
takes the outermost page coordinate modulo that shape's first dim (tensor_accessor.h get_bank_and_offset_from_page_id).
Built for a ladder cache's own 1-batch slab ([1, 1, 128, 512]), the program wrote every later engine slab's
(slot, layer) batch (page0 = batch x 128) into batch 0 = slot 0, layer 0.

Here: one cache on an own slab (1 batch) and one on an engine-like slab (3 batches, the cache at batch 2), the same
whole chunk written into both (the steady pattern: the last chip owns the 4 blocks), the own cache's snapshot first
(builds the pattern's program), then the engine-like one: batch 2 must hold the definition on every chip, batches 0
and 1 untouched. Then the host us of a call on the steady pattern (the key's cost).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_snapshot_send_geometry.py -s
"""

from __future__ import annotations

import statistics
import time

import torch

import ttnn
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

WIDTH = 512
LAYERS = 3


def _set(monkeypatch, **kw):
    for k, v in kw.items():
        monkeypatch.setenv(settings.env_name(k), str(v))
    settings.refresh()


@mesh_parametrize(timeout_s=1800)
def test_snapshot_send_geometry(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt.attention import SNAP, TtWindowCache, WindowTables, _upload
    from models.demos.deepseek_v41_flash_d_p.tt.served import CacheSlab, alloc_served

    _set(monkeypatch, WINDOW_KV_KEEP="snapshot128", KV_BLOCK_SIZE=64, CHEAP_CCL="on")
    mesh = mesh_device
    if mesh.get_num_devices() < 2:
        import pytest

        pytest.skip("the snapshot send needs at least two chips")
    chunk = int(spec().get("target.chunk"))
    tables = WindowTables(mesh, chunk)
    g = tables.geo
    send = tables.snapshot_send()
    assert send is not None
    slab_t = alloc_served(mesh, WIDTH, SNAP, 32, LAYERS, dedup=False)
    own = TtWindowCache(mesh, tables, 4 * chunk)  # (slab None: its own [1, 1, 128, 512], the ladder's)
    eng = TtWindowCache(mesh, tables, 4 * chunk, slab=CacheSlab(slab_t, slot=0, row=2, layers=LAYERS))
    assert int(own.snap.tensor.shape[0]) == 1 and int(eng.snap.tensor.shape[0]) == LAYERS
    gen = torch.Generator().manual_seed(32)
    full = (torch.randn(chunk, WIDTH, generator=gen) * 2).to(torch.bfloat16)

    def host_batches():
        return [ttnn.to_torch(t).reshape(LAYERS, SNAP, WIDTH) for t in ttnn.get_device_tensors(slab_t)]

    for c in (own, eng):
        t = _upload(mesh, full.view(g.sp, g.tp, g.q_rows, WIDTH), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (0, 1))
        c.write_own(t, 0, chunk)
        ttnn.deallocate(t)
    guard = host_batches()
    own.write_snapshot(chunk)  # the steady pattern's program, first built for the 1-batch slab
    eng.write_snapshot(chunk)
    ttnn.synchronize_device(mesh)
    ring = eng.ring
    owners = sorted({str(ring.own[(p // 32) % ring.BLOCKS]) for p in range(chunk - SNAP, chunk, 32)})
    want = torch.zeros(SNAP, WIDTH, dtype=torch.bfloat16)
    for p in range(chunk - SNAP, chunk):
        want[p % SNAP] = full[p]
    fails = []
    for q, got in enumerate(host_batches()):
        if not torch.equal(got[2].view(torch.int16), want.view(torch.int16)):
            fails.append(f"chip {q}: the engine slab's batch 2 is not the snapshot")
        for b in (0, 1):
            if not torch.equal(got[b].view(torch.int16), guard[q][b].view(torch.int16)):
                fails.append(f"chip {q}: batch {b} (another slot / layer) changed")
    own_rows = own.snapshot_host()
    for q, got in enumerate(own_rows):
        if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
            fails.append(f"chip {q}: the own slab is not the snapshot")
    print(f"[snapshot geometry] owners {owners}, programs {len(send._pp)}, failures {len(fails)}", flush=True)
    assert not fails, "\n".join(fails[:12])

    times = []
    for it in range(32):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        eng.write_snapshot(chunk)
        dt = time.perf_counter() - t0
        if it >= 2:
            times.append(dt)
    ttnn.synchronize_device(mesh)
    print(f"[snapshot geometry] host us a call (median of 30): {1e6 * statistics.median(times):.1f}", flush=True)
    own.free()
    eng.free()
    ttnn.deallocate(slab_t)
