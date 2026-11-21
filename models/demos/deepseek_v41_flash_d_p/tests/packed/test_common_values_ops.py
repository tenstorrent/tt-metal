# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.9X (COMMON_VALUE_SLOTS): the two calls that take their per-run values as common runtime args (prepare_program's
common value slots) give the same bytes as their per-core form, op level, on the 8x4 mesh (the forks' own tests open a
single card or their own mesh: agent rule 19 / the resident runner's held mesh):

    ttnn.bringup.ckv_bfp4(prepared, compact, dynamic_row0)   vs prepared (per-core row0 slot), the history pack's shape
                                                             (tt/history_gather._pack_fixed: n rows into a [1, 1, R,
                                                             320] record table) at several row0, rows outside untouched
    ttnn.bringup.unpack_fp8_rows(whole_grid, common_values)  vs whole_grid (per-core value slots), several ranges
    tt/lane_rows.GluedRows (the glued RoPE rows)             COMMON_VALUE_SLOTS on vs off, glued layouts

Also: one program for every row0 / range (prepared programs added after the first call == 0), and the warm host ms a
call of both forms. The held model gives the mesh only (no layer runs).

    <the walls env> scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_common_values_ops.py
"""

from __future__ import annotations

import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"


def _host_ms(fn, n: int = 20) -> float:
    """Host ms a call (enqueue only; the caller syncs after)."""
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3


@mesh_parametrize(timeout_s=1800)
def test_common_values_ops(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests.fresh_common import PrepCount
    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    chunk = int(S.data["target"]["chunk"])
    cap = -(-deploy // chunk) * chunk  # (the walls' one load)
    monkeypatch.setitem(S.data["target"], "seq", max(cap, int(S.data["target"]["seq"])))
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    mesh = model.mesh
    chips = mesh.get_num_devices()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    bad = []

    # ---- ckv_bfp4 dynamic_row0 (bfloat4_b records, the pack's piece shape)
    R, FILL = 8192, 0xA5
    for n in (1024, 320):
        torch.manual_seed(n)
        pc = PrepCount()
        for k, row0 in enumerate((0, 96, 2048 + 32, R - n)):
            x = torch.randn(1, 1, n, 512) * (1 + 3 * torch.rand(1, 1, n, 1))
            ckv = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            tabs = [
                ttnn.from_torch(
                    torch.full((1, 1, R, 320), FILL, dtype=torch.uint8), dtype=ttnn.uint8, device=mesh, mesh_mapper=rep
                )
                for _ in range(2)
            ]
            ttnn.bringup.ckv_bfp4(ckv, tabs[0], row0, prepared=True, max_grid=max_grid())
            with pc:
                p0 = pc.n
                ttnn.bringup.ckv_bfp4(
                    ckv, tabs[1], row0, prepared=True, compact=True, dynamic_row0=True, max_grid=max_grid()
                )
                if k > 0 and pc.n != p0:
                    bad.append(f"ckv_bfp4 n {n} row0 {row0}: a new prepare")
            a, b = (ttnn.to_torch(t, mesh_composer=comp).view(torch.uint8).reshape(chips, R, 320) for t in tabs)
            if not torch.equal(a, b):
                bad.append(f"ckv_bfp4 n {n} row0 {row0}: dynamic_row0 differs from the default")
            if not ((b[:, :row0] == FILL).all() and (b[:, row0 + n :] == FILL).all()):
                bad.append(f"ckv_bfp4 n {n} row0 {row0}: rows outside written")
            if k == 0:
                t_def = _host_ms(lambda: ttnn.bringup.ckv_bfp4(ckv, tabs[0], row0, prepared=True, max_grid=max_grid()))
                t_dyn = _host_ms(
                    lambda: ttnn.bringup.ckv_bfp4(
                        ckv, tabs[1], row0, prepared=True, compact=True, dynamic_row0=True, max_grid=max_grid()
                    )
                )
                ttnn.synchronize_device(mesh)
                print(f"ckv_bfp4 n {n}: host ms a call, per-core row0 {t_def:.3f}, dynamic_row0 {t_dyn:.3f}")
                metrics.record(f"ckv_bfp4_host_ms_n{n}_default", round(t_def, 4))
                metrics.record(f"ckv_bfp4_host_ms_n{n}_dynamic_row0", round(t_dyn, 4))
            for t in [ckv] + tabs:
                ttnn.deallocate(t)

    # ---- unpack_fp8_rows common_values (the served index-key rows' format)
    from ttnn.bringup.fp8_rows_ttnn.fp8_rows import ROW_BYTES

    Rr = 16384
    torch.manual_seed(5)
    raw = torch.randint(0, 256, (1, 1, Rr, ROW_BYTES), dtype=torch.uint8)
    raw[..., 128:132] = torch.tensor([0, 0, 128, 59], dtype=torch.uint8)  # (a sane fp32 scale: 2^-8 x 1.0)
    raw[..., 132:] = 0
    e = raw[..., :128]
    e[(e & 0x7F) == 0x7F] = 0  # (no e4m3 NaN)
    rows = ttnn.from_torch(raw, dtype=ttnn.uint8, device=mesh, mesh_mapper=rep)
    pc = PrepCount()
    for k, (lo, nb) in enumerate(((0, 1), (96, 3), (512, 37), (1024, 300))):
        outs = [
            ttnn.from_torch(
                torch.full((1, 1, Rr, 128), 3.0),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=rep,
            )
            for _ in range(2)
        ]
        ttnn.bringup.unpack_fp8_rows(
            rows, outs[0], row_lo=lo, row_hi=lo + 32 * nb, max_grid=max_grid(), whole_grid=True
        )
        with pc:
            p0 = pc.n
            ttnn.bringup.unpack_fp8_rows(
                rows, outs[1], row_lo=lo, row_hi=lo + 32 * nb, max_grid=max_grid(), whole_grid=True, common_values=True
            )
            if k > 0 and pc.n != p0:
                bad.append(f"unpack ({lo}, {nb} bands): a new prepare")
        a, b = (ttnn.to_torch(t, mesh_composer=comp).float() for t in outs)
        if not torch.equal(a, b):
            bad.append(f"unpack ({lo}, {nb} bands): common_values differs from the default")
        if k == 0:
            t_def = _host_ms(
                lambda: ttnn.bringup.unpack_fp8_rows(
                    rows, outs[0], row_lo=0, row_hi=64, max_grid=max_grid(), whole_grid=True
                )
            )
            t_cv = _host_ms(
                lambda: ttnn.bringup.unpack_fp8_rows(
                    rows, outs[1], row_lo=0, row_hi=64, max_grid=max_grid(), whole_grid=True, common_values=True
                )
            )
            ttnn.synchronize_device(mesh)
            print(f"unpack_fp8_rows whole grid: host ms a call, per-core values {t_def:.3f}, common values {t_cv:.3f}")
            metrics.record("unpack_host_ms_default", round(t_def, 4))
            metrics.record("unpack_host_ms_common_values", round(t_cv, 4))
        for t in outs:
            ttnn.deallocate(t)
    ttnn.deallocate(rows)
    # ---- GluedRows (the glued RoPE rows: pieces as common value slots) on vs off
    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import switch
    from models.demos.deepseek_v41_flash_d_p.tt.lane_rows import glued_rows

    T = 262144
    torch.manual_seed(9)
    table = ttnn.from_torch(
        torch.randn(1, 1, T, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
    )
    layouts = [
        [(139264, 2048), (139264 + 64, 2048), (4096, 2048), (64, 2048)],
        [(200000 // 64 * 64, 1600), (64, 3008), (8192, 768), (32768, 1344), (96, 1472)],
        [(5000 // 64 * 64, chunk)],
    ]
    for li, starts in enumerate(layouts):
        outs = {}
        for v in ("off", "on"):
            with switch("COMMON_VALUE_SLOTS", v):
                o = glued_rows(table, starts, chunk)
                outs[v] = ttnn.to_torch(o, mesh_composer=comp).float()
                ttnn.deallocate(o)
        if not torch.equal(outs["off"], outs["on"]):
            bad.append(f"GluedRows layout {li}: common values differ from the per-core ones")
    with switch("COMMON_VALUE_SLOTS", "off"):
        t_def = _host_ms(lambda: ttnn.deallocate(glued_rows(table, layouts[1], chunk)))
    with switch("COMMON_VALUE_SLOTS", "on"):
        t_cv = _host_ms(lambda: ttnn.deallocate(glued_rows(table, layouts[1], chunk)))
    ttnn.synchronize_device(mesh)
    print(f"GluedRows: host ms a call, per-core values {t_def:.3f}, common values {t_cv:.3f}")
    metrics.record("glued_rows_host_ms_default", round(t_def, 4))
    metrics.record("glued_rows_host_ms_common_values", round(t_cv, 4))
    ttnn.deallocate(table)
    metrics.record("common_values_mismatches", len(bad))
    for b in bad:
        print("  " + b)
    print(f"common value slots: {len(bad)} mismatches")
    assert not bad, "common value slots: not the same bytes / a new program"
