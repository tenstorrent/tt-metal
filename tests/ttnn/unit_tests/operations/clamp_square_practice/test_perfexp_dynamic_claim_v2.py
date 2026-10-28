# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""dynamic_claim_v2 perf experiment (clamp_square_practice).

test_faa_probe: DIAGNOSTIC claims-only fetch-and-add stress (env DC_REPS, DC_MODES=ret0_t0,...).
Perf is never asserted; correctness (no duplicated / missing claim, bit-exact output) is.
"""
import os
from collections import Counter

import pytest
import torch
import ttnn

from ttnn.operations.clamp_square_practice.perf_experiments.dynamic_claim_v2 import faa_probe


def _want(env, val):
    v = os.environ.get(env)
    return not v or val in v.split(",")


@pytest.mark.parametrize("mode", ["ret0_t0", "ret1_t0", "ret0_t1", "ret1_t1"])
def test_faa_probe(device, mode):
    if not _want("DC_MODES", mode):
        pytest.skip("filtered")
    ret_mode = int(mode[3])
    traffic = int(mode[-1])
    total = int(os.environ.get("DC_TOTAL", "2048"))
    reps = int(os.environ.get("DC_REPS", "50"))
    grid = device.compute_with_storage_grid_size()
    G = grid.x * grid.y
    src = ttnn.from_torch(torch.randn((1, 1, 1024, 1024), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    bad = 0
    for rep in range(reps):
        log = ttnn.from_torch(
            torch.full((G, 512), -7, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        desc, N = faa_probe.build(src, log, total, ret_mode=ret_mode, traffic=traffic)
        ttnn.generic_op([src, log], desc)
        st = ttnn.to_torch(log).to(torch.int64) & 0xFFFFFFFF
        vals = []
        for i in range(N):
            n = int(st[i, 0])
            assert n <= 511, f"core {i} log overflow n={n}"
            vals += st[i, 1 : 1 + n].tolist()
        good = [v for v in vals if v < total]
        dup = [k for k, c in Counter(good).items() if c > 1]
        missing = sorted(set(range(total)) - set(good))
        term = [v for v in vals if v >= total]
        if dup or missing or len(term) != N:
            bad += 1
            for i in range(N):
                n = int(st[i, 0])
                lv = st[i, 1 : 1 + n].tolist()
                if any(v in dup for v in lv) or any(v >= total for v in lv[:-1]) or (lv and lv[-1] > total + 4 * N):
                    print(f"  core {i}: {lv}")
            print(
                f"FAA_BAD mode={mode} rep={rep} dup={dup[:10]} missing={missing[:10]} nterm={len(term)} odd={[hex(v) for v in term if v > total + 4*N][:5]}"
            )
    print(f"FAA_RESULT mode={mode} reps={reps} bad={bad}")
    assert bad == 0


from ttnn.operations.clamp_square_practice.perf_experiments.dynamic_claim_v2 import descriptor as dc

SHAPES = [
    (1, 1, 4096, 4096),
    (8, 1, 1024, 1024),
    (2, 4, 256, 512),
    (1, 1, 32, 32),
    (1, 1, 32, 8192),
    (64, 128),
    (1, 1, 1056, 2080),
]
VARIANTS = [
    "baseline",
    "disp",
    "disp_f1",
    "disp_f4",
    "disp_k8",
    "disp_k32",
    "disp_a50",
    "disp_a50_k1",
    "disp_h4",
    "disp_h1",
    "disp_p1",
    "disp_b4",
    "disp_f1_k32",
    "disp_f4_k32",
    "disp_b4_h1",
    "disp_b4_f4",
    "disp_b4_f4_k32",
    "disp_b2",
    "disp_b2_h4",
    "disp_b4_h4",
    "disp_b4_p1",
    "disp_b4_f4_h1",
    "disp_b2_f4",
    "disp_b4_f8_k32",
    "disp_b1_h8",
    "disp_b2_f4_k64",
    "disp_r1",
    "disp_b4_r1",
    "disp_b4_r2",
    "disp_b4_r1_k32",
    "disp_b4_r1_f1",
    "disp_b4_r1_h1",
    "disp_b4_r1_h4",
    "disp_r1_k32",
    "disp_r2",
    "disp_b4_r1_f1_k32",
    "disp_b4_r1_p1",
    "disp_b2_r1",
    "disp_b2_r2_h4",
    "disp_f4_r2",
    "disp_f4_r1",
    "disp_f4_h4",
    "disp_f4_r4",
    "disp_f8",
    "disp_f4_k8",
    "disp_f4_h1",
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", VARIANTS)
def test_dynamic(device, shape, variant):
    """DC_VARIANTS / DC_SHAPES filter; DC_REPS = launches (stress). Bit-exact vs torch (== the op)."""
    if not _want("DC_VARIANTS", variant) or not _want("DC_SHAPES", "x".join(map(str, shape))):
        pytest.skip("filtered")
    torch.manual_seed(0)
    xt = torch.randn(shape, dtype=torch.bfloat16) * 2
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ref = torch.clamp(xt.float(), -1.0, 1.0).pow(2).to(torch.bfloat16).view(torch.int16)
    reps = int(os.environ.get("DC_REPS", "1"))
    nbad = 0
    info = None
    for rep in range(reps):
        out = ttnn.full(list(shape), float("nan"), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        stats = None
        if os.environ.get("DC_STATS") == "1" and variant != "baseline":
            stats = ttnn.from_torch(
                torch.zeros((128, 16), dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
        desc, info = dc.build(variant, x, out, -1.0, 1.0, stats=stats)
        ttnn.generic_op([x, out] + ([stats] if stats is not None else []), desc)
        if stats is not None:
            st_all = ttnn.to_torch(stats).to(torch.int64) & 0xFFFFFFFF
            st = st_all[: info["W"]]
            dsw, dt, dgr, desw, det = (int(v) for v in st_all[info["W"], :5])
            print(
                f"DC_DISP variant={variant} sweeps={dsw} serve_us={dt/1350:.1f} us_per_sweep={dt/1350/max(dsw,1):.3f} "
                f"grants={dgr} empty_sweeps={desw} cyc_per_empty_sweep={det/max(desw,1):.0f} "
                f"cyc_per_grant_est={(dt-det - (dsw-desw)*det/max(desw,1))/max(dgr,1):.0f}"
            )
            us = lambda c: (c.float() / 1350.0)
            gw, gt, ng, nb, te = us(st[:, 0]), us(st[:, 1]), st[:, 2], st[:, 3], us(st[:, 4])
            if os.environ.get("DC_STATS_CORES"):
                order = te.argsort()
                for i in order[-8:].tolist() + order[:3].tolist():
                    print(
                        f"  core {i}: blocks={int(nb[i])} grants={int(ng[i])} grant_wait={gw[i]:.1f} gate_wait={gt[i]:.1f} span={te[i]:.1f}"
                    )
            print(
                f"DC_STATS variant={variant} grants_total={int(ng.sum())} blocks_total={int(nb.sum())} "
                f"blocks/core min={int(nb.min())} max={int(nb.max())} grant_wait_us p50={gw.median():.1f} max={gw.max():.1f} "
                f"gate_wait_us p50={gt.median():.1f} max={gt.max():.1f} dyn_span_us p50={te.median():.1f} max={te.max():.1f}"
            )
        got = ttnn.to_torch(out).view(torch.int16)
        nmis = int((got != ref).sum())
        if nmis:
            nbad += 1
            bad = (got != ref).reshape(-1, shape[-1])
            rows, cols = bad.nonzero(as_tuple=True)
            W = shape[-1]
            tiles = sorted({int(r // 32) * (W // 32) + int(c // 32) for r, c in zip(rows.tolist(), cols.tolist())})
            print(f"DC_BAD variant={variant} shape={shape} rep={rep} nmis={nmis} tiles={tiles[:16]}")
            if nbad > 3:
                break
        out.deallocate()
    print(f"DC_RESULT variant={variant} shape={shape} reps={reps} bad={nbad} info={info}")
    assert nbad == 0
