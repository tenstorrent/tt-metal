# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load-based decode buckets, round level (8 layers, no prefill, fixed blocks as the tt_opt minimal gate): one eager verify + accept + commit + draft round of the full B=16 runner against the same
round of the bucket runner of the first Ub users of every mesh row (``SpecRunner(..., drafter=drafter_root.view_users(Ub), Ub=Ub)``) on the same state: per-layer stream PCC of the users u < Ub, logits
PCC, accepted ids / accept counts / new drafts (exact) of those users. Env: TT_OPT_K (3 or 5, the k of both runners), BUCKET_UBS (default "2,1"), TT_OPT_LAYER_IDS. Prints 'BUCKETTEST ...'.
"""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.tt_opt import test_tt_opt_decode as T
from models.demos.blackhole.deepseek_v41_flash.tt.spec_model import SpecRunner


def _round(runner, X, base, snaps):
    dec = runner.dec
    T._feed_eager(runner, X, base, -1)
    dec.dbg_store = []
    dec.forward()
    ttnn.synchronize_device(runner.md)
    out = {"streams": {tag: [r.clone() for r in rows] for tag, rows in dec.dbg_store}}
    dec.dbg_store = None
    cols = runner.cols
    lg = dec.dbg_logits
    out["logits"] = torch.cat(
        [
            torch.cat(
                [
                    ttnn.to_torch(ttnn.get_device_tensors(lg)[r * cols + c]).float().reshape(runner.U * runner.n, -1)
                    for c in range(cols)
                ],
                dim=1,
            )
            for r in range(runner.rows)
        ]
    )
    pk = runner._readback(dec.pack)
    out["a"], out["m"], out["d"] = pk[0].clone(), pk[1].clone(), pk[2].clone()
    dec.restore_states(snaps)
    return out


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", T.PARAMS, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_bucket_round(mesh_device):
    md = mesh_device
    gen, runner, X, base, _ = T._setup(md, minimal=True, tracy_layers=False)
    U, rows = runner.U, runner.rows
    n = runner.n
    runner.dec.set_force(-1)
    snaps = runner.dec.snapshot_states()
    full = _round(runner, X, base, snaps)
    for Ub in [int(x) for x in os.environ.get("BUCKET_UBS", "2,1").split(",")]:
        dv = runner.drafter_root.view_users(Ub)
        br = SpecRunner(gen.m, runner.k, drafter=dv, Ub=Ub)
        br.prepare(torch.zeros(rows * Ub, br.n, dtype=torch.long), torch.full((rows * Ub,), 130, dtype=torch.long))
        phys = br.phys  # runner row -> model user
        bsnaps = br.dec.snapshot_states()
        got = _round(br, X[phys], base[phys], bsnaps)
        worst, worst_tag = 1.0, ""
        for tag, rws in full["streams"].items():
            for r in range(rows):
                f, g = rws[r], got["streams"][tag][r]
                for u in range(Ub):
                    p = T._pcc(f[u * n : (u + 1) * n], g[u * n : (u + 1) * n])
                    if p < worst:
                        worst, worst_tag = p, f"{tag} row {r} user {u}"
        sel = phys
        lg_f = full["logits"].reshape(rows, U * n, -1)
        lg_g = got["logits"].reshape(rows, Ub * n, -1)
        lp = min(
            T._pcc(lg_f[r, u * n : (u + 1) * n], lg_g[r, u * n : (u + 1) * n]) for r in range(rows) for u in range(Ub)
        )
        eq_a = bool(torch.equal(full["a"][sel], got["a"]))
        eq_m = bool(torch.equal(full["m"][sel], got["m"]))
        eq_d = bool(torch.equal(full["d"][sel], got["d"]))
        T._log(
            f"BUCKETTEST Ub {Ub} k {runner.k}: streams worst PCC {worst:.6f} ({worst_tag}); logits PCC {lp:.6f}; argmax ids equal {eq_a}; accept counts equal {eq_m}; drafts equal {eq_d}"
        )
        assert worst >= 0.9995 and lp >= 0.9995 and eq_a and eq_m, (Ub, worst, lp, eq_a, eq_m, eq_d)
