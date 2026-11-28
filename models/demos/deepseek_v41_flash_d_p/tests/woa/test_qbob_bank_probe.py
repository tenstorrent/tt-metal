# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.42 (OWNER 2026-11-27): wq_b / wo_b bank-layout probe on the C++ ttnn.grouped_matmul, op level, no model load.

The two projections as the model runs them (ATTN_SP32_PROJ, replicated on every chip): wq_b x [rows, 1280] bfp8 @
w [1280, 32768] bfp8 -> bf16, wo_b x [rows, 8192] bfp8 @ w [8192, 5120] bfp8 -> bf16 (act_dtype), ATTN_FIDELITY / fp32
DEST, at 256 (8192 chunk) and 160 (5120 chunk) rows a chip. The bank path is grouped_matmul with G = 1: bank.plan splits
the N column tiles evenly over the 8 DRAM banks (NT / 8 a bank, no padding, OV-3), each bank's columns over a cluster
of cores next to it, each core's columns K-major in its bank (one contiguous read a K block); every cluster's sender
reads ALL of x (G = 1: one group) and multicasts it to its cluster.

Step 0 (stubs first, no weight reads; the owner's kill rule): today's call (mm_configs' program config, the stock op)
vs the bank path with diag ("no_w",), ("no_x",), ("no_compute",) on every layout at its default knobs. The weight
reads are the only thing a bank layout removes: if the best layout's no-weight-reads time is >= KILL x today's, the
projection is not read-bound on this design and the probe stops there for it (KILL). Otherwise step 1: a small knob
sweep of the full bank call (block_w, depth, w_inflight), its repeat (must be 0) and its error vs fp64 next to today's.

Records qbob_us_today_<p>_<rows>, qbob_us_<layout>_<stub>_<p>_<rows>, qbob_floor_ratio_<p>_<rows> (best no_w / today),
qbob_kill_<p>_<rows> (1 = killed), and when run: qbob_us_bank_<p>_<rows>, qbob_repeat_<p>_<rows>, qbob_rel_<side>_<p>_<rows>.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_qbob_bank_probe.py"
"""

from __future__ import annotations

import pytest
import torch
from ttnn.bringup.grouped_matmul_ttnn import bank

import ttnn
from models.demos.common.bringup.core.metrics import record
from models.demos.common.bringup.testing.harness import mesh_parametrize
from models.demos.deepseek_v41_flash_d_p.tests.woa.test_woa_probe import _chips, _maxdiff, _rep, time_us
from models.demos.deepseek_v41_flash_d_p.tt import mm_configs as mm
from models.demos.deepseek_v41_flash_d_p.tt.common import act_dtype, attn_compute_config, gemm_dtype
from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

BATCH_SHARED = True  # (a mesh-only op probe: the resident runner's held mesh, 0 model loads)
DRAM = ttnn.DRAM_MEMORY_CONFIG
PROJ = {"wq_b": (1280, 32768), "wo_b": (8192, 5120)}  # (K, N) a chip under ATTN_SP32_PROJ
ROWS = (256, 160)
KILL = 0.9  # the WA.2 step-0 rule: no-weight-reads >= 0.9 x today -> not read-bound -> kill
STUBS = ("no_w", "no_x", "no_compute")
LAYOUTS = [  # (name, bank.plan shape, cores a bank); every one <= 110 cores on the matmul-capped grid (rule 18)
    ("band10", "band", 10),
    ("row11", "row", 11),
    ("rect11", "rect", 11),
    ("band8", "band", 8),
]
DEFAULT_KNOBS = (4, 1, 1)  # (depth, w_inflight, x_ahead): wo_a's WOA_BANK_KNOBS at 256 rows
# step 0's knob set a layout: the first of these (block_w, knobs) that builds and fits L1 (run 1: wo_b at 256 rows
# overflowed L1 at the planner's block_w 32 with depth 4)
STEP0 = {"wq_b": [(8, DEFAULT_KNOBS), (4, DEFAULT_KNOBS), (8, (2, 1, 1)), (4, (2, 1, 1))],
         "wo_b": [(8, DEFAULT_KNOBS), (8, (2, 1, 1)), (4, (2, 1, 1))]}  # fmt: skip
SWEEP = [(d, wi, 1) for d in (2, 3, 4) for wi in (1, 2) if wi <= d]
BLOCK_W = {"wq_b": (None, 4, 10, 20), "wo_b": (None, 8, 16)}  # None: the planner's widest that fits


class Proj:
    def __init__(self, mesh, name: str, rows: int):
        K, N = PROJ[name]
        self.mesh, self.name, self.rows, self.K, self.N = mesh, name, rows, K, N
        g = torch.Generator().manual_seed(rows + K)
        self.xt = torch.randn(1, 1, rows, K, generator=g)
        self.wt = torch.randn(1, 1, K, N, generator=g) / K**0.5
        self.x = _rep(mesh, self.xt, gemm_dtype())  # (GEMM_INPUTS: bfp8)
        self.w = _rep(mesh, self.wt, ttnn.bfloat8_b)  # (DENSE_WEIGHTS: bfp8)
        self.ydt = ttnn.bfloat16 if name == "wq_b" else act_dtype()  # (attention: wq_b bf16, wo_b act_dtype)
        self.ckc = attn_compute_config(mesh)
        self.grid = max_grid(mesh)
        self.pc = mm.program_config(self.x, self.w, self.ydt, self.ckc)
        cap = self.grid.x * self.grid.y if self.grid is not None else 10**9
        self.lays = {
            n: bank.plan(mesh, 1, K, N, cap, c, shape=sh, grid=self.grid) for n, sh, c in LAYOUTS
        }  # (None: no layout on this grid)
        self.wbs = {n: bank.upload(self.wt, lay, mesh) for n, lay in self.lays.items() if lay is not None}

    def today(self):
        return mm._run(ttnn.linear, self.x, self.w, self.pc, self.ydt, self.ckc, DRAM)

    def bank(self, lay: str, knobs=DEFAULT_KNOBS, block_w=None, diag=()):
        d, wi, xa = knobs
        return ttnn.grouped_matmul(
            self.x, self.wbs[lay], w_layout=self.lays[lay], dtype=self.ydt, compute_kernel_config=self.ckc,
            max_grid=self.grid, block_w=block_w, depth=d, w_inflight=wi, x_ahead=xa, one_group=True, diag=list(diag),
        )  # fmt: skip

    def free(self):
        for t in [self.x, self.w, *self.wbs.values()]:
            ttnn.deallocate(t)


def _ok(fn) -> str | None:
    """None when fn() builds and runs, else the first line of its error (a knob set that does not fit L1)."""
    try:
        ttnn.deallocate(fn())
        return None
    except Exception as ex:  # (a refusal or an L1 overflow of one knob set: reported, not fatal)
        return str(ex).splitlines()[0][:200]


def _rel(t, truth) -> float:
    o = _chips(t)[0].double()
    return float((o - truth).norm() / truth.norm())


@mesh_parametrize(timeout_s=3600)
def test_qbob_bank_probe(mesh_device):
    mesh = mesh_device
    if mesh is None:
        pytest.skip("device probe")
    verdict = {}
    for name in PROJ:
        for rows in ROWS:
            p = Proj(mesh, name, rows)
            tag = f"{name}_{rows}"
            lays = {n: lay for n, lay in p.lays.items() if lay is not None}
            print(f"[qbob] {tag}: today's config {p.pc}")
            for n, lay in p.lays.items():
                if lay is None:
                    print(f"[qbob] {tag}: layout {n}: none on this grid")
                else:
                    print(
                        f"[qbob] {tag}: layout {n}: {len(lay.slots)} cores, pn max {lay.pn_max}, "
                        f"{lay.tiles_per_bank} tiles a bank"
                    )
            # step 0: today and the stubs at the default knobs
            cands = [("today", p.today)]
            for n in lays:
                pick = None
                for bw, k in STEP0[name]:
                    err = _ok(lambda n=n, bw=bw, k=k: p.bank(n, knobs=k, block_w=bw))
                    if err is None:
                        pick = (bw, k)
                        break
                    print(f"[qbob] {tag}: {n} bw{bw} {k} refused: {err}")
                if pick is None:
                    continue
                print(f"[qbob] {tag}: {n} step-0 knobs bw{pick[0]} (depth, w_inflight, x_ahead) {pick[1]}")
                for s in ("full",) + STUBS:
                    d = () if s == "full" else (s,)
                    fn = lambda n=n, d=d, bw=pick[0], k=pick[1]: p.bank(n, knobs=k, block_w=bw, diag=d)  # noqa: E731
                    err = _ok(fn)
                    if err is None:
                        cands.append((f"{n}/{s}", fn))
                    else:
                        print(f"[qbob] {tag}: {n}/{s} refused: {err}")
            t = {k: v[0] for k, v in time_us(mesh, [(f"{tag}/{k}", fn) for k, fn in cands]).items()}
            t = {k[len(tag) + 1 :]: v for k, v in t.items()}
            today = t["today"]
            record(f"qbob_us_today_{tag}", round(today, 2))
            for k, us in t.items():
                if k != "today":
                    n, s = k.split("/")
                    record(f"qbob_us_{n}_{s}_{tag}", round(us, 2))
            no_w = {k.split("/")[0]: v for k, v in t.items() if k.endswith("/no_w")}
            best = min(no_w, key=no_w.get) if no_w else None
            ratio = no_w[best] / today if best else float("nan")
            kill = best is not None and ratio >= KILL
            record(f"qbob_floor_ratio_{tag}", round(ratio, 3), layout=best)
            record(f"qbob_kill_{tag}", int(kill))
            print(f"[qbob] {tag}: today {today:.2f} us")
            for n in lays:
                row = "  ".join(
                    f"{s} {t[f'{n}/{s}']:.2f} ({t[f'{n}/{s}'] / today:.3f}x)"
                    for s in ("full",) + STUBS
                    if f"{n}/{s}" in t
                )
                print(f"[qbob] {tag}:   {n}: {row}")
            print(
                f"[qbob] {tag}: best no-weight floor {best} {no_w.get(best, float('nan')):.2f} us = {ratio:.3f} x today "
                f"-> {'KILL (not read-bound on this design)' if kill else 'read-bound: step 1'}"
            )
            res = dict(today=today, best=best, ratio=ratio, kill=kill, stubs=t)
            if best is not None and not kill:
                res.update(_step1(mesh, p, tag, lays, today))
            verdict[tag] = res
            p.free()
    print("[qbob] == verdict")
    for tag, r in verdict.items():
        line = f"[qbob] {tag}: today {r['today']:.2f} us, no-w floor {r['best']} {r['ratio']:.3f} x, kill {r['kill']}"
        if "bank_us" in r:
            line += f"; bank best {r['bank_cfg']} {r['bank_us']:.2f} us ({r['bank_us'] / r['today']:.3f} x)"
        print(line)
    for tag, r in verdict.items():
        assert r["best"] is not None, f"{tag}: no bank layout ran (step 0 has no answer)"
        if "repeat" in r:
            assert r["repeat"] == 0.0, f"{tag}: the bank path is not repeatable ({r['repeat']})"


def _step1(mesh, p: Proj, tag: str, lays: dict, today: float) -> dict:
    """The full bank call's knob sweep (every layout x block_w x SWEEP), the best's repeat and error vs fp64."""
    cands = []
    for n in lays:
        for bw in BLOCK_W[p.name]:
            for k in SWEEP:
                fn = lambda n=n, bw=bw, k=k: p.bank(n, knobs=k, block_w=bw)  # noqa: E731
                if _ok(fn) is None:
                    cands.append((f"{n}/bw{bw}_d{k[0]}_wi{k[1]}_xa{k[2]}", fn))
    t = {}
    for _ in range(2):  # (twice; the min of the two)
        got = time_us(mesh, [(f"{tag}/s/{k}", fn) for k, fn in cands])
        for k, v in got.items():
            k = k[len(tag) + 3 :]
            t[k] = min(t.get(k, float("inf")), v[0])
    for k, us in sorted(t.items(), key=lambda kv: kv[1])[:12]:
        print(f"[qbob] {tag}: sweep {k}: {us:.2f} us ({us / today:.3f} x today)")
    best = min(t, key=t.get)
    fn = dict(cands)[best]
    a, b = fn(), fn()
    rep = _maxdiff(a, b)
    truth = _chips(p.x)[0].double() @ _chips(p.w)[0].double()  # (the device's bfp8 inputs, fp64 product)
    y = p.today()
    rel_t, rel_b = _rel(y, truth), _rel(a, truth)
    for o in (a, b, y):
        ttnn.deallocate(o)
    record(f"qbob_us_bank_{tag}", round(t[best], 2), cfg=best, n_cfgs=len(t))
    record(f"qbob_repeat_{tag}", rep)
    record(f"qbob_rel_today_{tag}", rel_t)
    record(f"qbob_rel_bank_{tag}", rel_b)
    print(
        f"[qbob] {tag}: bank best {best} {t[best]:.2f} us vs today {today:.2f} ({t[best] / today:.3f} x); "
        f"repeat {rep}; rel vs fp64 today {rel_t:.3e} bank {rel_b:.3e}"
    )
    return dict(bank_us=t[best], bank_cfg=best, repeat=rep, rel_today=rel_t, rel_bank=rel_b)
