#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""numeric_pinpoint.py: after a NUMERIC FAIL, narrow the failing chip's fault down to a part.

run.sh (lib/pinpoint.sh) runs it once, before the end reset, only when numeric.txt or the sustained report has a NUMERIC
FAIL line. It opens the 4x8 mesh with no fabric (no ETH link is used: inside the box) and runs the splits on all 32 chips
at once; the other 31 chips are the control (a chip is compared with the known input, or with the output most chips
agree on, bit for bit). First, once, repeat: the failing shape again (heavy load op + checked C4 matmul, 400
launches or 60 s, until the fault comes back; one fault hit 1 in 194). Then the splits in order: dram, mm (L1 vs DRAM), heavy (a known trigger: back-to-back C4n matmul with x,
w, y in DRAM, plus an x-in-L1, a read-only and a write-only arm, --heavy-s a pass; x and w stay uploaded across passes), rw (once:
the C4n-size x/w/y-in-DRAM matmul under a PINPOINT_RW_S 60 s burst, numeric_dram.rw_burst), core (per-core FPU
and SFPU), clock (capped AICLK, once). The passes repeat until every failing chip is isolated (pinpoint_verdict.isolated) or the time budget is spent
(one pass once ended after 9.5 s of 300 s, 'not isolated', on a chip the heavy op hits 187/187).
pinpoint_verdict.py holds the split descriptions and turns --result into the NUMERIC-PINPOINT lines.
Every check runs under run.sh's max-load stage load on the whole mesh (pinpoint_load.py; --load-args off: solo),
with each chip's W, pre-fault temperature and the box kW per launch in <result>.load.jsonl and res['load'].

  numeric_pinpoint.py --numeric <numeric.txt> [--c7 <report.txt>] --result <json> [--budget-s 270] [--heavy-s 90]
                     [--heavy-op <run.sh's --heavy>] [--load-args "<fl_load_args heavy>"|off]

--result is rewritten after every split (a timeout keeps what ran). Exit 0 (mismatches never fail it), 1 an error.
"""
import argparse
import json
import os
import re
import sys
import time

import pinpoint_verdict as pv

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet"))
import chunk_sched as cs  # noqa: E402  core_of: the compute core owning a matmul output word
import stop_signal as ss  # noqa: E402


def asic_of(bdf):
    """'0000:45:00.0' or '45:00.0' -> the ASIC number 1-8 in its tray (bus & 0xf, as fleet_py.ta and the dashboard cards)."""
    try:
        return int(str(bdf).split(":")[-2], 16) & 0x0F or None
    except (ValueError, IndexError):
        return None


PATTERNS = ["rand", 0x5555, 0x2AAA, 0x7F7F, 0x3F80, 0x55AA]  # normal bf16 only (no Inf/NaN/denormal)
CAP_MHZ = int(os.environ.get("PINPOINT_CAP_MHZ", "800"))
# heavy split: the C4n shape (x MxK, w KxN, y MxN); N/HR_PIECES and N stay multiples of 8 tile columns, so
# its per-bank counts are tile index mod 8, a DRAM label only for the DRAM readback splits (a matmul output's
# wrong word is named by its owning compute core, cs.core_of). PINPOINT_HEAVY_SHAPE=M,K,N for tests.
HEAVY_SHAPE = tuple(int(v) for v in os.environ.get("PINPOINT_HEAVY_SHAPE", "2048,8192,16384").split(","))
HR_PIECES = 16  # read-only arm: w in 16 DRAM pieces, a group of pieces' outputs in L1 at a time
HR_GROUP = [4, 2, 1]  # pieces per group; a failed allocation retries at the next level
HW_K = [1024, 512, 256]  # write-only arm: x MxK and w KxN in L1, the full y written to DRAM
XL_DIV = [1, 2, 4]  # x-in-L1 arm: x in row chunks of M/1, M/2, M/4 rows in L1, w read, y written in DRAM
HEAVY_ARMS = (
    ("hv", "heavy"),
    ("xl", "heavy_xl"),
    ("hr", "heavy_read"),
    ("hw", "heavy_write"),
    ("hd", "heavy_dec"),
)  # arm -> its result slot
HEAVY_LEVELS = dict(hr=HR_GROUP, hw=HW_K, xl=XL_DIV)
HEAVY_ARM_MIN = 3  # checks each of the xl, read and write arms gets before the heavy split may stop early
# repeat split: launches and seconds of the failing shape before the component splits, acc read every WIN
REPEAT_N = int(os.environ.get("PINPOINT_REPEAT_N", "400"))
REPEAT_S = float(os.environ.get("PINPOINT_REPEAT_S", "60"))
REPEAT_WIN = int(os.environ.get("PINPOINT_REPEAT_WIN", "100"))


def log(msg):
    print(f"{time.strftime('%F %T')} PINPOINT {msg}", flush=True)


def stop_label(left_s, ran, errors):
    """Why a pass that found nothing new to run ends the run (say when earlier splits errored)."""
    if left_s < 60:
        return "budget"
    if ran:
        return "no split could run"
    return f"every split done ({len(errors)} errored)" if errors else "every split done"


def wx_bad_words(got, exp):
    """Words of a device readback (flat int16 values) that differ bit for bit from the host operand exp."""
    return int((got.reshape(-1) != exp.reshape(-1)).sum())


def record_wx_readback(chips, mesh_bdf, per_chip_got, exp_by_name):
    """chips[bdf]['wx_readback'] = {'bad_words': N}: the x and w operands read back from each chip's DRAM against the
    host copy (per_chip_got: name -> one int16 tensor per mesh chip; exp_by_name: name -> int16 host tensor)."""
    for i, bdf in mesh_bdf.items():
        n = sum(wx_bad_words(per_chip_got[k][i], e) for k, e in exp_by_name.items())
        chips[bdf]["wx_readback"] = dict(bad_words=n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--numeric", default="")
    ap.add_argument("--c7", default="")
    ap.add_argument("--result", required=True)
    ap.add_argument("--budget-s", type=float, default=270)
    ap.add_argument("--rounds", type=int, default=2, help="dram split rounds (patterns)")
    ap.add_argument("--reps", type=int, default=4, help="checks per op in the mm and core splits")
    ap.add_argument("--heavy-s", type=float, default=90, help="heavy split wall time a pass")
    ap.add_argument("--heavy", type=int, default=3, help="heavy matmuls queued before each heavy-split check")
    ap.add_argument(
        "--heavy-op",
        default="",
        help="the repeat split's load op: run.sh's --heavy (ub0_repro.py's; "
        "default env TRAY_HEAVY, which sudo may drop, else ub0_repro's default)",
    )
    ap.add_argument(
        "--load-args",
        default="",
        help="the max-load stage's chunk_load flags (lib/fleet.sh fl_load_args "
        "heavy): every check runs under that load on the whole mesh; '' the ub0_repro defaults, 'off' none",
    )
    ap.add_argument("--mesh", default="4x8")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    left = lambda: a.budget_s - (time.monotonic() - t0)
    tg = pv.targets(pv.read_lines(a.numeric), pv.read_lines(a.c7))
    res = dict(
        schema="pinpoint-1",
        started=time.strftime("%F %T %Z"),
        budget_s=a.budget_s,
        targets=sorted(tg),
        splits=[],
        stopped="",
        errors={},
        chips={},
        s=0,
    )
    L = T = None  # the max-load stage's load and its per-launch telemetry

    def save():
        res["s"] = round(time.monotonic() - t0, 1)
        if L is not None:
            res["load"] = dict(
                op=L.desc,
                off=L.off,
                bursts=L.bursts,
                cq=L.cq_why or "pcie on cq 1",
                jsonl=a.result + ".load.jsonl",
                **(T.summary() if T else {}),
            )
        with open(a.result + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(a.result + ".tmp", a.result)

    names = ", ".join(f"{b} tray {t['tray']}" for b, t in sorted(tg.items()))
    log(f"failing chips: {names or 'none'}; " f"budget {a.budget_s:.0f} s")
    save()
    if not tg:
        res["stopped"] = "no NUMERIC FAIL line"
        save()
        return 0

    import pinpoint_load as pl
    import torch
    import ttnn
    import ub0_repro as ub
    from numeric_check import umd_map_from_stdout
    from ub0_repro import chips, tray_of

    r, c = (int(v) for v in a.mesh.split("x"))
    st = dict(mesh=None, pass_=0, checks=0, clock_done=False, split="")
    la = pl.parse(a.load_args, a.heavy_op or os.environ.get("TRAY_HEAVY"), ub)
    cq = dict(why=None if la and la.cqs > 1 else "one command queue")

    def open_mesh():
        if (
            la and la.cqs > 1 and cq["why"] is None
        ):  # the load's PCIe write on cq 1, beside the checks on cq 0 (chunk_load.py)
            try:
                st["mesh"] = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(r, c), num_command_queues=la.cqs)
                return st["mesh"]
            except Exception as e:
                cq["why"] = f"2 command queues refused ({type(e).__name__}: {str(e)[:120]})"
        st["mesh"] = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(r, c))
        return st["mesh"]

    def close_mesh():
        if L is not None:
            L.free()
        ttnn.close_mesh_device(st["mesh"])
        st["mesh"] = None

    ss.install("PINPOINT")  # TERM/INT: the finally blocks below close the mesh and restore the clocks
    mesh = open_mesh()
    try:  # one finally for everything after the open: a TERM during setup closes the mesh too
        ss.arm()
        log(f"mesh open {tuple(mesh.shape)} in {time.monotonic() - t0:.0f} s")
        try:
            metal = [int(v) for v in mesh.get_device_ids()]
        except Exception:
            metal = list(range(r * c))
        umd, ids = umd_map_from_stdout(), chips()
        mesh_bdf = {}
        for i, m in enumerate(metal):
            dev = umd.get(m, m) if umd else m
            p = ids.get(dev)
            bdf = re.sub(r"^[0-9a-f]{4}:", "", p) if p else f"metal{m}"
            mesh_bdf[i] = bdf
            res["chips"][bdf] = dict(mesh=i, metal=m, dev=dev, tray=tray_of(p) if p else None)
        res["map_source"] = "umd_log" if umd else "assumed_identity"
        nb = 8
        for get in (lambda: mesh.get_devices()[0].num_dram_channels(), lambda: mesh.num_dram_channels()):
            try:
                nb = int(get()) or 8
                break
            except Exception:
                pass
        try:
            g = mesh.compute_with_storage_grid_size()
            gx, gy = int(g.x), int(g.y)
        except Exception:
            gx, gy = 0, 0
        res.update(banks=nb, grid=[gx, gy])

        def own(H, W):
            """(row, col) -> the compute core that wrote that word of an HxW matmul output on the gx x gy grid."""
            return (lambda r, c: cs.core_of(r, c, H, W, [gx, gy])) if gx and gy else None

        if la is not None:
            cq1 = None if cq["why"] else (ttnn.QueueId(1) if hasattr(ttnn, "QueueId") else 1)
            L = pl.Load(la, ttnn, torch, log, time.sleep, cq1=cq1, cq_why=cq["why"])
            T = pl.Tele(res["chips"], tg, a.result + ".load.jsonl", log)
            log(
                f"load: run.sh's max-load stage op ({la.heavy}) on all {r * c} chips around every check; "
                f"per-launch W, C and box kW to {a.result}.load.jsonl"
            )
        else:
            log("load: off (--load-args off); the checks run solo")

        def up(t, mc, dtype=None):
            return ttnn.from_torch(
                t,
                dtype=dtype or ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=st["mesh"],
                memory_config=mc,
                mesh_mapper=ttnn.ReplicateTensorToMesh(st["mesh"]),
            )

        def per_chip(t, H, W):
            """One launch: the check's readback, under the load (a burst ahead of the check op, one heavy op behind it), the
            chips' W and C read while it runs, then the gap and the next check's burst."""
            on = L is not None and L.before_read(st["mesh"])
            if T is not None:
                T.sample(st["split"] + ("" if on else " (no load)"))
            got = [
                ttnn.to_torch(d).reshape(H, W).contiguous().view(torch.int16) for d in ttnn.get_device_tensors(t.cpu())
            ]
            if on:
                L.after_read(st["mesh"])
            return got

        def majority(got):
            reps = []
            for i, t in enumerate(got):
                for rep in reps:
                    if torch.equal(rep[0], t):
                        rep[1].append(i)
                        break
                else:
                    reps.append((t, [i]))
            return max(reps, key=lambda x: len(x[1]))[0]

        def slot(split, i, **init):
            return res["chips"][mesh_bdf[i]].setdefault(split, dict(checks=0, bad_checks=0, bad=0, **init))

        def tiles(H, W):
            Wt = W // 32
            rr = torch.arange(H).view(H, 1).expand(H, W)
            cc = torch.arange(W).view(1, W).expand(H, W)
            return (((rr // 32) * Wt + cc // 32) % nb).reshape(-1), (cc % 2).reshape(-1), (cc * 8 // W).reshape(-1)

        def count(
            split,
            got,
            exp,
            bank_of=None,
            lane_of=None,
            extra=None,
            again=None,
            col_of=None,
            decoupled=False,
            owner=None,
        ):
            """Every chip's readback against exp (int16): per chip checks, bad checks, words, banks, lanes. owner(row, col) names
            the compute core that wrote a matmul output word (cs.core_of under the op's grid and per-core split);
            a compute split with an owner logs and records that core, never a tile-index 'bank'; the DRAM readback splits
            (mm_dram, heavy_read, heavy_write) keep their DRAM bank as well."""
            e = exp.reshape(-1)
            bank_ok = bank_of is not None and (owner is None or split not in pv.COMPUTE_SPLITS)
            second, n_bad = None, 0
            st["checks"] += 1
            for i, gt in enumerate(got):
                s = slot(split, i, per_bank=[0] * nb, per_lane=[0, 0], per_col=[0] * 8, inf=0, nan=0)
                s["checks"] += 1
                gf = gt.reshape(-1)
                bad = torch.nonzero(gf != e).flatten()
                if not bad.numel():
                    continue
                n_bad += 1
                s["bad_checks"] += 1
                s["bad"] += int(bad.numel())
                f = gf[bad].view(torch.bfloat16).float()
                s["inf"] += int(torch.isinf(f).sum())
                s["nan"] += int(torch.isnan(f).sum())
                if bank_of is not None:
                    s["per_bank"] = [
                        u + v for u, v in zip(s["per_bank"], torch.bincount(bank_of[bad], minlength=nb).tolist())
                    ]
                    s["per_lane"] = [
                        u + v for u, v in zip(s["per_lane"], torch.bincount(lane_of[bad], minlength=2).tolist())
                    ]
                    if col_of is not None:
                        s["per_col"] = [
                            u + v for u, v in zip(s["per_col"], torch.bincount(col_of[bad], minlength=8).tolist())
                        ]
                    if decoupled:
                        s["decoupled"] = True
                if T is not None:
                    T.faulted(mesh_bdf[i])
                if gt.dim() == 2 and len(s.setdefault("locs", [])) < 8:  # where: the first bad word
                    j = int(bad[0])
                    rw_, cl_ = divmod(j, int(gt.shape[1]))
                    ch = res["chips"].get(mesh_bdf[i], {})
                    lo = dict(
                        split=split,
                        launch=T.n if T else None,
                        row=rw_,
                        col=cl_,
                        tile=[rw_ // 32, cl_ // 32],
                        bank=int(bank_of[j]) if bank_ok else None,
                        words=int(bad.numel()),
                        bdf=mesh_bdf[i],
                        tray=ch.get("tray"),
                        asic=asic_of(mesh_bdf[i]),
                    )
                    if owner is not None:
                        o = owner(rw_, cl_)
                        lo.update(
                            core=o["core"],
                            element=o["element"],
                            face=o["face"],
                            in_face=o["in_face"],
                            grid=o["grid"],
                            per_core=o["per_core"],
                        )
                        s["per_core_bad"] = s.get("per_core_bad") or {}
                        for jj in bad[:4096].tolist():  # bounded: which cores own the wrong words
                            r_, c_ = divmod(int(jj), int(gt.shape[1]))
                            k_ = "%d,%d" % tuple(owner(r_, c_)["core"])
                            s["per_core_bad"][k_] = s["per_core_bad"].get(k_, 0) + 1
                    s["locs"].append(lo)
                if extra:
                    extra(s, bad)
                if again is not None:
                    second = second if second is not None else again()
                    s["reread_n"] = s.get("reread_n", 0) + 1
                    s["reread_same"] = s.get("reread_same", 0) + int(torch.equal(second[i].reshape(-1), gf))
                lc = (s.get("locs") or [None])[-1]
                # DRAM readback splits name both: the bank that held the word and the core that wrote it
                where = "; ".join(
                    ([f"DRAM banks {torch.bincount(bank_of[bad], minlength=nb).tolist()}"] if bank_ok else [])
                    + ([f"compute cores x,y {s['per_core_bad']}"] if owner is not None else [])
                )
                log(
                    f"BAD {split} mesh{i} {mesh_bdf[i]} tray {res['chips'][mesh_bdf[i]].get('tray')}: {int(bad.numel())} words"
                    + (f", {where}" if where else "")
                    + (
                        f"; first at row {lc['row']} col {lc['col']} (tile {lc['tile'][0]},{lc['tile'][1]}"
                        + (f", bank {lc['bank']}" if lc["bank"] is not None else "")
                        + (
                            f", compute core ({lc['core'][0]},{lc['core'][1]}) of the {lc['grid'][0]}x{lc['grid'][1]} grid, element "
                            f"({lc['element'][0]},{lc['element'][1]}), 16x16 face {lc['face'][0]},{lc['face'][1]} element "
                            f"{lc['in_face'][0]},{lc['in_face'][1]}"
                            if lc.get("core")
                            else ""
                        )
                        + f") launch {lc['launch']}"
                        if lc and lc["split"] == split
                        else ""
                    )
                )
            return n_bad

        def done_all():
            return all(pv.isolated(res["chips"].get(b)) for b in tg)

        # --- split dram: no compute kernel; per phase, bank and lane (a DRAM-only test, shortened)
        def split_dram():
            H, W = 512, 4096
            bank_of, lane_of, col_of = tiles(H, W)
            for rnd in range((st["pass_"] - 1) * a.rounds, st["pass_"] * a.rounds):  # a new pattern each pass
                if left() < 40:
                    return "budget"
                pat = PATTERNS[rnd % len(PATTERNS)]
                xb = (
                    torch.randn(H, W, generator=torch.Generator().manual_seed(1024000 + rnd)).to(torch.bfloat16)
                    if pat == "rand"
                    else torch.full((H, W), pat, dtype=torch.int16).view(torch.bfloat16)
                )
                x, exp = xb.reshape(1, 1, H, W), xb.view(torch.int16)

                def run(p, t):
                    before = [
                        slot("dram", i, per_bank=[0] * nb, per_lane=[0, 0], per_col=[0] * 8, inf=0, nan=0)["bad_checks"]
                        for i in range(len(mesh_bdf))
                    ]
                    count(
                        "dram", per_chip(t, H, W), exp, bank_of, lane_of, col_of=col_of, again=lambda: per_chip(t, H, W)
                    )
                    for i in range(len(mesh_bdf)):
                        s = slot("dram", i)
                        k = s.setdefault("per_phase", {}).setdefault(p, [0, 0])
                        k[1] += 1
                        k[0] += int(s["bad_checks"] > before[i])

                d = up(x, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.synchronize_device(st["mesh"])
                run("hr", d)
                l1 = ttnn.to_memory_config(d, ttnn.L1_MEMORY_CONFIG)
                ttnn.synchronize_device(st["mesh"])
                run("nr", l1)
                ttnn.deallocate(l1)
                l1 = up(x, ttnn.L1_MEMORY_CONFIG)
                ttnn.synchronize_device(st["mesh"])
                run("hl", l1)
                d2 = ttnn.to_memory_config(l1, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.synchronize_device(st["mesh"])
                ttnn.deallocate(l1)
                run("nw", d2)
                ttnn.deallocate(d2)
                d3 = ttnn.clone(d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.synchronize_device(st["mesh"])
                run("rw", d3)
                ttnn.deallocate(d3)
                ttnn.deallocate(d)
                log(
                    f"dram round {rnd} ({pat if isinstance(pat, str) else hex(pat)}) done at {time.monotonic() - t0:.0f} s"
                )
            return ""

        # --- split mm: the same matmul with its tensors in L1 and in DRAM; output tiles -> DRAM bank
        M, K, N = 1024, 2048, 2048
        gen = torch.Generator().manual_seed(1024)
        acts = [torch.randn(1, 1, M, K, generator=gen).to(torch.bfloat16) for _ in range(2)]
        wt = (torch.randn(1, 1, K, N, generator=gen) / K**0.5).to(torch.bfloat16)

        def mm_run(split, mc, reps):
            bank_of, lane_of, col_of = tiles(M, N)
            w = up(wt, mc)
            xs = [up(x, mc) for x in acts]
            gold = {}
            for k in range(reps):
                if left() < 15:
                    break
                y = ttnn.matmul(xs[k % 2], w, memory_config=mc)
                got = per_chip(y, M, N)
                ttnn.deallocate(y)
                if k % 2 not in gold:
                    gold[k % 2] = majority(got)
                count(split, got, gold[k % 2], bank_of, lane_of, col_of=col_of, owner=own(M, N))
            for t in xs + [w]:
                ttnn.deallocate(t)

        def split_mm():
            for split, mc in (("mm_l1", ttnn.L1_MEMORY_CONFIG), ("mm_dram", ttnn.DRAM_MEMORY_CONFIG)):
                if left() < 30:
                    return "budget"
                mm_run(split, mc, a.reps)
                log(f"{split} done at {time.monotonic() - t0:.0f} s")
            return ""

        # --- split heavy (a known trigger of a bank 3 fault): back-to-back C4n matmuls with x,
        # w, y in DRAM on all 32 chips, one check between bursts, in turn: hv the heavy op's own output (x, w read, y written),
        # xl x in L1 row chunks with w read and y written in DRAM (49/54 on one failing chip, first hit 22 s
        # into load, while hr only hit after ~4.3 min), hr x, w read from DRAM with y kept in L1 (read only), hw x, w in L1
        # with the full y written to DRAM (write only).
        # Bad words per DRAM bank (tile column mod banks) and lane, outputs against the majority of the 32 chips.
        # x, w and w's 16 pieces (~512 MB a chip) stay uploaded across passes and the repeat split (hdev; heavy_free).
        HM, HK, HN = HEAVY_SHAPE
        hst, hdev = {}, {}

        def heavy_inputs():
            if not hst:  # host inputs once per process (the w alone is 256 MB in bf16)
                g = torch.Generator().manual_seed(1411)
                rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16)
                hst.update(
                    x=rn(1, 1, HM, HK),
                    w=rn(1, 1, HK, HN, sc=0.02),
                    hw={k: (rn(1, 1, HM, k), rn(1, 1, k, HN, sc=0.03)) for k in HW_K},
                )

        def heavy_dev(pieces=False):
            """x and w in DRAM (and w's HR_PIECES pieces when asked), uploaded once while the mesh stays open."""
            heavy_inputs()
            D = ttnn.DRAM_MEMORY_CONFIG
            if "x" not in hdev:
                hdev["x"], hdev["w"] = up(hst["x"], D), up(hst["w"], D)
            if "wn" not in hdev:  # hd arm: w without its last tile column, N one tile short of a multiple of 8 (511)
                hdev["wn"] = up(hst["w"][..., : HN - 32].contiguous(), D)
            if pieces and "wp" not in hdev:
                pw = HN // HR_PIECES
                hdev["wp"] = [up(hst["w"][..., j * pw : (j + 1) * pw].contiguous(), D) for j in range(HR_PIECES)]
            return hdev["x"], hdev["w"], hdev.get("wp") or []

        def heavy_free(keys=("x", "w", "wp", "wn")):
            for key in keys:
                ts = hdev.pop(key, None)
                for t_ in ts if isinstance(ts, list) else [ts] if ts is not None else []:
                    try:
                        ttnn.deallocate(t_)
                    except Exception:
                        pass

        def heavy_decided():
            """Every failing chip hit at least twice in hv and xl together on one bank, and the xl, read and write arms ran
            or were dropped."""
            hs = res["heavy"]
            for b in tg:
                ch = res["chips"].get(b) or {}
                hv, xl = ch.get("heavy") or {}, ch.get("heavy_xl") or {}
                pb = [u + v for u, v in zip(hv.get("per_bank") or [0] * nb, xl.get("per_bank") or [0] * nb)]
                if hv.get("bad_checks", 0) + xl.get("bad_checks", 0) < 2 or not pv.bank_comp(pb)[0]:
                    return False
            return all(arm in hs["dropped"] or hs["checks"].get(arm, 0) >= HEAVY_ARM_MIN for arm in ("xl", "hr", "hw"))

        def split_heavy():
            if left() < 45:
                return "budget"
            hs = res.setdefault("heavy", dict(shape=[HM, HK, HN], checks={}, dropped={}, level={}, matmuls=0))
            D, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
            bank_of, lane_of, col_of = tiles(HM, HN)
            pw = HN // HR_PIECES
            # hd: y has HN/32 - 1 tile columns, so its tile column no longer maps to bank n mod 8: a DRAM bank fault keeps
            # to one bank while a compute fault spreads over all banks (pinpoint_verdict names the cause)
            bank_d, lane_d, col_d = tiles(HM, HN - 32)

            def free(ts):
                for t_ in ts:
                    try:
                        ttnn.deallocate(t_)
                    except Exception:
                        pass

            def burst(n):
                y = None
                for _ in range(n):
                    if y is not None:
                        ttnn.deallocate(y)
                    y = ttnn.matmul(xh, wh, memory_config=D)
                hs["matmuls"] += n
                return y

            try:
                xh, wh, whp = heavy_dev(pieces="hr" not in hs["dropped"])
                ttnn.synchronize_device(st["mesh"])
                if not any("wx_readback" in c for c in res["chips"].values()):
                    record_wx_readback(
                        res["chips"],
                        mesh_bdf,
                        dict(x=per_chip(xh, HM, HK), w=per_chip(wh, HK, HN)),
                        dict(
                            x=hst["x"].reshape(HM, HK).view(torch.int16), w=hst["w"].reshape(HK, HN).view(torch.int16)
                        ),
                    )
                t_end = time.monotonic() + min(a.heavy_s, left() - 40)
                k = 0
                while time.monotonic() < t_end and not heavy_decided() and len(hs["dropped"]) < len(HEAVY_ARMS):
                    arm, slot_name = HEAVY_ARMS[k % len(HEAVY_ARMS)]
                    k += 1
                    if arm in hs["dropped"]:
                        continue
                    lvl = hs["level"].get(arm, 0)
                    tmp, y0, out, outs, again = [], None, None, [], None
                    try:
                        if arm == "hw":
                            tmp = [up(hst["hw"][HW_K[lvl]][0], L1), up(hst["hw"][HW_K[lvl]][1], L1)]
                        y0 = burst(a.heavy)  # the load before the check (queued, no sync)
                        if arm == "hv":
                            out = ttnn.matmul(xh, wh, memory_config=D)
                            hs["matmuls"] += 1
                        elif arm == "hd":
                            out = ttnn.matmul(xh, hdev["wn"], memory_config=D)
                        elif arm == "hw":
                            out = ttnn.matmul(tmp[0], tmp[1], memory_config=D)
                        elif arm == "xl":  # x row chunks in L1, the full w read from DRAM, y row chunks written to DRAM
                            rows = HM // XL_DIV[lvl]
                            for m0 in range(0, HM, rows):
                                tmp.append(up(hst["x"][..., m0 : m0 + rows, :].contiguous(), L1))
                                outs.append(ttnn.matmul(tmp[-1], wh, memory_config=D))
                                ttnn.deallocate(tmp.pop())
                        else:
                            parts = []
                            for g0 in range(0, HR_PIECES, HR_GROUP[lvl]):
                                tmp = [
                                    ttnn.matmul(xh, whp[j], memory_config=L1)
                                    for j in range(g0, min(g0 + HR_GROUP[lvl], HR_PIECES))
                                ]
                                parts += [per_chip(y, HM, pw) for y in tmp]  # syncs
                                free(tmp)
                                tmp = []
                                ttnn.deallocate(y0)  # keep the load going between groups
                                y0 = burst(1)
                            got = [torch.cat([pt[i] for pt in parts], dim=1) for i in range(len(parts[0]))]
                        ttnn.synchronize_device(st["mesh"])
                        free(tmp)
                        tmp = []
                        ttnn.deallocate(y0)
                        y0 = None
                        if out is not None:
                            WO = HN - 32 if arm == "hd" else HN
                            got = per_chip(out, HM, WO)
                            again = lambda: per_chip(out, HM, WO)
                        elif outs:
                            rows = HM // XL_DIV[lvl]
                            cat = lambda: [torch.cat(c, dim=0) for c in zip(*(per_chip(o, rows, HN) for o in outs))]
                            got, again = cat(), cat
                    except Exception as e:
                        err = f"{type(e).__name__}: {str(e)[:300]}"
                        free(tmp + outs + [t_ for t_ in (y0, out) if t_ is not None])
                        if arm in HEAVY_LEVELS and lvl + 1 < len(HEAVY_LEVELS[arm]):
                            hs["level"][arm] = lvl + 1
                            log(f"heavy {arm} failed at level {lvl}, next level {lvl + 1}: {err}")
                        else:
                            hs["dropped"][arm] = err
                            log(f"heavy {arm} dropped: {err}")
                            if arm == "hr":
                                heavy_free(("wp",))
                        continue
                    hs["checks"][arm] = hs["checks"].get(arm, 0) + 1
                    if arm == "hd":
                        count(
                            slot_name,
                            got,
                            majority(got),
                            bank_d,
                            lane_d,
                            col_of=col_d,
                            again=again,
                            decoupled=True,
                            owner=own(HM, HN - 32),
                        )
                    elif arm == "xl":  # each row chunk is its own matmul over the whole grid
                        rows = HM // XL_DIV[lvl]
                        _o = own(rows, HN)  # no grid read: no owner, so count keeps the DRAM bank label
                        count(
                            slot_name,
                            got,
                            majority(got),
                            bank_of,
                            lane_of,
                            col_of=col_of,
                            again=again,
                            owner=None if _o is None else (lambda r, c, _o=_o, _n=rows: _o(r % _n, c)),
                        )
                    elif arm == "hr":  # each column piece is its own matmul over the whole grid
                        _o = own(HM, pw)
                        count(
                            slot_name,
                            got,
                            majority(got),
                            bank_of,
                            lane_of,
                            col_of=col_of,
                            again=again,
                            owner=None if _o is None else (lambda r, c, _o=_o, _n=pw: _o(r, c % _n)),
                        )
                    else:
                        count(
                            slot_name,
                            got,
                            majority(got),
                            bank_of,
                            lane_of,
                            col_of=col_of,
                            again=again,
                            owner=own(HM, HN),
                        )
                    free(outs + ([out] if out is not None else []))
                log(
                    f"heavy split pass {st['pass_']} done at {time.monotonic() - t0:.0f} s: checks {hs['checks']}, "
                    f"{hs['matmuls']} heavy matmuls, dropped {sorted(hs['dropped']) or 'none'}"
                )
            except Exception:
                heavy_free()  # an upload or sync failed: upload again next pass
                raise
            if len(hs["dropped"]) == len(HEAVY_ARMS):  # the loop skips a split in res['errors']: no upload again
                heavy_free()
                res["errors"]["heavy"] = "all arms dropped: " + "; ".join(
                    f"{k} {v}" for k, v in sorted(hs["dropped"].items())
                )
                return "error"
            return ""

        # --- split repeat (one chip's [heavy] fault hit 1 in 194
        # launches, and 26 tries in 9.2 s said 'not isolated'): once, before the component splits, the failing shape again:
        # run.sh's own heavy load op (ub0_repro.heavy_op, --heavy, no fabric) then the checked C4 matmul (x, w, y in DRAM),
        # REPEAT_N launches or REPEAT_S seconds. Each output is compared on the device (acc += ne(y, ref), no readback, as
        # chunk_load.py does), the acc read every REPEAT_WIN launches; it stops at the first window a failing chip is bad in.
        def split_repeat():
            if st["pass_"] > 1:
                return "once"
            rp = res.setdefault("repeat", dict(shape=[HM, HK, HN], load=None, launches=0, windows=0, s=0, back=None))
            D = ttnn.DRAM_MEMORY_CONFIG
            bank_of, lane_of, col_of = tiles(HM, HN)
            held = []

            def free(*ts):
                for t_ in ts:
                    if t_ is not None:
                        try:
                            ttnn.deallocate(t_)
                        except Exception:
                            pass

            def unhold(t):  # by identity: list.remove compares with ==, a device ttnn.eq on ttnn tensors
                held[:] = [h for h in held if h is not t]

            try:
                xh, wh, _ = heavy_dev()  # kept for the heavy split
                load = L.build(st["mesh"]) if L is not None else None
                if load is not None:  # the max-load stage's own op, in its bursts and gaps
                    rp["load"] = L.desc
                else:  # load off: run.sh's heavy op alone, back to back
                    try:
                        hv = a.heavy_op or os.environ.get("TRAY_HEAVY") or ub.HEAVY_DEFAULT
                        op, _, desc = ub.heavy_op(
                            argparse.Namespace(heavy=hv, m=HM, k=HK, n=HN, topo="none"),
                            ttnn,
                            torch,
                            st["mesh"],
                            xh,
                            wh,
                            fabric=False,
                        )
                        held += list(getattr(op, "tensors", ()))
                        free(op())
                        load, rp["load"] = op, f"{hv}: {desc}"
                    except Exception as e:  # the load op cannot run here: the checked matmul alone, back to back
                        rp["load"] = f"none ({type(e).__name__}: {str(e)[:160]}); checked matmul only"
                log(f"repeat: load {rp['load']}; up to {REPEAT_N} launches or {REPEAT_S:g} s")
                ref = ttnn.matmul(xh, wh, memory_config=D)
                held.append(ref)
                got = per_chip(ref, HM, HN)
                count(
                    "repeat", got, majority(got), bank_of, lane_of, col_of=col_of, owner=own(HM, HN)
                )  # wrong from launch 0 on
                for b in res["chips"]:
                    s0 = res["chips"][b].get("repeat")
                    if s0:
                        s0["launch0_bad"] = int(s0["bad_checks"] > 0)  # repeat_line: 'wrong from launch 0'
                acc = ttnn.ne(ref, ref, memory_config=D)
                held.append(acc)
                ts, n = time.monotonic(), 0
                t_end = ts + max(0.0, min(REPEAT_S, left() - 60))  # the cap counts launches only, not the setup
                while True:
                    stop = n >= REPEAT_N or time.monotonic() >= t_end
                    if n and (stop or n % REPEAT_WIN == 0):
                        ttnn.synchronize_device(st["mesh"])
                        zero = torch.zeros(HM * HN, dtype=torch.int16)
                        a_ = per_chip(acc, HM, HN)
                        before = {b: (res["chips"][b].get("repeat") or {}).get("bad_checks", 0) for b in tg}
                        count("repeat", a_, zero, bank_of, lane_of, col_of=col_of, owner=own(HM, HN))
                        rp["windows"] += 1
                        rp["launches"], rp["s"] = n, round(time.monotonic() - ts, 1)
                        save()
                        hit = [b for b in tg if (res["chips"][b].get("repeat") or {}).get("bad_checks", 0) > before[b]]
                        if hit:
                            rp["back"] = dict(launches=n, s=rp["s"], chips=hit)
                            log(f"repeat: fault back on {', '.join(hit)} within {n} launches ({rp['s']} s)")
                            break
                        if not stop:
                            unhold(acc)
                            free(acc)
                            acc = ttnn.ne(ref, ref, memory_config=D)  # a new window starts at zero
                            held.append(acc)
                    if stop:
                        break
                    if load is not None and L is not None and L.op is load:  # a burst of the load, then its gap
                        if n and n % L.la.burst_steps == 0:
                            ttnn.synchronize_device(st["mesh"])
                            T.sample("repeat")
                            T.commit()
                            time.sleep(L.la.burst_gap_ms / 1000.0)
                        free(load())
                    elif load is not None:
                        free(load())
                    y = ttnn.matmul(xh, wh, memory_config=D)
                    d = ttnn.ne(y, ref, memory_config=D)
                    s = ttnn.add(acc, d, memory_config=D)
                    free(y, d)
                    unhold(acc)
                    free(acc)
                    acc = s
                    held.append(acc)
                    n += 1
                if rp["back"] is None:
                    log(f"repeat: fault not back in {n} launches ({rp['s']} s)")
            finally:
                free(*held)
            return "back" if rp["back"] else "not back"

        # --- split rw: the known failing shape, the C4n-size matmul with x, w and y all in DRAM, back to back on every
        # chip for PINPOINT_RW_S s of load (one burst serves every suspect chip at once); output tiles -> DRAM bank. The small
        # mm split missed 03:00.0 (3/3) and 88:00.0 (2/2); an earlier run hit 03:00.0 in 187/187 checks of this shape, 7 s into load
        def split_rw():
            if left() < 45:
                return "budget"
            from numeric_dram import M as RM
            from numeric_dram import N as RN
            from numeric_dram import bank_lane, rw_burst

            bank_of, lane_of = bank_lane(torch, nb)
            col_of = (torch.arange(RN).view(1, RN).expand(RM, RN) * 8 // RN).reshape(-1)
            n = [0]

            def on_check(got, gold, t_load):
                n[0] += 1
                count("rw_dram", got, gold, bank_of, lane_of, col_of=col_of)

            got = rw_burst(
                ttnn,
                torch,
                st["mesh"],
                on_check,
                float(os.environ.get("PINPOINT_RW_S", "60")),
                max(left() - 20, 30),
                stop=lambda: n[0] >= 3 and done_all(),
                check_s=float(os.environ.get("PINPOINT_RW_CHECK_S", "1.5")),
            )
            log(
                f"rw split done at {time.monotonic() - t0:.0f} s: {got['checks']} checks, {got['load_s']:g} s of load, "
                f"{got['tflops']:g} TFLOP/s per chip"
            )
            return ""

        # --- split core: one shard a core, FPU (multiply) and SFPU (exp); bad elements -> core x,y
        def split_core():
            if not gx or not gy:
                raise RuntimeError("no compute grid size")
            n, W = gx * gy, 1024
            H = n * 32
            mc = ttnn.create_sharded_memory_config(
                shape=(H, W),
                core_grid=ttnn.CoreGrid(y=gy, x=gx),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            gen2 = torch.Generator().manual_seed(1025)
            xa = (torch.rand(1, 1, H, W, generator=gen2) * 2 - 1).to(torch.bfloat16)
            xb = (torch.rand(1, 1, H, W, generator=gen2) * 2 - 1).to(torch.bfloat16)
            ta, tb = up(xa, mc), up(xb, mc)
            core_of = (torch.arange(H).view(H, 1).expand(H, W) // 32).reshape(-1)  # shard index (ROW_MAJOR: x first)

            def per_core(s, bad):
                pc = s.setdefault("per_core", {})
                for sh, v in enumerate(torch.bincount(core_of[bad], minlength=n).tolist()):
                    if v:
                        key = f"{sh % gx},{sh // gx}"
                        pc[key] = pc.get(key, 0) + v
                s["grid"] = [gx, gy]

            gold = {}
            for k in range(a.reps):
                for op, fn in (
                    ("fpu", lambda: ttnn.multiply(ta, tb, memory_config=mc)),
                    ("sfpu", lambda: ttnn.exp(ta, memory_config=mc)),
                ):
                    if left() < 10:
                        return "budget"
                    y = fn()
                    got = per_chip(y, H, W)
                    ttnn.deallocate(y)
                    gold.setdefault(op, majority(got))
                    count(op, got, gold[op], extra=per_core)
            log(f"core split done at {time.monotonic() - t0:.0f} s")
            return ""

        # --- split clock: the DRAM matmul with the failing chips' AICLK capped (FORCE_AICLK 0x33; 0 = unforce)
        def force(devs, mhz):
            import tt_umd

            for n in devs:
                d = tt_umd.TTDevice.create(int(n))
                d.init_tt_device()
                rc = d.arc_msg(0x33, True, [mhz], 2000)
                got = d.arc_msg(0x34, True, [], 2000)
                log(f"dev {n} FORCE_AICLK {mhz}: exit {rc[0]}; now {got[1]} MHz mode {got[2]}")

        def split_clock():
            want = [b for b in tg if pv.wants_clock(res["chips"].get(b))]
            if st["clock_done"] or not want:
                return "not needed"
            if left() < 120:
                return "budget"
            st["clock_done"] = True  # once: it closes the mesh and caps the clock
            devs = [res["chips"][b]["dev"] for b in want if b in res["chips"]]
            for b in want:
                m = res["chips"][b].get("mm_dram") or {}
                res["chips"][b]["clock"] = dict(
                    mhz=CAP_MHZ, checks=0, bad_checks=0, base_checks=m.get("checks", 0), base_bad=m.get("bad_checks", 0)
                )
            heavy_free()  # the mesh closes: its tensors go with it
            close_mesh()
            try:
                force(devs, CAP_MHZ)
                open_mesh()
                mm_run("clock_mm", ttnn.DRAM_MEMORY_CONFIG, a.reps)
                for b in want:
                    s = res["chips"][b].get("clock_mm") or {}
                    res["chips"][b]["clock"].update(checks=s.get("checks", 0), bad_checks=s.get("bad_checks", 0))
            finally:
                if st["mesh"] is not None:
                    close_mesh()
                force(devs, 0)
            return ""

        splits = (
            ("repeat", split_repeat),
            ("dram", split_dram),
            ("mm", split_mm),
            ("heavy", split_heavy),
            ("rw", split_rw),
            ("core", split_core),
            ("clock", split_clock),
        )
        only = [v for v in os.environ.get("PINPOINT_SPLITS", "").split(",") if v]  # e.g. repeat: that split alone
        splits = tuple(sp for sp in splits if not only or sp[0] in only)
        try:
            while not res["stopped"]:  # passes until isolated or the budget is spent
                st["pass_"] += 1
                n0, ran = st["checks"], 0
                for name, fn in splits:
                    if done_all():
                        res["stopped"] = "isolated"
                        break
                    if left() < 20:
                        res["stopped"] = "budget"
                        break
                    if name in res["errors"] or (
                        name in ("repeat", "rw") and st["pass_"] > 1
                    ):  # failed once, or once only (rw: one 60 s burst)
                        continue
                    ts = time.monotonic()
                    ran += 1
                    st["split"] = name
                    try:
                        why = fn()
                    except Exception as e:  # one split's error does not end the others
                        why = "error"
                        res["errors"][name] = f"{type(e).__name__}: {str(e)[:300]}"
                        log(f"split {name} error: {res['errors'][name]}")
                    res["splits"].append(
                        dict(name=name, pass_=st["pass_"], s=round(time.monotonic() - ts, 1), note=why)
                    )
                    save()
                    if st["mesh"] is None:
                        res["stopped"] = f"error: split {name} left no mesh"
                        break
                else:
                    if st["checks"] == n0:
                        # every split ran once already (once-only or failed, e.g. PINPOINT_SPLITS=repeat): done, not stuck
                        res["stopped"] = stop_label(left(), ran, res["errors"])
            res["passes"] = st["pass_"]
        finally:
            if st["mesh"] is not None:
                heavy_free()
                close_mesh()
            if T is not None:
                T.commit()
            save()
        log(
            f"done in {res['s']} s, {st['pass_']} pass(es): splits {[s['name'] for s in res['splits']]}, "
            f"stopped: {res['stopped'] or '-'}"
        )
        return 0
    finally:
        if st["mesh"] is not None:
            close_mesh()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"error: {type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)
