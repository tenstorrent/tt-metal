#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""numeric_dram.py: C4d, the early numeric readback with the known failing DRAM shape.

The C4n heavy matmul with x, w and y all in DRAM (x 2048x8192, w 8192x16384, y 2048x16384, 512 tile columns), back
to back on all 32 chips of the 4x8 mesh (no fabric, no ETH link), with a readback of the last output every --per-check
matmuls. Each chip's output is compared bit for bit with the output most chips agree on (all chips get the same inputs).
One chip was found bad in this shape in 187/187 checks, first hit 7 s into load, where C4n's
own check catches it only 21-22 min into run.sh. run.sh no longer runs it as a stage (C4d dropped: the bus-bar stage caught
the same fault first); the numeric pinpoint's rw split (lib/pinpoint.sh) still uses rw_burst().

  numeric_dram.py --out <numeric.txt> [--burst-s 15] [--wall-s 150] [--per-check 8] [--check-s 1.5]

--out gets one line per mismatched chip, in numeric_report.py's form so pinpoint_verdict.py and fl_report read it:
  NUMERIC FAIL <BDF> (tray N) chip D: C4d x/w/y-in-DRAM matmul k/N checks mismatched, first at T s of load, W words,
                                       banks [...], Inf I NaN M
and one 'C4d numeric:' summary line. Exit 0 (mismatches never fail it), 1 an error. rw_burst() is also the numeric
pinpoint's rw split (numeric_pinpoint.py). rw_check() runs the same burst on an open mesh: G3, run.sh's
readback after bus-bar load chunk 1 (fleet/chunk_load.py --rw-after).
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet"))
import stop_signal as ss  # noqa: E402

M, K, N = 2048, 8192, 16384   # the C4n shape (the pinpoint's 'heavy')
FLOP = 2 * M * K * N


def log(msg):
    print(f"{time.strftime('%F %T')} C4D {msg}", flush=True)


def mesh_chips(mesh, metal_n):
    """mesh index -> dict(bdf, dev, tray, metal): the UMD map from this process's log, else the identity."""
    from numeric_check import umd_map_from_stdout
    from ub0_repro import chips, tray_of
    try:
        metal = [int(v) for v in mesh.get_device_ids()]
    except Exception:
        metal = list(range(metal_n))
    umd, ids = umd_map_from_stdout(), chips()
    out = {}
    for i, m in enumerate(metal):
        dev = umd.get(m, m) if umd else m
        p = ids.get(dev)
        out[i] = dict(bdf=re.sub(r"^[0-9a-f]{4}:", "", p) if p else f"metal{m}", dev=dev, metal=m,
                      tray=tray_of(p) if p else None)
    return out, ("umd_log" if umd else "assumed_identity")


def majority(torch, got):
    reps = []
    for i, t in enumerate(got):
        for rep in reps:
            if torch.equal(rep[0], t):
                rep[1].append(i)
                break
        else:
            reps.append((t, [i]))
    return max(reps, key=lambda x: len(x[1]))[0]


def rw_burst(ttnn, torch, mesh, on_check, burst_s, wall_s, per_check=8, seed=1197000, stop=None, check_s=0.0):
    """Back-to-back x/w/y-in-DRAM matmuls on every chip, in batches of per_check with a device sync after each; after the
    first batch, then once at least check_s s of load passed since the last check, and at the end, on_check(got, gold,
    t_load) with got the per-chip int16 outputs (M x N) and gold the majority. Ends at burst_s s of load (readbacks excluded) or wall_s s of
    wall clock, or when stop() is true. Returns dict(checks, matmuls, load_s, wall_s, tflops) (TFLOP/s per chip)."""
    D = ttnn.DRAM_MEMORY_CONFIG
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s, sc=1.0: (torch.randn(*s, generator=g, dtype=torch.float32) * sc).to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=D,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    xh, wh = up(rn(1, 1, M, K)), up(rn(1, 1, K, N, sc=0.02))
    ttnn.synchronize_device(mesh)
    t0 = time.monotonic()
    paused, nmm, checks, y, last = 0.0, 0, 0, None, None
    try:
        while time.monotonic() - t0 - paused < burst_s and time.monotonic() - t0 < wall_s and not (stop and stop()):
            for _ in range(per_check):
                old, y = y, ttnn.matmul(xh, wh, memory_config=D)
                nmm += 1
                if old is not None:
                    ttnn.deallocate(old)
            ttnn.synchronize_device(mesh)   # the matmuls run async: wait for them so their run time counts as load
            s0 = time.monotonic()
            t_load = s0 - t0 - paused
            if last is not None and t_load - last < check_s and t_load < burst_s:
                continue
            last = t_load
            got = [ttnn.to_torch(d).reshape(M, N).contiguous().view(torch.int16) for d in ttnn.get_device_tensors(y.cpu())]
            checks += 1
            on_check(got, majority(torch, got), t_load)
            paused += time.monotonic() - s0
    finally:
        for t in (xh, wh, y):
            if t is not None:
                ttnn.deallocate(t)
    el = time.monotonic() - t0
    return dict(checks=checks, matmuls=nmm, load_s=round(el - paused, 1), wall_s=round(el, 1),
                tflops=round(nmm * FLOP / max(el - paused, 1e-3) / 1e12, 1))


def bank_lane(torch, nb):
    """Per element of the M x N output: its DRAM bank (interleaved tile index mod banks) and 16-bit lane (column parity)."""
    rr = torch.arange(M).view(M, 1).expand(M, N)
    cc = torch.arange(N).view(1, N).expand(M, N)
    return (((rr // 32) * (N // 32) + cc // 32) % nb).reshape(-1), (cc % 2).reshape(-1)


def rw_check(ttnn, torch, mesh, chips, burst_s, wall_s, per_check=8, check_s=3.0, nb=8, beat=None, say=None):
    """G3: rw_burst() on a mesh another stage already opened (chunk_load.py, after bus-bar chunk
    1). chips: mesh index -> dict(pci, tray, dev). Returns rw_burst()'s dict plus banks and, per mesh index, checks,
    bad_checks, bad (words), first (s of load), first_t (epoch), per_bank, per_lane, inf, nan. beat(s) is called at
    every check (the stage's hang watch); say(line) for each mismatched chip at the check that saw it."""
    bank_of, lane_of = bank_lane(torch, nb)
    st = {i: dict(checks=0, bad_checks=0, bad=0, first=None, first_t=None, per_bank=[0] * nb, per_lane=[0, 0], inf=0,
                  nan=0) for i in chips}

    def on_check(got, gold, t_load):
        e = gold.reshape(-1)
        for i, gt in enumerate(got):
            s = st.setdefault(i, dict(checks=0, bad_checks=0, bad=0, first=None, first_t=None, per_bank=[0] * nb,
                                      per_lane=[0, 0], inf=0, nan=0))
            s["checks"] += 1
            gf = gt.reshape(-1)
            bad = torch.nonzero(gf != e).flatten()
            if not bad.numel():
                continue
            s["bad_checks"] += 1
            s["bad"] += int(bad.numel())
            if s["first"] is None:
                s["first"], s["first_t"] = round(t_load, 1), round(time.time(), 1)
            f = gf[bad].view(torch.bfloat16).float()
            s["inf"] += int(torch.isinf(f).sum())
            s["nan"] += int(torch.isnan(f).sum())
            s["per_bank"] = [u + v for u, v in zip(s["per_bank"], torch.bincount(bank_of[bad], minlength=nb).tolist())]
            s["per_lane"] = [u + v for u, v in zip(s["per_lane"], torch.bincount(lane_of[bad], minlength=2).tolist())]
            if say:
                c = chips.get(i, {})
                say(f"BAD mesh{i} {c.get('pci')} tray {c.get('tray') or '?'}: {int(bad.numel())} words at "
                    f"{t_load:.0f} s of load")
        if beat:
            beat(f"rw check {st[0]['checks'] if 0 in st else 0}")

    got = rw_burst(ttnn, torch, mesh, on_check, burst_s, wall_s, per_check, check_s=check_s)
    return dict(got, banks=nb, chips={str(i): dict(chips.get(i, {}), **s) for i, s in st.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--burst-s", type=float, default=float("15"))
    ap.add_argument("--wall-s", type=float, default=float("150"))
    ap.add_argument("--per-check", type=int, default=int("8"))
    ap.add_argument("--check-s", type=float, default=float("1.5"))
    ap.add_argument("--mesh", default="4x8")
    a = ap.parse_args(argv)
    import torch
    import ttnn
    r, c = (int(v) for v in a.mesh.split("x"))
    t0 = time.monotonic()
    ss.install("C4D")   # TERM/INT: the finally below closes the mesh
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(r, c))
    lines = []
    try:
        ss.arm()
        log(f"mesh open {tuple(mesh.shape)} in {time.monotonic() - t0:.0f} s")
        mc, src = mesh_chips(mesh, r * c)
        nb = 8
        try:
            nb = int(mesh.get_devices()[0].num_dram_channels()) or 8
        except Exception:
            pass
        bank_of, lane_of = bank_lane(torch, nb)
        st = {i: dict(checks=0, bad_checks=0, bad=0, first=None, per_bank=[0] * nb, per_lane=[0, 0], inf=0, nan=0)
              for i in mc}

        def on_check(got, gold, t_load):
            e = gold.reshape(-1)
            for i, gt in enumerate(got):
                s = st[i]
                s["checks"] += 1
                gf = gt.reshape(-1)
                bad = torch.nonzero(gf != e).flatten()
                if not bad.numel():
                    continue
                s["bad_checks"] += 1
                s["bad"] += int(bad.numel())
                s["first"] = s["first"] if s["first"] is not None else round(t_load, 1)
                f = gf[bad].view(torch.bfloat16).float()
                s["inf"] += int(torch.isinf(f).sum())
                s["nan"] += int(torch.isnan(f).sum())
                s["per_bank"] = [u + v for u, v in zip(s["per_bank"], torch.bincount(bank_of[bad], minlength=nb).tolist())]
                s["per_lane"] = [u + v for u, v in zip(s["per_lane"], torch.bincount(lane_of[bad], minlength=2).tolist())]
                log(f"BAD mesh{i} {mc[i]['bdf']} tray {mc[i]['tray']}: {int(bad.numel())} words at {t_load:.0f} s of load")

        log(f"x/w/y-in-DRAM matmul {M}x{K}x{N} on {len(mc)} chips: {a.burst_s:g} s of load, a readback every "
            f"{a.check_s:g} s of load (batches of {a.per_check} matmuls), wall cap {a.wall_s:g} s, banks {nb}, map {src}")
        got = rw_burst(ttnn, torch, mesh, on_check, a.burst_s, a.wall_s, a.per_check, check_s=a.check_s)
        for i, s in sorted(st.items(), key=lambda kv: (mc[kv[0]]["tray"] or 9, mc[kv[0]]["bdf"])):
            if s["bad_checks"]:
                lines.append(f"NUMERIC FAIL {mc[i]['bdf']} (tray {mc[i]['tray'] or '?'}) chip {mc[i]['dev']}: C4d x/w/y-in-DRAM "
                             f"matmul {s['bad_checks']}/{s['checks']} checks mismatched, first at {s['first']:g} s of load, "
                             f"{s['bad']} words, banks {s['per_bank']}, lanes even/odd {s['per_lane'][0]}/{s['per_lane'][1]}, "
                             f"Inf {s['inf']} NaN {s['nan']}")
        nbad = sum(1 for s in st.values() if s["bad_checks"])
        lines.append(f"C4d numeric: {got['checks']} checks on {len(mc)} chips, {got['load_s']:g} s of load in "
                     f"{got['wall_s']:g} s, {got['matmuls']} matmuls, {got['tflops']:g} TFLOP/s per chip; "
                     + (f"{nbad} chip(s) mismatched" if nbad else "every chip matched"))
    finally:
        ttnn.close_mesh_device(mesh)
        with open(a.out + ".tmp", "w") as f:
            f.write("".join(ln + "\n" for ln in lines))
            f.flush()
            os.fsync(f.fileno())
        os.replace(a.out + ".tmp", a.out)
    for ln in lines:
        log(ln)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"error: {type(e).__name__}: {str(e)[:300]}")
        sys.exit(1)
