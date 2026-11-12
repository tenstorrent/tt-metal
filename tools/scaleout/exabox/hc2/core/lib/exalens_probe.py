#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Reduced repro: can exalens still read the chip? One NOC per call.
#
# For every local chip (target first), reads: unique id, ARC firmware version, one word of the first
# Tensix's L1 and the base-FW / fabric-router heartbeat of each active ethernet core (twice, 1 s apart,
# to see whether it still counts). Every read is wrapped: a timeout on one core is a result, not a crash.
# Written against the exalens API that tools/triage uses on tt-metal main a869f55; stage 0 runs it on a
# healthy chip to confirm it against the H3 image's exalens.

import argparse
import json
import os
import sys
import time

ETH_HEARTBEAT_BH = 0x7CC70     # check_eth_status.py (Blackhole): heartbeat[0], 0xABCD base FW / 0xDCBA router
ETH_METAL_HEARTBEAT_BH = 0x7CC74
L1_PROBE_ADDR = 0x0


def safe(fn):
    try:
        return {"ok": True, "v": fn()}
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {str(e)[:300]}"}


def tray_of(bdf):   # tray 1-4 from the PCI bus byte, as lib/gddr_telem.py's tray and run.sh's rk_tray
    try:
        bus = int(bdf.split(":")[-2], 16)
    except (ValueError, IndexError):
        return "?"
    return {0x00: "1", 0x40: "2", 0xC0: "3", 0x80: "4"}.get(bus & 0xF0, "?")


def missing_module_warn(e, chips):
    """An exalens init that fails on a missing python module gets one loud WARN (stderr) naming
    the module and the chip(s), so a hang capture without NOC state is never only a JSON init_error. Returns the module."""
    if not isinstance(e, ImportError):
        return None
    mod = getattr(e, "name", None) or "ttexalens"
    msg = (f"WARN: hang capture: python module '{mod}' is missing in {sys.executable}: no exalens NOC state for {chips} "
           f"(the venv lacks tt-exalens: rerun traytest/setup.sh)")
    print(msg, file=sys.stderr, flush=True)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noc", type=int, choices=[0, 1], help="NOC to read through (required without --code; --code: 0)")
    ap.add_argument("--bdf", default="0000:02:00.0", help="target chip PCI BDF")
    ap.add_argument("--unique-id", default="10714525003553276259", help="target chip unique id (decimal)")
    ap.add_argument("--all", action="store_true", help="probe every chip, not only the target")
    ap.add_argument("--out", help="JSON result file (required without --code)")
    # N3 code readback (goal 54 stage 2): lib/code_readback.py over every chip's instruction memory; reads only
    ap.add_argument("--code", action="store_true", help="instruction-memory readback of every chip (lib/code_readback.py)")
    ap.add_argument("--budget-s", type=float, default=5.0, help="--code: seconds of reads before it stops at a core boundary")
    ap.add_argument("--cursor-file", help="--code: next core is kept here, so budgeted runs rotate over the box")
    ap.add_argument("--scope", choices=["plain", "full", "smoke"], default="plain")
    ap.add_argument("--elf-dir", help="--code: <RISC>.elf files to compare executable PT_LOAD segments with")
    ap.add_argument("--umd-log", help="--code: a log with UMD's chip open lines, for BDFs when exalens gives none")
    ap.add_argument("--region", action="append", default=[], help="--code: RISC:name:base:size (replaces the default table)")
    ap.add_argument("--read-timeout-s", type=float, default=2.0)
    a = ap.parse_args()
    if a.code:
        return code_main(a)
    if a.noc is None or not a.out:
        ap.error("--noc and --out are required without --code")

    res = {"noc": a.noc, "t": time.time(), "target_bdf": a.bdf, "chips": []}
    try:
        from ttexalens.context import to_noc_id
        from ttexalens.tt_exalens_init import init_ttexalens
        from ttexalens.tt_exalens_lib import read_word_from_device
        ctx = init_ttexalens(noc_id=to_noc_id(a.noc))
    except Exception as e:
        res["init_error"] = f"{type(e).__name__}: {str(e)[:1000]}"
        mod = missing_module_warn(e, f"{a.bdf} tray {tray_of(a.bdf)}")
        if mod:
            res["missing_module"] = mod
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps(res))
        return 3

    bdfs = safe(lambda: {int(k): str(v) for k, v in ctx.cluster_descriptor.get_chip_pci_bdfs().items()})
    res["bdfs"] = bdfs
    bdf_of = bdfs["v"] if bdfs["ok"] else {}
    devices = list(ctx.devices.values())

    def is_target(dev):
        return bdf_of.get(int(dev.id), "").lower().endswith(a.bdf.lower().split(":", 1)[-1]) or \
            str(getattr(dev, "unique_id", "")) == a.unique_id

    devices.sort(key=lambda d: (not is_target(d), int(d.id)))
    for dev in devices:
        tgt = is_target(dev)
        if not tgt and not a.all:
            continue
        c = {"id": int(dev.id), "bdf": bdf_of.get(int(dev.id)), "target": tgt}
        c["unique_id"] = safe(lambda: str(dev.unique_id))
        c["arc_fw"] = safe(lambda: str(dev.firmware_version))
        t0 = time.time()
        tensix = safe(lambda: dev.get_block_locations("functional_workers"))
        if tensix["ok"] and tensix["v"]:
            loc = tensix["v"][0]
            c["tensix_loc"] = str(loc)
            c["tensix_l1_word0"] = safe(lambda: hex(read_word_from_device(loc, L1_PROBE_ADDR, context=ctx)))
        else:
            c["tensix_locs"] = tensix if not tensix["ok"] else "none"
        eth = safe(lambda: dev.get_block_locations("active_eth"))
        if not eth["ok"]:
            # "active_eth" is not a block name on every tt-metal exalens build (KeyError):
            # fall back to "eth", then the ETH location source's method, and name what could not be read.
            eth2 = safe(lambda: dev.get_block_locations("eth"))
            if not eth2["ok"] and hasattr(dev, "get_eth_locations"):
                eth2 = safe(lambda: list(dev.get_eth_locations()))
            if eth2["ok"]:
                eth = {"ok": True, "v": eth2["v"], "source": "eth fallback for active_eth (KeyError/missing)"}
            else:
                eth = {"ok": False, "err": (eth["err"] or "") + "; eth fallbacks failed: " + (eth2["err"] or "no ETH location source")}
        c["eth"] = []
        if eth["ok"]:
            for loc in eth["v"]:
                e = {"loc": str(loc)}
                e["hb0"] = safe(lambda: hex(read_word_from_device(loc, ETH_HEARTBEAT_BH, context=ctx)))
                e["metal_hb0"] = safe(lambda: hex(read_word_from_device(loc, ETH_METAL_HEARTBEAT_BH, context=ctx)))
                c["eth"].append(e)
            time.sleep(1.0)
            for e, loc in zip(c["eth"], eth["v"]):
                e["hb1"] = safe(lambda: hex(read_word_from_device(loc, ETH_HEARTBEAT_BH, context=ctx)))
                e["counting"] = e["hb0"]["ok"] and e["hb1"]["ok"] and e["hb0"]["v"] != e["hb1"]["v"]
        else:
            c["eth_locs"] = eth
        c["probe_s"] = round(time.time() - t0, 2)
        reads = [c["unique_id"], c["arc_fw"], c.get("tensix_l1_word0", {"ok": False})] + \
            [e[k] for e in c["eth"] for k in ("hb0", "hb1") if k in e]
        c["reads_ok"] = sum(r["ok"] for r in reads)
        c["reads_total"] = len(reads)
        c["eth_counting"] = sum(1 for e in c["eth"] if e.get("counting"))
        res["chips"].append(c)
        print(f"noc{a.noc} chip {c['id']} bdf={c['bdf']} target={tgt} reads {c['reads_ok']}/{c['reads_total']} "
              f"eth counting {c['eth_counting']}/{len(c['eth'])}", flush=True)
    if not any(c["target"] for c in res["chips"]):
        res["target_missing"] = True
        print(f"noc{a.noc}: target {a.bdf} not among exalens devices", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    return 0


def code_main(a):
    """--code: CODE FAIL/WARN/INFO/SUMMARY lines on stdout; rc 0 once the pass ran (FAIL lines carry the verdict), 3 when
    exalens could not start (no FAIL: the caller reports the part as not run)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import code_readback as cr
    try:
        regions = [cr.parse_region(r) for r in a.region] or None
    except ValueError as e:
        print(f"CODE ERROR bad --region: {e}", flush=True)
        return 2
    try:
        from ttexalens.context import to_noc_id
        from ttexalens.tt_exalens_init import init_ttexalens
        from ttexalens import tt_exalens_lib as lib
        ctx = init_ttexalens(noc_id=to_noc_id(a.noc or 0))
    except Exception as e:
        missing_module_warn(e, "every chip (instruction-memory readback)")
        print(f"CODE ERROR exalens init: {type(e).__name__}: {str(e)[:300]}", flush=True)
        return 3
    bdfs = safe(lambda: {int(k): str(v) for k, v in ctx.cluster_descriptor.get_chip_pci_bdfs().items()})
    bdf_of = dict(bdfs["v"]) if bdfs["ok"] else {}
    devices = sorted(ctx.devices.values(), key=lambda d: int(d.id))
    # a chip the cluster descriptor gives no BDF for: the UMD open line of --umd-log, then of this probe's own stdout file
    # (run.log: exalens's UMD prints it there), mapped through sysfs; still none: its lines read 'chip<id> tray ?'
    sys_root = os.environ.get("UB0_SYS_ROOT", "/sys")
    for get in ((lambda: cr.bdf_from_umd(open(a.umd_log, errors="replace").read(), sys_root)) if a.umd_log else None,
                lambda: cr.bdf_from_stdout(sys_root)):
        if get is None or all(bdf_of.get(int(d.id)) for d in devices):
            continue
        try:
            got = get()
        except OSError:
            got = {}
        for k, v in got.items():
            if not bdf_of.get(int(k)):
                bdf_of[int(k)] = v
    res = cr.run(ctx, devices, bdf_of, budget_s=a.budget_s, cursor_file=a.cursor_file, scope=a.scope,
                 elf_dir=a.elf_dir, regions=regions, read_timeout_s=a.read_timeout_s,
                 read_word=lib.read_word_from_device, read_block=getattr(lib, "read_from_device", None),
                 out=lambda m: print(m, flush=True), fw_of=cr.fw_from_sysfs(bdf_of, sys_root))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
