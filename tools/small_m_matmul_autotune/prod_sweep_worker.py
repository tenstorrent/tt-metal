#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""One shape, fully-default production run: absolute PCC + N measurement blocks of kernel wall time.

DEFAULTS ONLY: config=None (the picker chooses), no diagnostic mask, no env override that changes behaviour.
TT_SMALL_M_LOG_CFG may be set by the driver -- it only makes the factory log which config / reduction strategy
/ placement it picked, and does not alter the program.

Kernel wall comes from the device-profiler CSV demuxed by run-host-id, so it is device time for the op alone,
not host wall. Each block is 2 warmup + TIMED timed iterations on resident inputs; multiple blocks in one
process give an iteration-level stability estimate.

argv: M K N [nblocks] [config]   config = "auto" (picker, default) or "Pk,Ns,Sm,kb,nsb"
Emits one line: SWEEP_JSON {...}
"""
import csv
import json
import os
import statistics
import sys

import torch
import ttnn

ROOT = os.environ.get("TT_METAL_HOME", os.getcwd())
CSV_PATH = f"{ROOT}/generated/profiler/.logs/profile_log_device.csv"
FREQ = 1.35e9
WARMUP, TIMED = 2, 12

M, K, N = (int(x) for x in sys.argv[1:4])
NBLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
CFG = sys.argv[5] if len(sys.argv) > 5 else "auto"


def parse_runids():
    if not os.path.exists(CSV_PATH):
        return {}
    raw = {}
    for row in csv.reader(open(CSV_PATH)):
        if len(row) < 12 or not row[10].strip().endswith("-KERNEL"):
            continue
        raw.setdefault(row[7].strip(), {}).setdefault((row[1], row[2], row[3]), []).append(
            (row[11].strip(), int(row[5]))
        )
    out = {}
    for runid, cr in raw.items():
        durs, start = {}, None
        for kk, lst in cr.items():
            st = None
            for t, c in lst:
                if t == "ZONE_START":
                    st = c
                    start = c if start is None else min(start, c)
                elif t == "ZONE_END" and st is not None:
                    durs[kk] = c - st
                    st = None
        if durs:
            out[runid] = {"wall": max(durs.values()) / FREQ * 1e6, "start": start or 0}
    return out


def pcc(a, b):
    # float64: in fp32 the dot/norm accumulation over millions of elements rounds enough to return values
    # slightly ABOVE 1.0 (seen: 1.000036), which reads as a broken metric in a report.
    x, y = a.flatten().double(), b.flatten().double()
    x = x - x.mean()
    y = y - y.mean()
    d = (x.norm() * y.norm()).item()
    return 1.0 if d == 0 else torch.dot(x, y).item() / d


def main():
    try:
        os.remove(CSV_PATH)
    except OSError:
        pass
    res = {"M": M, "K": K, "N": N, "outcome": "runtime", "err": "", "cfg_req": CFG}
    labels = []
    dev = ttnn.open_device(device_id=0)
    # Report the BOARD that actually ran this measurement. Perf goldens are board-specific -- the compute grid
    # differs between harvest configurations (an 11x10 dev part vs a 12x10 Galaxy chip), and a golden from one
    # board is not a threshold on another. Reporting it here, from the device that ran the op, means the
    # consumer cannot compare against the wrong board's numbers.
    _g = dev.compute_with_storage_grid_size()
    res["board"] = f"{_g.x}x{_g.y}"
    res["board_cores"] = _g.x * _g.y
    ran = False
    try:
        t0 = torch.randn(1, 1, M, K)
        t1 = torch.randn(1, 1, K, N)
        ref = (t0.bfloat16().float() @ t1.bfloat16().float())[0, 0]
        a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wc = ttnn.create_small_m_weight_memory_config(list(t1.shape), ttnn.bfloat16, dev)
        a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=wc)

        cfg = None
        if CFG != "auto":
            pk, ns, sm, kbt, nst = (int(x) for x in CFG.split(","))
            cfg = ttnn.SmallMMatmulConfig(
                k_slices=pk, n_slices=ns, m_slices=sm, k_block_tiles=kbt, n_subblock_tiles=nst
            )

        def call(label):
            out = ttnn.experimental.small_m_matmul(a0, a1, config=cfg)  # None => production picker
            labels.append(label)
            return out

        o = call("pcc")
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(ttnn.from_device(o))[0, 0]
        res["pcc"] = round(pcc(ref, got), 6)
        # EXPLICIT finite check, separate from PCC. A handful of NaN/Inf among millions of elements barely
        # moves PCC but is a hard correctness failure -- a reduce-scatter CB wrap bug was exactly that.
        # Count them rather than just asserting, so a regression says how bad it is.
        finite = torch.isfinite(got)
        res["n_nonfinite"] = int((~finite).sum())
        res["finite"] = bool(res["n_nonfinite"] == 0)
        res["out_absmax"] = round(float(got[finite].abs().max()) if finite.any() else float("nan"), 4)

        for b in range(NBLOCKS):
            for _ in range(WARMUP):
                call(f"b{b}_w")
            for _ in range(TIMED):
                call(f"b{b}_t")
            ttnn.synchronize_device(dev)
        ran = True
    except Exception as e:  # noqa: BLE001
        res["err"] = str(e)[:300]
    finally:
        ttnn.close_device(dev)  # flushes the profiler CSV

    if not ran:
        print("SWEEP_JSON " + json.dumps(res), flush=True)
        return
    rid = parse_runids()
    order = sorted(rid, key=lambda r: rid[r]["start"])
    if len(order) != len(labels):
        res["err"] = "demux misalign {} vs {}".format(len(order), len(labels))
        print("SWEEP_JSON " + json.dumps(res), flush=True)
        return
    blocks = []
    for b in range(NBLOCKS):
        w = [rid[i]["wall"] for i, lab in zip(order, labels) if lab == "b{}_t".format(b)]
        if w:
            blocks.append([round(x, 3) for x in w])
    allw = [x for b in blocks for x in b]
    if not allw:
        res["err"] = "no timed iterations recovered"
        print("SWEEP_JSON " + json.dumps(res), flush=True)
        return
    res["blocks"] = blocks
    res["block_medians"] = [round(statistics.median(b), 3) for b in blocks]
    res["median_us"] = round(statistics.median(allw), 3)
    res["min_us"] = round(min(allw), 3)
    res["max_us"] = round(max(allw), 3)
    res["n_iters"] = len(allw)
    res["outcome"] = "ok"
    print("SWEEP_JSON " + json.dumps(res), flush=True)


main()
