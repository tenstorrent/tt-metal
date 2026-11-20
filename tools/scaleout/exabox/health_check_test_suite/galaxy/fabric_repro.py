#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A loopable fabric-in-the-loop reduced repro for one chip's ethernet hang. Run with TT_VISIBLE_DEVICES set to
the chips under test only (the mesh opens over exactly those chips).
1xN mesh, FABRIC_1D: ttnn.all_gather(dim=2, cluster_axis=1) of a [1,1,16384,16384] bf16 DRAM tensor per chip
(512 MiB in per chip).
Link data: each chip sends its own host-built bit pattern (link_pattern.py: walking ones and
zeros, seeded by mesh index). Launch 0 and the last launch are compared bit for bit with it on the host, every slab on
every chip; each other launch is compared on the device with launch 0's checked output, and read back bit for bit only
when that differs. A mismatch writes a 'linkdata' row: sender and receiver mesh index, words wrong, the flipped bits
and their place in a 32 B chunk.
Runs launches until --seconds. One JSON row per launch (fsync) in --rows; --beat holds 'launch i' / 'done i'
for an outside watchdog (this process never returns while a chip is hung). Exit 0 done, 4 data mismatch, 1 setup error."""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2, help="mesh width")
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--rows", required=True)
ap.add_argument("--beat", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()
rows = open(a.rows, "a")


def row(**kw):
    kw = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label": a.label, **kw}
    rows.write(json.dumps(kw) + "\n")
    rows.flush()
    os.fsync(rows.fileno())
    print(json.dumps(kw), flush=True)


def beat(s):
    with open(a.beat + ".tmp", "w") as f:
        f.write(f"{s} {time.time():.0f}\n")
    os.replace(a.beat + ".tmp", a.beat)


import torch  # noqa: E402
import ttnn  # noqa: E402

# the sibling programs, also when run or imported through a link
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import link_pattern as lp  # noqa: E402
import stop_signal as ss  # noqa: E402

GiB = 2**30
vis = os.environ.get("TT_VISIBLE_DEVICES", "")
ss.install("FABRIC")  # TERM/INT: the finally blocks below close the device
beat("open")
bad = 0
t_end = time.time() + a.seconds

N = a.n
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, N))
try:
    ss.arm()
    ids = list(m.get_device_ids())
    # mesh id -> UMD index (TT_VISIBLE_DEVICES carries UMD indices). tt-umd numbers the chips it opens
    # in its own PCI-address order, which is UMD index order, not the argument order: a c4:00.0 + 84:00.0 pair opened
    # as "27,19" is mesh id 0 = UMD 19 (84:00.0), logged as 'Opening local chip ids/PCIe ids: {0, 1}/[27, 19]'
    vlist = sorted(
        (v.strip() for v in vis.split(",") if v.strip()), key=lambda v: (0, int(v)) if v.isdigit() else (1, 0)
    )
    phys = [vlist[i] if i < len(vlist) else str(i) for i in ids]
    H, W = 16384, 16384
    SEED = 4802
    exp = [lp.pattern(SEED, k, H, W) for k in range(N)]  # chip k's host-built pattern
    x = ttnn.from_torch(
        torch.cat([lp.to_torch(e).reshape(1, 1, H, W) for e in exp], dim=1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=m,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(m, mesh_shape=(1, N), dims=(0, 1)),
    )
    DRAM = ttnn.DRAM_MEMORY_CONFIG

    def got(t):  # one chip's output, read back as uint16 rows
        return lp.from_torch(ttnn.to_torch(t))

    def host_check(i, outs):
        """outs: [(receiver, tensor, [senders, one per H-row slab])]. Bitwise vs the host-built patterns."""
        n = 0
        for recv, t, sends in outs:
            g = got(t)
            for k, snd in enumerate(sends):
                d = lp.diff(g[k * H : (k + 1) * H], exp[snd])
                if d:
                    n += d["words"]
                    row(
                        event="linkdata",
                        i=i,
                        send=snd,
                        recv=recv,
                        send_node=phys[snd],
                        recv_node=phys[recv],
                        link=snd != recv,
                        detail=lp.fmt(d),
                        **d,
                    )
            del g
        return n

    def dev_flags(y, ref):
        """per receiver: did any element differ from the checked launch-0 output (ne, then max, on the device)."""
        f = ttnn.ne(y, ref, memory_config=DRAM)
        mx = ttnn.max(f)
        ttnn.deallocate(f)
        out = [float(ttnn.to_torch(t).float().max()) > 0 for t in ttnn.get_device_tensors(mx)]
        ttnn.deallocate(mx)
        return out

    devcmp, ref = True, None
    per = H * W * 2
    row(
        event="setup",
        fabric="FABRIC_1D",
        two_erisc_off=os.environ.get("TT_METAL_DISABLE_FABRIC_TWO_ERISC"),
        visible=vis,
        mesh_ids=ids,
        devnodes_in_mesh_order=phys,
        shape=[1, 1, H, W],
        MiB_in_per_chip=per // 2**20,
    )
    i = 0
    while i == 0 or time.time() < t_end:  # at least one launch (setup can outlast --seconds)
        beat(f"launch {i}")
        t0 = time.time()
        y = ttnn.all_gather(x, dim=2, cluster_axis=1, memory_config=DRAM)
        ttnn.synchronize_device(m)
        moved = (N - 1) * per * N
        dt = time.time() - t0
        beat(f"done {i}")
        r = {"event": "iter", "i": i, "s": round(dt, 4), "GBps_total": round(moved / dt / 1e9, 1)}
        last = time.time() >= t_end
        outs = [(j, t, list(range(N))) for j, t in enumerate(ttnn.get_device_tensors(y))]  # slab k is chip k's pattern
        mism = None
        if i == 0 or last:
            mism = host_check(i, outs)
        elif devcmp:
            try:
                fl = dev_flags(y, ref)
                want = [o for o in outs if fl[o[0]]]
                mism = host_check(i, want) if want else 0
                if want and not mism:  # the device saw a difference the readback no longer holds
                    row(event="linkdata_transient", i=i, recv=[o[0] for o in want])
                    mism = 1
            except Exception as e:  # noqa: BLE001
                devcmp = False
                row(event="devcmp_off", i=i, error=f"{type(e).__name__}: {str(e)[:300]}")
        if mism is not None:
            r.update(verify="ok" if mism == 0 else "MISMATCH", mismatches=mism)
            bad += mism > 0
        if i == 0 and mism == 0 and not last:  # launch 0 is the host-checked reference for the device compare
            ref = y
            r.update(devcmp=devcmp)
            row(**r)
            i += 1
            continue
        if i == 0:
            devcmp = False  # launch 0 wrong: no reference; launch 0 and the last are still host-checked
        ttnn.deallocate(y)
        row(**r)
        i += 1
    if ref is not None:
        ttnn.deallocate(ref)
    row(event="end", iters=i, verify_failures=bad, devcmp=devcmp)
finally:
    with ss.closing():  # a TERM during the close waits for it
        beat("close")
        ttnn.close_mesh_device(m)
        beat("closed")
sys.exit(4 if bad else 0)
