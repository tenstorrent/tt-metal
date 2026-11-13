#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""A loopable fabric-in-the-loop reduced repro for one chip's ethernet hang. Run with TT_VISIBLE_DEVICES set to
the chips under test only (the mesh opens over exactly those chips).
--mode p2p      1x2 mesh, FABRIC_1D: ttnn.point_to_point of a [1,1,8192,16384] bf16 DRAM tensor 0->1 then 1->0 per launch
--mode ag       1xN mesh (FABRIC_1D for N=2, FABRIC_1D_RING for --ring): ttnn.all_gather(dim=2, cluster_axis=1) of a
                [1,1,16384,16384] bf16 DRAM tensor per chip (the op that hangs in R1h/tray-test: 512 MiB in per chip)
--mode ethwrite one chip, fabric off: eth_write.cpp on all 120 Tensix cores writes 2 KB pages over the NOC into the
                chip's own 12 ETH cores' L1 (window 0x40000-0x60000), posted on VC+1 like unicast_writer.cpp
--hw H,W        p2p/ag: the per-chip tensor instead (run.sh --repro link-a: 2560,4096, an odd per-worker page count)
Link data: each chip sends its own host-built bit pattern (link_pattern.py: walking ones and
zeros, seeded by mesh index). Launch 0 and the last launch are compared bit for bit with it on the host, every slab on
every chip; each other launch is compared on the device with launch 0's checked output, and read back bit for bit only
when that differs. A mismatch writes a 'linkdata' row: sender and receiver mesh index, words wrong, the flipped bits
and their place in a 32 B chunk.
Runs --iters launches or until --seconds. One JSON row per launch (fsync) in --rows; --beat holds 'launch i' / 'done i'
for an outside watchdog (this process never returns while a chip is hung). Exit 0 done, 4 data mismatch, 1 setup error."""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["p2p", "ag", "ethwrite"], required=True)
ap.add_argument("--n", type=int, default=2, help="ag: mesh width")
ap.add_argument("--ring", action="store_true", help="ag: FABRIC_1D_RING (ring all-gather)")
ap.add_argument("--iters", type=int, default=100000)
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--noc", choices=["0", "1", "both"], default="0", help="ethwrite")
ap.add_argument("--posted", type=int, default=1, choices=[0, 1], help="ethwrite")
ap.add_argument("--writes", type=int, default=16384, help="ethwrite: 2 KB writes per writer per launch")
ap.add_argument(
    "--hw", default=None, help="p2p/ag: per-chip tensor H,W (bf16, multiples of 32; run.sh --repro link-a: 2560,4096)"
)
ap.add_argument("--out-mem", choices=["dram", "l1"], default="dram", help="ag: all-gather output memory")
ap.add_argument(
    "--planes",
    type=int,
    default=None,
    help="p2p/ag: set_fabric_config num_planes (routing planes = ETH channels per direction)",
)
ap.add_argument("--rows", required=True)
ap.add_argument("--beat", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
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

sys.path.insert(0, HERE)
import link_pattern as lp  # noqa: E402
import stop_signal as ss  # noqa: E402

GiB = 2**30
vis = os.environ.get("TT_VISIBLE_DEVICES", "")
ss.install("FABRIC")  # TERM/INT: the finally blocks below close the device
beat("open")
bad = 0
t_end = time.time() + a.seconds

if a.mode == "ethwrite":
    dev = ttnn.open_device(device_id=0)
    try:
        ss.arm()
        g = dev.compute_with_storage_grid_size()
        cores = [(x, y) for x in range(g.x) for y in range(g.y)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
        dummy = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        PAGE, NETH, X0, Y, BASE, SLOTS = 2048, 12, 20, 25, 0x40000, 64  # 12 unharvested ETH: translated (20..31, 25)
        procs = {
            "0": [(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)],
            "1": [(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)],
        }
        procs["both"] = procs["0"] + procs["1"]
        plist = procs[a.noc]
        nwr = len(cores) * len(plist)
        src = open(os.path.join(HERE, "eth_write.cpp")).read()
        cb = ttnn.CBDescriptor(
            total_size=PAGE,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=PAGE)],
        )
        nbytes = nwr * a.writes * PAGE
        row(
            event="setup",
            mode="ethwrite",
            visible=vis,
            writers=nwr,
            noc=a.noc,
            posted=a.posted,
            writes=a.writes,
            eth=[[X0 + i, Y] for i in range(NETH)],
            window=[hex(BASE), hex(BASE + SLOTS * PAGE)],
            bytes_per_launch=nbytes,
        )

        def program(fill):
            ks = []
            for pi, (proc, noc) in enumerate(plist):
                rt = ttnn.RuntimeArgs()
                for ci, (x, y) in enumerate(cores):
                    k = ci * len(plist) + pi
                    rt[x][y] = [k % NETH, BASE + (k // NETH % SLOTS) * PAGE, a.writes, fill]
                ks.append(
                    ttnn.KernelDescriptor(
                        kernel_source=src,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=crs,
                        compile_time_args=[PAGE, 8, 0, NETH, X0, Y],
                        defines=[("T76_POSTED", str(a.posted))],
                        runtime_args=rt,
                        config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc),
                    )
                )
            return ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=[cb])

        i = 0
        while i < a.iters and (i == 0 or time.time() < t_end):  # at least one launch (setup can outlast --seconds)
            beat(f"launch {i}")
            t0 = time.time()
            ttnn.generic_op([dummy, dummy], program(0x3F803F80 + i % 64))
            ttnn.synchronize_device(dev)
            dt = time.time() - t0
            beat(f"done {i}")
            row(event="iter", i=i, s=round(dt, 4), GBps=round(nbytes / dt / 1e9, 1))
            i += 1
        last_fill = 0x3F803F80 + (i - 1) % 64
    finally:
        with ss.closing():  # a TERM during the close waits for it
            beat("close")
            ttnn.close_device(dev)
            beat("closed")
    # read back the written slots over PCIe/NOC0 (tt_umd) after close: did the writes land in ETH L1?
    import tt_umd

    NOC0 = [(1, 1), (16, 1), (2, 1), (15, 1), (3, 1), (4, 1), (13, 1), (12, 1), (6, 1), (11, 1), (7, 1), (10, 1)]
    td = tt_umd.TTDevice.create(int(vis.split(",")[0]))
    td.init_tt_device()
    used = sorted({(k % NETH, k // NETH % SLOTS) for k in range(nwr)})
    got = {}
    for coords, name in ((NOC0, "noc0"), ([(X0 + e, Y) for e in range(NETH)], "translated")):
        miss = 0
        for e, sl in used:
            b = bytearray(4)
            td.noc_read(0, coords[e][0], coords[e][1], BASE + sl * PAGE, b)
            miss += int.from_bytes(b, "little") != last_fill
        got[name] = miss
    bad = min(got.values()) > 0
    row(
        event="end",
        iters=i,
        slots_checked=len(used),
        last_fill=hex(last_fill),
        readback_mismatches=got,
        verify_failures=int(bad),
    )
    sys.exit(4 if bad else 0)

N = 2 if a.mode == "p2p" else a.n
fab = "FABRIC_1D_RING" if (a.mode == "ag" and a.ring) else "FABRIC_1D"
if a.planes:
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fab), num_planes=a.planes)
else:
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fab))
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
    H, W = (8192, 16384) if a.mode == "p2p" else (16384, 16384)
    if a.hw:
        H, W = (int(v) for v in a.hw.split(","))
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

    def dev_flags(ys, refs):
        """per receiver: did any element differ from the checked launch-0 output (ne, then max, on the device)."""
        out = []
        for y, rf in zip(ys, refs):
            f = ttnn.ne(y, rf, memory_config=DRAM)
            mx = ttnn.max(f)
            ttnn.deallocate(f)
            out.append([float(ttnn.to_torch(t).float().max()) > 0 for t in ttnn.get_device_tensors(mx)])
            ttnn.deallocate(mx)
        return out

    devcmp, refs = True, None
    per = H * W * 2
    row(
        event="setup",
        mode=a.mode,
        fabric=fab,
        planes=a.planes,
        out_mem=a.out_mem,
        two_erisc_off=os.environ.get("TT_METAL_DISABLE_FABRIC_TWO_ERISC"),
        visible=vis,
        mesh_ids=ids,
        devnodes_in_mesh_order=phys,
        shape=[1, 1, H, W],
        MiB_in_per_chip=per // 2**20,
    )
    c0, c1 = ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(0, 1)
    i = 0
    while i < a.iters and (i == 0 or time.time() < t_end):  # at least one launch (setup can outlast --seconds)
        beat(f"launch {i}")
        t0 = time.time()
        if a.mode == "p2p":
            y1 = ttnn.point_to_point(x, sender_coord=c0, receiver_coord=c1, topology=ttnn.Topology.Linear)
            y2 = ttnn.point_to_point(x, sender_coord=c1, receiver_coord=c0, topology=ttnn.Topology.Linear)
            ttnn.synchronize_device(m)
            moved = 2 * per
        else:
            y = ttnn.all_gather(
                x,
                dim=2,
                cluster_axis=1,
                memory_config=(ttnn.L1_MEMORY_CONFIG if a.out_mem == "l1" else ttnn.DRAM_MEMORY_CONFIG),
            )
            ttnn.synchronize_device(m)
            moved = (N - 1) * per * N
        dt = time.time() - t0
        beat(f"done {i}")
        r = {"event": "iter", "i": i, "s": round(dt, 4), "GBps_total": round(moved / dt / 1e9, 1)}
        last = i + 1 >= a.iters or time.time() >= t_end
        ys = [y1, y2] if a.mode == "p2p" else [y]
        if a.mode == "p2p":  # receiver 1 holds sender 0's shard, receiver 0 sender 1's
            outs = [(1, ttnn.get_device_tensors(y1)[1], [0]), (0, ttnn.get_device_tensors(y2)[0], [1])]
        else:  # every chip's output: slab k is chip k's pattern
            outs = [(j, t, list(range(N))) for j, t in enumerate(ttnn.get_device_tensors(y))]
        mism = None
        if i == 0 or last:
            mism = host_check(i, outs)
        elif devcmp:
            try:
                fl = dev_flags(ys, refs)
                want = [o for o in outs if any(f[o[0]] for f in fl)]
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
            refs = [ttnn.to_memory_config(t, DRAM) if a.out_mem == "l1" else t for t in ys]
            if a.out_mem == "l1":
                for t in ys:
                    ttnn.deallocate(t)
            r.update(devcmp=devcmp)
            row(**r)
            i += 1
            continue
        if i == 0:
            devcmp = False  # launch 0 wrong: no reference; launch 0 and the last are still host-checked
        if a.mode == "p2p":
            ttnn.deallocate(y1)
            ttnn.deallocate(y2)
        else:
            ttnn.deallocate(y)
        row(**r)
        i += 1
    for t in refs or []:
        ttnn.deallocate(t)
    row(event="end", iters=i, verify_failures=bad, devcmp=devcmp)
finally:
    with ss.closing():  # a TERM during the close waits for it
        beat("close")
        ttnn.close_mesh_device(m)
        beat("closed")
sys.exit(4 if bad else 0)
