#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The bug b op loop (run.sh's C6b earlier, now run.sh --repro abort-fabric's all-gather).
The H3 soak hangs (4/4: a quad host Device 24/25, a second quad host, a third quad host) stall the to_out FSDP weight all-gather of a DiT attention block (AllGatherAsync [1,1,896,1344] ->
[1,1,7168,1344], dim 2, the 8-chip SP axis, persistent ping-pong buffer, no barrier) on the end-of-line chip pair. This loops
that op sequence as the h3-worker:0.3.9-cb9c6543 image calls it (attention_minimax_h3.py, layers/linear.py, CCLManager):
  h  = unsqueeze(concatenate_heads(spatial_BHNE [1,14,5856,128]), 0)          NLPConcatHeads -> [1,1,5856,1792]
  a1 = all_gather_async(h, dim 3, TP axis 0)                                  no persistent buffer, barrier semaphore
  w  = all_gather_async(W_l [1,1,896,1344], dim 2, SP axis 1)                persistent ping-pong buffer, NO barrier
  o  = minimal_matmul(a1, reshape(w, [7168,1344]))                            MinimalMatmul -> [1,1,5856,1344]
Mesh 4x8, FABRIC_1D + Topology.Linear (--topo ring: FABRIC_1D_RING on --ring-axes). One launch = --blocks iterations with
no host sync, then sync. Port of an earlier agloop.py (~1.2 M launches on a quad host with no stall, so no
synthetic repro of bug b is on record; its job is to name the stuck chips when this op stalls). Beat/rows protocol of
h3shape_repro.py. After an op timeout whose hook (on_hang.sh) wrote $FLEET_HANG_DIR/HANG it exits with no device
teardown (a hung mesh would hang the close). Exit 0 done, 4 mismatch/non-finite, 3 hang, 1 error."""
import argparse
import hashlib
import json
import os
import sys

sys.path.append(
    os.path.dirname(os.path.realpath(__file__))
)  # the sibling programs, also when run or imported through a link
import time

ap = argparse.ArgumentParser()
ap.add_argument("--rows-n", type=int, default=4)
ap.add_argument("--cols-n", type=int, default=8)
ap.add_argument("--topo", choices=["linear", "ring"], default="linear")
ap.add_argument("--ring-axes", default="0,1", help="with --topo ring: mesh axes whose CCLs use Topology.Ring")
ap.add_argument("--links", type=int, default=2)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--blocks", type=int, default=8)
ap.add_argument("--seq", type=int, default=5856)
ap.add_argument("--heads", type=int, default=14)
ap.add_argument("--head-dim", type=int, default=128)
ap.add_argument("--w-rows", type=int, default=896)
ap.add_argument("--w-cols", type=int, default=1344)
ap.add_argument("--no-mm", action="store_true", help="skip minimal_matmul (CCL only)")
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--iters", type=int, default=10**9)
ap.add_argument("--verify-every", type=int, default=100)
ap.add_argument("--rows", required=True)
ap.add_argument("--beat", required=True)
a = ap.parse_args()
rows = open(a.rows, "a")


def row(**kw):
    kw = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kw}
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

import stop_signal as ss  # noqa: E402

ss.install("AGLOOP")  # TERM/INT: closes the mesh below
beat("open")
R, C = a.rows_n, a.cols_n
TP, SP = 0, 1
ring_axes = {int(x) for x in a.ring_axes.split(",") if x} if a.topo == "ring" else set()
topo = {ax: (ttnn.Topology.Ring if ax in ring_axes else ttnn.Topology.Linear) for ax in (TP, SP)}
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if ring_axes else ttnn.FabricConfig.FABRIC_1D)
m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(R, C))
DRAM = ttnn.DRAM_MEMORY_CONFIG
bad = 0
try:
    ss.arm()
    g = m.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    sem = lambda: ttnn.create_global_semaphore(m, cores, 0)  # noqa: E731
    ag_s = {ax: [sem() for _ in range(4)] for ax in (TP, SP)}  # CCLManager: 2 sems x 2 ping-pong per axis
    bar_s = {ax: [sem() for _ in range(2)] for ax in (TP, SP)}  # 1 barrier x 2 ping-pong per axis
    idx = {"ag": [0, 0], "bar": [0, 0]}

    def nxt(kind, ax, pool, n):
        i = idx[kind][ax]
        idx[kind][ax] = (i + 1) % 2
        return pool[ax][i * n : (i + 1) * n] if n > 1 else pool[ax][i]

    pp = {}  # persistent AG output ping-pong buffers, keyed like CCLManager.get_ag_ping_pong_buffer

    def pp_buf(shape, dim, ax):
        out = list(shape)
        out[dim] *= m.shape[ax]
        key = (tuple(out), dim, ax)
        if key not in pp:
            ttnn.synchronize_device(m)
            pp[key] = [
                [
                    ttnn.allocate_tensor_on_device(ttnn.Shape(out), ttnn.bfloat16, ttnn.TILE_LAYOUT, m, DRAM)
                    for _ in range(2)
                ],
                0,
            ]
            ttnn.synchronize_device(m)
        bufs, i = pp[key]
        pp[key][1] = 1 - i
        return bufs[i]

    def ag(t, dim, ax):  # ccl_manager.all_gather(use_hyperparams=False)
        return ttnn.experimental.all_gather_async(
            t,
            persistent_output_buffer=None,
            barrier_semaphore=nxt("bar", ax, bar_s, 1),
            dim=dim,
            multi_device_global_semaphore=nxt("ag", ax, ag_s, 2),
            num_links=a.links,
            topology=topo[ax],
            cluster_axis=ax,
        )

    def ag_persist(t, dim, ax):  # ccl_manager.all_gather_persistent_buffer (no barrier, no hyperparams)
        return ttnn.experimental.all_gather_async(
            t,
            persistent_output_buffer=pp_buf(t.shape, dim, ax),
            barrier_semaphore=None,
            dim=dim,
            multi_device_global_semaphore=nxt("ag", ax, ag_s, 2),
            num_links=a.links,
            topology=topo[ax],
            cluster_axis=ax,
        )

    torch.manual_seed(0)
    heads = ttnn.from_torch(
        (torch.randn(1, a.heads, a.seq, a.head_dim)).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=m,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(m),
    )
    ws = []
    for _ in range(a.layers):  # each chip its own [1,1,896,1344] shard of the to_out weight
        full = (torch.randn(R, C, a.w_rows, a.w_cols) * 0.02).to(torch.bfloat16)
        ws.append(
            ttnn.from_torch(
                full,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=m,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensor2dMesh(m, mesh_shape=(R, C), dims=(0, 1)),
            )
        )
    ttnn.synchronize_device(m)  # as CCLManager: semaphores zeroed everywhere before the first CCL
    nit = [0]

    def one_iter():
        h = ttnn.transformer.concatenate_heads(heads)
        h4 = ttnn.unsqueeze(h, 0)
        a1 = ag(h4, 3, TP)
        ttnn.deallocate(h4)
        w = ag_persist(ws[nit[0] % a.layers], 2, SP)
        nit[0] += 1
        if a.no_mm:
            return a1, w
        w2 = ttnn.reshape(w, (w.shape[-2], w.shape[-1]))
        o = ttnn.experimental.minimal_matmul(a1, w2)
        ttnn.deallocate(a1)
        return o, None

    def one_launch():
        out = None
        for b in range(a.blocks):
            o, w = one_iter()
            if b < a.blocks - 1:
                ttnn.deallocate(o)
            else:
                out = o
        return out

    def digest(t):
        hs, fin = [], True
        for d in ttnn.get_device_tensors(t):
            v = ttnn.to_torch(d).float()
            fin = fin and bool(torch.isfinite(v).all())
            hs.append(hashlib.md5(v.numpy().tobytes()).hexdigest()[:12])
        return hs, fin

    # reference: one launch from a fixed start (iteration counter 0, so layer k at block k)
    beat("launch 0")
    t0 = time.time()
    out = one_launch()
    ttnn.synchronize_device(m)
    ref, fin = digest(out)
    ttnn.deallocate(out)
    beat("done 0")
    row(
        event="setup",
        mesh=[R, C],
        topo=a.topo,
        ring_axes=sorted(ring_axes),
        links=a.links,
        layers=a.layers,
        blocks=a.blocks,
        seq=a.seq,
        w_shape=[a.w_rows, a.w_cols],
        no_mm=a.no_mm,
        mesh_ids=list(m.get_device_ids()),
        visible=os.environ.get("TT_VISIBLE_DEVICES", ""),
        warm_s=round(time.time() - t0, 1),
        ref_finite=fin,
    )
    bad += not fin
    t_end = time.time() + a.seconds
    i, nblk = 1, a.blocks
    if a.blocks % a.layers:
        row(
            event="note", msg="blocks not a multiple of layers: verify compares launches with the same layer phase only"
        )
    while i < a.iters and time.time() < t_end:
        beat(f"launch {i}")
        t0 = time.time()
        out = one_launch()
        ttnn.synchronize_device(m)
        dt = time.time() - t0
        beat(f"done {i}")
        nblk += a.blocks
        r = {"event": "iter", "i": i, "s": round(dt, 4), "ag_sp": nblk}
        last = i + 1 >= a.iters or time.time() >= t_end
        same_phase = (i * a.blocks) % a.layers == 0
        if same_phase and (i % a.verify_every == 0 or last):
            hs, fin = digest(out)
            nmis = sum(1 for p, q in zip(hs, ref) if p != q)
            ok = fin and nmis == 0
            bad += not ok
            r.update(verify="ok" if ok else "MISMATCH", mismatched_chips=nmis, finite=fin)
            if nmis:
                r["mismatch_idx"] = [k for k, (p, q) in enumerate(zip(hs, ref)) if p != q][:40]
        ttnn.deallocate(out)
        if "verify" in r or i % 50 == 0 or dt > 2.0:
            row(**r)
        i += 1
    row(event="end", iters=i, ag_sp_total=nblk, verify_failures=bad)
except Exception as e:  # a TT_THROW after the op-timeout hook ran lands here (a StopSignal is no Exception)
    hd = os.environ.get("FLEET_HANG_DIR", "")
    if hd and os.path.exists(os.path.join(hd, "HANG")):
        row(
            event="hang_exit",
            error=str(e)[:300],
            msg="op timed out after the hook ran; exiting without device teardown",
        )
        os._exit(3)
    raise
finally:  # every other way out closes once; TERM/INT exits 143/130 after it, also one during the close
    with ss.closing():  # (the normal-path close ran armed, so a TERM there cut it)
        beat("close")
        ttnn.close_mesh_device(m)
        beat("closed")
sys.exit(4 if bad else 0)
