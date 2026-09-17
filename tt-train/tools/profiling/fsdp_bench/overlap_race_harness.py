# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The FSDP overlap schedule in isolation, checked bitwise. Run it before changing the schedule.

Moving parts of ``ttml.fsdp`` overlap mode, and nothing else:

  * K "blocks", each owning a shard of a ``[rows, 2048]`` bf16 weight (different data on every device).
  * Persistent full-shape slot buffers; block k gathers into its slot on hardware queue 1 (the CCL
    sub-device), issued one block ahead (block k's compute issues block k+1's gather), like pre_forward.
  * Block k's compute on queue 0: R matmuls reading the gathered weight from its slot.
  * At the end of a "step" every shard is rewritten in place on queue 0 (the optimizer).
  * The dependency policy between the two queues is pluggable:
      none      only the CCL->compute barrier before a block consumes its gather
      slot      + the gather waits for this device's release of the slot's last reader. Zero
                cross-device slack: a collective writes into every device's copy of the slot and a
                lagging peer may still be reading. The exploration branch's harness caught partial
                corruption with this policy on a one-column sub-device (a different subset of devices
                each step); with a one-row sub-device the window is small enough that it passes, so
                treat ``none`` as the negative control and ``slot`` as the "not proven safe" one
      schedule  + ``ttml.fsdp.SlotSchedule`` (the shipped design): 2 * lookahead slots, wait for the
                release ``lookahead`` positions after the slot's last one
      drain     + a compute->CCL drain before every gather (the conservative bound, lookahead 1)

Each block's output checksum is compared with a synchronous reference and a mismatch is classified
by which candidate it equals: the slot's previous occupant (stale read), the slot's next occupant
(early overwrite), or the block's pre-update weight (gather read the shard too early).

    python overlap_race_harness.py --deps none --steps 20        # expect corruption (negative control)
    python overlap_race_harness.py --deps schedule --steps 20    # expect 0 mismatches
"""

import argparse
import time

import numpy as np
import ttnn
import ttml
from ttml.fsdp import SlotSchedule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=8)
    ap.add_argument("--deps", choices=["none", "slot", "schedule", "drain"], default="schedule")
    ap.add_argument("--lookahead", type=int, default=2, help="schedule: units the CCL queue may run ahead")
    ap.add_argument("--slots", type=int, default=None, help="none/slot/drain: slot count (default 2*lookahead)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=4, help="matmuls per block reading the gathered weight")
    ap.add_argument("--rows", type=int, default=5632, help="rows of the full weight (w1 shape by default)")
    ap.add_argument("--m", type=int, default=2048, help="rows of the activation multiplied with the weight")
    ap.add_argument("--subdevice", default="columns=1", help="CCL sub-device: columns=N or rows=N")
    ap.add_argument("--no-update", action="store_true", help="skip the in-place shard update between steps")
    ap.add_argument("--per-device", action="store_true", help="report which devices hold a wrong result")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    n = args.mesh
    mesh = ttml.Mesh((n, 1), ("fsdp", "_1"))
    ttml.open_device_mesh(mesh, num_command_queues=2)
    ctx = ttml.autograd.AutoContext.get_instance()
    kind, count = args.subdevice.split("=")
    ctx.enable_ccl_sub_device(int(count) if kind == "columns" else 0, int(count) if kind == "rows" else 0)
    dev = ctx.get_device()
    compute_ids = [ttnn.SubDeviceId(ctx.compute_sub_device_index())]
    ccl_ids = [ttnn.SubDeviceId(ctx.ccl_sub_device_index())]
    cq0, cq1 = ttnn.QueueId(0), ttnn.QueueId(1)
    schedule = SlotSchedule(args.lookahead)
    K, S = args.blocks, (args.slots or schedule.num_slots)
    axis = mesh.axis_index("fsdp")
    grid = dev.compute_with_storage_grid_size()
    print(
        f"mesh={mesh.shape} compute grid={grid.x}x{grid.y} deps={args.deps} lookahead={args.lookahead} "
        f"blocks={K} slots={S} repeat={args.repeat} rows={args.rows}",
        flush=True,
    )

    rng = np.random.default_rng(0)
    bf16, tile = ttnn.DataType.BFLOAT16, ttnn.Layout.TILE
    shards = []
    for k in range(K):
        full = rng.standard_normal((1, 1, args.rows, 2048)).astype(np.float32) * 0.02
        shards.append(ttml.autograd.Tensor.from_numpy(full, tile, bf16, mesh.axis_mapper("fsdp", 2)).get_value())
    x = ttml.autograd.Tensor.from_numpy(
        rng.standard_normal((1, 1, args.m, args.rows)).astype(np.float32) * 0.02, tile, bf16, None
    ).get_value()
    pool = [ttml.core.distributed.all_gather(shards[0], 2, axis) for _ in range(S)]
    ttnn.synchronize_device(dev)

    def compute(w):
        y = ttnn.matmul(x, w)
        for _ in range(args.repeat - 1):
            y = ttnn.add(y, ttnn.matmul(x, w))
        return y

    def checksum(y):
        s = ttnn.sum(ttnn.typecast(y, ttnn.DataType.FLOAT32))
        per_device = [float(ttnn.to_torch(t).float().sum()) for t in ttnn.get_device_tensors(s)]
        ttnn.deallocate(s)
        return tuple(per_device) if args.per_device else per_device[0]

    def sync_reference(k):
        w = ttml.core.distributed.all_gather(shards[k], 2, axis)
        ttnn.synchronize_device(dev)
        y = compute(w)
        v = checksum(y)
        ttnn.deallocate(y)
        ttnn.deallocate(w)
        return v

    # --- the same dependency primitives as ttml/fsdp.py -----------------------------------------
    def ccl_barrier():
        ttnn.wait_for_event(cq0, ttnn.record_event(dev, cq_id=cq1, sub_device_ids=ccl_ids))

    def compute_drain():
        ttnn.wait_for_event(cq1, ttnn.record_event(dev, cq_id=cq0, sub_device_ids=compute_ids))

    release_events = {}  # policy "schedule": seq -> event; policy "slot": slot -> event

    def release(slot):
        ev = ttnn.record_event(dev, cq_id=cq0, sub_device_ids=compute_ids)
        release_events[schedule.release(slot) if args.deps == "schedule" else slot] = ev

    def before_gather(slot):
        if args.deps == "drain":
            compute_drain()
        elif args.deps == "slot":
            ev = release_events.get(slot)
            if ev is not None:
                ttnn.wait_for_event(cq1, ev)
        elif args.deps == "schedule":
            target = schedule.wait_target(slot)
            if target is None:
                compute_drain()
            else:
                ttnn.wait_for_event(cq1, release_events[target])

    def issue_gather(k):
        slot = k % S
        before_gather(slot)
        with ttnn.command_queue(cq1):
            ttml.core.distributed.all_gather(shards[k], 2, axis, pool[slot])

    # --- the loop --------------------------------------------------------------------------------
    total_bad = 0
    kinds = {"stale (previous occupant)": 0, "early overwrite (next occupant)": 0, "pre-update shard": 0, "unknown": 0}
    prev_refs = None
    t_start = time.perf_counter()
    for step in range(args.steps):
        refs = [sync_reference(k) for k in range(K)]
        outs = []
        t_host0 = time.perf_counter()
        compute_drain()  # the root's drain: the in-place update of the shards must be complete
        issue_gather(0)
        for k in range(K):
            ccl_barrier()  # block k's gather must be complete before its compute
            if k + 1 < K:
                issue_gather(k + 1)  # prefetch the next block, like pre_forward
            outs.append(compute(pool[k % S]))
            release(k % S)  # the compute enqueued so far was the slot's last reader
        t_host1 = time.perf_counter()
        ttnn.synchronize_device(dev)
        t_dev = time.perf_counter()
        got = [checksum(y) for y in outs]
        for y in outs:
            ttnn.deallocate(y)
        if not args.no_update:
            for k in range(K):
                ttnn.multiply(shards[k], 1.0 + 1e-3 * (k + 1), output_tensor=shards[k])  # in place, queue 0
        bad = []
        for k in range(K):
            if got[k] == refs[k]:
                continue
            if args.per_device:
                wrong = [d for d in range(len(got[k])) if got[k][d] != refs[k][d]]
                kinds["unknown"] += 1
                bad.append((k, f"devices {wrong}"))
                continue
            kind = "unknown"
            if k >= S and got[k] == refs[k - S]:
                kind = "stale (previous occupant)"
            elif k + S < K and got[k] == refs[k + S]:
                kind = "early overwrite (next occupant)"
            elif prev_refs is not None and got[k] == prev_refs[k]:
                kind = "pre-update shard"
            kinds[kind] += 1
            bad.append((k, kind))
        total_bad += len(bad)
        prev_refs = refs
        if bad or args.verbose or step < 2:
            host_ms, dev_ms = (t_host1 - t_host0) * 1e3, (t_dev - t_host0) * 1e3
            print(
                f"step {step:3d}: host issued in {host_ms:6.1f} ms, device done at {dev_ms:6.1f} ms; "
                f"{len(bad)} mismatching blocks {bad if bad else ''}",
                flush=True,
            )
    dt = time.perf_counter() - t_start
    print(
        f"deps={args.deps}: {total_bad} mismatching block results out of {args.steps * K} ({dt:.1f} s)"
        + ("; by kind: " + ", ".join(f"{k}={v}" for k, v in kinds.items() if v) if total_bad else ""),
        flush=True,
    )
    ttml.close_device_mesh()


if __name__ == "__main__":
    main()
