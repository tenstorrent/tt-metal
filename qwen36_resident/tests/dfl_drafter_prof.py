# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's draft() / extend_context() as the resident loop runs them (traced, nb-block context table,
context C): wall times (the call as the loop makes it, and the bare trace replay), the drafts of a fixed input set
(--save / --check: bit-identical drafts across a change), and, under the device profiler (python -m tracy -r), the
ops of --iters draft + extend pairs between the 'start' / 'stop' signposts.

usage: python -m qwen36_resident.tests.dfl_drafter_prof [--C 600,2100] [--iters 20] [--coexist] [--save p | --check p]
"""
import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlash, build_drafter, tap_buffers
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(*_a, **_k):
        pass


ap = argparse.ArgumentParser()
ap.add_argument("--C", default="600,2100", help="draft contexts (the first is profiled)")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--nb", type=int, default=65)
ap.add_argument(
    "--coexist", action="store_true", help="the drafter of a full ResidentDFlash (the resident model built)"
)
ap.add_argument("--save", default="", help="save the drafts of the fixed input set here")
ap.add_argument("--check", default="", help="compare the drafts of the fixed input set against this save")
ap.add_argument("--prof-iters", type=int, default=5, help="draft + extend pairs between the signposts")
a = ap.parse_args()
faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
mesh = open_mesh()
try:
    ck = Q.Checkpoint(Q.checkpoint_dir())
    if a.coexist:
        spec = ResidentDFlash(mesh, ck, max_pos=a.nb * 64)
        dr, taps = spec.drafter, spec.taps
    else:
        dr = build_drafter(mesh, ck)
        taps = tap_buffers(mesh, len(dr.taps), dr.block)
        dr.alloc(torch.arange(a.nb, dtype=torch.int32).reshape(1, a.nb), 64, taps)
    nt, R, H = len(dr.taps), dr.block, 5120
    Cs = [int(c) for c in a.C.split(",")]
    top = max(Cs) + 64
    g = torch.Generator().manual_seed(11)
    ctx = torch.randn(nt, top, H, generator=g) * 2
    for s0 in range(0, top // 64 * 64, 2048):
        rows = min(2048, top // 64 * 64 - s0)
        dev = tap_buffers(mesh, nt, rows, ctx[:, s0 : s0 + rows])
        dr.fill_context(dev, s0)
        for t in dev:
            ttnn.deallocate(t)
    rows8 = torch.randn(nt, R, H, generator=g) * 2
    host = tap_buffers(mesh, nt, R, rows8)
    for t, buf in zip(host, taps):
        ttnn.copy(t, buf)
        ttnn.deallocate(t)
    anchors = torch.randint(0, 150000, (64,), generator=g).tolist()
    # compile, then capture (as ResidentDFlash.__init__ does)
    dr.extend_context(Cs[0], R, traced=False)
    dr.draft(anchors[0], Cs[0], traced=False)
    ttnn.synchronize_device(mesh)
    dr.arm_traces()
    dr.extend_context(Cs[0], R)
    dr.draft(anchors[0], Cs[0])
    ttnn.synchronize_device(mesh)
    lay = dr.layout()

    # the fixed input set: drafts at each C for 8 anchors (the extend rewrites C .. C + 7 with the same rows first)
    drafts = []
    for C in Cs:
        for i in range(8):
            dr.extend_context(C, R)
            drafts.append(dr.draft(anchors[i], C))
    if a.save:
        torch.save(dict(drafts=drafts, Cs=Cs), a.save)
    if a.check:
        ref = torch.load(a.check)["drafts"]
        same = sum(x == y for x, y in zip(drafts, ref))
        log(f"drafts identical to {a.check}: {same} / {len(ref)}")

    for C in Cs:
        # the calls as the loop makes them
        t_d, t_x = [], []
        for i in range(a.iters):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            dr.draft(anchors[i % len(anchors)], C)
            t1 = time.perf_counter()
            dr.extend_context(C, 1 + i % R)
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
            t_d.append(t1 - t0)
            t_x.append(t2 - t1)
        # the bare trace replays (device time + dispatch)
        r_d, r_x = [], []
        for i in range(a.iters):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, lay.draft_tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t1 = time.perf_counter()
            ttnn.execute_trace(mesh, lay.ext_tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
            r_d.append(t1 - t0)
            r_x.append(t2 - t1)
        med = lambda v: sorted(v)[len(v) // 2] * 1e3  # noqa: E731
        log(
            f"C={C}: draft() {med(t_d):.2f} ms (trace replay {med(r_d):.2f}), extend {med(t_x):.2f} ms "
            f"(trace replay {med(r_x):.2f}); host part of draft() {med(t_d) - med(r_d):.2f} ms"
        )

    # the profiled window (the device profiler's buffers emptied first: a core's buffer holds a limited number of
    # ops, and the KV writes all run on one core)
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    signpost("start")
    for i in range(a.prof_iters):
        dr.draft(anchors[i], Cs[0])
        dr.extend_context(Cs[0], R)
    ttnn.synchronize_device(mesh)
    signpost("stop")
    ttnn.ReadDeviceProfiler(mesh)
finally:
    close_mesh(mesh)
