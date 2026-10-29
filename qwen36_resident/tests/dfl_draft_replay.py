# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replay a DFlash2 loop's drafter alone (dfl_taps_diag --save): the same context taps, extends and drafts in the
same order, against the drafts the loop got.

usage: python -m qwen36_resident.tests.dfl_draft_replay <saved .pt> [--nb 145]
"""
import argparse
import faulthandler
import sys

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import build_drafter, tap_buffers
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--nb", type=int, default=145)
ap.add_argument("--traced", type=int, default=1)
ap.add_argument("--ref-taps", default="", help="positions (a,b) whose taps come from the one-token reference instead")
ap.add_argument("--out", default="", help="save per-iteration drafts / hidden (+ the KV dump) here")
ap.add_argument(
    "--dump-iter", type=int, default=-1, help="dump the context KV (device 0) before this iteration's draft"
)
a = ap.parse_args()
faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)
rec = torch.load(a.path)
mesh = open_mesh()
try:
    ck = Q.Checkpoint(Q.checkpoint_dir())
    dr = build_drafter(mesh, ck)
    R, nt = dr.block, len(dr.taps)
    taps = tap_buffers(mesh, nt, R)
    dr.alloc(torch.arange(a.nb, dtype=torch.int32).reshape(1, a.nb), 64, taps)

    def put(rows):  # rows [taps, k, H] into the tap buffers' first k rows (the rest zero)
        host = torch.zeros(nt, R, 5120)
        host[:, : rows.shape[1]] = rows.float()
        for t, buf in zip(tap_buffers(mesh, nt, R, host), taps):
            ttnn.copy(t, buf)
            ttnn.deallocate(t)

    # the prompt: 8 rows per extend, as the loop's verify steps (positions from 0)
    P = len(rec["ids"])
    gen_taps = {p: t for p, t in rec["taps"]}
    # (the prompt's taps are not saved: the first saved record is the generation's) -> rebuild from records < P
    log(f"replay: {len(rec['drafts'])} iterations, prompt {P}")
    prompt = [(p, t) for p, t in rec["taps"] if p < P]
    for p0, t in prompt:
        put(t)
        dr.extend_context(p0, t.shape[1], traced=False)
    dr.draft(0, P, traced=False)
    if a.traced:
        dr.arm_traces()
    ref = {p: t[:, 0] for p, t in rec.get("ref_taps", [])}
    lo, hi = (int(x) for x in a.ref_taps.split(",")) if a.ref_taps else (0, 0)
    seq = rec["ids"] + rec["gen"]
    same = good = 0
    out = {"drafts": [], "hidden": [], "kv": None}
    for i, (p, tok, drafts, acc) in enumerate(rec["drafts"]):
        if i == a.dump_iter:
            ttnn.synchronize_device(mesh)
            nblk = (p + 8) // 64 + 1
            out["kv"] = (p, [[ttnn.to_torch(ttnn.get_device_tensors(c)[0])[:nblk].float() for c in kv] for kv in dr.kv])
        d = dr.draft(tok, p, traced=bool(a.traced))
        out["drafts"].append(d)
        out["hidden"].append(dr.last_hidden.clone())
        same += d == drafts
        n = 0
        while n < len(d) and p + 1 + n < len(seq) and d[n] == seq[p + 1 + n]:
            n += 1
        good += n
        if i < 60:
            log(f"iter {i} p={p}: replay drafts right {n} (loop accepted {acc})")
        t = gen_taps[p].clone()
        for r in range(t.shape[1]):
            if lo <= p + r < hi:
                t[:, r] = ref[p + r]
        put(t)
        dr.extend_context(p, t.shape[1], traced=bool(a.traced))
    log(f"replay drafts equal to the loop's: {same} / {len(rec['drafts'])}, right-prefix total {good}")
    if a.out:
        torch.save(out, a.out)
finally:
    close_mesh(mesh)
