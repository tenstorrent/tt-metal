# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTFT of ResidentDFlash's prefill path by stage (synchronized): the initial-state reset, the pipeline-parallel
prefill of all but the last token, its handoff, the drafter's context fill from the prefill's taps (on device), the
last token's verify step, and start()'s logits readback. The plain resident path runs the same prefill + handoff +
one step, so the fill is the DFlash2 overhead. Also: the fill through the host (read_taps) for reference, its time and
the drafter context KV it leaves against the device's.

--ab: the reset A/B (dflash.FULL_RESET, RESIDENT_PREFILL_FULL_RESET): per T, after a polluting request, start()'s
logits and the next --ab-tokens served tokens with the prefill path's reset skipped against the full reset; they
must be identical (exit 1 otherwise).

--sync-ab: the cost of the fill's synchronization (dflash.FILL_SYNC, RESIDENT_DFLASH_FILL_SYNC): per T, start()'s wall
time (the served path, no stage marks: the stage timings above synchronize after every stage) without and with it,
--reps each, interleaved.

usage: python -m qwen36_resident.tests.dfl_ttft [--T 128,1024,2048,4096,8192] [--reps 3] [--ab] [--sync-ab]
    [--no-host-fill]
"""
import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import dflash
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlash
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--T", default="128,1024,2048,4096,8192")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt")
ap.add_argument("--ab", action="store_true")
ap.add_argument("--ab-tokens", type=int, default=64)
ap.add_argument("--sync-ab", action="store_true")
ap.add_argument("--no-host-fill", action="store_true")
a = ap.parse_args()
faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
mesh = open_mesh()
ok = True
try:
    Ts = [int(t) for t in a.T.split(",")]
    spec = ResidentDFlash(mesh, Q.Checkpoint(Q.checkpoint_dir()), max_pos=max(Ts) + 256, prefill=True)
    corpus = torch.load(a.corpus)["ids"]

    def timed(ids):
        """one prompt through _prompt with the stage marks, then start()'s logits readback: seconds per stage"""
        time.sleep(1)
        spec.prompt_times = []
        t0 = time.perf_counter()
        spec._prompt(ids)
        t1 = time.perf_counter()
        spec.m.logits()
        t2 = time.perf_counter()
        pt = dict(spec.prompt_times)
        spec.prompt_times = None
        return dict(
            reset=pt["start"] - pt["begin"],
            prefill=pt["prefill"] - pt["start"],
            handoff=pt["handoff"] - pt["prefill"],
            fill=pt["fill"] - pt["handoff"],
            step=t1 - pt["fill"],
            logits=t2 - t1,
            total=t2 - t0,
        )

    def served(ids, n):
        """start() + next_block until n tokens, as the served class: (logits, tokens)"""
        lg = spec.start(ids)
        out = [spec.first_token]
        while len(out) < n:
            out += spec.next_block(8, spec.max_pos)
        spec.stop()
        return lg, out[:n]

    for T in Ts:
        ids = corpus[:T].tolist()
        for full in [False, True] if a.ab else [dflash.FULL_RESET]:
            dflash.FULL_RESET = full
            rows = [timed(ids) for _ in range(a.reps)]
            med = {k: sorted(r[k] for r in rows)[len(rows) // 2] * 1e3 for k in rows[0]}
            plain = med["prefill"] + med["handoff"] + med["step"]
            log(
                f"T={T} full_reset={int(full)}: " + ", ".join(f"{k} {v:.1f}" for k, v in med.items()) + " ms; "
                f"DFlash2 overhead (fill) {med['fill']:.1f} ms = {100 * med['fill'] / plain:.1f}% of prefill + handoff "
                f"+ step {plain:.1f} ms"
            )
        if a.ab:
            # a different request first, so the skipped reset meets another request's state
            other = corpus[T + 1000 : 2 * T + 1000].tolist()
            res = {}
            for full in (False, True):
                dflash.FULL_RESET = full
                served(other, 16)
                res[full] = served(ids, a.ab_tokens)
            d = float((res[False][0].float() - res[True][0].float()).abs().max())
            same = res[False][1] == res[True][1]
            ok &= same and d == 0
            log(
                f"T={T} reset A/B: start logits max |d| {d:.4g}, {a.ab_tokens} served tokens "
                f"{'identical' if same else 'DIFFER: ' + str(res[False][1]) + ' vs ' + str(res[True][1])}"
            )
            dflash.FULL_RESET = False
        if a.sync_ab:
            walls = {False: [], True: []}
            for _ in range(a.reps):
                for sync in (False, True):
                    dflash.FILL_SYNC = sync
                    time.sleep(1)
                    t0 = time.perf_counter()
                    spec.start(ids)
                    walls[sync].append(time.perf_counter() - t0)
                    spec.stop()
            dflash.FILL_SYNC = True
            med = {k: 1e3 * sorted(v)[len(v) // 2] for k, v in walls.items()}
            log(
                f"T={T} fill sync A/B: start() median {med[False]:.2f} ms without, {med[True]:.2f} ms with "
                f"({med[True] - med[False]:+.2f} ms); without {[round(1e3 * w, 2) for w in walls[False]]}, "
                f"with {[round(1e3 * w, 2) for w in walls[True]]}"
            )
        if a.no_host_fill:
            continue
        # the context KV the device fill left (positions s0 .. T - 2), then the host fill's
        spec._prompt(ids)
        n = T - 1
        s0 = max(0, n - spec.window) // 64 * 64

        def context_kv():
            ttnn.synchronize_device(mesh)
            out = []
            for kc, vc in spec.drafter.kv:
                for c in (kc, vc):
                    x = ttnn.to_torch(ttnn.get_device_tensors(c)[0])[: -(-n // 64)].float()
                    out.append(x.permute(1, 0, 2, 3).reshape(x.shape[1], -1, x.shape[-1])[:, s0:n])
            return out

        kv_dev = context_kv()
        t2 = time.perf_counter()
        spec._fill_context(n, host=True)
        ttnn.synchronize_device(mesh)
        host_ms = (time.perf_counter() - t2) * 1e3
        kv_host = context_kv()
        diff = max(float((x - y).abs().max()) for x, y in zip(kv_dev, kv_host))
        log(f"T={T}: host fill {host_ms:.1f} ms, context KV device vs host fill max |d| {diff:.4g}")
finally:
    close_mesh(mesh)
if a.ab:
    log(f"reset A/B: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
