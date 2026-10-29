# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PPPrefill.reset A/B: the host-zero upload (RESIDENT_PP_HOST_ZERO=1, pp_prefill.HOST_ZERO) against the on-device
zero fill, through ResidentDFlash's serving path (PP prefill of all but the last token, handoff, drafter fill, verify
steps). Per prompt length T (the PP run's tokens; the prompt is T + 1), each arm after another request has left its
state in the prefill: start()'s logits, the PP buffers reset() zeroes after the run (every GDN layer's S and conv carry,
the chunk size's x_last) and --tokens served tokens. Pass: all bit-identical at every T. Also the time of reset()
alone and of start() per arm (synchronized, median of --reps).

usage: python -m qwen36_resident.tests.pp_reset_ab [--T 1,128,2048,32768] [--tokens 32] [--reps 3]
"""
import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlash
from qwen36_resident.prefill import pp_prefill
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--T", default="1,128,2048,32768")
ap.add_argument("--tokens", type=int, default=32)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt")
a = ap.parse_args()
faulthandler.dump_traceback_later(1200, repeat=True, file=sys.stderr)
mesh = open_mesh()
ok = True
try:
    Ts = [int(t) for t in a.T.split(",")]
    spec = ResidentDFlash(mesh, Q.Checkpoint(Q.checkpoint_dir()), max_pos=max(Ts) + 512, prefill=True)
    pp = spec.pp
    corpus = torch.load(a.corpus)["ids"]
    other = corpus[200000:203000].tolist()  # the polluting request
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)

    def pollute():
        spec.start(other)
        for _ in range(2):
            spec.next_block(8, spec.max_pos)
        spec.stop()

    def snapshot():
        ttnn.synchronize_device(mesh)
        out = {f"S{j}": ttnn.to_torch(pp.S[j], mesh_composer=comp) for j in pp.S}
        out.update({f"carry{j}": ttnn.to_torch(pp.carry[j], mesh_composer=comp) for j in pp.carry})
        out["x_last"] = ttnn.to_torch(pp.x_last, mesh_composer=comp)
        return out

    def ms(fn):
        t = []
        for _ in range(a.reps):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(mesh)
            t.append(time.perf_counter() - t0)
        return sorted(t)[len(t) // 2] * 1e3

    for host in (True, False):
        pp_prefill.HOST_ZERO = host
        log(f"pp.reset() host_zero={int(host)}: {ms(pp.reset):.2f} ms (median of {a.reps})")

    for T in Ts:
        prompt = corpus[: T + 1].tolist()
        res = {}
        for host in (True, False):
            pp_prefill.HOST_ZERO = host
            pollute()
            lg = spec.start(prompt)
            snap = snapshot()
            toks = [spec.first_token]
            while len(toks) < a.tokens:
                toks += spec.next_block(8, spec.max_pos)
            spec.stop()
            t_start = ms(lambda: (spec.start(prompt), spec.stop()))
            res[host] = (lg, snap, toks[: a.tokens], t_start)
        (lg1, s1, k1, t1), (lg0, s0, k0, t0) = res[True], res[False]
        d_lg = float((lg1.float() - lg0.float()).abs().max())
        d_buf = max(float((s1[k].float() - s0[k].float()).abs().max()) for k in s1)
        same = k1 == k0
        ok &= d_lg == 0 and d_buf == 0 and same
        log(
            f"T={T}: start logits max |d| {d_lg:.4g}, PP S / carry / x_last max |d| {d_buf:.4g}, {a.tokens} tokens "
            f"{'identical' if same else 'DIFFER'}; start() {t1:.1f} -> {t0:.1f} ms (host zero -> device zero)"
        )
finally:
    close_mesh(mesh)
log(f"PP reset A/B: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
