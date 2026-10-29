# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTFT of ResidentDFlash's prefill path by stage (synchronized): the pipeline-parallel prefill of all but the last
token, its handoff, the drafter's context fill from the prefill's taps (on device) and the last token's verify step.
The plain resident path runs the same prefill + handoff + one step, so the fill is the DFlash2 overhead. Also: the
fill through the host (read_taps) for reference, its time and the drafter context KV it leaves against the device's.

usage: python -m qwen36_resident.tests.dfl_ttft [--T 2048,8192,16000] [--reps 3]
"""
import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlash
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--T", default="2048,8192,16000")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt")
a = ap.parse_args()
faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
mesh = open_mesh()
try:
    Ts = [int(t) for t in a.T.split(",")]
    spec = ResidentDFlash(mesh, Q.Checkpoint(Q.checkpoint_dir()), max_pos=max(Ts) + 256, prefill=True)
    corpus = torch.load(a.corpus)["ids"]
    for T in Ts:
        ids = corpus[:T].tolist()
        rows = []
        for _ in range(a.reps):
            time.sleep(1)
            spec.prompt_times = []
            t0 = time.perf_counter()
            spec._prompt(ids)
            t1 = time.perf_counter()
            pt = dict(spec.prompt_times)
            spec.prompt_times = None
            st = dict(
                prefill=pt["prefill"] - pt["start"],
                handoff=pt["handoff"] - pt["prefill"],
                fill=pt["fill"] - pt["handoff"],
                step=t1 - pt["fill"],
                total=t1 - t0,
            )
            rows.append(st)
        # the context KV the device fill left (positions s0 .. T - 2), then the host fill's
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
        med = {k: sorted(r[k] for r in rows)[len(rows) // 2] * 1e3 for k in rows[0]}
        plain = med["prefill"] + med["handoff"] + med["step"]
        log(
            f"T={T}: " + ", ".join(f"{k} {v:.1f}" for k, v in med.items()) + f" ms; DFlash2 overhead (fill) "
            f"{med['fill']:.1f} ms = {100 * med['fill'] / plain:.1f}% of prefill + handoff + step {plain:.1f} ms; "
            f"host fill {host_ms:.1f} ms, context KV device vs host fill max |d| {diff:.4g}"
        )
finally:
    close_mesh(mesh)
