# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The R-row verify step (DFlash2) with the contiguous and the paged KV cache: the Qwen3.8 checkpoint, a corpus
sequence through verify steps, bit-identical outputs expected between the modes (and between builds: --save /
--against).

Per mode: from position 0 the sequence goes through verify steps with every row accepted; at --at positions (each
a multiple of R) it runs R steps with a = 0 .. R - 1 drafts right (the others wrong), keeping every row's logits and
the taps of those steps, then continues teacher-forced. Then the traced verify step's device time at --speed
positions. Paged: a shuffled block table from a pool 1.5x the user's blocks.

usage: python -m qwen36_resident.tests.dfl_paged [--modes contiguous,paged] [--at 128,8192] [--save out.pt]
       [--against base.pt]
"""
import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, is_attn
from qwen36_resident.tests.dfl_verify import Verifier
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--modes", default="contiguous,paged")
ap.add_argument("--rows", type=int, default=8)
ap.add_argument("--at", default="128,8192")
ap.add_argument("--speed", default="128,2048,8192,32768")
ap.add_argument("--save", default="")
ap.add_argument("--against", default="", help="a --save of another build: the contiguous outputs compared with it")
ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt")
a = ap.parse_args()
faulthandler.dump_traceback_later(1200, repeat=True, file=sys.stderr)
mesh = open_mesh()
try:
    R = a.rows
    ats = [int(x) for x in a.at.split(",")]
    speeds = [int(x) for x in a.speed.split(",") if x]
    assert all(p % R == 0 for p in ats)
    ck = Q.Checkpoint(Q.checkpoint_dir())
    cfg = ck.config
    layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
    d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=vocab)
    w = Q.load(ck, d, layers, interval)
    emb = Q.embedding(ck).float()
    seq = torch.load(a.corpus)["ids"][: max(ats) + R * R + 2 * R].tolist()  # (R steps of a = 0 .. R - 1 at each)
    max_pos = max(speeds + [len(seq)]) + 2 * R + 64
    block = 64
    bpu = -(-max_pos // block)
    blocks = int(1.5 * bpu)
    table = torch.randperm(blocks, generator=torch.Generator().manual_seed(5))[:bpu].tolist()
    out = {}
    for mode in a.modes.split(","):
        kw = dict(kv_pool_blocks=blocks, block_tables=[table]) if mode == "paged" else {}
        t0 = time.time()
        ver = Verifier(mesh, d, w, layers, interval, R, max_pos, emb, **kw)
        log(f"{mode}: built in {time.time() - t0:.0f} s")
        kept = []  # (p, a, logits [R, vocab], taps [taps, R, H])
        p = 0
        for at in ats:
            while p < at:  # teacher-forced, all accepted
                ver.step(seq[p : p + R], p)
                ver.commit(R - 1)
                p += R
            for acc in range(R):
                drafts = [seq[p + i] if i <= acc else (seq[p + i] + 1) % vocab for i in range(1, R)]
                ver.step([seq[p]] + drafts, p)
                kept.append((p, acc, ver.logits(vocab), ver.tap_rows()))
                ver.commit(acc)
                p += acc + 1
        # the traced step's device time (positions written into the token state; contents irrelevant)
        ver.reset()
        ver.step(seq[:R], 0)
        ttnn.synchronize_device(mesh)
        ver.m.capture_trace()
        sp = {}
        for q in speeds:
            ver.ring = ver.slot = 0
            ver.prev = None
            time.sleep(1.5)
            ver.step(seq[:R], q)
            ttnn.synchronize_device(mesh)
            tb = time.perf_counter()
            for _ in range(16):
                ver.m.step(None)
            ttnn.synchronize_device(mesh)
            sp[q] = 1e3 * (time.perf_counter() - tb) / 16
        ver.m.release_trace()
        log(f"{mode}: verify step device ms " + ", ".join(f"{q}: {v:.2f}" for q, v in sp.items()))
        out[mode] = dict(kept=kept, speed=sp)
        del ver
    modes = list(out)

    def same(x, y):
        n = sum(torch.equal(u[2], v[2]) and torch.equal(u[3], v[3]) for u, v in zip(x["kept"], y["kept"]))
        dl = max(float((u[2] - v[2]).abs().max()) for u, v in zip(x["kept"], y["kept"]))
        dt = max(float((u[3].float() - v[3].float()).abs().max()) for u, v in zip(x["kept"], y["kept"]))
        return f"{n}/{len(x['kept'])} steps bit-identical (logits max |d| {dl:.4g}, taps max |d| {dt:.4g})"

    if len(modes) == 2:
        log(f"{modes[1]} vs {modes[0]}: {same(out[modes[0]], out[modes[1]])}")
    if a.against:
        base = torch.load(a.against)
        for m in modes:
            if m in base:
                log(f"{m} vs {a.against}: {same(base[m], out[m])}")
    if a.save:
        torch.save(out, a.save)
finally:
    close_mesh(mesh)
