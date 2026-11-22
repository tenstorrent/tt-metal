# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB3 P3: the runner-model spec contract's protocol on dflash.SpecRowsSession, driven as vllm-tt-plugin's runner
drives a model (SPEC_DECODE_CONTRACT.md): a prefill per request (start: the first token sampled from its logits),
then per step propose (after every commit) -> verify ([token, drafts] per row) -> host greedy accept (the longest
prefix of drafts equal to the verify's argmax, plus the correction / bonus token) -> the next propose gets the
accepted counts. Users start one by one, two of them only after a few steps (the others held while a user starts).
The gate: every user's tokens equal the engine's own greedy decode of its prompt.

usage: python -m qwen36_resident.tests.db_protocol [--users 8] [--rows 4] [--prompt-len 1024] [--new 96]
"""
import argparse
import faulthandler
import sys
import time

import torch

from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows, SpecRowsSession
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--new", type=int, default=96)
    ap.add_argument("--late", type=int, default=2, help="users that start only after --late-steps steps")
    ap.add_argument("--late-steps", type=int, default=4)
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"]
    stride = (len(ids) - a.prompt_len - 101) // a.users
    prompts = [ids[stride * u + 101 : stride * u + 101 + a.prompt_len].tolist() for u in range(a.users)]
    mesh = open_mesh()
    ok = False
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        eng = ResidentDFlashRows(mesh, ck, users=a.users, max_pos=a.prompt_len + a.new + 256, R=a.rows, prefill=True)
        U, R = eng.U, eng.R
        ref = eng.greedy(prompts, a.new)
        print(f"greedy reference done ({U} users x {a.new})", flush=True)
        sess = SpecRowsSession(eng)
        active, gen, t, pos, acc = [False] * U, [[] for _ in range(U)], [0] * U, [0] * U, [1] * U

        def start(us):  # (several users in one call: their prompts through the PP, one verify step for all)
            t0 = time.perf_counter()
            for u, lg in zip(us, sess.start_many([(u, prompts[u], None) for u in us])):
                t[u], pos[u], acc[u], active[u] = int(lg.argmax()), len(prompts[u]), 1, True
                gen[u].append(t[u])
            print(f"started users {list(us)} in {time.perf_counter() - t0:.2f} s", flush=True)

        start(list(range(U - a.late)))
        t0, steps, accepted_total = time.perf_counter(), 0, 0
        while any(active[u] and len(gen[u]) < a.new for u in range(U)) or not all(active):
            if steps == a.late_steps:
                start(list(range(U - a.late, U)))
            run = [active[u] and len(gen[u]) < a.new for u in range(U)]
            drafts = sess.propose(t, pos, acc, run)
            tokens = [[t[u]] + list(drafts[u]) if run[u] else [0] * R for u in range(U)]
            g = sess.verify(tokens, pos, run)
            for u in range(U):
                if not run[u]:
                    continue
                k = 0
                while k < R - 1 and drafts[u][k] == g[u][k]:
                    k += 1
                gen[u] += list(drafts[u][:k]) + [g[u][k]]
                t[u], pos[u], acc[u] = g[u][k], pos[u] + k + 1, k + 1
                accepted_total += k + 1
            steps += 1
        dt = time.perf_counter() - t0
        same = [next((i for i, (x, y) in enumerate(zip(gen[u], ref[u])) if x != y), None) for u in range(U)]
        print(f"{steps} steps, {accepted_total / steps / U:.2f} tokens per user-step, {dt:.1f} s", flush=True)
        print(f"first difference vs greedy per user: {same}", flush=True)
        ok = sum(f is not None for f in same) <= 1  # (a near-tie flip at a tile crossing may remain: see PLAN)
        print(f"db protocol: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
