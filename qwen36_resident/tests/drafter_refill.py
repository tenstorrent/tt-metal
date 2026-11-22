# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB3 phase 2, P2-0c: the drafter's acceptance when its context is rebuilt approximately, as a user returning from
the plain geometry to speculation would get it (option B): the target state exact (the whole prompt through the PP
prefill and the handoff), the drafter's window filled from a SEPARATE PP run over only the sequence's last `window`
tokens as a fresh prompt (approximate taps: hidden states without the earlier context), placed at the tokens' absolute
positions. vs the exact fill (taps of the whole prompt's prefill). Spec decode loop as tests/db_protocol.py (propose ->
verify -> host greedy accept); reported: tokens per user-step for both and the tokens' agreement (the target is exact
either way: only near-tie flips may differ).

usage: python -m qwen36_resident.tests.drafter_refill [--users 8] [--prompt-len 6145] [--new 128]
"""
import argparse
import faulthandler
import time

import torch

from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows, SpecRowsSession
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"


def run(eng, prompts, n_new):
    sess = SpecRowsSession(eng)
    U, R = eng.U, eng.R
    t, pos, acc, gen = [0] * U, [0] * U, [1] * U, [[] for _ in range(U)]
    for u, lg in zip(range(U), sess.start_many([(u, prompts[u], None) for u in range(U)])):
        t[u], pos[u] = int(lg.argmax()), len(prompts[u])
        gen[u].append(t[u])
    user_steps = accepted = 0
    run_ = [True] * U
    while any(len(g) < n_new for g in gen):
        run_ = [len(gen[u]) < n_new for u in range(U)]
        drafts = sess.propose(t, pos, acc, run_)
        tokens = [[t[u]] + list(drafts[u]) if run_[u] else [0] * R for u in range(U)]
        g = sess.verify(tokens, pos, run_)
        for u in range(U):
            if not run_[u]:
                continue
            k = 0
            while k < R - 1 and drafts[u][k] == g[u][k]:
                k += 1
            gen[u] += list(drafts[u][:k]) + [g[u][k]]
            t[u], pos[u], acc[u] = g[u][k], pos[u] + k + 1, k + 1
            accepted += k + 1
            user_steps += 1
    return [x[:n_new] for x in gen], accepted / max(1, user_steps)


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=6145)
    ap.add_argument("--new", type=int, default=128)
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"]
    stride = (len(ids) - a.prompt_len - 101) // a.users
    prompts = [ids[stride * u + 101 : stride * u + 101 + a.prompt_len].tolist() for u in range(a.users)]
    mesh = open_mesh()
    try:
        eng = ResidentDFlashRows(
            mesh, Q.Checkpoint(Q.checkpoint_dir()), users=a.users, max_pos=a.prompt_len + a.new + 256, R=4, prefill=True
        )
        W = eng.window
        exact_fill = eng._fill_context
        t0 = time.perf_counter()
        g_exact, tpi_exact = run(eng, prompts, a.new)
        print(f"exact fill: {tpi_exact:.2f} tokens per user-step ({time.perf_counter() - t0:.1f} s)", flush=True)

        cur = {}

        def refill(n_tokens, phys=0, offset=0):  # (the tail as a fresh prompt; called after the whole prompt's handoff)
            ids_u = cur[phys]
            s0 = max(0, n_tokens - W) // 64 * 64
            eng.pp.reset()
            eng.pp.run(torch.as_tensor(ids_u[s0:n_tokens]))
            exact_fill(n_tokens - s0, phys=phys, offset=s0)

        for u in range(a.users):
            cur[u] = prompts[u]
        eng._fill_context = refill
        t0 = time.perf_counter()
        g_refill, tpi_refill = run(eng, prompts, a.new)
        print(
            f"refill (last {W} tokens as a fresh prompt): {tpi_refill:.2f} tokens per user-step "
            f"({time.perf_counter() - t0:.1f} s)",
            flush=True,
        )
        first = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None) for p, q in zip(g_exact, g_refill)]
        print(f"tokens exact vs refill, first difference per user: {first}", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
