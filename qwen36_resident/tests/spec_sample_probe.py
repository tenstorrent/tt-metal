# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sampled speculation on the verify build (RESIDENT_SPEC_SAMPLE=1), the device half: every verify row draws its token
with its user's (temperature, top_k, top_p, seed) at its own position (the hub's stateless (seed, position) RNG), and a
draft is accepted while the drawn token equals it — lossless for any draft. Gate: per user, the tokens of the spec
decode WITH drafts == the same build's decode WITHOUT drafts (row 0 only: plain sequential sampled decoding), and both
differ from greedy (the sampling is live). Reported: tokens per user-step sampled vs greedy.

usage: RESIDENT_SPEC_SAMPLE=1 python -m qwen36_resident.tests.spec_sample_probe [--users 8] [--new 128] [--temp 0.7]
"""
import argparse
import faulthandler

import torch

from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows, SpecRowsSession
from qwen36_resident.model import SPEC_SAMPLE
from qwen36_resident.tests.drafter_refill import CORPUS
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh


def run(eng, prompts, n_new, drafts_on=True):
    sess = SpecRowsSession(eng)
    U, R = eng.U, eng.R
    t, pos, acc, gen = [0] * U, [0] * U, [1] * U, [[] for _ in range(U)]
    for u, lg in zip(range(U), sess.start_many([(u, prompts[u], None) for u in range(U)])):
        t[u], pos[u] = int(lg.argmax()), len(prompts[u])  # (the first token: the prefill's argmax, both runs alike)
        gen[u].append(t[u])
    steps = accepted = 0
    while any(len(g) < n_new for g in gen):
        run_ = [len(gen[u]) < n_new for u in range(U)]
        drafts = sess.propose(t, pos, acc, run_)
        if not drafts_on:
            drafts = [[-1] * (R - 1) for _ in range(U)]  # (never equal to a drawn token: row 0 commits alone)
        tokens = [[t[u]] + [max(0, d) for d in drafts[u]] if run_[u] else [0] * R for u in range(U)]
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
            steps += 1
    return [x[:n_new] for x in gen], accepted / max(1, steps)


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=2049)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    a = ap.parse_args()
    assert SPEC_SAMPLE, "RESIDENT_SPEC_SAMPLE=1"
    ids = torch.load(CORPUS)["ids"]
    stride = (len(ids) - a.prompt_len - 101) // a.users
    prompts = [ids[stride * u + 101 : stride * u + 101 + a.prompt_len].tolist() for u in range(a.users)]
    mesh = open_mesh()
    try:
        eng = ResidentDFlashRows(
            mesh, Q.Checkpoint(Q.checkpoint_dir()), users=a.users, max_pos=a.prompt_len + a.new + 256, R=4, prefill=True
        )
        U, R = eng.U, eng.R
        assert eng.m.sample, "the verify build samples (model.py sample with RESIDENT_SPEC_SAMPLE)"
        g_greedy, tpi_greedy = run(eng, prompts, a.new)  # (params zero: greedy)
        rows = [(a.temp, a.top_k, a.top_p, 1000 + u) for u in range(U) for _ in range(R)]  # (row u * R + r)
        eng.m.set_sampling(rows=rows)
        g_spec, tpi_spec = run(eng, prompts, a.new, drafts_on=True)
        g_plain, tpi_plain = run(eng, prompts, a.new, drafts_on=False)
        same = [p == q for p, q in zip(g_spec, g_plain)]
        first = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None) for p, q in zip(g_spec, g_plain)]
        diff_greedy = [p != q for p, q in zip(g_spec, g_greedy)]
        print(f"greedy: {tpi_greedy:.2f} tokens per user-step", flush=True)
        print(
            f"sampled (T {a.temp}, top-k {a.top_k}, top-p {a.top_p}): {tpi_spec:.2f} tokens per user-step with drafts, "
            f"{tpi_plain:.2f} without",
            flush=True,
        )
        print(f"with drafts == without, per user: {same} (first difference {first})", flush=True)
        print(f"sampled != greedy, per user: {diff_greedy}", flush=True)
        print("PASS" if all(same) and any(diff_greedy) else "FAIL", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
