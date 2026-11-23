# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB3 phase 2, P2-2: AdaptiveEngine (qwen36_resident/adaptive.py) through a scripted ramp, greedy:
  4 users start in the spec build and speculate; 6 more arrive into the plain build (mixed steps: a spec step and a
  plain step each); the 4 move to the plain build (10 plain users); 7 leave; the 3 left return to the spec build
  (drafter refill) and speculate again.
Reference: every user's tokens decoded in the plain build alone (the same prompts, as many tokens). Reported: per user
the first difference (switches and speculation change the summation order: near-ties may flip), tokens per spec step,
each move's time.

usage: python -m qwen36_resident.tests.db_adaptive [--prompt-len 256] [--steps 12]
"""
import argparse
import faulthandler
import time

import torch

from qwen36_resident import weights as Q
from qwen36_resident.adaptive import AdaptiveEngine
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"]
    mesh = open_mesh()
    try:
        P, S = a.prompt_len, a.steps
        max_pos = P + 4 * 4 * S + 512
        bpu = -(-(max_pos + 72) // 64) + 2
        eng = AdaptiveEngine(mesh, Q.Checkpoint(Q.checkpoint_dir()), max_pos, kv_pool_blocks=24 * bpu + 8)
        prompt = {k: ids[900 * k + 101 : 900 * k + 101 + P].tolist() for k in range(10)}
        table = {k: list(range(k * bpu, (k + 1) * bpu)) for k in range(10)}
        spec_tokens = spec_steps = 0

        def spec(n):
            nonlocal spec_tokens, spec_steps
            for _ in range(n):
                out = eng.spec_step()
                spec_tokens += sum(len(v) for v in out.values())
                spec_steps += len(out)

        t0 = time.perf_counter()
        eng.add([(k, prompt[k], table[k]) for k in range(4)], "spec")
        spec(S)
        eng.add([(k, prompt[k], table[k]) for k in range(4, 10)], "plain")
        for _ in range(4):  # (mixed: the spec users speculate, the plain users step)
            eng.spec_step()
            eng.plain_step()
        for k in range(4):
            eng.to_plain(k)
        for _ in range(S):
            eng.plain_step()
        gen = {}
        for k in range(3, 10):
            gen[k] = eng.users[k].tokens[P:]
            eng.remove(k)
        for k in range(3):
            eng.to_spec(k)
        spec(S)
        for k in range(3):
            gen[k] = eng.users[k].tokens[P:]
            eng.remove(k)
        print(
            f"ramp done in {time.perf_counter() - t0:.1f} s; {spec_tokens / max(1, spec_steps):.2f} tokens per spec "
            f"user-step",
            flush=True,
        )
        for what, uid, sec in eng.switch_log:
            print(f"  {what} user {uid}: {sec * 1000:.0f} ms", flush=True)
        # ---- reference: each user in the plain build alone, as many tokens
        ref = {}
        for k in range(10):
            eng.add([(k, prompt[k], table[k])], "plain")
            while len(eng.users[k].tokens) - P < len(gen[k]):
                eng.plain_step()
            ref[k] = eng.users[k].tokens[P : P + len(gen[k])]
            eng.remove(k)
        firsts = {k: next((i for i, (x, y) in enumerate(zip(gen[k], ref[k])) if x != y), None) for k in range(10)}
        # (a user whose FIRST token differs: the spec build's start logits vs the plain build's first-step logits)
        for k in [k for k in range(10) if firsts[k] == 0]:
            gaps = []
            for geo in ("spec", "plain"):
                grp = list(range(4)) if (geo == "spec" and k < 4) else [k]  # (spec: as the ramp started it, index k)
                eng.free_spec.sort()
                eng.free_plain = [int(r) for r in eng.plain.row_order() if r in eng.free_plain]
                eng.add([(i, prompt[i], table[i]) for i in grp], geo)
                lg = eng.users[k].first_logits
                top = lg.topk(2)
                gaps.append(
                    f"{geo} (index {eng.users[k].idx}) top-2 {top.indices.tolist()} gap "
                    f"{float(top.values[0] - top.values[1]):.4f}"
                )
                for i in grp:
                    eng.remove(i)
            print(f"  user {k} first token: " + "; ".join(gaps), flush=True)
        print(
            f"first difference vs the plain-only reference per user (of its tokens): "
            f"{ {k: (firsts[k], len(gen[k])) for k in range(10)} }",
            flush=True,
        )
        n_eq = sum(f is None for f in firsts.values())
        print(f"db adaptive: {n_eq} / 10 users equal to the plain-only reference over every token", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
