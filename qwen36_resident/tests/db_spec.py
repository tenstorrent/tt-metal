# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB1 prototype end to end: greedy DFlash2 speculative decode of U users in one launch (dflash.ResidentDFlashRows,
the Qwen3.8 checkpoint and the DFlash2 drafter on the 1 x 4 mesh).

Per user a corpus prompt (one length for all); first the verify model's own greedy decode (one token per step, row 0),
then the speculative decode; the gate: every user's speculative tokens equal its greedy tokens. Reported: tokens/s per
user and in total, drafts accepted per iteration, the draft / verify / extend split (medians, synchronized).

usage: python -m qwen36_resident.tests.db_spec [--users 4] [--prompt-len 256] [--new 128] [--json out.json]
"""
import argparse
import faulthandler
import json
import sys
import time

import torch

from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--rows", type=int, default=8, help="verify rows per user (8: 4 users at most, 4: 8 users)")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--json", default=None)
    ap.add_argument("--gaps", action="store_true", help="log logits; at each divergence the top-2 gaps")
    ap.add_argument("--prefill", action="store_true", help="prompts through the PP prefill (handoff + drafter fill)")
    a = ap.parse_args()
    ids = torch.load(a.corpus)["ids"]
    stride = (len(ids) - a.prompt_len - 101) // a.users  # (distinct corpus slices that fit, 8 users included)
    prompts = [ids[stride * u + 101 : stride * u + 101 + a.prompt_len].tolist() for u in range(a.users)]
    mesh = open_mesh()
    out = dict(users=a.users, prompt_len=a.prompt_len, new=a.new)
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        t0 = time.time()
        eng = ResidentDFlashRows(
            mesh, ck, users=a.users, max_pos=a.prompt_len + a.new + 256, R=a.rows, prefill=a.prefill
        )
        print(f"engine: {a.users} users x {eng.R} verify rows + drafter, built in {time.time() - t0:.0f} s", flush=True)
        t0 = time.time()
        eng.logit_log = {} if a.gaps else None
        ref = eng.greedy(prompts, a.new)
        print(f"greedy reference: {a.users} x {a.new} tokens in {time.time() - t0:.1f} s", flush=True)
        if a.prefill:  # (the handoff: the same greedy with the prompts through verify steps instead of the PP prefill)
            pp, eng.pp = eng.pp, None
            nopp = eng.greedy(prompts, a.new)
            eng.pp = pp
            pre = [next((i for i, (x, y) in enumerate(zip(g, r)) if x != y), a.new) for g, r in zip(ref, nopp)]
            out["pp_vs_verify_prefix"] = pre
            print(f"PP-prefilled greedy vs verify-step prompts: equal prefix per user {pre} of {a.new}", flush=True)
        gen, stats = eng.generate(prompts, a.new)
        same = [sum(1 for x, y in zip(g, r) if x == y) for g, r in zip(gen, ref)]
        first_diff = [next((i for i, (x, y) in enumerate(zip(g, r)) if x != y), None) for g, r in zip(gen, ref)]
        out.update(stats=stats, same=same, first_diff=first_diff, ok=all(f is None for f in first_diff))
        print(
            f"spec decode: {stats['tok_s_per_user']:.1f} tok/s per user, {stats['tok_s_total']:.1f} total, "
            f"{stats['tokens_per_iter']:.2f} tokens per iteration (mean accept {stats['mean_accept']:.2f}), "
            f"{stats['ms_per_iter']:.1f} ms per iteration (draft {stats['draft_ms']:.1f} / verify "
            f"{stats['verify_ms']:.1f} / extend {stats['extend_ms']:.1f})",
            flush=True,
        )
        print(f"tokens equal to greedy per user: {same} of {a.new} (first difference {first_diff})", flush=True)
        if a.gaps:  # (at each user's first difference: how close the two decodes' top-2 are there)
            for u, f in enumerate(first_diff):
                if f is None:
                    continue
                pos = a.prompt_len + f - 1  # (the logits at position pos predict generated token f)
                gl, sl = eng.logit_log.get(("greedy", u, pos)), (eng.logit_log.get(("spec", u, pos)) or (0, None))[1]
                if gl is None or sl is None:
                    print(f"  user {u}: token {f}: logits not logged at {pos}")
                    continue
                g2, s2 = gl.topk(2), sl.topk(2)
                print(
                    f"  user {u}: token {f} (pos {pos}): greedy top-2 {g2.indices.tolist()} gap "
                    f"{float(g2.values[0] - g2.values[1]):.4f}; spec top-2 {s2.indices.tolist()} gap "
                    f"{float(s2.values[0] - s2.values[1]):.4f}; max |dlogit| {float((gl - sl).abs().max()):.4f}"
                )
        if a.gaps:  # (before each user's first difference: max |dlogit| vs the greedy decode, by the spec row index)
            by_row = {}
            for (kind, u, pos), v in list(eng.logit_log.items()):
                if kind != "spec" or (first_diff[u] is not None and pos >= a.prompt_len + first_diff[u] - 1):
                    continue
                g_ = eng.logit_log.get(("greedy", u, pos))
                if g_ is not None:
                    r, lg_ = v
                    d_ = float((g_ - lg_).abs().max())
                    n_, m_, z_ = by_row.get(r, (0, 0.0, 0))
                    by_row[r] = (n_ + 1, max(m_, d_), z_ + int(d_ == 0.0))
            for r in sorted(by_row):
                n_, m_, z_ = by_row[r]
                print(f"  spec row {r}: {n_} positions, bit-identical {z_}, max |dlogit| {m_:.4f}")
        print(f"db spec: {'PASS' if out['ok'] else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)
    sys.exit(0 if out.get("ok") else 1)


if __name__ == "__main__":
    main()
