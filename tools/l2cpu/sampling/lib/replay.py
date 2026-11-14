#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Replay recorded logit rows through the host reference library (libx280s_host.so).

    replay.py ROWS.npy --out expected.npz [--seed 1234] [--user 0] [--step0 0] [--vocab 151936]
    replay.py --make-synthetic ROWS.npy --n 64 [--bf16]     # write Qwen3-shaped synthetic rows

ROWS.npy: shape (N, Vpad), float32 logits or uint16 bfloat16 bits. Columns >= --vocab are padding
and are never read. Row i is sampled with step = step0 + i and the given user, for the three reference
settings, plus greedy:
    setting 0: T 0.7, top-k 50, top-p 0.9
    setting 1: T 1.0, top-k 0,  top-p 1.0
    setting 2: T 0.6, top-k 20, top-p 0.95
The output .npz holds tokens[int32, (3, N)], greedy[int32, (N,)], the settings, the seed/user/step0
and the fp32 intermediates (S, kept_sum, target as uint32 bits, (3, N)) for bit-exact checks.
A plain-text copy (one line per row: greedy t0 t1 t2) is written next to it as <out>.txt.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x280s_ref import X280S  # noqa: E402

SAMPLED_SETTINGS = [(0.7, 50, 0.9), (1.0, 0, 1.0), (0.6, 20, 0.95)]
V_QWEN = 151936


def synthetic_rows(n, V=V_QWEN, seed=0, bf16=False):
    """Qwen3-like rows: normal(0, 3) plus a few large logits (a peaked head)."""
    from x280s_ref import bf16_bits

    rng = np.random.default_rng(seed)
    rows = rng.normal(0.0, 3.0, (n, V)).astype(np.float32)
    for r in rows:
        k = int(rng.integers(1, 40))
        idx = rng.choice(V, k, replace=False)
        r[idx] += rng.uniform(6.0, 20.0, k).astype(np.float32)
    return bf16_bits(rows) if bf16 else rows


def replay(rows, seed, user=0, step0=0, vocab=None, lib=None):
    lib = lib or X280S()
    N, Vpad = rows.shape
    V = min(Vpad, V_QWEN) if vocab is None else vocab
    tokens = np.zeros((len(SAMPLED_SETTINGS), N), np.int32)
    inter = np.zeros((len(SAMPLED_SETTINGS), N, 3), np.uint32)
    greedy = np.zeros(N, np.int32)
    t0 = time.perf_counter()
    for i in range(N):
        row = np.ascontiguousarray(rows[i])
        greedy[i] = lib.argmax(row, vocab=V)[0]
        for s, (T, k, p) in enumerate(SAMPLED_SETTINGS):
            tok, st = lib.sample(row, T, k, p, seed, user=user, step=step0 + i, vocab=V)
            assert tok >= 0, (i, s, tok)
            tokens[s, i] = tok
            b = st.float_bits()
            inter[s, i] = (b["S"], b["kept_sum"], b["target"])
    dt = time.perf_counter() - t0
    return {
        "tokens": tokens,
        "greedy": greedy,
        "intermediates_S_kept_target": inter,
        "settings": np.array(SAMPLED_SETTINGS, np.float64),
        "seed": np.uint64(seed),
        "user": np.uint32(user),
        "step0": np.uint64(step0),
        "vocab": np.uint32(V),
        "seconds": dt,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=1234)
    ap.add_argument("--user", type=int, default=0)
    ap.add_argument("--step0", type=int, default=0)
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--make-synthetic", metavar="PATH")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()

    if a.make_synthetic:
        rows = synthetic_rows(a.n, bf16=a.bf16)
        np.save(a.make_synthetic, rows)
        print("wrote %s %s %s" % (a.make_synthetic, rows.shape, rows.dtype))
        return
    if not a.rows or not a.out:
        ap.error("ROWS.npy and --out are required")
    rows = np.load(a.rows, mmap_mode="r")
    if rows.ndim != 2 or rows.dtype not in (np.float32, np.uint16):
        sys.exit("expected a 2-D float32 or uint16 (bf16) array, got %s %s" % (rows.shape, rows.dtype))
    res = replay(rows, a.seed, a.user, a.step0, a.vocab)
    np.savez(a.out, **{k: v for k, v in res.items() if k != "seconds"})
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w") as f:
        f.write(
            "# row greedy T0.7/k50/p0.9 T1.0/k0/p1.0 T0.6/k20/p0.95  seed=%#x user=%d step0=%d vocab=%d\n"
            % (a.seed, a.user, a.step0, res["vocab"])
        )
        for i in range(rows.shape[0]):
            f.write("%d %d %d %d %d\n" % (i, res["greedy"][i], *res["tokens"][:, i]))
    n = rows.shape[0]
    print(
        "replayed %d rows (%s) in %.2f s (%.0f us per row per setting incl. ctypes) -> %s, %s"
        % (n, rows.dtype, res["seconds"], 1e6 * res["seconds"] / (n * 4), a.out, txt)
    )


if __name__ == "__main__":
    main()
