# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Independent NumPy model of the sampling specification and its corner-case decisions (SPEC_NOTES.md).
It is written from the spec text in NumPy idiom (stable lexsort, cumsum, np.exp) and is NOT bit-exact with the C library (np.exp differs from x280s_expf by
an ulp now and then). It guards the specification, not the bits: see test_numpy_reference.
"""

import numpy as np

K_MAX = 1024
F32 = np.float32


def splitmix64(x):
    m = (1 << 64) - 1
    z = (x + 0x9E3779B97F4A7C15) & m
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
    return z ^ (z >> 31)


def draw_u(seed, user, step):
    m = (1 << 64) - 1
    r = splitmix64((seed ^ (user << 32) ^ step) & m)
    return r, F32(r >> 40) * F32(2.0**-24)


def sample(logits, temperature, top_k, top_p, seed, user=0, step=0):
    """Returns (token, info). logits: float32 1-D array of length V."""
    x = np.array(logits, dtype=F32, copy=True)
    x[np.isnan(x)] = -np.inf
    V = x.shape[0]
    T = F32(temperature)
    if not T > 0:
        return int(np.argmax(x)), {"greedy": True}  # np.argmax returns the first maximum
    T = min(T, np.finfo(F32).max)
    s = (x / T).astype(F32)

    k = K_MAX if (top_k == 0 or top_k > K_MAX) else top_k
    k = min(k, V)
    # value descending, then index ascending
    order = np.lexsort((np.arange(V), -s))[:k]
    vals = s[order]
    with np.errstate(invalid="ignore", over="ignore"):
        w = np.where(vals == vals[0], F32(1), np.exp((vals - vals[0]).astype(F32))).astype(F32)
    c = np.cumsum(w, dtype=F32)  # numpy cumsum is a sequential running sum
    S = c[-1]
    p = F32(1) if not top_p < 1 else F32(top_p)
    thr = F32(p * S)
    n_kept = int(np.argmax(c >= thr)) + 1
    kept = c[n_kept - 1]
    r, u = draw_u(seed, user, step)
    target = F32(u * kept)
    hit = np.nonzero(c[:n_kept] > target)[0]
    pick = int(hit[0]) if hit.size else n_kept - 1
    info = {
        "greedy": False,
        "order": order,
        "w": w,
        "c": c,
        "S": S,
        "n_kept": n_kept,
        "kept": kept,
        "u": u,
        "target": target,
        "pick": pick,
        "p": p,
    }
    return int(order[pick]), info
