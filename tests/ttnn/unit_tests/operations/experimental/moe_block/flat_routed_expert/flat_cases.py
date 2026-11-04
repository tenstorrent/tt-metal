# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the flat routed expert's tests and its reference dump share: the model shapes, the per-chip load at a stage,
the seeded inputs of a case. Imports neither ttnn nor the op at import time (the dump runs in the reference tree)."""

from __future__ import annotations

import hashlib

import torch

# The routed-expert shapes (H: the width the experts see; K3: its latent width) and the Galaxy stage (SP x TP chips).
MODELS = {
    "glm53": dict(name="GLM-5.3", H=6144, I=2048, E=256, K=8, act="silu"),
    "glm53_flash": dict(name="GLM-5.3-Flash", H=4096, I=2048, E=288, K=8, act="silu"),
    "k3": dict(name="Kimi K3", H=3584, I=3072, E=896, K=16, act="situ"),
    "k27": dict(name="Kimi K2.7", H=7168, I=2048, E=384, K=8, act="silu"),
    "m3": dict(name="MiniMax M3", H=6144, I=3072, E=128, K=4, act="swigluoai"),
    "dsv41_flash": dict(name="DSv4.1-Flash", H=5120, I=2304, E=384, K=6, act="clamped_silu"),
}
STAGES = {"8x4": (8, 4), "4x4": (4, 4), "2x4": (2, 4)}
N_W = 4  # distinct weight sets, round-robin over the experts (a neighbour's weights read by mistake shows)
W_SCALE = 0.02
W_SEED = 0

# (model, chunk, y_row_major): the reference cases, each one chip at the Galaxy 8x4 per-chip load
REF_CASES = [("glm53", 5120, False), ("glm53", 5120, True), ("k3", 5120, False)]


def case_name(model, chunk, y_row_major):
    return f"{model}-{chunk}" + ("-rm" if y_row_major else "")


def per_chip(model, stage, chunk):
    """dict(chips, epc (experts per chip), S (home tokens per chip), M (tokens per expert, uniform))."""
    m = MODELS[model]
    sp, tp = STAGES[stage]
    n = sp * tp
    assert m["E"] % n == 0 and chunk % n == 0, (model, stage, chunk)
    return dict(chips=n, epc=m["E"] // n, S=chunk // n, M=chunk * m["K"] / m["E"])


def weight_sets(H, I, seed=W_SEED):
    """The N_W (Wg [H, I], Wu [H, I], Wd [I, H]) float32 weight sets of a case, regenerated from the seed."""
    torch.manual_seed(seed)
    return [tuple(torch.randn(*sh) * W_SCALE for sh in ((H, I), (H, I), (I, H))) for _ in range(N_W)]


def weights_digest(wsets):
    """sha256 of the weight sets' bytes: the reference records it, the test checks it regenerated the same weights."""
    h = hashlib.sha256()
    for ws in wsets:
        for w in ws:
            h.update(w.contiguous().numpy().tobytes())
    return h.hexdigest()


def uniform_counts(E_glob, K, epc, chunk, seed):
    """The chip's experts' shares of a uniform routing of the chunk (binomial, about chunk x K / E each)."""
    g = torch.Generator().manual_seed(seed)
    return [
        int(v) for v in torch.binomial(torch.full((epc,), float(chunk)), torch.full((epc,), K / E_glob), generator=g)
    ]


def case_inputs(model, chunk, seed=1):
    """The host inputs of one call: (x [rows, H] float32, counts [1, NG] int32, regions [1, NG] int32, gids, the
    per-expert (offset, count))."""
    md, pc = MODELS[model], per_chip(model, "8x4", chunk)
    H, E, NG = md["H"], pc["epc"], md["E"]
    gids = [(NG // E) * e + 1 for e in range(E)]
    counts = uniform_counts(NG, md["K"], E, chunk, seed)
    offs = [sum(-(-c // 32) * 32 for c in counts[:e]) for e in range(E)]
    rows = offs[-1] + -(-counts[-1] // 32) * 32 + 64
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.zeros(rows, H)
    c_ = torch.zeros(1, NG, dtype=torch.int32)
    r_ = torch.zeros(1, NG, dtype=torch.int32)
    for e, gid in enumerate(gids):
        x[offs[e] : offs[e] + counts[e]] = torch.randn(counts[e], H, generator=g)
        c_[0, gid], r_[0, gid] = counts[e], offs[e]
    return x, c_, r_, gids, list(zip(offs, counts))


# The plan fields the reference records and the test compares (the layout the weights and the program follow).
PLAN_KEYS = ("n_rd", "np", "nu", "rg", "mt", "nk_gu", "nd", "pcds", "col0s", "banks", "wd_rot_tiles")
PLAN_CORE_KEYS = ("gu", "down", "relays", "coords")
