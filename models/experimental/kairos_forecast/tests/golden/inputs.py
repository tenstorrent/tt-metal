# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Realistic input mix (skill §1, CONTRACT §5). numpy only, so it runs in either venv. Seeds go in MANIFEST."""
import numpy as np

CONTEXT = 2048
KINDS = ("rw_seasonal", "intermittent", "level_shift", "multi_seasonal", "trend", "spikes")


def make_series(rng, n, kind):
    t = np.arange(n, dtype=np.float64)
    if kind == "rw_seasonal":
        x = np.cumsum(rng.normal(size=n)) + 3 * np.sin(2 * np.pi * t / rng.choice([24, 48, 168]))
    elif kind == "intermittent":  # non-negative counts: exercises the clamp
        x = rng.poisson(rng.uniform(1, 5), n) * (rng.random(n) < rng.uniform(0.05, 0.4))
    elif kind == "level_shift":
        x = rng.normal(size=n) + np.repeat(rng.normal(0, 5, 8), -(-n // 8))[:n]
    elif kind == "multi_seasonal":
        x = sum(rng.uniform(1, 4) * np.sin(2 * np.pi * t / p + rng.uniform(0, 6)) for p in (24, 168, 365))
        x = x + rng.normal(0, 1, n)
    elif kind == "trend":
        x = rng.uniform(-0.02, 0.02) * t + 0.5 * np.sin(t / rng.uniform(50, 400)) + rng.normal(0, 0.1, n)
    else:
        x = rng.normal(size=n)
        spikes = rng.random(n) < 0.01
        x[spikes] += rng.normal(0, 20, int(spikes.sum()))
    if kind != "intermittent" and rng.random() < 0.25:  # some non-negative series
        x = x - x.min() + rng.uniform(0, 1)
    return (x * rng.uniform(0.1, 1000) if kind != "intermittent" else x).astype(np.float32)


def pad_or_truncate(x, length=CONTEXT):
    """The notebook's pad_or_truncate: keep the last `length` steps, left-pad with NaN."""
    x = np.asarray(x, dtype=np.float32)[-length:]
    return np.concatenate([np.full(length - len(x), np.nan, np.float32), x])


def make_batch(seed, n_series, lengths=None, nan_gap_frac=0.05):
    rng = np.random.default_rng(seed)
    out, kinds = [], []
    for i in range(n_series):
        kind = KINDS[i % len(KINDS)]
        n = CONTEXT if lengths is None else int(lengths[i])
        x = make_series(rng, n, kind)
        if rng.random() < nan_gap_frac and n > 256:
            s = rng.integers(0, n - 64)
            x[s : s + rng.integers(8, 64)] = np.nan
        out.append(pad_or_truncate(x))
        kinds.append(kind)
    return np.stack(out), kinds
