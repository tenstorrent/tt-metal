# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact top-k sampler compiled with numba (Blaze L5).

Same distribution as ``sample_token``'s torch path: keep every logit >= the k-th largest
(ties kept), softmax of logits / temperature over the kept ones, one draw. The draw is an
inverse-CDF lookup of a uniform from torch's generator, so a request seeded with
``torch.manual_seed`` stays reproducible. ~5 us per 2048-way draw instead of ~115 us of
torch ops; 16 draws sit on every frame's critical path.

``available()`` is False when numba cannot be imported; callers then use the torch path.
"""

import math

import numpy as np

try:
    from numba import njit

    @njit(cache=False, nogil=True)
    def _topk_draw(logits, inv_temp, k, u):
        n = logits.shape[0]
        if k > n:
            k = n
        kth = np.partition(logits, n - k)[n - k]
        m = -np.inf
        for i in range(n):
            if logits[i] >= kth and logits[i] > m:
                m = logits[i]
        total = 0.0
        for i in range(n):
            if logits[i] >= kth:
                total += math.exp((logits[i] - m) * inv_temp)
        target = u * total
        acc = 0.0
        last = -1
        for i in range(n):
            if logits[i] >= kth:
                acc += math.exp((logits[i] - m) * inv_temp)
                last = i
                if acc > target:
                    return i
        return last

    _OK = True
except Exception:  # noqa: BLE001 -- numba missing or broken: torch path
    _OK = False


def available() -> bool:
    return _OK


def warm() -> None:
    """Compile the kernel (first call ~1 s) so no request pays it."""
    if _OK:
        _topk_draw(np.zeros(64, dtype=np.float32), 1.0, 8, 0.5)


def draw(logits_f32: np.ndarray, temperature: float, top_k: int, u: float) -> int:
    """One draw from the top-k softmax of ``logits_f32`` (1-D float32, -inf allowed)."""
    return int(_topk_draw(logits_f32, 1.0 / temperature, int(top_k), float(u)))
