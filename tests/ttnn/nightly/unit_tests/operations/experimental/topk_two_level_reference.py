# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Oracle and input generators for the two-level topk_large_indices mode.

Values are compared the way topk_large_indices loads them on device: subnormals of either sign and -0 are the same
value as +0 (the load path flushes subnormals, and the sort ranks every member of that class equal, breaking ties by
position). normalize() maps a bf16 row to that order as float32.
"""

import numpy as np
import torch

BLOCK = 32
K = 2048
SENTINEL = 0xFFFFFFFF
_MIN_NORMAL = 2.0**-126


def normalize(values: torch.Tensor) -> torch.Tensor:
    """bf16 -> float32 in the device's load order: subnormals and -0 become +0."""
    x = values.float().clone()
    x[x.abs() < _MIN_NORMAL] = 0.0
    return x


def block_maxima(row: torch.Tensor, n: int, block: int = BLOCK) -> torch.Tensor:
    """Maximum of each `block`-key block of row[:n] in the device's load order (float32); the last partial block is
    padded with -inf."""
    num_blocks = (n + block - 1) // block
    padded = torch.full((num_blocks * block,), float("-inf"))
    padded[:n] = normalize(row[:n])
    return padded.reshape(num_blocks, block).amax(dim=1)


def exact_topk(row: torch.Tensor, n: int, k: int = K) -> torch.Tensor:
    """An exact top-k index list for one row in the device's load order, padded with sentinels."""
    x = normalize(row[:n])
    order = torch.topk(x, min(k, n), sorted=True).indices
    order = order[x[order] != float("-inf")]
    return torch.cat([order, torch.full((k - order.numel(),), SENTINEL, dtype=torch.int64)])


def expected_sentinels(row: torch.Tensor, n: int, k: int = K) -> int:
    live = int((normalize(row[:n]) != float("-inf")).sum())
    return max(0, k - live)


def assert_topk_exact(row: torch.Tensor, indices, n: int, k: int = K) -> None:
    """C1 for one row: distinct in-range indices, values descending, value multiset equal to the true top-k under
    the device's load order, and a contiguous sentinel tail of the right length."""
    idx = torch.as_tensor(indices, dtype=torch.int64).reshape(-1)
    assert idx.numel() == k, f"expected {k} indices, got {idx.numel()}"
    sentinel_mask = (idx & 0xFFFFFFFF) == SENTINEL
    num_sentinels = int(sentinel_mask.sum())
    want_sentinels = expected_sentinels(row, n, k)
    assert num_sentinels == want_sentinels, f"sentinel count {num_sentinels} != expected {want_sentinels}"
    if num_sentinels:
        assert bool(sentinel_mask[k - num_sentinels :].all()), "sentinels are not a contiguous tail"
    live_idx = idx[: k - num_sentinels]
    assert bool((live_idx >= 0).all() and (live_idx < n).all()), "index outside [0, n)"
    assert live_idx.unique().numel() == live_idx.numel(), "duplicate indices"
    x = normalize(row[:n])
    got = x[live_idx]
    assert bool((got[:-1] >= got[1:]).all()), "values are not descending"
    want = torch.topk(x, min(k, n), sorted=True).values
    want = want[want != float("-inf")][: k - num_sentinels]
    assert torch.equal(got, want), "value multiset differs from the exact top-k"


def _words_to_bf16(words: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((words.astype(np.uint32) << 16).view(np.float32).copy()).to(torch.bfloat16)


def generate(kind: str, rows: int, width: int, seed: int, n: int | None = None) -> torch.Tensor:
    """Input rows of `width` keys. `n` (valid length) places features that must sit inside the valid prefix."""
    n = width if n is None else n
    g = torch.Generator().manual_seed(seed)
    if kind == "iid":
        return torch.randn(rows, width, generator=g).to(torch.bfloat16)
    if kind == "narrow":
        return (25.0 + 10.0 * torch.rand(rows, width, generator=g)).to(torch.bfloat16)
    if kind.startswith("ties"):
        levels = int(kind[4:])
        words = (0x4100 + np.arange(levels, dtype=np.uint32)).astype(np.uint32)
        picks = torch.randint(0, levels, (rows, width), generator=g).numpy()
        return _words_to_bf16(words[picks])
    if kind == "rising":
        base = torch.linspace(-4.0, 4.0, width).expand(rows, width)
        return (base + 0.5 * torch.randn(rows, width, generator=g)).to(torch.bfloat16)
    if kind == "sinks":
        x = torch.randn(rows, width, generator=g)
        x[:, :4] += 30.0
        return x.to(torch.bfloat16)
    if kind == "causal":
        x = torch.randn(rows, width, generator=g)
        for r in range(rows):
            x[r, max(0, n - rows + r + 1) :] = float("-inf")
        return x.to(torch.bfloat16)
    if kind == "all_neginf":
        return torch.full((rows, width), float("-inf"), dtype=torch.bfloat16)
    if kind == "few_live":
        x = torch.full((rows, width), float("-inf"))
        live = torch.randint(0, n, (rows, 1000), generator=g)
        x.scatter_(1, live, torch.randn(rows, 1000, generator=g))
        return x.to(torch.bfloat16)
    if kind == "posinf":
        x = torch.randn(rows, width, generator=g)
        x[:, torch.randint(0, n, (5,), generator=g)] = float("inf")
        return x.to(torch.bfloat16)
    if kind == "zeros_subnormal":
        # Mostly -1.0 with a sea of +-0 and subnormals and 1000 positive normals scattered, so the 2048th value
        # sits inside the zero class.
        words = np.full((rows, width), 0xBF80, dtype=np.uint32)
        rng = np.random.default_rng(seed)
        zero_class = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F], dtype=np.uint32)
        for r in range(rows):
            pos = rng.choice(n, size=min(n, 6000), replace=False)
            words[r, pos] = zero_class[rng.integers(0, len(zero_class), size=pos.size)]
            win = rng.choice(n, size=min(n, 1000), replace=False)
            words[r, win] = 0x3F80 + rng.integers(0, 64, size=win.size)
        return _words_to_bf16(words)
    raise ValueError(kind)


GENERATORS = [
    "iid",
    "narrow",
    "ties2",
    "ties8",
    "ties64",
    "rising",
    "sinks",
    "causal",
    "all_neginf",
    "few_live",
    "posinf",
    "zeros_subnormal",
]
