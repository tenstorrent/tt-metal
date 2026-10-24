# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the dispatch chunk's per-expert capacity contract (``_DispatchChunkPlan``: the GeGLU op's max_tokens is
the chunk's rows, the maximum when every row names an expert at most once). The sentinel pad of the sparse ids keeps
a skewed routing plus pad rows within it; the old zero pad (expert 0 in all K slots) exceeded it on the CI shape
(bh_quietbox_2, 2026-10-24: 128 real tokens on a 256-row chunk -> 1032 rows on expert 0 vs 256, the kernels'
ASSERT(count_tiles == count_tiles_raw) under the watcher). No device, no conftest."""

import torch

from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants
from models.demos.gemma4.tt.experts.dispatch_prefill import Gemma4DispatchExperts

E, K = 128, 8


def _capacity(seq_per_chip, dgs=4):
    _, _, max_buf, max_tok = compute_constants(seq_per_chip, E, K, dgs, dgs, K)
    assert max_buf >= dgs * seq_per_chip * K, "drop-free buffer contract"
    return max_tok


def _ids(n, skew_to=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(n)])
    if skew_to is not None:
        ids[:, 0] = skew_to  # every real token's first slot on one expert, the other slots distinct
        for r in range(n):
            if (ids[r, 1:] == skew_to).any():
                ids[r, 1:] = torch.tensor([e for e in range(E) if e != skew_to][: K - 1])
    return ids


def test_capacity_is_the_chunk_rows():
    assert _capacity(64) == 256 and _capacity(256) == 1024 and _capacity(1024) == 4096


def test_ci_shape_zero_pad_overflowed_sentinel_pad_does_not():
    n, chunk = 128, 256  # the fitted chunk of a 128-token call on the (4,1) column: 64 rows/chip
    cap = _capacity(chunk // 4)
    ids = _ids(n)
    zero = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "zero")
    assert int(zero[0]) > cap and int(zero[0]) >= (chunk - n) * K, int(zero[0])  # the CI assert
    sentinel = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "sentinel")
    assert int(sentinel.max()) <= cap and int(sentinel.sum()) == n * K
    dense = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "dense")
    assert int(dense.max()) <= cap  # the dense twin passed on CI: +pad per expert 0..K-1


def test_skewed_routing_stays_within_capacity_for_every_pad():
    for n, chunk in ((129, 256), (300, 512), (2065, 2304), (4096, 4096)):
        cap = _capacity(chunk // 4)
        ids = _ids(n, skew_to=5)
        sentinel = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "sentinel")
        assert int(sentinel[5]) == n and int(sentinel.max()) <= cap, (n, chunk, int(sentinel.max()), cap)
        dense = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "dense")
        assert int(dense[5]) == n + (chunk - n) and int(dense.max()) <= cap, (n, chunk, int(dense.max()), cap)
        if chunk > n:
            zero = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "zero")
            assert int(zero[0]) == int(torch.bincount(ids.reshape(-1), minlength=E)[0]) + (chunk - n) * K


def test_served_shapes_old_pad_overflow_boundary():
    """The old zero pad exceeded the capacity when the real rows filled less than ~7/8 of the fitted chunk
    (K x pad + real_0 > chunk): a 300-token prompt on its 512-row chunk did, a 2065-token prompt on its 2304-row
    chunk did not; the sentinel pad never does."""
    for n, chunk, overflowed in ((300, 512, True), (2065, 2304, False), (129, 256, True)):
        cap = _capacity(chunk // 4)
        ids = _ids(n)
        zero = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "zero")
        assert (int(zero[0]) > cap) == overflowed, (n, chunk, int(zero[0]), cap)
        sentinel = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "sentinel")
        assert int(sentinel.max()) <= cap
