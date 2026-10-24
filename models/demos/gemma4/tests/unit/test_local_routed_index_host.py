# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the local routed prefill's index algebra (tt/experts/local_routed_index.py): the sorted pair
order, exclusive region offsets, inverse permutation and the per-chip routed golden against a plain dense sum.
No device, no ttnn import (run with ``--noconftest``: the directory conftest imports the dFlash host harness).

    pytest models/demos/gemma4/tests/unit/test_local_routed_index_host.py --noconftest -p no:cacheprovider
"""

import torch
import torch.nn.functional as F

from models.demos.gemma4.tt.experts.local_routed_index import (
    aligned_layout,
    index_pipeline,
    local_expert_mask,
    local_routed_golden,
)


def _routing(S, E, K, seed=0, skew=None):
    g = torch.Generator().manual_seed(seed)
    indices = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(S)])
    if skew is not None:
        # most tokens take the same few experts: large regions for a handful of experts, empty ones elsewhere
        hot = torch.arange(K) + skew
        n_hot = int(0.9 * S)
        indices[:n_hot] = hot
    values = torch.rand(S, K, generator=g) + 0.1
    values = values / values.sum(-1, keepdim=True)
    return values.to(torch.bfloat16), indices.to(torch.int64)


def test_index_pipeline_properties():
    S, E, K = 256, 128, 8
    _, indices = _routing(S, E, K)
    pos, counts, offsets, inv = index_pipeline(indices, E)
    N = S * K
    gid = indices.reshape(-1)
    assert pos.shape == (N,) and inv.shape == (N,)
    # sorted by expert id, ties by pair index (stable)
    sorted_gid = gid[pos]
    assert torch.all(sorted_gid[1:] >= sorted_gid[:-1])
    same = sorted_gid[1:] == sorted_gid[:-1]
    assert torch.all(pos[1:][same] > pos[:-1][same])
    # counts / exclusive offsets delimit every expert's region in the sorted order
    assert int(counts.sum()) == N
    assert torch.equal(offsets, torch.cumsum(counts, 0) - counts)
    for g_ in (0, 5, E - 1):
        s, n = int(offsets[g_]), int(counts[g_])
        assert torch.all(sorted_gid[s : s + n] == g_)
    # inverse permutation: every pair's buffer row lies inside its expert's region
    assert torch.equal(inv[pos], torch.arange(N))
    rows = inv  # row of pair p = t*K + k
    assert torch.all(rows >= offsets[gid]) and torch.all(rows < offsets[gid] + counts[gid])


def test_aligned_layout_regions():
    S, E, K = 256, 128, 8
    for seed, skew in ((0, None), (1, 40)):
        _, indices = _routing(S, E, K, seed=seed, skew=skew)
        pos, counts, _, _ = index_pipeline(indices, E)
        aligned_offsets, dest, M = aligned_layout(indices, E)
        gid = indices.reshape(-1).to(torch.int64)
        assert M == S * K + 32 * E
        # every region starts on a tile row, holds its expert's rows contiguously and ends inside the buffer
        assert torch.all(aligned_offsets % 32 == 0)
        assert int(aligned_offsets[-1] + counts[-1]) <= M
        assert torch.unique(dest).numel() == dest.numel()
        for g in (0, 40, 41, E - 1):
            s, n = int(aligned_offsets[g]), int(counts[g])
            rows = dest[gid[pos] == g]
            assert torch.equal(rows, torch.arange(s, s + n))
        # the buffer row of pair p is dest[inv[p]]
        inv = torch.empty_like(pos)
        inv[pos] = torch.arange(pos.numel())
        rows_of_pair = dest[inv]
        assert torch.all(rows_of_pair >= aligned_offsets[gid])
        assert torch.all(rows_of_pair < aligned_offsets[gid] + counts[gid])


def test_local_expert_mask():
    m = local_expert_mask(128, 64, 32)
    assert m.shape == (129,) and m.dtype == torch.int32
    assert torch.all(m[64:96] == 0) and torch.all(m[:64] == -1) and torch.all(m[96:] == -1)


def _dense_reference(x, values, indices, base, num_local, gate, up, down):
    """sum over the local slots of w * expert(x) in token order (what the device chip must produce)."""
    S, K = indices.shape
    out = torch.zeros(S, x.shape[-1])
    xf = x.float()
    for t in range(S):
        for k in range(K):
            g = int(indices[t, k])
            if base <= g < base + num_local:
                e = g - base
                h = F.gelu(xf[t] @ gate[e].float(), approximate="tanh") * (xf[t] @ up[e].float())
                out[t] += float(values[t, k]) * (h @ down[e].float())
    return out


def test_local_routed_golden_matches_dense_sum():
    S, E, K, H, I = 128, 32, 4, 64, 48
    num_local = 8
    for base, skew in ((0, None), (8, 8), (24, None)):
        values, indices = _routing(S, E, K, seed=base, skew=skew)
        x = torch.randn(S, H)
        gate = [torch.randn(H, I) * 0.1 for _ in range(num_local)]
        up = [torch.randn(H, I) * 0.1 for _ in range(num_local)]
        down = [torch.randn(I, H) * 0.1 for _ in range(num_local)]
        got = local_routed_golden(x, values, indices, base, num_local, gate, up, down)
        ref = _dense_reference(x, values, indices, base, num_local, gate, up, down)
        assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (base, skew, (got - ref).abs().max())
        if skew is not None:
            # the hot experts of this chip carry 90% of the tokens; the others of this chip see only the random tail
            _, counts, _, _ = index_pipeline(indices, E)
            assert int(counts[base]) >= int(0.9 * S)
