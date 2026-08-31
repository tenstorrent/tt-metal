# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE (32 chips, BH Galaxy): how far does bf8 KV-cache accuracy fall at a REAL prefix depth?

``test_attention_vs_ref`` tops out at 2048 tokens, where the bf8 cache still looks fine. The decay is
a function of attended prefix length, so the question that matters for a 128K production prefill is
what happens at tens of thousands of tokens. This drives a full 10-chunk prefill to **51200 tokens**
at the production chunk size (5120) and PCCs the LAST chunk against torch.

**Why only the last chunk.** A full-sequence float reference at this length is impossible: 96 heads x
51200^2 scores is ~960 GB. But only the final rows carry the longest prefix, which is exactly the
worst case, so the reference computes Q for the last 5120 rows only, against K/V for all 51200 --
96 x 5120 x 51200 done one head at a time, ~1 GB peak.

Slow by construction (minutes of CPU for the reference). It is a measurement, not a gate: the
assert is loose and deliberately so, because the point is to REPORT the number.
"""

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_d_p.reference.torch_reference import apply_rope_hf
from models.demos.mistral_medium_d_p.tt.attention import allocate_kv_cache
from models.demos.mistral_medium_d_p.tt.rope import build_indexed_rope
from models.demos.mistral_medium_d_p.tt.rope_tables import build_hf_cos_sin

from ..test_factory import mesh_setup, parametrize_mesh_with_fabric
from .shapes import HEAD_DIM, HIDDEN, N_KV, N_Q, YARN, per_chip
from .test_attention_vs_ref import _build_attention, _chunk_order, _gather_sp_tp, _place_sp, _random_attn_weights


def _reference_tail(x, w, total, tail):
    """Float reference for the LAST `tail` rows of a `total`-token sequence, head by head."""
    cos, sin = build_hf_cos_sin(total, HEAD_DIM, **YARN)
    xf = x[0].float()

    k = (xf @ w["k"].t().float()).view(total, N_KV, HEAD_DIM).transpose(0, 1)  # [N_KV, total, hd]
    v = (xf @ w["v"].t().float()).view(total, N_KV, HEAD_DIM).transpose(0, 1)
    k = apply_rope_hf(k, cos, sin)

    lo = total - tail
    q = (xf[lo:] @ w["q"].t().float()).view(tail, N_Q, HEAD_DIM).transpose(0, 1)  # [N_Q, tail, hd]
    q = apply_rope_hf(q, cos[lo:], sin[lo:])
    del xf

    # Query row i sits at global position lo+i and may not see beyond it.
    mask = torch.arange(total)[None, :] > (lo + torch.arange(tail))[:, None]
    rep = N_Q // N_KV
    out = torch.empty(N_Q, tail, HEAD_DIM)
    for h in range(N_Q):
        s = (q[h] @ k[h // rep].t()) * (HEAD_DIM**-0.5)
        s.masked_fill_(mask, float("-inf"))
        out[h] = torch.softmax(s, dim=-1) @ v[h // rep]
        del s
    o = out.transpose(0, 1).reshape(tail, N_Q * HEAD_DIM)
    return (o @ w["o"].t().float())[None]


@pytest.mark.slow
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("n_chunks", [10], ids=["50k"])
def test_attention_accuracy_at_depth(mesh_device, device_params, n_chunks, reset_seeds):
    """Prefill 51200 tokens in 5120-token chunks; PCC the final chunk against torch."""
    torch.manual_seed(0)
    mesh_config, ccl = mesh_setup(mesh_device)
    sp, tp = mesh_config.sp, mesh_config.tp

    chunk_local = 640  # production chunk: chunk_global = 8 * 640 = 5120
    chunk_global = sp * chunk_local
    total = n_chunks * chunk_global
    cache_global = total

    w = _random_attn_weights(seed=7)
    x = torch.randn(1, total, HIDDEN) * 0.1

    attn = _build_attention(mesh_device, mesh_config, ccl, w, cache_global)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=cache_global,
        sp_axis=mesh_config.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        n_kv_local=per_chip(tp)["n_kv"],
    )
    rope_mats = build_indexed_rope(
        mesh_device,
        head_dim=HEAD_DIM,
        max_seq_len=cache_global,
        chunk_size=chunk_global,
        sp_axis=mesh_config.sp_axis,
        **YARN,
    )

    got = None
    for c in range(n_chunks):
        cached_len = c * chunk_global
        idx, inv = _chunk_order(cached_len, sp, chunk_local)
        xc = x[:, cached_len : cached_len + chunk_global, :].reshape(1, 1, chunk_global, HIDDEN)
        out_tt = attn(
            _place_sp(xc, mesh_device, mesh_config, idx),
            rope_mats=rope_mats,
            kv_cache=kv_cache,
            cached_len=cached_len,
            indexed_rope=True,
        )
        if c == n_chunks - 1:
            got = _gather_sp_tp(out_tt, mesh_device, mesh_config, inv).reshape(1, chunk_global, HIDDEN)

    logger.info(f"device prefill done: {total} tokens in {n_chunks} chunks of {chunk_global}; building reference...")
    ref = _reference_tail(x, w, total, chunk_global)

    _, pcc_all = comp_pcc(ref, got, 0.0)
    logger.info(f"ACCURACY@DEPTH total={total} last-chunk rows {total - chunk_global}..{total - 1}: PCC {pcc_all}")
    # Per-block, to show the decay inside the final chunk itself.
    nb = 8
    bs = chunk_global // nb
    for b in range(nb):
        _, p = comp_pcc(ref[:, b * bs : (b + 1) * bs, :], got[:, b * bs : (b + 1) * bs, :], 0.0)
        lo = total - chunk_global + b * bs
        logger.info(f"ACCURACY@DEPTH   rows {lo}..{lo + bs - 1} (prefix {lo + bs}): {p}")

    assert pcc_all > 0.5, f"catastrophic accuracy loss at depth {total}: {pcc_all}"
