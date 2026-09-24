# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ring-joint SDPA at this model's attention shape (24 q / 4 kv heads, head_dim 256), SP=8 x TP=4.

* nocache: live Q/K/V, the sequence SP-sharded across rows — gathered by online softmax over the ring,
  vs unsharded causal GQA. (pattern: minimax_m3/tests/unit/test_ring_joint_sp_vs_ref.py)
* cache-read: the last chunk's Q against the accumulated block-cyclic bf8 cache prefix — the mechanism
  chunked prefill depends on. (pattern: minimax_m3/tests/unit/test_ring_joint_cache_read_sp_vs_ref.py)
"""

import pytest
import torch

import ttnn
from models.demos.qwen_3_8_27b.config import QWEN38
from models.demos.qwen_3_8_27b.tests.common import assert_pcc, from_sp, to_sp
from models.demos.qwen_3_8_27b.tt.kv_cache import allocate_caches, cache_capacity, write_kv_chunk
from models.demos.qwen_3_8_27b.tt.sdpa import ring_sdpa_cache, ring_sdpa_nocache

NQ, NKV, HD = QWEN38.num_attention_heads, QWEN38.num_key_value_heads, QWEN38.head_dim


def torch_gqa_causal(q, k, v, q_start=0):
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    T, S = q.shape[2], k.shape[2]
    sc = (q @ k.transpose(-1, -2)) * HD**-0.5
    mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=q_start)
    return torch.softmax(sc.masked_fill(~mask, float("-inf")), -1) @ v


@pytest.mark.parametrize("seq", [2048, 10240], ids=["2k", "10k_oneshot"])
def test_ring_joint_sp_nocache(mesh, mesh_config, ccl_manager, seq):
    torch.manual_seed(0)
    q = torch.randn(1, NQ, seq, HD) * 0.5
    k = torch.randn(1, NKV, seq, HD) * 0.5
    v = torch.randn(1, NKV, seq, HD)
    ref = torch_gqa_causal(q, k, v)
    tq, tk, tv = (to_sp(t, mesh_config, seq_dim=2, tp_dim=1) for t in (q, k, v))
    out = ring_sdpa_nocache(
        tq, tk, tv, mesh_device=mesh, ccl=ccl_manager, n_kv=NKV, head_dim=HD, logical_n=seq, scale=HD**-0.5
    )
    assert_pcc(f"ring_nocache_{seq}", from_sp(out, mesh_config, seq_dim=2, tp_dim=1), ref)


@pytest.mark.parametrize("n_chunks,chunk", [(2, 2048), (2, 5120)], ids=["2x2048", "2x5120"])
def test_ring_joint_cache_read_sp(mesh, mesh_config, ccl_manager, n_chunks, chunk):
    """Prior chunks written through update_padded_kv_cache; the last chunk's Q reads the whole prefix.
    Two layers so a wrong layer index reads zeros (layer 0) instead of the data (layer 1)."""
    torch.manual_seed(1)
    S = n_chunks * chunk
    q = torch.randn(1, NQ, S, HD) * 0.5
    k = torch.randn(1, NKV, S, HD) * 0.5
    v = torch.randn(1, NKV, S, HD)
    caches = allocate_caches(
        mesh, num_attn_layers=2, gdn_layers=[], max_seq_len=cache_capacity(16384, [chunk]), head_dim=HD
    )
    layer = 1
    for c in range(n_chunks):
        sl = slice(c * chunk, (c + 1) * chunk)
        write_kv_chunk(
            caches,
            to_sp(k[:, :, sl], mesh_config, tp_dim=1),
            to_sp(v[:, :, sl], mesh_config, tp_dim=1),
            slot_idx=0,
            layer_idx=layer,
            kv_actual=c * chunk,
        )
    last = (n_chunks - 1) * chunk
    tq = to_sp(q[:, :, last:], mesh_config, tp_dim=1)
    out = ring_sdpa_cache(
        tq,
        caches,
        mesh_device=mesh,
        ccl=ccl_manager,
        n_kv=NKV,
        kv_actual=last,
        logical_n=S,
        slot_idx=0,
        layer_idx=layer,
        scale=HD**-0.5,
    )
    ref = torch_gqa_causal(q[:, :, last:], k, v, q_start=last)
    assert_pcc(f"ring_cache_read_{n_chunks}x{chunk}", from_sp(out, mesh_config, seq_dim=2, tp_dim=1), ref)
    ttnn.deallocate(caches.k)
    ttnn.deallocate(caches.v)
