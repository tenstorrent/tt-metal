# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from models.demos.deepseek_v4_flash.cpu_reference import (
    combine_routed_experts,
    compressor_prefill,
    compress_topk_indices,
    hc_split_sinkhorn,
    hyperconnection_post,
    hyperconnection_pre,
    indexer_topk,
    sparse_attention,
    swiglu_expert,
    v4_router,
    window_topk_indices,
)


def test_v4_router_hash_and_sqrtsoftplus_top6():
    x = torch.tensor(
        [
            [1.0, 0.5, -0.5, 2.0],
            [0.0, 1.0, 1.5, -1.0],
            [2.0, -1.0, 0.25, 0.5],
        ]
    )
    gate_weight = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4) / 17.0
    bias = torch.tensor([0.0, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0])
    weights, indices = v4_router(x, gate_weight, topk=6, route_scale=1.5, bias=bias)

    scores = F.softplus(x @ gate_weight.T).sqrt()
    expected_indices = (scores + bias).topk(6, dim=-1).indices
    expected_weights = scores.gather(-1, expected_indices)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True) * 1.5
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights)

    tid2eid = torch.tensor([[0, 2, 4, 6, 1, 3], [1, 3, 5, 7, 0, 2], [7, 6, 5, 4, 3, 2]], dtype=torch.int32)
    hash_weights, hash_indices = v4_router(
        x, gate_weight, topk=6, route_scale=1.5, input_ids=torch.tensor([2, 0, 1]), tid2eid=tid2eid
    )
    torch.testing.assert_close(hash_indices, tid2eid[torch.tensor([2, 0, 1])].long())
    expected_hash_weights = scores.gather(-1, hash_indices)
    expected_hash_weights = expected_hash_weights / expected_hash_weights.sum(dim=-1, keepdim=True) * 1.5
    torch.testing.assert_close(hash_weights, expected_hash_weights)


def test_hyperconnection_split_sinkhorn_pre_post():
    torch.manual_seed(0)
    batch, seq, hc_mult, hidden = 2, 3, 4, 5
    mix_hc = (2 + hc_mult) * hc_mult
    x = torch.randn(batch, seq, hc_mult, hidden, dtype=torch.bfloat16)
    hc_fn = torch.randn(mix_hc, hc_mult * hidden)
    hc_scale = torch.tensor([0.5, 1.25, 0.75])
    hc_base = torch.linspace(-0.2, 0.3, mix_hc)

    y, post, comb = hyperconnection_pre(
        x, hc_fn, hc_scale, hc_base, norm_eps=1e-6, hc_mult=hc_mult, sinkhorn_iters=20, hc_eps=1e-6
    )
    assert y.shape == (batch, seq, hidden)
    assert post.shape == (batch, seq, hc_mult)
    assert comb.shape == (batch, seq, hc_mult, hc_mult)
    torch.testing.assert_close(comb.sum(dim=-1), torch.ones(batch, seq, hc_mult), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(comb.sum(dim=-2), torch.ones(batch, seq, hc_mult), atol=2e-5, rtol=2e-5)

    mixes = torch.randn(batch, seq, mix_hc)
    pre, split_post, split_comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=8)
    assert torch.all(pre > 0)
    assert torch.all((split_post >= 0) & (split_post <= 2))
    assert split_comb.shape == (batch, seq, hc_mult, hc_mult)

    out = hyperconnection_post(y, x, post, comb)
    manual = post.unsqueeze(-1) * y.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * x.unsqueeze(-2), dim=2)
    torch.testing.assert_close(out, manual.to(out.dtype))


def test_compressor_and_topk_indices_prefill_paths():
    x = torch.arange(1 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4) / 10.0
    wkv = torch.eye(4)
    wgate = torch.zeros(4, 4)
    ape = torch.zeros(2, 4)
    norm_weight = torch.ones(4)
    compressed = compressor_prefill(
        x, wkv, wgate, ape, norm_weight, compress_ratio=2, head_dim=4, norm_eps=1e-6, overlap=False
    )
    manual = x.reshape(1, 2, 2, 4).mean(dim=2)
    manual = manual * torch.rsqrt(manual.square().mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(compressed, manual)

    assert window_topk_indices(3, batch_size=1, seq_len=4, start_pos=0).tolist() == [
        [[0, -1, -1], [0, 1, -1], [0, 1, 2], [1, 2, 3]]
    ]
    assert compress_topk_indices(2, batch_size=1, seq_len=4, start_pos=0, offset=3).tolist() == [
        [[-1, -1], [3, -1], [3, -1], [3, 4]]
    ]


def test_indexer_topk_and_sparse_attention_decode():
    q = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    kv = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    weights = torch.tensor([[[1.0, 0.5]]])
    topk = indexer_topk(q, kv, weights, index_topk=2, compress_ratio=1, start_pos=2, offset=5)
    assert topk.tolist() == [[[7, 5]]]

    topk_idxs = torch.tensor([[[0, 2, -1]]], dtype=torch.int64)
    attn_sink = torch.tensor([0.25, -0.5])
    out = sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=1.0)

    manual = torch.zeros_like(out)
    for head in range(q.shape[2]):
        gathered = kv[0, [0, 2]]
        scores = torch.tensor([(q[0, 0, head] * gathered[0]).sum(), (q[0, 0, head] * gathered[1]).sum()])
        probs = torch.softmax(torch.cat([scores, attn_sink[head : head + 1]]), dim=0)[:2]
        manual[0, 0, head] = probs @ gathered
    torch.testing.assert_close(out, manual)


def test_shared_and_routed_expert_debug_weights():
    x = torch.tensor([[[1.0, -1.0], [0.5, 2.0]]])
    w1 = torch.eye(2)
    w3 = torch.tensor([[2.0, 0.0], [0.0, -1.0]])
    w2 = torch.eye(2)
    shared = swiglu_expert(x.reshape(-1, 2), w1, w2, w3).view_as(x)
    manual_hidden = F.silu(x.float()) * F.linear(x.float(), w3)
    torch.testing.assert_close(shared, manual_hidden)

    route_weights = torch.tensor([[[0.25], [0.5]]])
    route_indices = torch.tensor([[[3], [3]]])
    combined = combine_routed_experts(
        x,
        route_weights,
        route_indices,
        {3: (w1, w2, w3)},
        shared_expert=(w1, w2, w3),
    )
    torch.testing.assert_close(combined, shared + shared * route_weights)
