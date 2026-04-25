# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.cpu_reference import (
    combine_routed_experts,
    compress_topk_indices,
    compressor_prefill,
    hc_split_sinkhorn,
    hyperconnection_post,
    hyperconnection_pre,
    indexer_topk,
    sparse_attention,
    swiglu_expert,
    v4_router,
    window_topk_indices,
)
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.ttnn_attention_projection import (
    AttentionProjectionWeights,
    grouped_output_projection_a,
    load_attention_projection_weights,
    validate_attention_projection_weights,
)
from models.demos.deepseek_v4_flash.ttnn_decode_cache import (
    Batch1DecodeCache,
    Batch1DecodeLayerCache,
    advance_batch1_decode_layer_cache,
)
from models.demos.deepseek_v4_flash.ttnn_decoder_layer import (
    DecoderLayerNormWeights,
    load_decoder_layer_norm_weights,
    validate_decoder_layer_decode_step_input,
    validate_decoder_layer_input,
    validate_decoder_layer_norm_weights,
)
from models.demos.deepseek_v4_flash.ttnn_prefill_attention_block import (
    load_attention_sink,
    validate_prefill_attention_block_config,
    validate_prefill_attention_block_input,
)
from models.demos.deepseek_v4_flash.ttnn_prefill_indexer import (
    PrefillIndexerWeights,
    load_prefill_indexer_weights,
    validate_prefill_indexer_config,
    validate_prefill_indexer_input,
)
from models.demos.deepseek_v4_flash.ttnn_router import select_router_scores, validate_router_config
from models.demos.deepseek_v4_flash.ttnn_sparse_attention import validate_sparse_attention_contract


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


def test_ttnn_router_host_scoring_matches_cpu_reference_shapes():
    x = torch.arange(1 * 3 * 4, dtype=torch.float32).reshape(1, 3, 4) / 11.0
    gate_weight = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4) / 13.0
    scores = x @ gate_weight.T
    bias = torch.tensor([0.0, 2.0, 0.0, 1.0, -1.0])

    weights, indices = select_router_scores(scores, topk=2, route_scale=1.25, bias=bias)
    expected_weights, expected_indices = v4_router(x, gate_weight, topk=2, route_scale=1.25, bias=bias)

    assert weights.shape == (1, 3, 2)
    assert indices.shape == (1, 3, 2)
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights)

    tid2eid = torch.tensor([[0, 2], [1, 3], [4, 0]], dtype=torch.int32)
    input_ids = torch.tensor([[2, 0, 1]], dtype=torch.int64)
    hash_weights, hash_indices = select_router_scores(
        scores,
        topk=2,
        route_scale=1.25,
        input_ids=input_ids,
        tid2eid=tid2eid,
    )
    expected_hash_weights, expected_hash_indices = v4_router(
        x,
        gate_weight,
        topk=2,
        route_scale=1.25,
        input_ids=input_ids,
        tid2eid=tid2eid,
    )

    assert hash_weights.shape == (1, 3, 2)
    assert hash_indices.shape == (1, 3, 2)
    torch.testing.assert_close(hash_indices, expected_hash_indices)
    torch.testing.assert_close(hash_weights, expected_hash_weights)


def test_ttnn_router_rejects_bad_api_inputs():
    scores = torch.zeros(1, 3, 4)
    gate_weight = torch.zeros(4, 8)
    tid2eid = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    with pytest.raises(ValueError, match="Unsupported DeepSeek V4 Flash scoring_func"):
        validate_router_config(gate_weight, topk=2, route_scale=1.0, scoring_func="aux_loss")
    with pytest.raises(ValueError, match="bias must have shape"):
        select_router_scores(scores, topk=2, route_scale=1.0, bias=torch.zeros(3))
    with pytest.raises(ValueError, match="input_ids is required"):
        select_router_scores(scores, topk=2, route_scale=1.0, tid2eid=tid2eid)
    with pytest.raises(ValueError, match="input_ids must have shape"):
        select_router_scores(
            scores, topk=2, route_scale=1.0, input_ids=torch.zeros(3, dtype=torch.int64), tid2eid=tid2eid
        )
    with pytest.raises(ValueError, match="input_ids values must be"):
        select_router_scores(
            scores,
            topk=2,
            route_scale=1.0,
            input_ids=torch.full((1, 3), 2, dtype=torch.int64),
            tid2eid=tid2eid,
        )


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


def test_attention_projection_weight_loading_and_grouped_wo_a_host_scaffold(tmp_path):
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=1)
    output = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")
    weights = load_attention_projection_weights(output, layer=0, include_output_projection=True)

    validate_attention_projection_weights(
        weights,
        hidden_size=32,
        q_lora_rank=16,
        num_heads=4,
        head_dim=8,
        o_groups=4,
        o_lora_rank=16,
    )
    assert weights.wq_a.shape == (16, 32)
    assert weights.q_norm.shape == (16,)
    assert weights.wq_b.shape == (32, 16)
    assert weights.wo_a is not None and weights.wo_a.shape == (64, 8)
    assert weights.wo_b is not None and weights.wo_b.shape == (32, 64)

    attention_output = torch.arange(1 * 2 * 32, dtype=torch.float32).reshape(1, 2, 32) / 31
    grouped_rank = grouped_output_projection_a(attention_output, weights.wo_a, o_groups=4)
    manual = []
    for group in range(4):
        group_input = attention_output[:, :, group * 8 : (group + 1) * 8]
        group_weight = weights.wo_a[group * 16 : (group + 1) * 16]
        manual.append(F.linear(group_input.float(), group_weight.float()))
    torch.testing.assert_close(grouped_rank, torch.cat(manual, dim=-1))

    with pytest.raises(ValueError, match="Expected q_norm shape"):
        validate_attention_projection_weights(
            AttentionProjectionWeights(
                wq_a=weights.wq_a,
                q_norm=torch.ones(8),
                wq_b=weights.wq_b,
                wo_a=weights.wo_a,
                wo_b=weights.wo_b,
            ),
            hidden_size=32,
            q_lora_rank=16,
            num_heads=4,
            head_dim=8,
            o_groups=4,
            o_lora_rank=16,
        )


def test_prefill_indexer_weight_loading_and_contract_validation(tmp_path):
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=3)
    output = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")
    weights = load_prefill_indexer_weights(output, layer=2)

    assert weights.wq_b.shape == (32, 16)
    assert weights.weights_proj.shape == (4, 32)
    assert weights.compressor.wkv.shape == (16, 32)
    assert weights.compressor.wgate.shape == (16, 32)
    assert weights.compressor.ape.shape == (4, 16)
    assert weights.compressor.norm_weight.shape == (8,)

    device = object()
    dtype = object()
    memory_config = object()
    projection = SimpleNamespace(
        device=device,
        dtype=dtype,
        memory_config=memory_config,
        hidden_size=32,
        q_lora_rank=16,
    )
    validate_prefill_indexer_config(
        attention_projection=projection,
        weights=weights,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=8,
        compress_ratio=4,
        overlap=True,
        dtype=dtype,
        memory_config=memory_config,
    )

    hidden_states = torch.zeros(1, 1, 8, 32)
    q_rank = torch.zeros(1, 1, 8, 16)
    validate_prefill_indexer_input(
        hidden_states,
        q_rank=q_rank,
        hidden_size=32,
        q_lora_rank=16,
        compress_ratio=4,
    )
    with pytest.raises(ValueError, match="offset must be 0"):
        validate_prefill_indexer_input(
            hidden_states,
            hidden_size=32,
            q_lora_rank=16,
            compress_ratio=4,
            offset=8,
        )
    with pytest.raises(ValueError, match="Expected indexer wq_b shape"):
        validate_prefill_indexer_config(
            attention_projection=projection,
            weights=PrefillIndexerWeights(
                wq_b=weights.wq_b[:8],
                weights_proj=weights.weights_proj,
                compressor=weights.compressor,
            ),
            index_n_heads=4,
            index_head_dim=8,
            index_topk=8,
            compress_ratio=4,
            overlap=True,
            dtype=dtype,
            memory_config=memory_config,
        )


def test_prefill_attention_block_contract_validation(tmp_path):
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=3)
    output = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")
    attn_sink = load_attention_sink(output, layer=2)
    torch.testing.assert_close(attn_sink, torch.zeros(4))

    hidden_states = torch.zeros(1, 1, 8, 32)
    topk_idxs = torch.zeros(1, 8, 2, dtype=torch.int64)
    validate_prefill_attention_block_input(
        hidden_states,
        topk_idxs,
        hidden_size=32,
        compress_ratio=4,
    )
    validate_prefill_attention_block_input(hidden_states, None, hidden_size=32, compress_ratio=4)
    with pytest.raises(ValueError, match="Expected topk_idxs batch/seq"):
        validate_prefill_attention_block_input(
            hidden_states,
            torch.zeros(1, 7, 2, dtype=torch.int64),
            hidden_size=32,
            compress_ratio=4,
        )

    device = object()
    dtype = object()
    memory_config = object()
    projection = SimpleNamespace(
        device=device,
        dtype=dtype,
        memory_config=memory_config,
        num_heads=4,
        head_dim=8,
        hidden_size=32,
        q_lora_rank=16,
    )
    compressor = SimpleNamespace(device=device, dtype=dtype, memory_config=memory_config, head_dim=8, compress_ratio=4)
    indexer = SimpleNamespace(
        device=device,
        dtype=dtype,
        memory_config=memory_config,
        hidden_size=32,
        q_lora_rank=16,
        compress_ratio=4,
    )
    sparse_attention = SimpleNamespace(
        device=device,
        dtype=dtype,
        memory_config=memory_config,
        num_heads=4,
        head_dim=8,
    )
    validate_prefill_attention_block_config(
        attention_projection=projection,
        compressor=compressor,
        indexer=indexer,
        sparse_attention=sparse_attention,
        attn_sink=attn_sink,
    )
    bad_compressor = SimpleNamespace(
        device=device, dtype=dtype, memory_config=memory_config, head_dim=4, compress_ratio=4
    )
    with pytest.raises(ValueError, match="compressor head_dim"):
        validate_prefill_attention_block_config(
            attention_projection=projection,
            compressor=bad_compressor,
            indexer=indexer,
            sparse_attention=sparse_attention,
            attn_sink=attn_sink,
        )


def test_decoder_layer_norm_loading_and_contract_validation(tmp_path):
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=3)
    output = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")
    weights = load_decoder_layer_norm_weights(output, layer=2)

    assert weights.attn_norm.shape == (32,)
    assert weights.ffn_norm.shape == (32,)
    torch.testing.assert_close(weights.attn_norm, torch.ones(32))
    torch.testing.assert_close(weights.ffn_norm, torch.ones(32))
    validate_decoder_layer_norm_weights(weights, hidden_size=32)

    hidden_states = torch.zeros(1, 1, 8, 32)
    input_ids = torch.zeros(1, 8, dtype=torch.int64)
    topk_idxs = torch.zeros(1, 8, 2, dtype=torch.int64)
    validate_decoder_layer_input(
        hidden_states,
        input_ids=input_ids,
        topk_idxs=topk_idxs,
        hidden_size=32,
        compress_ratio=4,
    )

    with pytest.raises(ValueError, match="Expected attn_norm shape"):
        validate_decoder_layer_norm_weights(
            DecoderLayerNormWeights(attn_norm=torch.ones(16), ffn_norm=weights.ffn_norm),
            hidden_size=32,
        )
    with pytest.raises(ValueError, match="hidden_states must have shape"):
        validate_decoder_layer_input(torch.zeros(2, 1, 8, 32), hidden_size=32, compress_ratio=4)
    with pytest.raises(ValueError, match="hidden_states tokens"):
        validate_decoder_layer_input(torch.zeros(1, 1, 3, 32), hidden_size=32, compress_ratio=4)
    with pytest.raises(ValueError, match="input_ids must have shape"):
        validate_decoder_layer_input(
            hidden_states,
            input_ids=torch.zeros(8, dtype=torch.int64),
            hidden_size=32,
            compress_ratio=4,
        )
    with pytest.raises(ValueError, match="topk_idxs batch/tokens"):
        validate_decoder_layer_input(
            hidden_states,
            topk_idxs=torch.zeros(1, 7, 2, dtype=torch.int64),
            hidden_size=32,
            compress_ratio=4,
        )


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


def test_batch1_decode_cache_shape_contract_and_transition():
    cache = Batch1DecodeLayerCache(
        layer_id=2,
        current_position=8,
        batch_size=1,
        hidden_size=32,
        compress_ratio=4,
        head_dim=8,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=8,
        attention_input_history=torch.zeros(1, 8, 32, dtype=torch.bfloat16),
        compressed_kv=torch.zeros(1, 2, 8, dtype=torch.bfloat16),
        index_compressed_kv=torch.zeros(1, 2, 8, dtype=torch.bfloat16),
    )
    model_cache = Batch1DecodeCache(layer_caches=(cache,))

    assert model_cache.batch_size == 1
    assert model_cache.current_position == 8
    assert model_cache.layer_ids == (2,)
    assert cache.compressed_cache_length == 2

    next_cache = advance_batch1_decode_layer_cache(
        cache,
        attention_input_token=torch.ones(1, 1, 32, dtype=torch.bfloat16),
        compressed_kv=torch.zeros(1, 2, 8, dtype=torch.bfloat16),
        index_compressed_kv=torch.zeros(1, 2, 8, dtype=torch.bfloat16),
        last_topk_idxs=torch.tensor([[[1, 0]]], dtype=torch.int64),
    )

    assert next_cache.current_position == 9
    assert next_cache.attention_input_history.shape == (1, 9, 32)
    assert next_cache.compressed_kv.shape == (1, 2, 8)
    torch.testing.assert_close(next_cache.attention_input_history[:, -1], torch.ones(1, 32, dtype=torch.bfloat16))
    torch.testing.assert_close(next_cache.last_topk_idxs, torch.tensor([[[1, 0]]], dtype=torch.int64))

    with pytest.raises(ValueError, match="compressed_kv must have shape"):
        Batch1DecodeLayerCache(
            layer_id=2,
            current_position=8,
            batch_size=1,
            hidden_size=32,
            compress_ratio=4,
            head_dim=8,
            index_n_heads=4,
            index_head_dim=8,
            index_topk=8,
            attention_input_history=torch.zeros(1, 8, 32),
            compressed_kv=torch.zeros(1, 3, 8),
            index_compressed_kv=torch.zeros(1, 2, 8),
        )
    with pytest.raises(ValueError, match="same current_position"):
        Batch1DecodeCache(
            layer_caches=(
                cache,
                Batch1DecodeLayerCache(
                    layer_id=3,
                    current_position=9,
                    batch_size=1,
                    hidden_size=32,
                    compress_ratio=4,
                    head_dim=8,
                    index_n_heads=4,
                    index_head_dim=8,
                    index_topk=8,
                    attention_input_history=torch.zeros(1, 9, 32),
                    compressed_kv=torch.zeros(1, 2, 8),
                    index_compressed_kv=torch.zeros(1, 2, 8),
                ),
            )
        )
    with pytest.raises(ValueError, match="attention_input_token must have shape"):
        advance_batch1_decode_layer_cache(
            cache,
            attention_input_token=torch.ones(1, 2, 32),
            compressed_kv=torch.zeros(1, 2, 8),
            index_compressed_kv=torch.zeros(1, 2, 8),
            last_topk_idxs=torch.zeros(1, 1, 1, dtype=torch.int64),
        )


def test_decoder_layer_decode_step_contract_validation():
    hidden_states = torch.zeros(1, 1, 1, 32)
    input_ids = torch.zeros(1, 1, dtype=torch.int64)
    cache = Batch1DecodeLayerCache(
        layer_id=2,
        current_position=8,
        batch_size=1,
        hidden_size=32,
        compress_ratio=4,
        head_dim=8,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=8,
        attention_input_history=torch.zeros(1, 8, 32),
        compressed_kv=torch.zeros(1, 2, 8),
        index_compressed_kv=torch.zeros(1, 2, 8),
    )

    validate_decoder_layer_decode_step_input(
        hidden_states,
        input_ids=input_ids,
        cache=cache,
        hidden_size=32,
        layer=2,
    )

    with pytest.raises(ValueError, match="decode hidden_states must have shape"):
        validate_decoder_layer_decode_step_input(
            torch.zeros(1, 1, 2, 32),
            input_ids=input_ids,
            cache=cache,
            hidden_size=32,
            layer=2,
        )
    with pytest.raises(ValueError, match="decode input_ids must have shape"):
        validate_decoder_layer_decode_step_input(
            hidden_states,
            input_ids=torch.zeros(1, 2, dtype=torch.int64),
            cache=cache,
            hidden_size=32,
            layer=2,
        )
    with pytest.raises(ValueError, match="decode cache layer_id"):
        validate_decoder_layer_decode_step_input(
            hidden_states,
            input_ids=input_ids,
            cache=cache,
            hidden_size=32,
            layer=1,
        )


def test_sparse_attention_contract_validation_errors():
    q = torch.zeros(1, 2, 3, 4)
    kv = torch.zeros(1, 5, 4)
    attn_sink = torch.zeros(3)
    topk_idxs = torch.tensor([[[0, 1], [2, -1]]], dtype=torch.int64)

    validate_sparse_attention_contract(q, kv, attn_sink, topk_idxs, num_heads=3, head_dim=4)

    with pytest.raises(ValueError, match="Expected q heads/dim"):
        validate_sparse_attention_contract(q, kv, attn_sink, topk_idxs, num_heads=2, head_dim=4)
    with pytest.raises(ValueError, match="topk_idxs values must be < cache_len 5"):
        validate_sparse_attention_contract(q, kv, attn_sink, torch.full((1, 2, 1), 5), num_heads=3, head_dim=4)
    with pytest.raises(ValueError, match="topk_idxs values must be -1 or non-negative"):
        validate_sparse_attention_contract(q, kv, attn_sink, torch.full((1, 2, 1), -2), num_heads=3, head_dim=4)


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
