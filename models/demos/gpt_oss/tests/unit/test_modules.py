# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.tt_transformers.tt.common import gather_cos_sin, precompute_freqs, rope_scaling_model_factory
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format
from models.tt_transformers.tt.rope import RotarySetup

from ...tt.layer import DecoderLayer
from ...utils.general_utils import throughput_experts_supported_on_arch
from ..test_factory import TestFactory, compare_tensors, parametrize_batch_seq, parametrize_mesh_with_fabric


# Helper Functions for Common Test Patterns
def run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.99):
    """Standard component output comparison"""
    tt_output_tensors = ttnn.get_device_tensors(tt_output)

    passing_final = True
    for i in range(len(tt_output_tensors)):
        tt_output_torch = ttnn.to_torch(tt_output_tensors[i])
        passing, output = compare_tensors(tt_output_torch, reference_output, mesh_device, pcc_threshold=pcc_threshold)
        passing_final = passing_final and passing
    if passing_final:
        return True, output
    else:
        return False, output


def run_attention_component(
    mesh_device,
    hidden_shape,
    mask,
    position_embeddings,
    rope_mats,
    tt_position_idx,
    reference_layer,
    decoder_layer,
    is_decode,
    is_row_sharded,
    pcc_threshold,
    page_table=None,
):
    """Test attention component - extracted from decoder layer.

    When ``page_table`` is provided, attention runs through the paged kv-cache
    code path; the decoder_layer must have been constructed with a matching
    ``paged_attention_config``.
    """

    # Create input
    batch_size, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    # Convert to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )

    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(1, 1, -1, hidden_states.shape[-1]),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    reference_attention = reference_layer.self_attn

    # Reference attention forward
    with torch.no_grad():
        reference_out, _ = reference_attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            use_cache=True,
        )

    # TTNN attention forward (no mask needed, causal masking handled internally)
    attention_module = decoder_layer.self_attn
    tt_out = attention_module(
        tt_hidden_states,
        rope_mats=rope_mats,
        position_idx=tt_position_idx,
        page_table=page_table,
        kv_cache=None,
        is_decode=is_decode,
    )

    # Compare outputs
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=mesh_composer)[..., : batch_size * seq_len, :hidden_size]

    # Compare outputs
    passing, output = compare_tensors(tt_output_torch, reference_out, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"Attention test passed. Output: {output}")
    else:
        assert passing, f"Attention test failed. Output: {output}"


def run_rms_norm_component(
    mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """Test RMSNorm component - extracted from decoder layer"""

    # Create input
    batch_size, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    # Extract reference RMSNorm from reference layer
    reference_rms_norm = reference_layer.input_layernorm

    # Reference RMSNorm forward
    with torch.no_grad():
        ref_output = reference_rms_norm(hidden_states)

    # Convert to TTNN tensors
    # tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    # TTNN RMSNorm forward
    rms_norm_module = decoder_layer.input_layernorm
    tt_output = rms_norm_module(tt_hidden_states)

    # Compare outputs
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[:batch_size, :seq_len, :]
    # Compare outputs
    passing, output = compare_tensors(tt_output_torch, ref_output, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"RMS Norm test passed. Output: {output}")
    else:
        assert passing, f"RMS Norm test failed. Output: {output}"


def run_topk_router_component(
    mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """Test TopK router component - extracted from decoder layer"""

    # Create input
    batch, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    # Extract reference TopK router from reference layer
    reference_router = reference_layer.mlp.router
    # transformers 5.x GptOssTopKRouter.forward returns (router_logits, router_scores, router_indices) with
    # router_scores already SPARSE [num_tokens, top_k] (softmax over the top_k logits, in top-k order);
    # <5 returned (router_scores_dense, router_indices) with router_scores DENSE [num_tokens, num_experts].
    # GptOssMLP flattens (batch,seq,hidden) -> (num_tokens, hidden) before the router, so flatten here too
    # (also matches the TT router, which operates on flattened tokens) and normalise both versions to a
    # sparse [num_tokens, top_k] weight tensor aligned to router_indices.
    _router_out = reference_router(hidden_states.reshape(-1, hidden_size))
    if len(_router_out) == 3:
        router_scores, router_indices = _router_out[1], _router_out[2]
    else:
        router_scores_dense, router_indices = _router_out
        router_scores = torch.gather(router_scores_dense, 1, router_indices)

    # Convert to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )

    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(1, 1, -1, 2880),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    # Extract TT TopK router from decoder layer
    tt_router = decoder_layer.mlp.router
    tt_router_indices, tt_router_weights = tt_router(tt_hidden_states, decoder_layer.mlp.use_throughput_experts)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape))
    top_k = router_indices.shape[1]
    tt_router_indices_torch = ttnn.to_torch(tt_router_indices, mesh_composer=mesh_composer)[:batch, :top_k]
    tt_router_weights_full = ttnn.to_torch(tt_router_weights, mesh_composer=mesh_composer)[:batch]
    if decoder_layer.mlp.use_throughput_experts:
        # throughput path returns sparse [batch, top_k] weights (top-k order)
        tt_router_weights_torch = tt_router_weights_full[:, :top_k]
    else:
        # non-throughput path returns DENSE [batch, num_experts] (ttnn.scatter of the top_k weights at the
        # selected expert ids); gather the weights at the selected indices so both sides are sparse
        # [batch, top_k] aligned to their indices. Reading [:, :top_k] here would grab unselected experts.
        tt_router_weights_torch = torch.gather(tt_router_weights_full, 1, tt_router_indices_torch.long())

    # Compare outputs
    # We will sort the indices here as the order of the indices is not guaranteed to be the same in the reference and TT implementation.
    sorted_tt_indices, sorted_tt_indices_order = torch.sort(tt_router_indices_torch, dim=-1)
    sorted_ref_indices, sorted_ref_indices_order = torch.sort(router_indices, dim=-1)
    indices_passing, indices_output = compare_tensors(
        sorted_tt_indices, sorted_ref_indices, mesh_device, pcc_threshold=pcc_threshold
    )
    # Reorder each token's weights into ascending-expert-id order so the two sides line up even when
    # TT (bf16) and the reference (fp32) emit the same top-k experts in a different (value-sorted) order.
    # gather along the top_k axis is the correct reorder; `weights.squeeze()[order]` indexes dim 0 and
    # mangles the comparison for batch > 1.
    weights_passing, weights_output = compare_tensors(
        torch.gather(tt_router_weights_torch, -1, sorted_tt_indices_order),
        torch.gather(router_scores, -1, sorted_ref_indices_order),
        mesh_device,
        pcc_threshold=pcc_threshold,
    )
    if not (indices_passing and weights_passing):
        assert (
            False
        ), f"\nTopK Router test (indices) {indices_passing}. Output: {indices_output}\nTopK Router test (weights) {weights_passing}. Output: {weights_output}"
    else:
        logger.info(f"TopK Router indices test passed. Output: {indices_output}")
        logger.info(f"TopK Router weights test passed. Output: {weights_output}")


def run_throughput_experts_component(
    mesh_device, hidden_shape, config, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """Test experts component - extracted from decoder layer"""

    # Create input
    _, _, num_tokens, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    router_indices = torch.zeros(num_tokens, config.num_experts_per_tok, dtype=torch.long)
    routing_weights = torch.zeros(num_tokens, config.num_local_experts)

    for t in range(num_tokens):
        active_experts = torch.randperm(config.num_local_experts)[: config.num_experts_per_tok]
        router_indices[..., t, :] = active_experts
        weights = torch.rand(config.num_experts_per_tok)
        weights = weights / weights.sum()  # Normalize
        routing_weights[..., t, active_experts] = weights
    topk_weights_dense = torch.tensor(
        [[routing_weights[..., i, j].item() for j in b] for i, b in enumerate(router_indices.squeeze())]
    )
    # Extract reference experts from reference layer
    reference_experts = reference_layer.mlp.experts.eval()  # Set to eval mode for inference
    # transformers 5.x GptOssExperts.forward indexes `hidden_states[token_idx]` over a flattened
    # [num_tokens, hidden] token axis and `routing_weights[token_idx, top_k_pos]` over a by-position
    # [num_tokens, top_k] layout. Passing the unflattened 4-D `hidden_states` ([1,1,num_tokens,hidden])
    # or the dense [num_tokens, num_experts] `routing_weights` raises
    # "IndexError: index N out of bounds for dimension 0 with size 1" at modeling_gpt_oss.py. Flatten the
    # hidden states and pass the by-position weights (`topk_weights_dense`), matching run_experts_component.
    reference_output = reference_experts(
        hidden_states.reshape(-1, hidden_size),
        router_indices=router_indices.squeeze(),
        routing_weights=topk_weights_dense,
    )

    # Convert to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_routing_weights = ttnn.from_torch(
        topk_weights_dense,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_mapper,
    )
    tt_router_indices = ttnn.from_torch(
        router_indices,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint16,
        mesh_mapper=mesh_mapper,
    )

    # Extract TT experts from decoder layer
    tt_experts = decoder_layer.mlp.experts
    tt_output = tt_experts(
        hidden_states=tt_hidden_states,
        topk_expert_indices=tt_router_indices,
        topk_expert_weights=tt_routing_weights,
        is_decode=is_decode,
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., :num_tokens, :hidden_size]
    # Compare outputs
    passing, output = compare_tensors(tt_output, reference_output, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"High Throughput Experts test passed. Output: {output}")
    else:
        assert passing, f"High Throughput Experts test failed. Output: {output}"


def run_fused_throughput_experts_component(
    mesh_device, hidden_shape, config, reference_layer, decoder_layer, is_row_sharded
):
    """Test fused experts: all_to_all_dispatch_metadata → moe_gpt → selective_reduce_combine.

    Uses global expert IDs (0..num_experts-1). Dispatch (cluster_axis=0, column rings of
    4 devices) silently skips experts not on the local column ring; moe_gpt processes only
    the tokens it receives. routing_weight_map=None → unweighted sum over processed experts.
    """
    from models.demos.gpt_oss.tt.experts_throughput import create_fused_moe_gpt_config, fused_decode_forward
    from models.demos.gpt_oss.utils.general_utils import get_default_num_links

    _, _, num_tokens, hidden_size = hidden_shape

    cluster_axis = 0
    tokens_per_device = num_tokens // mesh_device.shape[cluster_axis]  # e.g., 128 // 4 = 32

    # Extract TT experts from decoder layer
    tt_experts = decoder_layer.mlp.experts
    tt_config = tt_experts.config

    ref_state = reference_layer.state_dict()
    fused_state_dict = {
        "gate_up_proj": ref_state["mlp.experts.gate_up_proj"],  # [E, hidden_size, 2*intermediate_size]
        "down_proj": ref_state["mlp.experts.down_proj"],  # [E, intermediate_size, hidden_size]
    }

    fused_config = create_fused_moe_gpt_config(
        mesh_device=mesh_device,
        config=tt_config,
        state_dict=fused_state_dict,
        tokens_per_device=tokens_per_device,
        weight_dtype=ttnn.bfloat4_b,
        cluster_axis=cluster_axis,
        num_links=get_default_num_links(mesh_device),
    )

    # Create routing with global expert IDs (0..num_experts-1).
    # Each column ring only processes experts on devices in that ring; the all_reduce
    # across columns (cluster_axis=1) combines all partial results.
    num_experts_per_tok = tt_config.num_experts_per_tok
    total_experts = config.num_local_experts  # 128
    indices_list = []
    scores_list = []

    for _ in range(num_tokens):
        selected = torch.randperm(total_experts)[:num_experts_per_tok].sort().values
        indices_list.append(selected.to(torch.int64))
        scores = torch.rand(num_experts_per_tok, dtype=torch.float32) + 1e-5
        scores = scores / scores.sum()
        scores_list.append(scores)
    indices_torch = torch.stack(indices_list, dim=0).reshape(num_tokens, 1, 1, num_experts_per_tok)
    scores_torch = torch.stack(scores_list, dim=0).reshape(num_tokens, 1, 1, num_experts_per_tok)

    # Create hidden states - tokens on dim 0 matching e2e test format
    hidden_states_torch = torch.randn(num_tokens, 1, 1, hidden_size, dtype=torch.float32)

    # Zero out bias in reference model since moe_gpt kernel doesn't add bias.
    reference_experts = reference_layer.mlp.experts.eval()
    with torch.no_grad():
        reference_experts.gate_up_proj_bias.zero_()
        reference_experts.down_proj_bias.zero_()
        # transformers 5.x GptOssExperts.forward indexes hidden_states[token_idx] over a flattened
        # [num_tokens, hidden] axis and routing_weights[token_idx, top_k_pos] over a by-position
        # [num_tokens, top_k] layout. Pass the flattened hidden states and the by-position scores
        # (aligned to indices_torch) instead of the 4-D tensor + dense [num_tokens, num_experts] weights,
        # which raised "IndexError: index N out of bounds for dimension 0 with size 1" at modeling_gpt_oss.py.
        reference_output = reference_experts(
            hidden_states_torch.reshape(-1, hidden_size),
            router_indices=indices_torch.squeeze(),
            routing_weights=scores_torch.reshape(num_tokens, num_experts_per_tok),
        )

    # Upload to device: shard tokens (dim 0) across mesh rows, replicate across cols
    mesh_mapper_tokens = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape))

    tt_hidden = ttnn.from_torch(
        hidden_states_torch.bfloat16(),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper_tokens,
    )

    # Use DRAM INTERLEAVED for indices/scores (matches real router output format).
    # fused_decode_forward handles the conversion to the format dispatch needs.
    tt_indices = ttnn.from_torch(
        indices_torch.reshape(num_tokens, 1, 1, num_experts_per_tok).to(torch.int16),
        dtype=ttnn.uint16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper_tokens,
    )
    tt_scores = ttnn.from_torch(
        scores_torch.reshape(num_tokens, 1, 1, num_experts_per_tok),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper_tokens,
    )

    try:
        tt_output = fused_decode_forward(
            hidden_states=tt_hidden,
            topk_expert_indices=tt_indices,
            topk_expert_scores=tt_scores,
            config=tt_config,
            fused_config=fused_config,
            mesh_device=mesh_device,
        )

        # After all_reduce across cols, all col devices in the same row have identical
        # output [1, 1, M, H]. Pick col=0 from each row and concat tokens along dim -2.
        mesh_rows, mesh_cols = mesh_device.shape
        dev_tensors = ttnn.get_device_tensors(tt_output)
        per_row = [ttnn.to_torch(dev_tensors[r * mesh_cols]) for r in range(mesh_rows)]
        tt_output_torch = torch.cat(per_row, dim=-2)[..., :hidden_size]

        assert not torch.isnan(tt_output_torch).any(), "NaN detected in fused expert output"
        assert not torch.isinf(tt_output_torch).any(), "Inf detected in fused expert output"

        # reference_output: [num_tokens, hidden_size]
        tt_flat = tt_output_torch.reshape(-1, hidden_size)[:num_tokens].float()
        ref_flat = reference_output.reshape(-1, hidden_size)[:num_tokens].float()

        passing, pcc_str = compare_tensors(tt_flat, ref_flat, mesh_device, pcc_threshold=0.98)
        logger.info(
            f"Fused throughput experts PCC (global expert IDs): {pcc_str}. "
            f"Output range: [{tt_flat.min():.4f}, {tt_flat.max():.4f}]."
        )
    finally:
        for attr in [
            "dispatch_sparse",
            "dispatch_indices",
            "dispatch_scores",
            "combine_preallocated",
            "tt_dispatch_mapping",
            "tt_moe_gpt_mapping",
            "tt_w0_w1",
            "tt_w2",
        ]:
            tensor = getattr(fused_config, attr, None)
            if tensor is not None:
                try:
                    ttnn.deallocate(tensor)
                except Exception as e:
                    logger.debug(f"Failed to deallocate {attr}: {e}")
        ttnn.synchronize_device(mesh_device)


def run_experts_component(mesh_device, hidden_shape, config, reference_layer, decoder_layer, is_decode, pcc_threshold):
    """Test experts component - extracted from decoder layer"""

    # Create input
    batch_size, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)
    import itertools

    router_indices = torch.zeros(batch_size * seq_len, config.num_experts_per_tok, dtype=torch.long)
    routing_weights = torch.zeros(batch_size * seq_len, config.num_local_experts)
    # transformers 5.x GptOssExperts indexes routing_weights[token_idx, top_k_pos] — i.e. it expects a
    # by-position [num_tokens, top_k] layout, not the dense [num_tokens, num_experts] by-expert layout.
    # Build both: the dense one for the TT experts (unchanged), the by-position one for the 5.x reference.
    routing_weights_topk = torch.zeros(batch_size * seq_len, config.num_experts_per_tok)

    for b, s in itertools.product(range(batch_size), range(seq_len)):
        active_experts = torch.randperm(config.num_local_experts)[: config.num_experts_per_tok]
        router_indices[b * seq_len + s, :] = active_experts
        weights = torch.rand(config.num_experts_per_tok)
        weights = weights / weights.sum()  # Normalize
        routing_weights[b * seq_len + s, active_experts] = weights  # dense, by expert id (TT)
        routing_weights_topk[b * seq_len + s, :] = weights  # by top-k position (5.x reference)

    # Extract reference experts from reference layer
    reference_experts = reference_layer.mlp.experts.eval()  # Set to eval mode for inference
    # transformers 5.x GptOssExperts expects flattened [num_tokens, hidden] (GptOssMLP reshapes
    # (batch,seq,hidden)->(-1,hidden) before calling experts), and indexes hidden_states[token_idx] and
    # routing_weights[token_idx, top_k_pos]. Flatten hidden_states and pass by-position routing weights;
    # output is then [batch*seq, hidden], matching the flat tt_output below. For transformers <5 the
    # reference used the dense by-expert layout — GPT-OSS is now unpinned to 5.x so we target 5.x.
    reference_output = reference_experts(
        hidden_states.reshape(-1, hidden_size), router_indices=router_indices, routing_weights=routing_weights_topk
    )

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(0),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
    )
    tt_routing_weights = ttnn.from_torch(
        routing_weights,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
    )

    # Extract TT experts from decoder layer
    tt_experts = decoder_layer.mlp.experts
    tt_output = tt_experts(
        hidden_states=tt_hidden_states,
        topk_expert_weights=tt_routing_weights,
        is_decode=is_decode,
    )
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., : batch_size * seq_len, :hidden_size]
    # Compare outputs
    passing, output = compare_tensors(tt_output, reference_output, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"Low Latency Experts test passed. Output: {output}")
    else:
        assert passing, f"Low Latency Experts test failed. Output: {output}"


def run_full_mlp_pipeline(
    mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """Test complete MLP (router + experts) - essential MoE functionality"""

    # Create input
    batch, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    reference_model = reference_layer.mlp
    reference_output, routing_scores = reference_model(hidden_states)

    # Convert to TTNN tensors
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        # hidden_states.reshape(-1, 1, 2880),
        hidden_states.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
    )

    # Create TT MLP using TestFactory setup
    tt_mlp = decoder_layer.mlp
    tt_output = tt_mlp(tt_hidden_states, is_decode=is_decode)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., : batch * seq_len, :hidden_size]

    # Compare outputs
    passing, output = compare_tensors(tt_output_torch, reference_output, mesh_device, pcc_threshold=pcc_threshold)
    if passing:
        logger.info(f"MLP Pipeline test passed. Output: {output}")
    else:
        assert passing, f"MLP Pipeline test failed. Output: {output}"


def setup_reference_layer(setup, layer_idx=0):
    logger.info("Setting up reference layer...")
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer

    # transformers 5.x dispatches MoE experts via config._experts_implementation; a standalone layer
    # leaves it None, so GptOssExperts uses the default flat-token forward that indexes
    # hidden_states[token_idx] and IndexErrors on batched hidden_states. Pin it to eager (mirrors the
    # existing _attn_implementation="eager" the decoder setup uses).
    setup["config"]._experts_implementation = "eager"
    reference_layer = GptOssDecoderLayer(setup["config"], layer_idx=layer_idx)
    # Initialize random weights at the model's real scale (config.initializer_range, 0.02 for GPT-OSS)
    # rather than the unit-normal default. The router gate is the reason this matters: at std=1 the
    # router logits reach ~±150 (logit_std ≈ sqrt(hidden_size) ≈ 54), a magnitude where the bf16-only
    # ttnn.topk ties adjacent top experts and the gate softmax collapses to ~0.5/0.5 — a pure artifact
    # of the unrealistic weight scale, not a real-model issue. At the real scale logits are ~±3, well
    # within bf16 resolution, so the gate matches the fp32 reference. (See #47970.)
    init_std = getattr(setup["config"], "initializer_range", 0.02)
    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if any(proj in name for proj in ["router", "experts", "sinks"]):
                param.data.normal_(0, init_std)
    return reference_layer


def setup_decoder_layer(setup, reference_layer, local_batch_size, seq_len, layer_idx=0, paged_attention_config=None):
    logger.info("Setting up TT decoder layer...")
    reference_state = reference_layer.state_dict()
    config = setup["config"]
    # Convert HF QKV weights to Meta format for RoPE compatibility
    reference_state_swizzled = convert_hf_qkv_to_meta_format(reference_state, config.head_dim)
    max_seq_len = getattr(config, "max_position_embeddings", 131072)
    rope_scaling = rope_scaling_model_factory(config.rope_scaling)
    rope_theta = getattr(config, "rope_theta", None) or getattr(config, "default_theta", 10000.0)
    rope_setup = RotarySetup(
        device=setup["mesh_device"],
        batch_size=1,
        head_dim=config.head_dim,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        datatype=ttnn.bfloat16,
    )
    transformation_mats = rope_setup.get_both_trans_mats()
    decoder_layer = DecoderLayer(
        setup["mesh_device"],
        config,
        reference_state_swizzled,
        layer_idx=layer_idx,
        ccl_manager=setup["ccl_manager"],
        dtype=setup["dtype"],
        mesh_config=setup["mesh_config"],
        transformation_mats=transformation_mats,
        max_seq_len=max(seq_len, 128),
        max_local_batch_size=local_batch_size,
        paged_attention_config=paged_attention_config,
        # Mirror production `create_tt_model` (models/demos/gpt_oss/demo/text_demo.py):
        # throughput experts is gated on global_batch_size > 1, not tokens-per-step > 1.
        # Single-user prefill on a multi-row mesh runs the low-throughput path in
        # production (multi-user prefill is staged through `batched_prefill` with one
        # user per row). The DeepSeek prefill kernels assume each row contributes a
        # disjoint slice of tokens; feeding them a single user's seq either replicated
        # across rows (8× duplication of all-reduced expert outputs) or sharded across
        # rows (correct for experts but breaks the layer's self-attention, which needs
        # the full seq per chip) gives a fundamentally inconsistent test setup. Gate
        # on `local_batch_size > 1` so the throughput path is only exercised in the
        # batched configurations it's actually designed for.
        use_throughput_experts=setup["mesh_device"].shape[0] > 1
        and local_batch_size > 1
        and throughput_experts_supported_on_arch(),
    )
    return decoder_layer


@parametrize_mesh_with_fabric([(1, 1), (1, 8), (4, 8)])
@parametrize_batch_seq(
    [
        (1, 1),  # decode
        (128, 1),  # decode
        (1, 128),  # prefill
        (1, 1024),  # prefill 1k
        (1, 4096),  # prefill 4k
    ],
    ids=[
        "decode_low_latency",
        "decode_high_throughput",
        "prefill_128",
        "prefill_1024",
        "prefill_4096",
    ],
)
@pytest.mark.parametrize(
    # We want to test the first two layers so we capture both sliding and global attention layers
    "layer_idx",
    [0],
    ids=[
        "layer_0",
    ],
)
@pytest.mark.parametrize(
    # Cover both the legacy non-paged kv-cache path and the paged path that vLLM
    # and the hybrid kv-cache-groups manager exercise. The paged path was a real
    # test gap — it goes through paged_fill_cache / paged_update_cache /
    # paged_scaled_dot_product_attention_decode, none of which the non-paged
    # path touches.
    "paged",
    [False, True],
    ids=["unpaged", "paged"],
)
def test_decoder(
    mesh_device, device_params, batch_size, seq_len, layer_idx, paged, test_modules, test_thresholds, reset_seeds
):
    """
    Test decoder layer components.

    Args:
        test_modules: Which modules to test (from --test-modules flag). Options:
            - "all": Test all components (default)
            - "attention": Test attention only
            - "rms_norm": Test RMS normalization only
            - "router": Test TopK router only (decode mode only)
            - "mlp": Test full MLP pipeline (router + experts)
            - "decoder": Test full decoder layer only
            - Comma-separated: "attention,mlp" or "router,experts" etc.

    Usage:
        pytest test_modules.py  # runs all tests
        pytest test_modules.py --test-modules=attention
        pytest test_modules.py --test-modules=attention,mlp
    """
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] == 1 and batch_size > 1:
        pytest.skip(
            f"Skipping batch size {batch_size} for mesh shape {tuple(mesh_device.shape)}. "
            "Only batch size 1 is supported for mesh shape without row-sharding."
        )

    if is_blackhole() and mesh_device.shape[0] > 1 and batch_size * seq_len > 1:
        pytest.skip(
            f"Skipping batch={batch_size} seq_len={seq_len} on Blackhole {tuple(mesh_device.shape)}: "
            "this configuration uses throughput experts which are not supported on Blackhole."
        )

    assert batch_size == 1 or seq_len == 1, "Only single user prefill or single token decode is supported"
    is_decode = seq_len == 1
    mode = "decode" if is_decode else "prefill"

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    pcc_thresholds = test_thresholds[setup["model_args"].model_name][mode]
    # Set attention implementation for transformers compatibility
    config = setup["config"]
    config._attn_implementation = "eager"
    # transformers 5.x also needs the MoE experts dispatch pinned for standalone reference layers.
    config._experts_implementation = "eager"

    if batch_size > 32:
        if mesh_device.shape[0] == 1:
            pytest.skip(f"Batch size > 32 is not supported for mesh shape {tuple(mesh_device.shape)}")
        is_row_sharded = True
        assert batch_size % mesh_device.shape[0] == 0, "Batch size must be evenly divisible by mesh device shape"
        local_batch_size = batch_size // mesh_device.shape[0]
    else:
        is_row_sharded = False
        local_batch_size = batch_size

    # Paged attention: when enabled, allocate a PagedAttentionConfig sized to fit the
    # longest seq_len in the parametrize matrix (one user per call here), then build
    # a page_table tensor with sequential block ids. Both the decoder layer's kv cache
    # allocation and the per-call paged_fill_cache / paged_sdpa_decode invocations are
    # gated by the same config + page_table, so the test exercises the full paged path
    # end-to-end (in contrast to the legacy `paged=False` path which goes through
    # ttnn.fill_cache + non-paged SDPA).
    paged_attention_config = None
    page_table_tt = None
    if paged:
        from models.tt_transformers.tt.common import PagedAttentionConfig

        paged_block_size = 64
        effective_seq_len = max(seq_len, 128)
        paged_blocks_per_seq = max((effective_seq_len + paged_block_size - 1) // paged_block_size, 1)
        paged_max_blocks = local_batch_size * paged_blocks_per_seq
        paged_attention_config = PagedAttentionConfig(block_size=paged_block_size, max_num_blocks=paged_max_blocks)
        page_table_torch = torch.arange(paged_max_blocks, dtype=torch.int32).reshape(
            local_batch_size, paged_blocks_per_seq
        )
        page_table_tt = ttnn.from_torch(
            page_table_torch,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # Create reference model
    reference_layer = setup_reference_layer(setup, layer_idx=layer_idx)
    decoder_layer = setup_decoder_layer(
        setup,
        reference_layer,
        local_batch_size,
        seq_len,
        layer_idx=layer_idx,
        paged_attention_config=paged_attention_config,
    )

    # Create input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create position IDs first
    position_ids = torch.concat([torch.arange(seq_len, dtype=torch.long) for _ in range(batch_size)])

    # Create attention mask like the working attention test
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)
    # Newer transformers expose layer type via ``config.layer_types[i]`` rather than
    # a ``GptOssDecoderLayer.attention_type`` attribute. Probe both.
    layer_attn_type = (
        getattr(reference_layer, "attention_type", None)
        or getattr(config, "layer_types", [None] * (layer_idx + 1))[layer_idx]
    )
    if layer_attn_type == "sliding_attention":
        mask += torch.tril(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=-config.sliding_window)

    # Handle decode mode for TT model like original
    if seq_len == 1:  # decode
        mask = None

    # Create RoPE embeddings using Meta format (matching tt-transformers test)
    max_seq_len = seq_len

    # For TTNN: use precompute_freqs and gather_cos_sin to get cos/sin tensors
    # TODO: To test longer sequences we will need to change this so we can apply rope scaling using Yarn implementation
    # Newer GptOssConfig drops the top-level ``rope_theta`` attribute (it's now bundled
    # in ``rope_parameters``); the class still exposes ``default_theta`` as the
    # canonical base.
    rope_theta = getattr(config, "rope_theta", None) or getattr(config, "default_theta", 150000.0)
    cos_full, sin_full = precompute_freqs(
        dim=config.head_dim,
        end=max_seq_len * 2,
        theta=rope_theta,
        scale_factor=None,
        orig_context_len=131072,
    )

    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

    rope_embeddings_ref = GptOssRotaryEmbedding(config)
    cos_hf_ref, sin_hf_ref = rope_embeddings_ref(hidden_states, position_ids.unsqueeze(1 if is_decode else 0))
    position_embeddings_ref = (cos_hf_ref, sin_hf_ref)

    # Create TTNN RoPE embeddings in Meta format using gather_cos_sin
    cos_meta, sin_meta = gather_cos_sin(position_ids, cos_full, sin_full)
    cos_meta = cos_meta.reshape(1, batch_size, seq_len, config.head_dim)
    sin_meta = sin_meta.reshape(1, batch_size, seq_len, config.head_dim)

    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-3, None), mesh_shape=setup["mesh_device"].shape, mesh_device=setup["mesh_device"])
        if is_row_sharded
        else None
    )
    tt_cos = ttnn.from_torch(
        cos_meta, device=setup["mesh_device"], mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tt_sin = ttnn.from_torch(
        sin_meta, device=setup["mesh_device"], mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # For decode mode, convert cos/sin to HEIGHT_SHARDED to match Q/K/V from nlp_create_qkv_heads_decode
    if mode == "decode":
        grid_size = ttnn.CoreCoord(8, 8)  # Safe limit: max 8 per dimension to avoid Galaxy hangs
        batch_grid = ttnn.num_cores_to_corerangeset(local_batch_size, grid_size, row_wise=True)
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, config.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        tt_cos = ttnn.interleaved_to_sharded(tt_cos, mem_config)
        tt_sin = ttnn.interleaved_to_sharded(tt_sin, mem_config)

    # rope_mats is now [cos, sin] in Meta format
    rope_mats = [tt_cos, tt_sin]

    # Create position index for TTNN
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=setup["mesh_device"].shape, mesh_device=setup["mesh_device"])
        if is_row_sharded
        else None
    )
    tt_position_idx = ttnn.from_torch(
        position_ids.squeeze(),
        device=setup["mesh_device"],
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )

    # Parse test_modules (supports comma-separated values)
    modules_to_test = set(test_modules.split(","))
    run_all = "all" in modules_to_test

    # The paged/unpaged dimension only changes behavior for components that touch
    # the kv cache (attention + the full decoder layer). Skip the paged variant
    # for runs that ask for only non-kv components — and inside ``should_test``,
    # gate non-kv components on ``not paged`` so an "all" invocation doesn't run
    # router / experts / mlp / rms_norm twice with identical inputs. The kv-using
    # components still execute under both paged settings.
    KV_USING_MODULES = {"attention", "decoder"}
    if paged and not run_all and not (modules_to_test & KV_USING_MODULES):
        pytest.skip(
            f"paged variant only exercises kv-cache-using components ({sorted(KV_USING_MODULES)}); "
            f"requested modules {sorted(modules_to_test)} don't touch the kv cache"
        )

    logger.info(f"Running tests: {test_modules} (paged={paged})")

    # Helper to check if a module should be tested. Non-kv components only run on
    # the unpaged variant (their behavior is independent of paged kv-cache state).
    def should_test(module_name):
        if paged and module_name not in KV_USING_MODULES:
            return False
        return run_all or module_name in modules_to_test

    if should_test("router"):
        # Only run router test in decode mode
        if seq_len == 1:
            logger.info("Testing TopK Router...")
            run_topk_router_component(
                setup["mesh_device"],
                hidden_states.shape,
                reference_layer,
                decoder_layer,
                is_decode=is_decode,
                is_row_sharded=is_row_sharded,
                pcc_threshold=pcc_thresholds["router"],
            )
        else:
            logger.info("Router test only runs in decode mode (seq_len=1). Skipping...")

    if should_test("fused_experts"):
        if decoder_layer.mlp.use_throughput_experts and is_decode and is_row_sharded:
            logger.info(f"Testing Fused Throughput Experts for mesh shape {tuple(mesh_device.shape)}...")
            hidden_states_throughput_experts = hidden_states.reshape(1, 1, batch_size * seq_len, -1)
            run_fused_throughput_experts_component(
                setup["mesh_device"],
                hidden_states_throughput_experts.shape,
                config,
                reference_layer,
                decoder_layer,
                is_row_sharded=is_row_sharded,
            )
        else:
            logger.info("Fused experts test requires throughput experts + decode mode + row-sharded. Skipping...")

    if should_test("experts"):
        if decoder_layer.mlp.use_throughput_experts:
            logger.info(f"Testing High Throughput Experts (EP=32) for mesh shape {mesh_shape}...")
            ttnn.synchronize_device(setup["mesh_device"])
            hidden_states_throughput_experts = hidden_states.reshape(1, 1, batch_size * seq_len, -1)
            run_throughput_experts_component(
                setup["mesh_device"],
                hidden_states_throughput_experts.shape,
                config,
                reference_layer,
                decoder_layer,
                is_decode=is_decode,
                is_row_sharded=is_row_sharded,
                pcc_threshold=pcc_thresholds["experts"],
            )
        else:
            logger.info(f"Testing Low Throughput Experts (EP=4) for mesh shape {tuple(mesh_device.shape)}...")
            run_experts_component(
                setup["mesh_device"],
                hidden_states.shape,
                config,
                reference_layer,
                decoder_layer,
                is_decode=is_decode,
                pcc_threshold=pcc_thresholds["experts"],
            )

    if should_test("attention"):
        logger.info("Testing Attention (paged={})...", paged)
        run_attention_component(
            setup["mesh_device"],
            hidden_states.shape,
            mask,
            position_embeddings_ref,
            rope_mats,
            tt_position_idx,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["attention"],
            page_table=page_table_tt,
        )

    if should_test("rms_norm"):
        logger.info("Testing RMS Norm...")
        run_rms_norm_component(
            setup["mesh_device"],
            hidden_states.shape,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["rms_norm"],
        )

    if should_test("mlp"):
        logger.info("Testing Full MLP Pipeline...")
        run_full_mlp_pipeline(
            setup["mesh_device"],
            hidden_states.shape,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["mlp"],
        )

    if should_test("decoder"):
        logger.info("Testing Full Decoder Layer...")
        # Create TTNN tensors for decoder layer test
        if is_row_sharded:
            mesh_mapper = ttnn.ShardTensor2dMesh(
                dims=(-2, None), mesh_shape=setup["mesh_device"].shape, mesh_device=setup["mesh_device"]
            )
        else:
            mesh_mapper = None
        tt_hidden_states = ttnn.from_torch(
            hidden_states.reshape(1, 1, batch_size * seq_len, -1),
            device=setup["mesh_device"],
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
        )

        # Test full decoder layer integration
        with torch.no_grad():
            reference_output = reference_layer(hidden_states, position_embeddings=position_embeddings_ref)

        tt_output = decoder_layer(
            tt_hidden_states,
            position_embeddings=rope_mats,
            position_idx=tt_position_idx,
            page_table=page_table_tt,
            is_decode=is_decode,
        )

        # Compare outputs
        pcc_threshold = pcc_thresholds["decoder"]
        mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[
            ..., : batch_size * seq_len, : config.hidden_size
        ]
        passing, output = compare_tensors(
            tt_output_torch.squeeze(), reference_output.squeeze(), mesh_device, pcc_threshold=pcc_threshold
        )
        if passing:
            logger.info(f"Decoder Layer test passed. Output: {output}")
        else:
            assert passing, f"Decoder Layer test failed. Output: {output}"

    tested_modules = [m for m in modules_to_test if m != "router" or seq_len == 1]
    logger.info(f"✓ Tests completed successfully: {', '.join(tested_modules)}")


def run_model_forward_test(
    mesh_device,
    config,
    state_dict_meta,
    reference_model,
    mesh_config,
    batch_size,
    seq_len,
    is_decode,
    pcc_threshold=0.88,
    tensor_cache_path=None,
):
    """
    Run a single forward pass test comparing TT model to reference model.

    Args:
        mesh_device: TTNN mesh device
        config: HuggingFace config (with num_hidden_layers already modified)
        state_dict_meta: Model weights in meta format for TT model
        reference_model: Already-instantiated HuggingFace reference model (eval mode)
        mesh_config: Mesh configuration
        batch_size: Batch size
        seq_len: Sequence length
        is_decode: True for decode mode (seq_len=1), False for prefill mode
        pcc_threshold: PCC threshold for comparison
    """
    from models.demos.gpt_oss.tt.ccl import CCLManager
    from models.demos.gpt_oss.tt.model import Model
    from models.demos.gpt_oss.utils.general_utils import get_default_num_links

    # Determine local batch size for row sharding
    if batch_size > 32:
        is_row_sharded = True
        assert batch_size % mesh_device.shape[0] == 0, "Batch size must be divisible by mesh rows"
        local_batch_size = batch_size // mesh_device.shape[0]
    else:
        is_row_sharded = False
        local_batch_size = batch_size

    # Create CCL manager
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))

    # Create TT model with meta format weights
    # Use throughput experts for row-sharded batches (batch > 32 on multi-row mesh)
    use_throughput_experts = is_row_sharded and mesh_device.shape[0] > 1 and throughput_experts_supported_on_arch()
    tt_model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict_meta,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=tensor_cache_path,
        paged_attention_config=None,
        mesh_config=mesh_config,
        create_kv_cache=True,
        max_local_batch_size=local_batch_size,
        users_row_sharded=is_row_sharded,
        use_throughput_experts=use_throughput_experts,
    )

    # Create random input tokens
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Create position IDs
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    # Run reference forward pass
    with torch.no_grad():
        reference_output = reference_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,  # Let the model handle masking
            use_cache=False,
        )
        reference_logits = reference_output.logits  # [batch_size, seq_len, vocab_size]

    # Prepare inputs for TT model
    if is_decode:
        # Decode mode: use ttnn_decode_forward
        # Flatten tokens for decode
        tokens_flat = input_ids.reshape(-1)  # [batch_size]
        current_pos = torch.zeros(batch_size, dtype=torch.long)

        # Prepare inputs using model's method
        tt_tokens, tt_current_pos, tt_rope_idxs, _ = tt_model.prepare_inputs_decode(
            tokens_flat, current_pos, page_table=None
        )

        # Run TT decode forward
        tt_logits, _ = tt_model.ttnn_decode_forward(
            tokens=tt_tokens,
            current_pos=tt_current_pos,
            rot_mat_idxs=tt_rope_idxs,
            page_table=None,
            kv_cache=None,
        )
    else:
        # Prefill mode: use ttnn_prefill_forward
        # Embed tokens first
        tt_tokens = ttnn.from_torch(
            input_ids.unsqueeze(0).unsqueeze(0),  # [1, 1, batch, seq]
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tt_embeds = ttnn.embedding(tt_tokens, tt_model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        if len(tt_embeds.shape) == 3:
            tt_embeds = ttnn.unsqueeze_to_4D(tt_embeds)

        # Run TT prefill forward
        tt_logits = tt_model.ttnn_prefill_forward(
            x=tt_embeds,
            user_id=0,
            rot_mats_global=None,  # Let model compute RoPE
            page_table=None,
            kv_cache=None,
            get_last_token=-1,  # Get all tokens
        )

    # Convert TT output to torch
    mesh_composer_dims = (-2, -1) if is_row_sharded else (0, -1)
    tt_logits_torch = ttnn.to_torch(
        tt_logits,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=mesh_composer_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    # Slice to match reference shape
    tt_logits_torch = tt_logits_torch[0, 0]

    # Reshape to match reference
    tt_logits_torch = tt_logits_torch.reshape(batch_size, seq_len, -1)

    # Truncate to vocab_size — lm_head weight may be padded to padded_vocab_size
    # for on-device sampling alignment, producing extra columns in the output.
    vocab_size = config.vocab_size
    if tt_logits_torch.shape[-1] > vocab_size:
        tt_logits_torch = tt_logits_torch[:, :, :vocab_size]

    # Compare outputs
    passing, output = compare_tensors(tt_logits_torch, reference_logits, mesh_device, pcc_threshold=pcc_threshold)
    return passing, output


@parametrize_mesh_with_fabric([(1, 1), (1, 8), (4, 8)])
@pytest.mark.parametrize(
    "batch_size, seq_len, mode",
    [
        (1, 128, "prefill"),
        (128, 1, "decode"),
    ],
    ids=[
        "prefill_b1_s128",
        "decode_b128_s1",
    ],
)
@pytest.mark.parametrize(
    "num_layers",
    [1],
    ids=["1_layer"],
)
def test_model(mesh_device, device_params, batch_size, seq_len, mode, num_layers, reset_seeds):
    """
    Test full model forward pass comparing TT implementation to HuggingFace reference.

    This test:
    1. Loads model config and overrides num_hidden_layers
    2. Creates both TT model and reference model with real weights
    3. Runs prefill (batch=1, seq=128) or decode (batch=128, seq=1) forward pass
    4. Compares outputs using PCC

    Args:
        mesh_device: TTNN mesh device fixture
        device_params: Device parameters fixture
        batch_size: Batch size for the test
        seq_len: Sequence length for the test
        mode: "prefill" or "decode"
        num_layers: Number of layers to use (overrides config.num_hidden_layers)
        reset_seeds: Fixture to reset random seeds
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    from models.demos.gpt_oss.config import MeshConfig, ModeConfig

    mesh_shape = tuple(mesh_device.shape)

    if mesh_shape[0] == 1 and batch_size > 1:
        pytest.skip(
            f"Skipping batch size {batch_size} for mesh shape {mesh_shape}. Only batch size 1 is supported when mesh rows = 1."
        )

    if is_blackhole() and batch_size > 32 and mesh_device.shape[0] > 1:
        pytest.skip(
            f"Skipping batch={batch_size} on Blackhole {tuple(mesh_device.shape)}: row-sharded batches "
            "use throughput experts which are not supported on Blackhole."
        )

    is_decode = mode == "decode"

    # Setup test using TestFactory
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Override number of layers
    original_num_layers = config.num_hidden_layers
    config.num_hidden_layers = num_layers
    logger.info(f"Overriding num_hidden_layers from {original_num_layers} to {num_layers}")

    # Set attention implementation
    config._attn_implementation = "eager"

    # Create mesh config
    mesh_shape = tuple(mesh_device.shape)
    mesh_config = MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0]))

    # Build a small random-init reference. ``setup_test`` already pulled the
    # HF config (a tiny JSON) so we have the correct architecture; we don't
    # download or load any actual checkpoint. Accuracy testing against real
    # weights lives in ``tests/accuracy/test_model.py`` — keep this unit test
    # cheap so it can run anywhere the HF config resolves (offline or via
    # cache).
    tensor_cache_path = None
    reference_model_hf = GptOssForCausalLM(config)
    reference_model_hf.eval()
    state_dict_hf = reference_model_hf.state_dict()
    state_dict_meta = convert_hf_qkv_to_meta_format(state_dict_hf, config.head_dim)

    logger.info(f"Running {mode} test with batch_size={batch_size}, seq_len={seq_len}, num_layers={num_layers}")

    # Run the forward test
    passing, output = run_model_forward_test(
        mesh_device=mesh_device,
        config=config,
        state_dict_meta=state_dict_meta,
        reference_model=reference_model_hf,
        mesh_config=mesh_config,
        batch_size=batch_size,
        seq_len=seq_len,
        is_decode=is_decode,
        pcc_threshold=0.95 if num_layers == 1 else 0.85,
        tensor_cache_path=tensor_cache_path,
    )

    if passing:
        logger.info(f"✓ Model {mode} test passed. PCC: {output}")
    else:
        assert passing, f"Model {mode} test failed. PCC: {output}"
