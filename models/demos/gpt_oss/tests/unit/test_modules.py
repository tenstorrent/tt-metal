# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
    GptOssTopKRouter,
)

import ttnn
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import precompute_freqs_cis
from models.tt_transformers.tt.common import gather_cos_sin, precompute_freqs
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format
from models.tt_transformers.tt.rope import RotarySetup

from ...tt.layer import DecoderLayer
from ..test_factory import TestFactory, compare_tensors, parametrize_batch_seq, parametrize_mesh_with_fabric


# Helper Functions for Common Test Patterns
def gather_sharded_output(tt_output, mesh_config, logical_shape):
    """
    All-gather row-sharded output for comparison.
    Returns a torch Tensor (on host).

    Args:
        logical_shape: Tuple (batch_size, seq_len, hidden_size)
    """
    batch_size, seq_len, hidden_size = logical_shape
    num_rows = mesh_config.mesh_shape[0]

    # 1. Gather full tensor if distributed
    if num_rows > 1:
        tt_output = ttnn.all_gather(
            tt_output,
            dim=-1,
            topology=ttnn.Topology.Linear,
            cluster_axis=0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    return tt_output

    # 2. Convert to torch (host)
    # Use get_device_tensors and take the first one since it's replicated
    device_tensors = ttnn.get_device_tensors(tt_output)
    tt_output_torch = ttnn.to_torch(device_tensors[0])

    current_shape = tt_output_torch.shape
    padded_seq_len = current_shape[-2]

    # Handle reshaping if 4D -> 3D
    if len(current_shape) == 4:
        tt_output_torch = tt_output_torch.reshape(batch_size, padded_seq_len, current_shape[-1])

    # Slice sequence length if needed
    if padded_seq_len > seq_len:
        tt_output_torch = tt_output_torch[:, :seq_len, :]

    return tt_output_torch


def initialize_decoder_layer(layer, config):
    """Mirror HF weight init for standalone decoder layer."""
    std = getattr(config, "initializer_range", 0.02)

    for module in layer.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GptOssRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, GptOssExperts):
            module.gate_up_proj.data.normal_(0.0, std)
            module.gate_up_proj_bias.data.zero_()
            module.down_proj.data.normal_(0.0, std)
            module.down_proj_bias.data.zero_()
        elif isinstance(module, GptOssAttention):
            module.sinks.data.normal_(0.0, std)
        elif isinstance(module, GptOssTopKRouter):
            module.weight.data.normal_(0.0, std)
            module.bias.data.normal_(0.0, std)


def run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.99):
    """Standard component output comparison"""
    if isinstance(tt_output, torch.Tensor):
        return compare_tensors(tt_output, reference_output, mesh_device, pcc_threshold=pcc_threshold)

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
    mesh_config,
    config,
):
    """Test attention component - extracted from decoder layer"""

    batch_size, seq_len, _ = hidden_shape
    hidden_size = config.hidden_size

    # Create inputs for reference and hardware paths
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(-1, None))
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=input_mapper,
    )

    reference_attention = reference_layer.self_attn
    with torch.no_grad():
        reference_out, _ = reference_attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=mask,
            use_cache=True,
        )

    attention_module = decoder_layer.self_attn
    tt_out = attention_module(tt_hidden_states, rope_mats, tt_position_idx)
    tt_out = ttnn.typecast(tt_out, ttnn.bfloat16)
    tt_out_gathered = gather_sharded_output(tt_out, mesh_config, hidden_states.shape)

    passing, output = run_component_comparison(tt_out_gathered, reference_out, mesh_device, pcc_threshold=0.94)
    logger.info(f"Attention test passed: {passing} with output: {output}")
    assert passing, f"Attention test failed. Output: {output}"


def run_rms_norm_component(mesh_device, hidden_shape, reference_layer, decoder_layer, mesh_config, config):
    """Test RMSNorm component - extracted from decoder layer"""

    batch_size, seq_len, _ = hidden_shape
    hidden_size = config.hidden_size

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    reference_rms_norm = reference_layer.input_layernorm
    with torch.no_grad():
        ref_output = reference_rms_norm(hidden_states)

    input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(-1, None))
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=input_mapper,
    )

    rms_norm_module = decoder_layer.input_layernorm
    tt_output = rms_norm_module(tt_hidden_states)
    tt_output = ttnn.typecast(tt_output, ttnn.bfloat16)
    tt_output_gathered = gather_sharded_output(tt_output, mesh_config, hidden_states.shape)

    passing, output = run_component_comparison(tt_output_gathered, ref_output, mesh_device, pcc_threshold=0.99)
    logger.info(f"RMS norm test passed: {passing} with output: {output}")
    assert passing, f"RMS norm test failed. Output: {output}"


def run_topk_router_component(mesh_device, hidden_shape, reference_layer, decoder_layer, config, mesh_config):
    """Test TopK router component - extracted from decoder layer"""

    batch_size, seq_len, _ = hidden_shape
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    reference_router = reference_layer.mlp.router
    with torch.no_grad():
        router_scores, router_indices = reference_router(hidden_states)

    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_router = decoder_layer.mlp.router
    tt_router_scores, tt_router_indices, tt_router_logits = tt_router(tt_hidden_states)

    for tt_output, reference_output in zip(tt_router_scores, router_scores):
        passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.945)
        logger.info(f"TopK router test passed: {passing} with output: {output}")
        assert passing, f"TopK router test failed. Output: {output}"


def run_experts_component(mesh_device, hidden_shape, config, reference_layer, decoder_layer, mesh_config):
    """Test experts component - extracted from decoder layer"""

    hidden_states = torch.randn(hidden_shape)
    seq_len = hidden_shape[1]
    batch_size = hidden_shape[0]
    if seq_len == 1:
        import itertools

        routing_weights = torch.zeros(batch_size * seq_len, config.num_local_experts)

        for b, s in itertools.product(range(batch_size), range(seq_len)):
            active_experts = torch.randperm(config.num_local_experts)[: config.num_experts_per_tok]
            weights = torch.rand(config.num_experts_per_tok)
            weights = weights / weights.sum()
            routing_weights[b * seq_len + s, active_experts] = weights
    else:
        routing_weights = torch.ones(hidden_states.shape[-2], config.num_local_experts) / config.num_local_experts

    reference_experts = reference_layer.mlp.experts.eval()
    with torch.no_grad():
        reference_output = reference_experts(hidden_states, routing_weights=routing_weights)

    tt_hidden_states = ttnn.from_torch(
        hidden_states,
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

    tt_experts = decoder_layer.mlp.experts
    tt_output = tt_experts(tt_hidden_states, tt_routing_weights)
    tt_output = ttnn.typecast(tt_output, ttnn.bfloat16)

    passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.93)
    logger.info(f"Experts test passed: {passing} with output: {output}")
    assert passing, f"Experts test failed. Output: {output}"


def run_full_mlp_pipeline(mesh_device, hidden_shape, reference_layer, decoder_layer, config, mesh_config):
    """Test complete MLP (router + experts) - essential MoE functionality"""

    hidden_states = torch.randn(hidden_shape)
    # hidden_states = torch.load("gpt_oss_debug/tt_input_mlp.pt").squeeze()[0, :2880].reshape(1, 1, 2880)
    # hidden_states = torch.load("gpt_oss_debug/ref_input_mlp.pt").squeeze().reshape(1, 1, 2880)
    reference_model = reference_layer.mlp
    with torch.no_grad():
        reference_output, routing_scores = reference_model(hidden_states)

    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    tt_mlp = decoder_layer.mlp
    tt_output, routing_scores = tt_mlp(tt_hidden_states)
    tt_output = ttnn.typecast(tt_output, ttnn.bfloat16)
    if hidden_shape[-2] == 1:
        tt_output_gathered = gather_sharded_output(tt_output, mesh_config, hidden_states.shape)
    else:
        tt_output_gathered = tt_output
    print("shapes, ref vs tt", reference_output.shape, tt_output_gathered.shape)
    passing, output = run_component_comparison(tt_output_gathered, reference_output, mesh_device, pcc_threshold=0.88)

    logger.info(f"MLP test passed: {passing} with output: {output}")
    assert passing, f"MLP test failed. Output: {output}"


@parametrize_mesh_with_fabric()
@parametrize_batch_seq(
    [
        (1, 128),  # decode
        # (1, 128),  # prefill
        # (1, 4096),  # prefill 4k
    ],
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        (4, 8),
    ],
    ids=["mesh_4x8"],
)
def test_decoder(mesh_device, device_params, batch_size, seq_len, mesh_shape, test_modules, reset_seeds):
    """Test decoder layer components."""
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(mesh_shape))
    logger.info(f"Requested mesh_shape={mesh_shape}, actual submesh shape={mesh_device.shape}")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    hidden_size = config.hidden_size

    config._attn_implementation = "eager"

    reference_layer = GptOssDecoderLayer(config, layer_idx=0)
    initialize_decoder_layer(reference_layer, config)
    reference_layer.eval()

    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if any(proj in name for proj in ["router", "experts", "sinks"]):
                param.data.normal_(0, 1)

    reference_state = reference_layer.state_dict()
    reference_state_swizzled = convert_hf_qkv_to_meta_format(reference_state, config.head_dim)

    max_seq_len = getattr(config, "max_position_embeddings", 131072)
    rope_setup = RotarySetup(
        device=setup["mesh_device"],
        batch_size=1,
        head_dim=config.head_dim,
        max_seq_len=max_seq_len,
        rope_theta=getattr(config, "rope_theta", 10000.0),
        rope_scaling=None,
        datatype=ttnn.bfloat16,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    decoder_layer = DecoderLayer(
        setup["mesh_device"],
        config,
        reference_state_swizzled,
        layer_idx=0,
        ccl_manager=setup["ccl_manager"],
        dtype=setup["dtype"],
        mesh_config=setup["mesh_config"],
        transformation_mats=transformation_mats,
    )

    hidden_states_ref = torch.randn(batch_size, seq_len, hidden_size)
    hidden_states = hidden_states_ref.clone()

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)
    if reference_layer.attention_type == "sliding_attention":
        mask += torch.tril(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=-config.sliding_window)

    if seq_len == 1:
        mask = None

    max_seq_len = seq_len
    position_ids_1d = position_ids.squeeze(0)
    freqs_cis_full = precompute_freqs_cis(
        dim=config.head_dim,
        end=max_seq_len * 2,
        theta=config.rope_theta,
    )

    cos_full, sin_full = precompute_freqs(
        dim=config.head_dim,
        end=max_seq_len * 2,
        theta=config.rope_theta,
        scale_factor=None,
        orig_context_len=131072,
    )

    rope_embeddings_ref = GptOssRotaryEmbedding(config)
    cos_hf_ref, sin_hf_ref = rope_embeddings_ref(hidden_states_ref, position_ids)
    position_embeddings_ref = (cos_hf_ref, sin_hf_ref)
    with torch.no_grad():
        reference_output = reference_layer(
            hidden_states_ref,
            position_embeddings=position_embeddings_ref,
            attention_mask=mask,
            use_cache=True,
        )
        if isinstance(reference_output, tuple):
            reference_output = reference_output[0]

    cos_meta, sin_meta = gather_cos_sin(position_ids_1d, cos_full, sin_full)

    tt_cos = ttnn.from_torch(cos_meta, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin_meta, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    if seq_len == 1:
        grid_size = setup["mesh_device"].compute_with_storage_grid_size()
        batch_grid = ttnn.num_cores_to_corerangeset(1, grid_size, row_wise=True)

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, config.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        tt_cos = ttnn.interleaved_to_sharded(tt_cos, mem_config)
        tt_sin = ttnn.interleaved_to_sharded(tt_sin, mem_config)

    rope_mats = [tt_cos, tt_sin]

    tt_position_idx = ttnn.from_torch(
        position_ids, device=setup["mesh_device"], layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    )

    input_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(-1, None))
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=setup["mesh_device"],
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=input_mapper,
    )

    modules_to_test = set(test_modules.split(","))
    run_all = "all" in modules_to_test

    logger.info(f"Running tests: {test_modules}")

    def should_test(module_name):
        return run_all or module_name in modules_to_test

    if should_test("router"):
        if seq_len == 1:
            logger.info("Testing TopK Router...")
            run_topk_router_component(
                setup["mesh_device"],
                hidden_states_ref.shape,
                reference_layer,
                decoder_layer,
                config,
                setup["mesh_config"],
            )
        elif "router" in modules_to_test:
            pytest.skip("Router test only runs in decode mode (seq_len=1)")

    if should_test("attention"):
        logger.info("Testing Attention...")
        run_attention_component(
            setup["mesh_device"],
            hidden_states_ref.shape,
            mask,
            position_embeddings_ref,
            rope_mats,
            tt_position_idx,
            reference_layer,
            decoder_layer,
            setup["mesh_config"],
            config,
        )

    if should_test("rms_norm"):
        logger.info("Testing RMS Norm...")
        run_rms_norm_component(
            setup["mesh_device"],
            hidden_states_ref.shape,
            reference_layer,
            decoder_layer,
            setup["mesh_config"],
            config,
        )

    if should_test("mlp"):
        logger.info("Testing Full MLP Pipeline...")
        run_full_mlp_pipeline(
            setup["mesh_device"], hidden_states_ref.shape, reference_layer, decoder_layer, config, setup["mesh_config"]
        )

    if should_test("decoder"):
        logger.info("Testing Full Decoder Layer...")
        tt_output = decoder_layer(tt_hidden_states, position_embeddings=rope_mats, position_idx=tt_position_idx)
        tt_output = ttnn.typecast(tt_output, ttnn.bfloat16)

        tt_output_gathered = gather_sharded_output(tt_output, setup["mesh_config"], hidden_states_ref.shape)

        pcc_threshold = 0.924 if seq_len == 1 else 0.88
        passing, output = run_component_comparison(
            tt_output_gathered, reference_output, setup["mesh_device"], pcc_threshold=pcc_threshold
        )

        logger.info(f"Decoder layer test: {passing} with output: {output}")
        assert passing, f"Decoder layer test failed. Output: {output}"

    tested_modules = [m for m in modules_to_test if m != "router" or seq_len == 1]
    logger.info(f"✓ Tests completed successfully: {', '.join(tested_modules)}")
