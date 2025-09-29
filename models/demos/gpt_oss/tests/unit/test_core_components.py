"""
Minimal but comprehensive core component tests - works for any GPT model size
"""

import pytest
import torch

import ttnn

from ...tt.attention import Attention
from ...tt.experts import Experts
from ...tt.mlp import MLP
from ...tt.rms_norm import RMSNorm
from ...tt.topk import TopKRouter
from ..test_factory import TestFactory, compare_tensors, parametrize_batch_seq, parametrize_mesh_with_fabric


# Helper Functions for Common Test Patterns
def run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.99):
    """Standard component output comparison"""
    tt_output_tensors = ttnn.get_device_tensors(tt_output)

    for i in range(len(tt_output_tensors)):
        tt_output_torch = ttnn.to_torch(tt_output_tensors[i])
        passing, output = compare_tensors(tt_output_torch, reference_output, mesh_device, pcc_threshold=pcc_threshold)
        if passing:
            return True, output
    else:
        return False, output


# Core MoE Tests - Essential for any model size


@parametrize_mesh_with_fabric(["1x8"])
@pytest.mark.parametrize("seq_len", [1, 32, 64])
def test_topk_router(mesh_device, device_params, seq_len, reset_seeds):
    """Test TopK routing - essential for MoE"""

    # Create submesh
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 2))

    # Use TestFactory for clean setup
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    hidden_states = torch.randn(seq_len, config.hidden_size)

    # Create reference model
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssTopKRouter

    reference_model = GptOssTopKRouter(config)
    router_scores, router_indices = reference_model(hidden_states)

    # Create TT model
    tt_model = TopKRouter(mesh_device, config, reference_model.state_dict())
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_router_scores, tt_router_indices, tt_router_logits = tt_model(tt_hidden_states)

    # Compare outputs
    for tt_output, reference_output in zip(tt_router_scores, router_scores):
        passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.99)
        assert passing, f"TopK router test failed. Output: {output}"


@parametrize_mesh_with_fabric(["1x8"])
@pytest.mark.parametrize("seq_len", [1, 128])
def test_experts_adaptive(mesh_device, device_params, seq_len, reset_seeds):
    """Test experts with adaptive routing - sparse for seq_len=1, dense for seq_len>1"""

    # Choose submesh based on routing type
    if seq_len == 1:
        # Sparse routing - use 1x8 submesh
        mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 8))
        use_sparse = True
    else:
        # Dense routing - use 1x2 submesh
        mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 2))
        use_sparse = False

    print(f"ROUTING TYPE: {'SPARSE' if use_sparse else 'DENSE'}")

    # Use TestFactory for clean setup
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Create input
    batch_size = 1
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create routing weights based on type
    if use_sparse:
        # Sparse routing weights - only few experts active per token
        import itertools

        routing_weights = torch.zeros(batch_size * seq_len, config.num_local_experts)

        for b, s in itertools.product(range(batch_size), range(seq_len)):
            active_experts = torch.randperm(config.num_local_experts)[: config.num_experts_per_tok]
            weights = torch.rand(config.num_experts_per_tok)
            weights = weights / weights.sum()  # Normalize
            routing_weights[b * seq_len + s, active_experts] = weights
    else:
        # Dense routing weights - all experts get some weight
        routing_weights = torch.randn(batch_size * seq_len, config.num_local_experts)
        routing_weights = torch.nn.functional.softmax(routing_weights, dim=-1)

    # Create reference model using real implementation
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssExperts

    reference_model = GptOssExperts(config).eval()  # Set to eval mode for inference
    state_dict = reference_model.state_dict()
    reference_output = reference_model(hidden_states, routing_weights=routing_weights)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_routing_weights = ttnn.from_torch(
        routing_weights, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Create TT model using TestFactory setup
    tt_model = Experts(mesh_device, config, state_dict, setup["ccl_manager"])
    tt_output = tt_model(tt_hidden_states, tt_routing_weights)

    # Compare outputs
    passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.99)
    assert passing, f"Experts test failed. Output: {output}"


@parametrize_mesh_with_fabric(["1x8"])
@parametrize_batch_seq(
    [
        (1, 1),
        (1, 128),
    ]
)
def test_attention_core(mesh_device, device_params, batch_size, seq_len, reset_seeds):
    """Test attention mechanism - core for any transformer (like original)"""

    # Create submesh like original (4x8 base -> 1x8 submesh)
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 8))
    print(f"MESH DEVICE: {mesh_device}")
    print(f"MESH SHAPE: {mesh_device.shape}")

    # Use our abstractions but with original test parameters
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Create input tensors like original
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)

    # Create mask like original
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)

    # Create RoPE embeddings like original
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssRotaryEmbedding

    RopeEmbeddings = GptOssRotaryEmbedding(config)
    cos, sin = RopeEmbeddings(hidden_states, position_ids)
    position_embeddings = (cos, sin)

    # Create reference model like original
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssAttention

    reference_model = GptOssAttention(config, layer_idx=0)
    state_dict = reference_model.state_dict()

    # Reference model forward like original (use original mask)
    reference_out, _ = reference_model(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=mask,
        use_cache=True,
    )

    # Handle decode mode for TT model like original
    if seq_len == 1:  # decode
        from models.demos.gpt_oss.utils.general_utils import get_decode_mask

        sliding_window = 0  # No sliding window for this test
        mask = get_decode_mask(position_ids[0].item(), sliding_window)
        mask = mask.repeat(1, config.num_attention_heads // mesh_device.shape[1], 1, 1).transpose(1, 2)

    # Convert to TTNN tensors like original
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_mask = ttnn.from_torch(mask, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_cos = ttnn.from_torch(cos.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin.unsqueeze(-2), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_position_idx = ttnn.from_torch(position_ids, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    # Create RoPE like original
    from models.demos.gpt_oss.tt.rope import ApplyRotaryPosEmb

    apply_rope = ApplyRotaryPosEmb(config)
    rope_stuff = (apply_rope, tt_cos, tt_sin)

    # Create TT model like original
    tt_model = Attention(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict,
        layer_idx=0,
        ccl_manager=setup["ccl_manager"],
        # tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    tt_out = tt_model(tt_hidden_states, tt_mask, rope_stuff, tt_position_idx)

    # Compare outputs
    passing, output = run_component_comparison(tt_out, reference_out, mesh_device, pcc_threshold=0.99)
    assert passing, f"Attention test failed. Output: {output}"


@parametrize_mesh_with_fabric(["1x8"])
@pytest.mark.parametrize("seq_len", [1, 32, 64])
def test_rms_norm(mesh_device, device_params, seq_len, reset_seeds):
    """Test RMS normalization - used throughout model"""

    # Create submesh
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 8))

    # Use TestFactory for clean setup
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Create input
    torch_input = torch.randn(1, seq_len, config.hidden_size) * 2 + 3

    # Create reference model using real implementation
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssRMSNorm

    reference_model = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps).eval()
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    # Convert to TTNN tensors
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Create TT model with mesh_config for current implementation
    from models.demos.gpt_oss.moe import MeshConfig

    mesh_config = MeshConfig(mesh_device.shape, tp=1)  # Disable sharding
    tt_rms_norm = RMSNorm(mesh_device, config, reference_model.state_dict(), mesh_config=mesh_config)
    tt_output = tt_rms_norm(tt_input)

    # Compare outputs
    passing, output = run_component_comparison(tt_output, ref_output, mesh_device, pcc_threshold=0.99)
    assert passing, f"RMS norm test failed. Output: {output}"


@parametrize_mesh_with_fabric(["1x8"])
@pytest.mark.parametrize(
    "seq_len",
    [
        128,
    ],
)
def test_full_mlp_pipeline(mesh_device, device_params, seq_len, reset_seeds):
    """Test complete MLP (router + experts) - essential MoE functionality"""

    # Create submesh
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 8))

    # Use TestFactory for clean setup
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Create input
    hidden_states = torch.randn(1, seq_len, config.hidden_size)

    # Create reference model using real implementation
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssMLP

    reference_model = GptOssMLP(config).eval()

    # Fix the router weights - they're all zeros!
    with torch.no_grad():
        # Initialize router weights properly
        torch.nn.init.normal_(reference_model.router.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(reference_model.router.bias)

    state_dict = reference_model.state_dict()

    reference_output, routing_scores = reference_model(hidden_states)

    # Convert to TTNN tensors
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Create TT MLP using TestFactory setup
    tt_mlp = MLP(
        mesh_device,
        config,
        state_dict,
        setup["ccl_manager"],
        dtype=setup["dtype"],
        tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    tt_output, routing_scores = tt_mlp(tt_hidden_states)

    # Compare outputs
    passing, output = run_component_comparison(tt_output, reference_output, mesh_device, pcc_threshold=0.99)
    assert passing, f"MLP test failed. Output: {output}"


@parametrize_mesh_with_fabric(
    [
        "1x8",
    ]
)
@parametrize_batch_seq(
    [
        (1, 1),
    ]
)
def test_decoder_layer_integration(mesh_device, device_params, batch_size, seq_len, reset_seeds):
    """Test complete decoder layer - combines attention + MLP + norms"""
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 8))

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    # Create reference model
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssDecoderLayer

    reference_layer = GptOssDecoderLayer(config, layer_idx=0)
    reference_state = reference_layer.state_dict()

    decoder_layer = DecoderLayer(
        setup["mesh_device"],
        config,
        reference_state,
        layer_idx=0,
        ccl_manager=setup["ccl_manager"],
        dtype=setup["dtype"],
        tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    # Create input
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create position IDs first
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # Create attention mask like the working attention test
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)

    # Handle decode mode for TT model like original
    if seq_len == 1:  # decode
        from models.demos.gpt_oss.utils.general_utils import get_decode_mask

        sliding_window = 0  # No sliding window for this test
        mask = get_decode_mask(position_ids[0].item(), sliding_window)
        mask = mask.repeat(1, config.num_attention_heads // setup["mesh_device"].shape[1], 1, 1).transpose(1, 2)

    # Create position embeddings for reference model
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssRotaryEmbedding

    rope_embeddings = GptOssRotaryEmbedding(config)
    cos, sin = rope_embeddings(hidden_states, position_ids)
    position_embeddings = (cos, sin)

    # Reference forward pass
    with torch.no_grad():
        reference_output = reference_layer(hidden_states, position_embeddings=position_embeddings)

    # Create TTNN RoPE embeddings for decoder layer
    tt_cos = ttnn.from_torch(
        cos.unsqueeze(-2), device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tt_sin = ttnn.from_torch(
        sin.unsqueeze(-2), device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    from models.demos.gpt_oss.tt.rope import ApplyRotaryPosEmb

    apply_rope = ApplyRotaryPosEmb(config)
    rope_stuff = (apply_rope, tt_cos, tt_sin)

    # Create position index for TTNN
    tt_position_idx = ttnn.from_torch(
        position_ids, device=setup["mesh_device"], layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    )

    # Create TTNN mask
    tt_mask = ttnn.from_torch(mask, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # TTNN forward pass
    tt_hidden_states = ttnn.from_torch(
        hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16  # setup["dtype"]
    )
    print(f"tt_hidden_states shape: {tt_hidden_states.shape}", "memory: ", tt_hidden_states.memory_config())
    tt_output = decoder_layer(
        tt_hidden_states, attention_mask=tt_mask, position_embeddings=rope_stuff, position_idx=tt_position_idx
    )

    # Compare outputs
    passing, output = run_component_comparison(tt_output, reference_output, setup["mesh_device"], pcc_threshold=0.99)
    assert passing, f"Decoder layer test failed. Output: {output}"
