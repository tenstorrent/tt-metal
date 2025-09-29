"""
Minimal but comprehensive core component tests - works for any GPT model size
"""

import pytest
import torch

import ttnn

from ...tt.attention import Attention
from ...tt.mlp import MLP
from ...tt.rms_norm import RMSNorm
from ...tt.topk import TopKRouter
from ..test_factory import TestFactory, parametrize_batch_seq, parametrize_mesh, parametrize_mesh_with_fabric

# Core MoE Tests - Essential for any model size


@parametrize_mesh_with_fabric(["1x8"])
@pytest.mark.parametrize("seq_len", [1, 32, 64])
def test_topk_router(mesh_device, device_params, seq_len, reset_seeds):
    """Test TopK routing - essential for MoE (like original)"""

    # Create submesh like original (4x8 base -> 1x2 submesh)
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 2))
    print(f"MESH DEVICE: {mesh_device}")
    print(f"MESH SHAPE: {mesh_device.shape}")

    # Get config like original (don't use TestFactory)
    from models.demos.gpt_oss.tt.model_config import ModelArgs

    model_args = ModelArgs(mesh_device=None, dummy_weights=True)

    # Create config like original
    from models.demos.gpt_oss.reference.configuration_gpt_oss import GptOssConfig

    config = GptOssConfig()

    # Create input like original
    hidden_states = torch.randn(seq_len, config.hidden_size)

    # Create reference model like original
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssTopKRouter

    reference_model = GptOssTopKRouter(config)

    # Reference model forward like original
    router_scores, router_indices = reference_model(hidden_states)

    # Convert to TTNN tensors like original
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Create TT model like original
    tt_model = TopKRouter(mesh_device, config, reference_model.state_dict())
    tt_router_scores, tt_router_indices, tt_router_logits = tt_model(tt_hidden_states)

    # Compare outputs like original
    tt_router_scores_tensors = ttnn.get_device_tensors(tt_router_scores)
    tt_router_indices_tensors = ttnn.get_device_tensors(tt_router_indices)
    tt_router_logits_tensors = ttnn.get_device_tensors(tt_router_logits)

    for i in range(len(tt_router_scores_tensors)):
        tt_router_scores = ttnn.to_torch(tt_router_scores_tensors[i])
        tt_router_indices = ttnn.to_torch(tt_router_indices_tensors[i])
        tt_router_logits = ttnn.to_torch(tt_router_logits_tensors[i])

        # Compare router scores like original
        from models.utility_functions import comp_pcc

        passing, output = comp_pcc(router_scores, tt_router_scores, pcc=0.99)
        mse = torch.nn.functional.mse_loss(router_scores, tt_router_scores)
        print(f"router_scores: {output}, mse: {mse}")

        if passing:
            break
    else:
        assert False, f"All device comparisons failed. Last output: {output}"


@parametrize_mesh_with_fabric(["1x8"])
@pytest.mark.parametrize("seq_len", [1, 32, 64])
def test_experts_adaptive(mesh_device, device_params, seq_len, reset_seeds):
    """Test experts with adaptive routing - sparse for seq_len=1, dense for seq_len>1"""

    # Choose submesh based on routing type
    if seq_len == 1:
        # Sparse routing - use 1x8 submesh like original test_sparse_experts
        mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 8))
        use_sparse = True
    else:
        # Dense routing - use 1x2 submesh like original test_experts
        mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 2))
        use_sparse = False

    print(f"MESH DEVICE: {mesh_device}")
    print(f"MESH SHAPE: {mesh_device.shape}")
    print(f"ROUTING TYPE: {'SPARSE' if use_sparse else 'DENSE'}")

    # Get config like original (don't use TestFactory)
    from models.demos.gpt_oss.tt.model_config import ModelArgs

    model_args = ModelArgs(mesh_device=None, dummy_weights=True)

    # Create config like original
    from models.demos.gpt_oss.reference.configuration_gpt_oss import GptOssConfig

    config = GptOssConfig()
    # config.num_local_experts = 32
    # config.intermediate_size = 2880
    # config.hidden_size = 2880
    # config.num_experts_per_tok = 4

    # Create input like original
    batch_size = 1
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Create routing weights based on type
    if use_sparse:
        # Sparse routing weights like original test_sparse_experts
        import itertools

        routing_weights = torch.zeros(batch_size * seq_len, config.num_local_experts)

        for b, s in itertools.product(range(batch_size), range(seq_len)):
            # Randomly select which experts are active for this token
            active_experts = torch.randperm(config.num_local_experts)[: config.num_experts_per_tok]
            weights = torch.rand(config.num_experts_per_tok)
            weights = weights / weights.sum()  # Normalize
            routing_weights[b * seq_len + s, active_experts] = weights
    else:
        # Dense routing weights like original test_experts
        routing_weights = torch.randn(batch_size * seq_len, config.num_local_experts)
        routing_weights = torch.nn.functional.softmax(routing_weights, dim=-1)

    # Create reference model like original (use ReferenceExperts from test factory)
    from models.demos.gpt_oss.tests.test_factory import ReferenceExperts

    reference_model = ReferenceExperts(config)
    state_dict = reference_model.state_dict()

    # Reference model forward like original
    reference_output = reference_model(hidden_states, routing_weights)

    # Convert to TTNN tensors like original
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_routing_weights = ttnn.from_torch(
        routing_weights, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Create TT model based on routing type
    from models.demos.gpt_oss.tt.ccl import CCLManager

    ccl_manager = CCLManager(mesh_device)

    # Use Experts for dense routing
    from models.demos.gpt_oss.tt.experts import Experts

    tt_model = Experts(mesh_device, config, state_dict, ccl_manager)

    tt_output = tt_model(tt_hidden_states, tt_routing_weights)

    # Compare outputs like original
    tt_output_tensors = ttnn.get_device_tensors(tt_output)

    for i in range(len(tt_output_tensors)):
        tt_output_torch = ttnn.to_torch(tt_output_tensors[i])

        # Compare outputs like original
        from models.utility_functions import comp_pcc

        passing, output = comp_pcc(reference_output, tt_output_torch, pcc=0.99)
        mse = torch.nn.functional.mse_loss(reference_output, tt_output_torch)

        routing_type = "sparse" if use_sparse else "dense"
        print(f"{routing_type}_experts_output: {output}, mse: {mse}")

        if passing:
            break
    else:
        assert False, f"All device comparisons failed. Last output: {output}"


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

    # Compare outputs like original
    tt_out_tensors = ttnn.get_device_tensors(tt_out)
    for i in range(len(tt_out_tensors)):
        tt_out_torch = ttnn.to_torch(tt_out_tensors[i])

        # Compare outputs like original
        from models.utility_functions import comp_pcc

        pcc = 0.99
        passed, pcc_message = comp_pcc(reference_out, tt_out_torch, pcc)
        print(f"Test passed: {passed}, PCC: {pcc_message}")

        if passed:
            break
    else:
        assert False, f"All device comparisons failed. Last output: {pcc_message}"


@parametrize_mesh_with_fabric(["1x8"])
@pytest.mark.parametrize("seq_len", [1, 32, 64])
def test_rms_norm(mesh_device, device_params, seq_len, reset_seeds):
    """Test RMS normalization - used throughout model (like original)"""

    # Create submesh like original (4x8 base -> 1x8 submesh)
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(1, 8))
    print(f"MESH DEVICE: {mesh_device}")
    print(f"MESH SHAPE: {mesh_device.shape}")

    # Get config like original (don't use TestFactory)
    from models.demos.gpt_oss.tt.model_config import ModelArgs

    model_args = ModelArgs(mesh_device=None, dummy_weights=True)

    # Create config like original
    from models.demos.gpt_oss.reference.configuration_gpt_oss import GptOssConfig

    config = GptOssConfig()

    # Create input like original
    torch_input = torch.randn(1, seq_len, config.hidden_size) * 2 + 3

    # Create reference model like original
    from models.demos.gpt_oss.reference.modeling_gpt_oss import GptOssRMSNorm

    ref_rms_norm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps).eval()

    # Reference model forward like original
    with torch.no_grad():
        ref_output = ref_rms_norm(torch_input)

    # Convert to TTNN tensors like original
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Create TT model like original (but with mesh_config for current implementation)
    from models.demos.gpt_oss.moe import MeshConfig

    # Disable sharding like original test (tp=1 means no tensor parallel)
    mesh_config = MeshConfig(mesh_device.shape, tp=1)
    tt_rms_norm = RMSNorm(mesh_device, config, ref_rms_norm.state_dict(), mesh_config=mesh_config)
    tt_output = tt_rms_norm(tt_input)

    # Compare outputs like original
    tt_output_tensors = ttnn.get_device_tensors(tt_output)
    expected_pcc = 0.99

    for i in range(len(tt_output_tensors)):
        tt_output_torch = ttnn.to_torch(tt_output_tensors[i])
        from models.utility_functions import comp_pcc

        passing, pcc_message = comp_pcc(tt_output_torch, ref_output, pcc=expected_pcc)
        mse = torch.nn.functional.mse_loss(tt_output_torch, ref_output)
        print(f"PCC: {pcc_message}, MSE: {mse}")

        if passing:
            break
    else:
        assert False, f"All device comparisons failed. Last PCC: {pcc_message}"


@parametrize_mesh(["1x8", "4x4"])
def test_full_mlp_pipeline(mesh_device, reset_seeds):
    """Test complete MLP (router + experts) - essential MoE functionality"""

    setup = TestFactory.setup_test(mesh_device)
    config = setup["config"]

    # Complete MLP state dict - use flat structure expected by substate function
    mlp_state = {
        "router.weight": torch.randn(config.num_local_experts, config.hidden_size),
        "router.bias": torch.randn(config.num_local_experts),
    }
    # Add experts state dict with proper prefix
    for key, value in setup["state_dict"].items():
        mlp_state[f"experts.{key}"] = value

    tt_mlp = MLP(
        setup["mesh_device"],
        config,
        mlp_state,
        setup["ccl_manager"],
        dtype=setup["dtype"],
        tensor_cache_path=setup["tensor_cache_path"],
        mesh_config=setup["mesh_config"],
    )

    # Test with various sequence lengths
    for seq_len in [1, 32]:
        hidden_states = torch.randn(1, seq_len, config.hidden_size)
        tt_hidden_states = ttnn.from_torch(
            hidden_states, device=setup["mesh_device"], layout=ttnn.TILE_LAYOUT, dtype=setup["dtype"]
        )

        mlp_out, routing_scores = tt_mlp(tt_hidden_states)

        # Verify output shapes - works for any model size
        assert mlp_out.shape == tt_hidden_states.shape
        assert routing_scores.shape[:-1] == tt_hidden_states.shape[:-1]  # Same batch/seq, different last dim
