import os

import pytest
import torch
import ttnn
from transformers import AutoConfig

from models.experimental.tt_symbiote.modules.moe import Glm4MoeConfig


def pytest_configure(config):
    """Register custom markers for incremental test levels and CI/nightly."""
    for i in range(1, 7):
        config.addinivalue_line("markers", f"level{i}: Incremental test level {i}")
    config.addinivalue_line("markers", "nightly: marks tests as nightly-only (deselect with -m 'not nightly')")
    config.addinivalue_line("markers", "ci: marks tests for CI runs")


@pytest.fixture
def default_glm_config():
    """Default GLM configuration for testing."""
    return Glm4MoeConfig(
        hidden_size=2048,
        intermediate_size=10240,
        moe_intermediate_size=1536,
        num_local_experts=64,
        num_experts_per_tok=4,
        n_shared_experts=1,
        routed_scaling_factor=1.8,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )


# ============================================================================
# Qwen3.5 shared fixtures
# ============================================================================

# ---------- Constants ----------

QWEN_MESH_DEVICE_PARAM = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))

QWEN_DEVICE_PARAMS = {
    "trace_region_size": 500_000_000,
    "num_command_queues": 1,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
}

QWEN_DEVICE_PARAMS_200MB = {
    "trace_region_size": 200_000_000,
    "num_command_queues": 1,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
}


# ---------- Session-scoped fixtures ----------


@pytest.fixture(scope="session")
def qwen_config():
    """Load Qwen3.5-27B text config (no weights). Session-scoped for reuse."""
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-27B-FP8", trust_remote_code=True)
    return config.text_config


@pytest.fixture(scope="session")
def qwen_rope(qwen_config):
    """Create RoPE embedding. Session-scoped (stateless)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    return Qwen3_5TextRotaryEmbedding(config=qwen_config)


# ---------- Shared helpers ----------


def col_sharded_to_torch(tensor, mesh_device):
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    return ttnn.to_torch(tensor, mesh_composer=mesh_composer)


def extract_output(output, mesh_device):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    hs = output[0] if isinstance(output, tuple) else output
    if isinstance(hs, TorchTTNNTensor):
        return hs.to_torch
    if isinstance(hs, torch.Tensor):
        return hs
    if isinstance(hs, ttnn.Tensor):
        return col_sharded_to_torch(hs, mesh_device)
    raise TypeError(f"Unexpected output type: {type(hs)}")


def assert_valid(hs, expected_shape, label):
    assert hs.shape == expected_shape, f"{label}: expected {expected_shape}, got {hs.shape}"
    assert not torch.isnan(hs).any(), f"{label}: NaN in output"
    assert not torch.isinf(hs).any(), f"{label}: Inf in output"


def create_paged_kv_cache(config, mesh_device, layer_indices, batch_size=1):
    from models.experimental.tt_symbiote.modules.attention import PagedAttentionConfig
    from models.experimental.tt_symbiote.modules.qwen_attention import TTNNQwenPagedAttentionKVCache

    num_kv_heads = getattr(config, "num_key_value_heads", 4)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=batch_size)
    return TTNNQwenPagedAttentionKVCache(
        num_layers=len(layer_indices),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        config=paged_config,
        device=None,
        layer_indices=layer_indices,
    ).to_device(mesh_device)


def reset_all_states(ttnn_layers, paged_cache):
    """Reset GDN states on all layers and paged KV cache."""
    for layer in ttnn_layers:
        if layer.layer_type == "linear_attention":
            layer.attention.reset_state()
    paged_cache.reset()


def create_layer_stack(qwen_config, mesh_device, layer_indices):
    """Create N decoder layers, preprocess weights, move to device.

    Returns list of TTNNQwen35DecoderLayer objects.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    from models.experimental.tt_symbiote.modules.qwen35_decoder_layer import TTNNQwen35DecoderLayer
    from models.experimental.tt_symbiote.utils.device_management import set_device

    ttnn_layers = []
    for idx in layer_indices:
        torch_layer = Qwen3_5DecoderLayer(qwen_config, layer_idx=idx).to(torch.bfloat16)
        ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
        set_device(ttnn_layer, mesh_device)
        ttnn_layer.preprocess_weights()
        ttnn_layer.move_weights_to_device()
        ttnn_layers.append(ttnn_layer)
    return ttnn_layers


def simulate_generate_multi_layer(
    ttnn_layers,
    mesh_device,
    config,
    rope,
    paged_cache,
    prefill_seq_len,
    max_new_tokens,
    hidden_size,
    full_attn_layer_indices,
    run_label,
    hang_timeout=120,
):
    """Simulate generate(): 1 prefill + N decode steps chained through all layers.

    Args:
        full_attn_layer_indices: set or list of layer indices (within ttnn_layers)
            that are full_attention and need RoPE/KV cache args.
    """
    batch_size = 1
    prefill_x = torch.randn(batch_size, prefill_seq_len, hidden_size, dtype=torch.bfloat16)

    # Prefill
    hs = prefill_x
    prefill_pos_ids = torch.arange(prefill_seq_len).unsqueeze(0)
    prefill_cos, prefill_sin = rope(prefill_x, prefill_pos_ids)

    for i, layer in enumerate(ttnn_layers):
        if layer.layer_type == "linear_attention":
            hs = layer(hs)
        else:
            hs = layer(
                hs,
                position_embeddings=(prefill_cos, prefill_sin),
                past_key_values=paged_cache,
                cache_position=torch.arange(prefill_seq_len).unsqueeze(0),
            )
        hs_torch = extract_output(hs, mesh_device)
        assert_valid(hs_torch, (batch_size, prefill_seq_len, hidden_size), f"{run_label} prefill layer {i}")
        hs = hs_torch

    # Decode
    for step in range(max_new_tokens):
        pos = prefill_seq_len + step
        decode_x = torch.randn(batch_size, 1, hidden_size, dtype=torch.bfloat16)
        decode_pos = torch.tensor([[pos]])
        decode_cos, decode_sin = rope(decode_x, decode_pos)

        hs = decode_x
        for i, layer in enumerate(ttnn_layers):
            if layer.layer_type == "linear_attention":
                hs = layer(hs)
            else:
                hs = layer(
                    hs,
                    position_embeddings=(decode_cos, decode_sin),
                    past_key_values=paged_cache,
                    cache_position=decode_pos,
                )
            hs_torch = extract_output(hs, mesh_device)
            assert_valid(hs_torch, (batch_size, 1, hidden_size), f"{run_label} decode step {step} layer {i}")
            hs = hs_torch
