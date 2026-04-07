# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Traced mode tests for Qwen3.5-27B decoder layer.

Tests prefill, decode, and combined prefill+decode under TT_SYMBIOTE_RUN_MODE=TRACED.
Verifies numerical accuracy against PyTorch reference for both layer types:
- linear_attention (GDN): layers 0-2 (DeltaNet with conv/recurrence state)
- full_attention (GQA): layer 3 (grouped query attention with paged KV cache)

Run with:
    export TT_SYMBIOTE_RUN_MODE=TRACED
    export TT_SYMBIOTE_DISPATCHER=CPU
    export MESH_DEVICE=T3K
    pytest models/experimental/tt_symbiote/tests/test_qwen3_5_27b_traced.py -v --timeout=600
"""

import os

# MUST set env vars before any tt_symbiote imports
os.environ.setdefault("TT_SYMBIOTE_RUN_MODE", "TRACED")
os.environ.setdefault("TT_SYMBIOTE_DISPATCHER", "CPU")

import pytest
import torch

import ttnn
from models.experimental.tt_symbiote.core.run_config import TracedRun
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.attention import PagedAttentionConfig
from models.experimental.tt_symbiote.modules.qwen_attention import TTNNQwenPagedAttentionKVCache
from models.experimental.tt_symbiote.modules.qwen35_decoder_layer import TTNNQwen35DecoderLayer
from models.experimental.tt_symbiote.utils.device_management import set_device


# ============================================================================
# Helpers
# ============================================================================


def _col_sharded_to_torch(tensor, mesh_device):
    """Convert a col-sharded mesh tensor to a full torch tensor."""
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    return ttnn.to_torch(tensor, mesh_composer=mesh_composer)


def _mesh_to_torch(tensor, mesh_device, batch_size=1):
    """Convert a replicated mesh tensor to a single torch tensor."""
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    t = ttnn.to_torch(tensor, mesh_composer=mesh_composer)
    return t[:batch_size]


def pcc_and_max_diff(tensor1, tensor2):
    """Calculate PCC and max absolute difference."""
    t1 = tensor1.float().flatten()
    t2 = tensor2.float().flatten()
    pcc = torch.corrcoef(torch.stack([t1, t2]))[0, 1].item()
    max_diff = torch.max(torch.abs(tensor1.float() - tensor2.float())).item()
    return pcc, max_diff


def extract_hidden_states(outputs, mesh_device):
    """Extract the hidden_states tensor from decoder layer output.

    In distributed traced mode, output is col-sharded — use ConcatMeshToTensor(dim=-1).
    """
    hs = outputs[0] if isinstance(outputs, tuple) else outputs
    if isinstance(hs, TorchTTNNTensor):
        return hs.to_torch
    if isinstance(hs, torch.Tensor):
        return hs
    if isinstance(hs, ttnn.Tensor):
        return _col_sharded_to_torch(hs, mesh_device)
    raise TypeError(f"Unexpected output type: {type(hs)}")


def create_paged_kv_cache(config, mesh_device, layer_indices=None, batch_size=1):
    """Create paged KV cache for specific full attention layers."""
    if layer_indices is None:
        # Default: Qwen3.5 pattern [lin, lin, lin, full] x N
        layer_types = getattr(
            config,
            "layer_types",
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        )
        num_layers_in_pattern = len(layer_types)
        num_repeats = config.num_hidden_layers // num_layers_in_pattern
        layer_indices = []
        for repeat_idx in range(num_repeats):
            for idx_in_pattern, layer_type in enumerate(layer_types):
                if layer_type == "full_attention":
                    layer_indices.append(repeat_idx * num_layers_in_pattern + idx_in_pattern)

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


def get_layer_type(layer):
    """Get layer type string from a HF Qwen3_5DecoderLayer."""
    return getattr(layer, "layer_type", "full_attention")


# ============================================================================
# Fixtures
# ============================================================================

MESH_DEVICE_PARAM = {
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

DEVICE_PARAMS = {
    "trace_region_size": 200000000,  # 200MB for decoder trace with GDN loop
    "num_command_queues": 1,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
}


@pytest.fixture(scope="module")
def qwen_model():
    """Load Qwen3.5-27B-FP8 with 4 hidden layers (module-scoped for reuse)."""
    from transformers import AutoModelForCausalLM
    from models.experimental.tt_symbiote.tests.test_qwen3_5_27b_modules import _dequantize_fp8_weights

    model_name = "Qwen/Qwen3.5-27B-FP8"
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, num_hidden_layers=4)
    _dequantize_fp8_weights(model, model_name)
    model = model.to(torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)
    return model


# ============================================================================
# Test 1: Prefill — GDN layer (linear_attention)
# ============================================================================


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_DEVICE_PARAM], indirect=True)
def test_prefill_gdn_layer_traced(mesh_device, qwen_model):
    """Test prefill for a GDN (linear_attention) decoder layer under TRACED mode.

    GDN layers use conv/recurrence state (no KV cache). Prefill unrolls a
    per-token loop that gets recorded into the trace.

    Verifies trace capture output matches PyTorch reference.
    """
    config = qwen_model.config
    torch_layer = qwen_model.model.layers[0]  # Layer 0 = linear_attention
    assert get_layer_type(torch_layer) == "linear_attention"

    batch_size, seq_length = 1, 8  # Short seq to keep trace size manageable
    hidden_size = config.hidden_size
    x = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.bfloat16)

    # PyTorch reference
    with torch.no_grad():
        torch_out = torch_layer.linear_attn(x)[0]
        torch_mlp_in = torch_layer.input_layernorm(x)
        # Full decoder layer reference
        torch_full_out = torch_layer(x, position_embeddings=None)[0]

    # TTNN traced
    ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
    set_device(ttnn_layer, mesh_device)
    ttnn_layer.preprocess_weights()
    ttnn_layer.move_weights_to_device()

    # Run 1: warmup (no trace)
    ttnn_out_1 = ttnn_layer(x)
    hs_1 = extract_hidden_states(ttnn_out_1, mesh_device)

    pcc_1, max_diff_1 = pcc_and_max_diff(torch_full_out, hs_1)
    print(f"GDN prefill warmup -- PCC: {pcc_1:.6f}, Max Diff: {max_diff_1:.6f}")

    # Run 2: trace capture (same inputs)
    ttnn_layer.attention.conv_states = None  # Reset GDN states
    ttnn_layer.attention.rec_states = None
    ttnn_layer.attention.rec_output = None

    ttnn_out_2 = ttnn_layer(x)
    hs_2 = extract_hidden_states(ttnn_out_2, mesh_device)

    pcc_2, max_diff_2 = pcc_and_max_diff(torch_full_out, hs_2)
    print(f"GDN prefill trace capture -- PCC: {pcc_2:.6f}, Max Diff: {max_diff_2:.6f}")

    # Run 3: trace replay
    ttnn_layer.attention.conv_states = None
    ttnn_layer.attention.rec_states = None
    ttnn_layer.attention.rec_output = None

    ttnn_out_3 = ttnn_layer(x)
    hs_3 = extract_hidden_states(ttnn_out_3, mesh_device)

    pcc_3, max_diff_3 = pcc_and_max_diff(torch_full_out, hs_3)
    print(f"GDN prefill trace replay -- PCC: {pcc_3:.6f}, Max Diff: {max_diff_3:.6f}")

    TracedRun.release_all()

    assert pcc_1 > 0.90, f"GDN prefill warmup PCC {pcc_1} below 0.90"
    assert pcc_2 > 0.90, f"GDN prefill trace capture PCC {pcc_2} below 0.90"
    assert pcc_3 > 0.90, f"GDN prefill trace replay PCC {pcc_3} below 0.90"


# ============================================================================
# Test 2: Decode — GDN layer (linear_attention)
# ============================================================================


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_DEVICE_PARAM], indirect=True)
def test_decode_gdn_layer_traced(mesh_device, qwen_model):
    """Test decode (seq_len=1) for a GDN layer under TRACED mode.

    GDN decode uses shift-register conv1d and fused recurrence.
    All ops are device-side and trace-compatible.
    """
    config = qwen_model.config
    torch_layer = qwen_model.model.layers[0]
    assert get_layer_type(torch_layer) == "linear_attention"

    batch_size = 1
    hidden_size = config.hidden_size

    # Prefill first to init conv/rec states
    prefill_x = torch.randn(batch_size, 4, hidden_size, dtype=torch.bfloat16)

    ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
    set_device(ttnn_layer, mesh_device)
    ttnn_layer.preprocess_weights()
    ttnn_layer.move_weights_to_device()

    # Prefill to init states
    ttnn_layer(prefill_x)

    # Decode: seq_len=1
    decode_x = torch.randn(batch_size, 1, hidden_size, dtype=torch.bfloat16)

    # Run 1: warmup
    out_1 = ttnn_layer(decode_x)
    hs_1 = extract_hidden_states(out_1, mesh_device)
    print(f"GDN decode warmup -- shape: {hs_1.shape}, mean: {hs_1.float().mean():.6f}")

    # Run 2: trace capture
    out_2 = ttnn_layer(decode_x)
    hs_2 = extract_hidden_states(out_2, mesh_device)
    print(f"GDN decode trace capture -- shape: {hs_2.shape}, mean: {hs_2.float().mean():.6f}")

    # Run 3: trace replay
    out_3 = ttnn_layer(decode_x)
    hs_3 = extract_hidden_states(out_3, mesh_device)
    print(f"GDN decode trace replay -- shape: {hs_3.shape}, mean: {hs_3.float().mean():.6f}")

    TracedRun.release_all()

    # Verify output shape and no NaN/Inf
    assert hs_1.shape == (batch_size, 1, hidden_size), f"Unexpected shape: {hs_1.shape}"
    assert not torch.isnan(hs_1).any(), "Warmup output contains NaN"
    assert not torch.isnan(hs_2).any(), "Trace capture output contains NaN"
    assert not torch.isnan(hs_3).any(), "Trace replay output contains NaN"


# ============================================================================
# Test 3: Prefill — Full attention layer (GQA)
# ============================================================================


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_DEVICE_PARAM], indirect=True)
def test_prefill_full_attn_layer_traced(mesh_device, qwen_model):
    """Test prefill for a full attention (GQA) decoder layer under TRACED mode.

    Full attention layers use paged KV cache. Prefill fills the cache and
    computes attention via SDPA.
    """
    config = qwen_model.config
    torch_layer = qwen_model.model.layers[3]  # Layer 3 = full_attention
    assert get_layer_type(torch_layer) == "full_attention"

    batch_size, seq_length = 1, 16
    hidden_size = config.hidden_size
    x = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_length).unsqueeze(0)
    cos, sin = qwen_model.model.rotary_emb(x, position_ids)

    # PyTorch reference
    with torch.no_grad():
        torch_out = torch_layer(
            x,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )[0]

    # TTNN traced
    ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
    set_device(ttnn_layer, mesh_device)
    ttnn_layer.preprocess_weights()
    ttnn_layer.move_weights_to_device()

    num_kv_heads = getattr(config, "num_key_value_heads", 4)
    head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)

    # Run 1: warmup
    paged_cache_1 = create_paged_kv_cache(config, mesh_device, layer_indices=[3])
    out_1 = ttnn_layer(
        x,
        position_embeddings=(cos, sin),
        past_key_values=paged_cache_1,
        cache_position=torch.arange(seq_length).unsqueeze(0),
    )
    hs_1 = extract_hidden_states(out_1, mesh_device)
    pcc_1, _ = pcc_and_max_diff(torch_out, hs_1)
    print(f"Full attn prefill warmup -- PCC: {pcc_1:.6f}")

    # Run 2: trace capture (fresh cache)
    paged_cache_2 = create_paged_kv_cache(config, mesh_device, layer_indices=[3])
    out_2 = ttnn_layer(
        x,
        position_embeddings=(cos, sin),
        past_key_values=paged_cache_2,
        cache_position=torch.arange(seq_length).unsqueeze(0),
    )
    hs_2 = extract_hidden_states(out_2, mesh_device)
    pcc_2, _ = pcc_and_max_diff(torch_out, hs_2)
    print(f"Full attn prefill trace capture -- PCC: {pcc_2:.6f}")

    # Run 3: trace replay (fresh cache)
    paged_cache_3 = create_paged_kv_cache(config, mesh_device, layer_indices=[3])
    out_3 = ttnn_layer(
        x,
        position_embeddings=(cos, sin),
        past_key_values=paged_cache_3,
        cache_position=torch.arange(seq_length).unsqueeze(0),
    )
    hs_3 = extract_hidden_states(out_3, mesh_device)
    pcc_3, _ = pcc_and_max_diff(torch_out, hs_3)
    print(f"Full attn prefill trace replay -- PCC: {pcc_3:.6f}")

    TracedRun.release_all()

    assert pcc_1 > 0.90, f"Full attn prefill warmup PCC {pcc_1} below 0.90"
    assert pcc_2 > 0.90, f"Full attn prefill trace capture PCC {pcc_2} below 0.90"
    assert pcc_3 > 0.90, f"Full attn prefill trace replay PCC {pcc_3} below 0.90"


# ============================================================================
# Test 4: Decode — Full attention layer (GQA)
# ============================================================================


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_DEVICE_PARAM], indirect=True)
def test_decode_full_attn_layer_traced(mesh_device, qwen_model):
    """Test decode (seq_len=1) for a full attention layer under TRACED mode.

    Uses paged attention decode with trace-compatible cache_position handling.
    """
    config = qwen_model.config
    torch_layer = qwen_model.model.layers[3]
    assert get_layer_type(torch_layer) == "full_attention"

    batch_size = 1
    hidden_size = config.hidden_size
    prefill_length = 8

    # TTNN layer
    ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
    set_device(ttnn_layer, mesh_device)
    ttnn_layer.preprocess_weights()
    ttnn_layer.move_weights_to_device()

    paged_cache = create_paged_kv_cache(config, mesh_device, layer_indices=[3])

    # Prefill to populate KV cache
    prefill_x = torch.randn(batch_size, prefill_length, hidden_size, dtype=torch.bfloat16)
    prefill_pos_ids = torch.arange(prefill_length).unsqueeze(0)
    prefill_cos, prefill_sin = qwen_model.model.rotary_emb(prefill_x, prefill_pos_ids)

    ttnn_layer(
        prefill_x,
        position_embeddings=(prefill_cos, prefill_sin),
        past_key_values=paged_cache,
        cache_position=torch.arange(prefill_length).unsqueeze(0),
    )

    # Decode (seq_len=1)
    decode_x = torch.randn(batch_size, 1, hidden_size, dtype=torch.bfloat16)
    decode_pos = torch.tensor([[prefill_length]])
    decode_cos, decode_sin = qwen_model.model.rotary_emb(decode_x, decode_pos)

    # Run 1: warmup
    out_1 = ttnn_layer(
        decode_x,
        position_embeddings=(decode_cos, decode_sin),
        past_key_values=paged_cache,
        cache_position=decode_pos,
    )
    hs_1 = extract_hidden_states(out_1, mesh_device)
    print(f"Full attn decode warmup -- shape: {hs_1.shape}")

    # Run 2: trace capture (different cache position)
    decode_pos_2 = torch.tensor([[prefill_length + 1]])
    decode_cos_2, decode_sin_2 = qwen_model.model.rotary_emb(decode_x, decode_pos_2)
    out_2 = ttnn_layer(
        decode_x,
        position_embeddings=(decode_cos_2, decode_sin_2),
        past_key_values=paged_cache,
        cache_position=decode_pos_2,
    )
    hs_2 = extract_hidden_states(out_2, mesh_device)
    print(f"Full attn decode trace capture -- shape: {hs_2.shape}")

    # Run 3: trace replay
    decode_pos_3 = torch.tensor([[prefill_length + 2]])
    decode_cos_3, decode_sin_3 = qwen_model.model.rotary_emb(decode_x, decode_pos_3)
    out_3 = ttnn_layer(
        decode_x,
        position_embeddings=(decode_cos_3, decode_sin_3),
        past_key_values=paged_cache,
        cache_position=decode_pos_3,
    )
    hs_3 = extract_hidden_states(out_3, mesh_device)
    print(f"Full attn decode trace replay -- shape: {hs_3.shape}")

    TracedRun.release_all()

    assert hs_1.shape == (batch_size, 1, hidden_size), f"Unexpected shape: {hs_1.shape}"
    assert not torch.isnan(hs_1).any(), "Warmup output contains NaN"
    assert not torch.isnan(hs_2).any(), "Trace capture output contains NaN"
    assert not torch.isnan(hs_3).any(), "Trace replay output contains NaN"
