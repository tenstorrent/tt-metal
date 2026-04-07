# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Module-by-module accuracy tests for Qwen3.5-27B-FP8 TTNN modules.

Tests cover individual components of the hybrid DeltaNet + softmax attention
architecture used in Qwen/Qwen3.5-27B-FP8:

1. Linear projections for full and linear attention
2. Partial rotary position embeddings (RoPE)
3. Q/K RMSNorm with (1+weight) adjustment
4. Paged KV cache with layer_indices mapping
5. Full attention (prefill + decode)
6. Linear attention at various sequence lengths
7. End-to-end decoder layer accuracy

Model architecture:
- 64 layers, pattern [linear, linear, linear, full] x 16
- Linear attention: Qwen3_5GatedDeltaNet -> TTNNQwen3LinearAttention
- Full attention: GQA with 4 KV heads, head_dim=256 -> TTNNQwen3FullAttention
- Dense SwiGLU MLP (NO MoE)
- FP8 weights cast to bfloat16
- Q gating: q_proj outputs 2x dim, split into Q and gate
- Q/K normalization: RMSNorm with (1+weight) adjustment
- Partial RoPE: rotary_dim=64, head_dim=256
- Flat config (no text_config nesting)
"""

import os
import pytest
import torch

# Try importing TTNN; skip tests if unavailable
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

# Try importing transformers (5.0+ required for Qwen3.5)
try:
    import transformers

    TRANSFORMERS_5 = transformers.__version__.startswith("5.")
except ImportError:
    TRANSFORMERS_5 = False

# Try importing tt_symbiote modules
try:
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
    from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
    from models.experimental.tt_symbiote.modules.qwen_attention import (
        TTNNQwen3LinearAttention,
        TTNNQwen3FullAttention,
        TTNNQwenPagedAttentionKVCache,
    )
    from models.experimental.tt_symbiote.modules.attention import PagedAttentionConfig
    from models.experimental.tt_symbiote.modules.linear import TTNNLinear
    from models.experimental.tt_symbiote.modules.rope import TTNNRotaryPositionEmbedding
    from models.experimental.tt_symbiote.utils.device_management import set_device
    from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

    TT_SYMBIOTE_AVAILABLE = True
except ImportError:
    TT_SYMBIOTE_AVAILABLE = False
    TorchTTNNTensor = None
    compare_fn_outputs = None
    TTNNQwen3LinearAttention = None
    TTNNQwen3FullAttention = None
    TTNNQwenPagedAttentionKVCache = None
    PagedAttentionConfig = None
    TTNNLinear = None
    TTNNRotaryPositionEmbedding = None
    set_device = None
    register_module_replacement_dict = None


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"

# Accuracy thresholds (PCC = Pearson Correlation Coefficient)
PCC_LINEAR_PROJ = 0.99
PCC_ROPE = 0.99
PCC_QK_NORM = 0.98
PCC_LINEAR_ATTN = 0.95
PCC_FULL_ATTN_PREFILL = 0.95
PCC_FULL_ATTN_DECODE = 0.90
PCC_DECODER_LAYER = 0.90


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def get_config_attr(config, attr):
    """Get config attribute, handling both flat and nested (text_config) styles.

    Qwen3.5 models use config.text_config.{attr} while other models
    may use config.{attr} directly.
    """
    value = getattr(config, attr, None)
    if value is None and hasattr(config, "text_config"):
        value = getattr(config.text_config, attr, None)
    return value


def set_config_attr(config, attr, value):
    """Set config attribute, handling both flat and nested (text_config) styles.

    For nested configs, we need to set on text_config for model initialization.
    """
    if hasattr(config, "text_config"):
        setattr(config.text_config, attr, value)
    else:
        setattr(config, attr, value)


def compute_pcc(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    a = tensor_a.float().flatten()
    b = tensor_b.float().flatten()

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if (a == 0).all() and (b == 0).all():
        return 1.0

    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1]

    if pcc.isnan():
        return 0.0

    return pcc.item()


def assert_with_pcc(torch_output, ttnn_output, pcc_threshold=0.95):
    """Assert TTNN output matches torch using Pearson Correlation Coefficient."""
    if TorchTTNNTensor is not None and isinstance(ttnn_output, TorchTTNNTensor):
        ttnn_np = ttnn_output.to_torch.to(torch.float32)
    elif TTNN_AVAILABLE and isinstance(ttnn_output, ttnn.Tensor):
        ttnn_np = ttnn.to_torch(ttnn_output).to(torch.float32)
    else:
        ttnn_np = ttnn_output.to(torch.float32)

    if TorchTTNNTensor is not None and isinstance(torch_output, TorchTTNNTensor):
        torch_out = torch_output.to_torch.to(torch.float32)
    else:
        torch_out = torch_output.to(torch.float32)

    torch_flat = torch_out.flatten()
    ttnn_flat = ttnn_np.flatten()

    mean_t = torch_flat.mean()
    mean_n = ttnn_flat.mean()
    diff_t = torch_flat - mean_t
    diff_n = ttnn_flat - mean_n
    pcc = (diff_t * diff_n).sum() / (torch.sqrt((diff_t**2).sum()) * torch.sqrt((diff_n**2).sum()) + 1e-12)

    pcc_val = pcc.item()
    print(f"  PCC = {pcc_val:.6f} (threshold = {pcc_threshold})")
    assert pcc_val >= pcc_threshold, f"PCC {pcc_val:.6f} < {pcc_threshold}"
    return pcc_val


def _dequantize_fp8_weights(model, model_name):
    """Dequantize FP8 block-quantized weights in-place.

    Transformers drops the weight_scale_inv tensors during loading, so the
    FP8 weights are cast to bfloat16 without proper scaling.  This function
    reads the raw FP8 weights and their block-wise scale factors from the
    safetensors files and applies the dequantization: bf16 = fp8 * scale_inv.
    """
    import json
    from safetensors import safe_open
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(model_name)

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return  # Not a sharded model or no FP8 weights

    with open(index_path) as f:
        index = json.load(f)

    # Build map: model param name -> (scale_key, weight_key, shard files)
    scale_map = {}
    for k, shard in index["weight_map"].items():
        if k.endswith(".weight_scale_inv"):
            weight_key = k.replace(".weight_scale_inv", ".weight")
            # Strip 'model.language_model.' prefix to match model param names
            param_key = weight_key.replace("model.language_model.", "model.")
            scale_map[param_key] = (
                k,
                os.path.join(model_dir, shard),
                weight_key,
                os.path.join(model_dir, index["weight_map"].get(weight_key, shard)),
            )

    if not scale_map:
        return

    shard_cache = {}
    dequantized = 0
    block_size = 128

    for param_name, param in model.named_parameters():
        if param_name not in scale_map:
            continue

        scale_key, scale_shard, weight_key, weight_shard = scale_map[param_name]

        if scale_shard not in shard_cache:
            shard_cache[scale_shard] = safe_open(scale_shard, framework="pt")
        if weight_shard not in shard_cache:
            shard_cache[weight_shard] = safe_open(weight_shard, framework="pt")

        scale_inv = shard_cache[scale_shard].get_tensor(scale_key)
        raw_weight = shard_cache[weight_shard].get_tensor(weight_key)

        rows, cols = raw_weight.shape
        weight_float = raw_weight.to(torch.float32)
        scale_rows, scale_cols = scale_inv.shape

        for i in range(scale_rows):
            for j in range(scale_cols):
                r_s = i * block_size
                r_e = min(r_s + block_size, rows)
                c_s = j * block_size
                c_e = min(c_s + block_size, cols)
                weight_float[r_s:r_e, c_s:c_e] *= scale_inv[i, j].float()

        param.data.copy_(weight_float.to(torch.bfloat16))
        dequantized += 1

    del shard_cache
    print(f"  Dequantized {dequantized}/{len(scale_map)} FP8 weights")


def load_model(num_hidden_layers: int):
    """Load Qwen3.5-27B-FP8 model with a limited number of layers.

    FP8 weights are block-quantized (128x128) with scale factors.  We load
    the model normally, then apply manual dequantization from the safetensors
    scale_inv tensors that transformers drops during loading.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    set_config_attr(config, "num_hidden_layers", num_hidden_layers)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        trust_remote_code=True,
    )

    print("  Applying FP8 dequantization...")
    _dequantize_fp8_weights(model, MODEL_NAME)

    model = model.to(torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    return model, config


def get_layer_type(layer) -> str:
    """Return 'linear_attention' or 'full_attention' for a model layer."""
    layer_type = getattr(layer, "layer_type", None)
    if layer_type is not None:
        return layer_type
    # Fallback: check for attribute presence
    if hasattr(layer, "linear_attn"):
        return "linear_attention"
    return "full_attention"


def get_attention_module(layer):
    """Get the attention submodule from a decoder layer."""
    layer_type = get_layer_type(layer)
    if layer_type == "linear_attention":
        return layer.linear_attn
    return layer.self_attn


# ──────────────────────────────────────────────────────────────────────
# Skip decorators
# ──────────────────────────────────────────────────────────────────────

skip_no_ttnn = pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")
skip_no_transformers = pytest.mark.skipif(not TRANSFORMERS_5, reason="Requires transformers 5.0+ for Qwen3.5")
skip_no_symbiote = pytest.mark.skipif(not TT_SYMBIOTE_AVAILABLE, reason="tt_symbiote modules not available")


# ──────────────────────────────────────────────────────────────────────
# Test 1: Linear projections — full attention (q/k/v/o_proj)
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_linear_projections_full_attention(device):
    """Test q/k/v/o_proj as individual TTNN matmuls for full attention layer.

    Full attention is layer 3 (pattern: linear, linear, linear, full).
    Tests each projection independently against PyTorch reference.
    PCC >= 0.99 expected.
    """
    model, config = load_model(num_hidden_layers=4)
    # Layer 3 is the first full attention layer
    torch_attn = model.model.layers[3].self_attn

    batch_size, seq_len = 1, 32
    hidden_states = torch.randn(batch_size, seq_len, get_config_attr(config, "hidden_size"), dtype=torch.bfloat16)

    # Test each projection independently
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        torch_proj = getattr(torch_attn, proj_name)

        # Determine input size for o_proj (num_heads * head_dim) vs others (hidden_size)
        if proj_name == "o_proj":
            proj_input = torch.randn(
                batch_size,
                seq_len,
                get_config_attr(config, "num_attention_heads") * get_config_attr(config, "head_dim"),
                dtype=torch.bfloat16,
            )
        else:
            proj_input = hidden_states

        # PyTorch reference
        with torch.no_grad():
            torch_out = torch_proj(proj_input)

        # TTNN version
        ttnn_proj = TTNNLinear.from_torch(torch_proj)
        set_device(ttnn_proj, device)
        ttnn_proj.preprocess_weights()
        ttnn_proj.move_weights_to_device()

        ttnn_input = TorchTTNNTensor(proj_input)
        ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
        ttnn_out = ttnn_proj(ttnn_input.ttnn_tensor)

        print(f"  {proj_name}: torch_out shape={torch_out.shape}")
        pcc = assert_with_pcc(torch_out, ttnn_out, PCC_LINEAR_PROJ)
        print(f"  {proj_name} PCC = {pcc:.6f}")

    print("PASS: test_linear_projections_full_attention")


# ──────────────────────────────────────────────────────────────────────
# Test 2: Linear projections — linear attention (in_proj_qkv/in_proj_z/out_proj)
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_linear_projections_linear_attention(device):
    """Test in_proj_qkv, in_proj_z, and out_proj for linear attention layer.

    Linear attention is layer 0 (pattern: linear, linear, linear, full).
    Tests each TTNN-accelerated projection independently.
    PCC >= 0.99 expected.
    """
    model, config = load_model(num_hidden_layers=1)
    torch_layer = model.model.layers[0].linear_attn

    batch_size, seq_len = 1, 32
    hidden_states = torch.randn(batch_size, seq_len, get_config_attr(config, "hidden_size"), dtype=torch.bfloat16)

    # Test in_proj_qkv (hidden_size -> key_dim * 2 + value_dim)
    proj_pairs = [
        ("in_proj_qkv", hidden_states),
        ("in_proj_z", hidden_states),
    ]

    for proj_name, proj_input in proj_pairs:
        torch_proj = getattr(torch_layer, proj_name)

        with torch.no_grad():
            torch_out = torch_proj(proj_input)

        ttnn_proj = TTNNLinear.from_torch(torch_proj)
        set_device(ttnn_proj, device)
        ttnn_proj.preprocess_weights()
        ttnn_proj.move_weights_to_device()

        ttnn_input = TorchTTNNTensor(proj_input)
        ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
        ttnn_out = ttnn_proj(ttnn_input.ttnn_tensor)

        print(f"  {proj_name}: torch_out shape={torch_out.shape}")
        pcc = assert_with_pcc(torch_out, ttnn_out, PCC_LINEAR_PROJ)
        print(f"  {proj_name} PCC = {pcc:.6f}")

    # Test out_proj (value_dim -> hidden_size)
    torch_out_proj = torch_layer.out_proj
    value_dim = torch_layer.value_dim
    out_proj_input = torch.randn(batch_size, seq_len, value_dim, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_out_proj(out_proj_input)

    ttnn_proj = TTNNLinear.from_torch(torch_out_proj)
    set_device(ttnn_proj, device)
    ttnn_proj.preprocess_weights()
    ttnn_proj.move_weights_to_device()

    ttnn_input = TorchTTNNTensor(out_proj_input)
    ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
    ttnn_out = ttnn_proj(ttnn_input.ttnn_tensor)

    print(f"  out_proj: torch_out shape={torch_out.shape}")
    pcc = assert_with_pcc(torch_out, ttnn_out, PCC_LINEAR_PROJ)
    print(f"  out_proj PCC = {pcc:.6f}")

    print("PASS: test_linear_projections_linear_attention")


# ──────────────────────────────────────────────────────────────────────
# Test 3: RoPE with partial rotary (64/256 dims)
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_rope_partial_rotary(device):
    """Test RoPE with partial rotary (rotary_dim=64, head_dim=256).

    Qwen3.5-27B-FP8 uses partial rotary factor=0.25, meaning only the first 64
    of 256 head dimensions get rotary embeddings. The remaining 192 dimensions
    pass through unchanged.
    PCC >= 0.99 expected.
    """
    model, config = load_model(num_hidden_layers=4)

    batch_size, seq_len = 1, 32
    num_heads = get_config_attr(config, "num_attention_heads")  # 24
    num_kv_heads = get_config_attr(config, "num_key_value_heads")  # 4
    head_dim = get_config_attr(config, "head_dim")  # 256

    # Create Q and K in [batch, num_heads, seq_len, head_dim] format
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

    # Get position embeddings from the model's rotary_emb
    hidden_dummy = torch.randn(batch_size, seq_len, get_config_attr(config, "hidden_size"), dtype=torch.bfloat16)
    pos = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden_dummy, pos)
    # cos, sin shape: [1, seq_len, rotary_dim] where rotary_dim=64

    # PyTorch reference: apply_rotary_pos_emb from transformers
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb as qwen_apply_rotary

    torch_q_rot, torch_k_rot = qwen_apply_rotary(q, k, cos, sin)

    # TTNN version
    rope = TTNNRotaryPositionEmbedding()
    set_device(rope, device)

    # Convert inputs to TTNN
    q_tt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    k_tt = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Expand cos/sin to [1, 1, seq_len, rotary_dim] for broadcasting
    cos_4d = cos.unsqueeze(1) if len(cos.shape) == 3 else cos
    sin_4d = sin.unsqueeze(1) if len(sin.shape) == 3 else sin
    cos_tt = ttnn.from_torch(cos_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_tt = ttnn.from_torch(sin_4d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_q_rot, ttnn_k_rot = rope(q_tt, k_tt, cos_tt, sin_tt)

    print(f"  Q rotated shape: torch={torch_q_rot.shape}")
    pcc_q = assert_with_pcc(torch_q_rot, ttnn_q_rot, PCC_ROPE)
    print(f"  Q RoPE PCC = {pcc_q:.6f}")

    pcc_k = assert_with_pcc(torch_k_rot, ttnn_k_rot, PCC_ROPE)
    print(f"  K RoPE PCC = {pcc_k:.6f}")

    print("PASS: test_rope_partial_rotary")


# ──────────────────────────────────────────────────────────────────────
# Test 4: Q/K normalization — RMSNorm with (1+weight)
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_qk_normalization(device):
    """Test Q/K RMSNorm with (1+weight) adjustment.

    Qwen3.5 applies RMSNorm to Q and K before RoPE. The PyTorch layer uses
    weight initialized near zero with forward: output = rms_norm(x) * (1.0 + weight).
    TTNN rms_norm uses: output = rms_norm(x) * weight, so we pre-add 1.0.
    PCC >= 0.98 expected.
    """
    model, config = load_model(num_hidden_layers=4)
    torch_attn = model.model.layers[3].self_attn

    batch_size, seq_len = 1, 32
    head_dim = get_config_attr(config, "head_dim")  # 256
    num_heads = get_config_attr(config, "num_attention_heads")  # 24
    num_kv_heads = get_config_attr(config, "num_key_value_heads")  # 4

    # Test Q normalization
    q_states = torch.randn(batch_size * seq_len * num_heads, head_dim, dtype=torch.bfloat16)

    # PyTorch reference: rms_norm(x) * (1 + weight)
    torch_q_norm = torch_attn.q_norm
    with torch.no_grad():
        torch_q_out = torch_q_norm(q_states)

    # TTNN version: pre-adjust weight to (1 + weight), then use ttnn.rms_norm
    q_norm_weight = torch_q_norm.weight.detach().clone()
    q_norm_adjusted = (1.0 + q_norm_weight.float()).to(q_norm_weight.dtype)
    eps = getattr(torch_q_norm, "eps", 1e-6)

    tt_q_norm_weight = ttnn.from_torch(
        q_norm_adjusted.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_q_states = ttnn.from_torch(
        q_states.unsqueeze(0) if q_states.dim() == 1 else q_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn_q_out = ttnn.rms_norm(tt_q_states, weight=tt_q_norm_weight, epsilon=eps)

    print(f"  Q norm: input shape={q_states.shape}")
    pcc_q = assert_with_pcc(torch_q_out, ttnn_q_out, PCC_QK_NORM)
    print(f"  Q norm PCC = {pcc_q:.6f}")

    # Test K normalization
    k_states = torch.randn(batch_size * seq_len * num_kv_heads, head_dim, dtype=torch.bfloat16)

    torch_k_norm = torch_attn.k_norm
    with torch.no_grad():
        torch_k_out = torch_k_norm(k_states)

    k_norm_weight = torch_k_norm.weight.detach().clone()
    k_norm_adjusted = (1.0 + k_norm_weight.float()).to(k_norm_weight.dtype)
    k_eps = getattr(torch_k_norm, "eps", 1e-6)

    tt_k_norm_weight = ttnn.from_torch(
        k_norm_adjusted.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_k_states = ttnn.from_torch(
        k_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn_k_out = ttnn.rms_norm(tt_k_states, weight=tt_k_norm_weight, epsilon=k_eps)

    print(f"  K norm: input shape={k_states.shape}")
    pcc_k = assert_with_pcc(torch_k_out, ttnn_k_out, PCC_QK_NORM)
    print(f"  K norm PCC = {pcc_k:.6f}")

    print("PASS: test_qk_normalization")


# ──────────────────────────────────────────────────────────────────────
# Test 5: Paged KV cache — layer_indices mapping
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_paged_kv_cache_layer_mapping(device):
    """Test that TTNNQwenPagedAttentionKVCache maps layer_indices correctly.

    Qwen3.5-27B-FP8 has 64 layers with pattern [linear, linear, linear, full] x 16.
    Full attention layers are at indices 3, 7, 11, 15, ..., 63.
    The KV cache should map layer_idx=3 -> cache_idx=0, layer_idx=7 -> cache_idx=1, etc.
    """
    # Full attention layer indices for 64-layer model
    full_attn_indices = [i * 4 + 3 for i in range(16)]  # [3, 7, 11, ..., 63]

    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=1)
    paged_cache = TTNNQwenPagedAttentionKVCache(
        num_layers=16,  # 16 full attention layers
        num_kv_heads=4,
        head_dim=256,
        config=paged_config,
        device=None,
        layer_indices=full_attn_indices,
    )

    # Verify mapping: layer_idx -> cache_idx
    for cache_idx, layer_idx in enumerate(full_attn_indices):
        mapped = paged_cache._get_cache_idx(layer_idx)
        assert mapped == cache_idx, f"Layer {layer_idx} should map to cache_idx {cache_idx}, got {mapped}"

    # Verify that unmapped indices fall through to identity
    unmapped_idx = 0  # linear attention layer, not in layer_indices
    result = paged_cache._get_cache_idx(unmapped_idx)
    assert result == unmapped_idx, f"Unmapped layer {unmapped_idx} should return itself ({unmapped_idx}), got {result}"

    # Verify has_previous_state starts as False
    assert not paged_cache.has_previous_state(), "Cache should start with no previous state"

    print(f"  Layer indices: {full_attn_indices}")
    print(f"  All {len(full_attn_indices)} mappings verified correctly")
    print("PASS: test_paged_kv_cache_layer_mapping")


# ──────────────────────────────────────────────────────────────────────
# Test 6: Paged KV cache — fill and decode cycle
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_paged_kv_cache_fill_and_decode(device):
    """Test paged KV cache fill, update, and sequence length tracking.

    Verifies that the cache correctly:
    1. Fills KV entries during prefill (paged_fill_on_device)
    2. Updates single tokens during decode (paged_update_on_device)
    3. Tracks sequence lengths through the fill/update cycle
    """
    layer_indices = [3]  # Single full attention layer for simplicity
    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=1)
    paged_cache = TTNNQwenPagedAttentionKVCache(
        num_layers=1,
        num_kv_heads=4,
        head_dim=256,
        config=paged_config,
        device=None,
        layer_indices=layer_indices,
    ).to_device(device)

    # Prefill: fill with 5 tokens
    prefill_len = 5
    key_states = torch.randn(1, 4, prefill_len, 256, dtype=torch.bfloat16)
    value_states = torch.randn(1, 4, prefill_len, 256, dtype=torch.bfloat16)

    key_tt = ttnn.from_torch(key_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    value_tt = ttnn.from_torch(value_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Fill using absolute layer_idx=3 (should map to cache_idx=0)
    paged_cache.paged_fill_on_device(key_tt, value_tt, layer_idx=3, batch_idx=0)

    # Check sequence length after fill
    seq_len_after_fill = paged_cache.get_seq_length(0)
    assert seq_len_after_fill == prefill_len, f"Expected seq_length={prefill_len} after fill, got {seq_len_after_fill}"

    # Decode: update with 1 token
    decode_key = torch.randn(1, 1, 4, 256, dtype=torch.bfloat16)
    decode_value = torch.randn(1, 1, 4, 256, dtype=torch.bfloat16)

    # paged_update expects [S, B, H, D] layout for decode
    decode_key_tt = ttnn.from_torch(decode_key, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    decode_value_tt = ttnn.from_torch(decode_value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Need to shard for paged_update_on_device
    tile_size = 32
    shard_h = ((4 + tile_size - 1) // tile_size) * tile_size  # 32
    core_grid = ttnn.CoreGrid(y=1, x=1)
    shard_cfg = ttnn.create_sharded_memory_config(
        shape=(shard_h, 256),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    decode_key_tt = ttnn.to_memory_config(decode_key_tt, shard_cfg)
    decode_value_tt = ttnn.to_memory_config(decode_value_tt, shard_cfg)

    cur_pos = torch.tensor([prefill_len], dtype=torch.int32)
    cur_pos_tt = ttnn.from_torch(cur_pos, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    paged_cache.paged_update_on_device(decode_key_tt, decode_value_tt, layer_idx=3, current_pos=cur_pos_tt)

    # Check sequence length after decode
    seq_len_after_decode = paged_cache.get_seq_length(0)
    assert (
        seq_len_after_decode == prefill_len + 1
    ), f"Expected seq_length={prefill_len + 1} after decode, got {seq_len_after_decode}"

    print(f"  Seq length after fill: {seq_len_after_fill}")
    print(f"  Seq length after decode: {seq_len_after_decode}")
    print("PASS: test_paged_kv_cache_fill_and_decode")


# ──────────────────────────────────────────────────────────────────────
# Test 7: Linear attention accuracy (TTNNQwen3LinearAttention)
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 32, 64], ids=["decode", "short_prefill", "prefill"])
def test_linear_attention_accuracy(device, seq_len):
    """Test TTNNQwen3LinearAttention output matches PyTorch reference.

    Tests with different sequence lengths:
    - T=1: Decode mode (recurrent path)
    - T=32: Short prefill
    - T=64: Full prefill (chunk_size=64)

    PCC >= 0.95 expected.
    """
    # Enable TTNN projections BEFORE layer creation (flag is read at __init__ time)
    original_env = os.environ.get("TTNN_LINEAR_ATTN_PROJECTIONS", "0")
    os.environ["TTNN_LINEAR_ATTN_PROJECTIONS"] = "1"

    try:
        model, config = load_model(num_hidden_layers=1)
        torch_layer = model.model.layers[0].linear_attn

        # Create TTNN version (reads TTNN_LINEAR_ATTN_PROJECTIONS at init)
        ttnn_layer = TTNNQwen3LinearAttention.from_torch(torch_layer, distributed=False)
        set_device(ttnn_layer, device)
        ttnn_layer.preprocess_weights()
        ttnn_layer.move_weights_to_device()

        batch_size = 1
        hidden_states = torch.randn(batch_size, seq_len, get_config_attr(config, "hidden_size"), dtype=torch.bfloat16)

        # PyTorch reference
        with torch.no_grad():
            torch_out = torch_layer(hidden_states)

        # TTNN forward
        ttnn_input = TorchTTNNTensor(hidden_states)
        ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
        ttnn_out = ttnn_layer(ttnn_input)

        compare_fn_outputs(torch_out, ttnn_out, f"TTNNQwen3LinearAttention_T={seq_len}")
        pcc = assert_with_pcc(torch_out, ttnn_out, PCC_LINEAR_ATTN)
        print(f"  Linear attention T={seq_len} PCC = {pcc:.6f}")
    finally:
        os.environ["TTNN_LINEAR_ATTN_PROJECTIONS"] = original_env

    print(f"PASS: test_linear_attention_accuracy T={seq_len}")


# ──────────────────────────────────────────────────────────────────────
# Test 8: Full attention prefill (TTNNQwen3FullAttention)
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_full_attention_prefill(device):
    """Test TTNNQwen3FullAttention prefill path matches PyTorch reference.

    Tests the full attention layer (layer 3) with a short prefill sequence.
    This exercises Q gating, Q/K normalization, partial RoPE, and SDPA.
    PCC >= 0.95 expected.
    """
    model, config = load_model(num_hidden_layers=4)
    torch_attn = model.model.layers[3].self_attn

    # Create TTNN version
    ttnn_attn = TTNNQwen3FullAttention.from_torch(torch_attn, distributed=False)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    batch_size, seq_len = 1, 32
    hidden_states = torch.randn(batch_size, seq_len, get_config_attr(config, "hidden_size"), dtype=torch.bfloat16)

    # Position embeddings
    pos = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden_states, pos)

    # PyTorch reference with DynamicCache
    from transformers.cache_utils import DynamicCache

    dynamic_cache = DynamicCache()

    with torch.no_grad():
        torch_out = torch_attn(
            hidden_states,
            position_embeddings=(cos, sin),
            past_key_values=dynamic_cache,
            attention_mask=None,
        )

    # TTNN version with paged KV cache
    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=1)
    paged_cache = TTNNQwenPagedAttentionKVCache(
        num_layers=1,
        num_kv_heads=get_config_attr(config, "num_key_value_heads"),
        head_dim=get_config_attr(config, "head_dim"),
        config=paged_config,
        device=None,
        layer_indices=[3],
    ).to_device(device)

    ttnn_input = TorchTTNNTensor(hidden_states)
    ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)

    cache_position = torch.arange(seq_len).unsqueeze(0)

    ttnn_out = ttnn_attn(
        ttnn_input.ttnn_tensor,
        position_embeddings=(cos, sin),
        past_key_values=paged_cache,
        cache_position=cache_position,
    )

    # torch_out is a tuple (attn_output, attn_weights); ttnn_out is also a tuple
    torch_attn_out = torch_out[0] if isinstance(torch_out, (tuple, list)) else torch_out
    ttnn_attn_out = ttnn_out[0] if isinstance(ttnn_out, (tuple, list)) else ttnn_out

    compare_fn_outputs(torch_attn_out, ttnn_attn_out, "TTNNQwen3FullAttention_prefill")
    pcc = assert_with_pcc(torch_attn_out, ttnn_attn_out, PCC_FULL_ATTN_PREFILL)
    print(f"  Full attention prefill PCC = {pcc:.6f}")
    print("PASS: test_full_attention_prefill")


# ──────────────────────────────────────────────────────────────────────
# Test 9: Full attention decode (after prefill)
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_full_attention_decode(device):
    """Test TTNNQwen3FullAttention decode path after prefill.

    Performs a prefill with 32 tokens, then a decode step with 1 token.
    Compares the decode output against PyTorch reference.
    PCC >= 0.90 expected (lower due to accumulated KV cache differences).
    """
    model, config = load_model(num_hidden_layers=4)
    torch_attn = model.model.layers[3].self_attn

    # Create TTNN version
    ttnn_attn = TTNNQwen3FullAttention.from_torch(torch_attn, distributed=False)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    batch_size = 1
    prefill_len = 32

    # --- Prefill phase ---
    prefill_input = torch.randn(batch_size, prefill_len, get_config_attr(config, "hidden_size"), dtype=torch.bfloat16)
    prefill_pos = torch.arange(prefill_len).unsqueeze(0)
    prefill_cos, prefill_sin = model.model.rotary_emb(prefill_input, prefill_pos)

    # PyTorch prefill
    from transformers.cache_utils import DynamicCache

    dynamic_cache = DynamicCache()
    with torch.no_grad():
        _ = torch_attn(
            prefill_input,
            position_embeddings=(prefill_cos, prefill_sin),
            past_key_values=dynamic_cache,
            attention_mask=None,
        )

    # TTNN prefill
    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=1)
    paged_cache = TTNNQwenPagedAttentionKVCache(
        num_layers=1,
        num_kv_heads=get_config_attr(config, "num_key_value_heads"),
        head_dim=get_config_attr(config, "head_dim"),
        config=paged_config,
        device=None,
        layer_indices=[3],
    ).to_device(device)

    ttnn_prefill_input = TorchTTNNTensor(prefill_input)
    ttnn_prefill_input.ttnn_tensor = ttnn.to_device(ttnn_prefill_input.to_ttnn, device)
    prefill_cache_pos = torch.arange(prefill_len).unsqueeze(0)

    _ = ttnn_attn(
        ttnn_prefill_input.ttnn_tensor,
        position_embeddings=(prefill_cos, prefill_sin),
        past_key_values=paged_cache,
        cache_position=prefill_cache_pos,
    )

    # --- Decode phase ---
    decode_input = torch.randn(batch_size, 1, get_config_attr(config, "hidden_size"), dtype=torch.bfloat16)
    decode_pos = torch.tensor([[prefill_len]])
    decode_cos, decode_sin = model.model.rotary_emb(decode_input, decode_pos)

    # PyTorch decode
    with torch.no_grad():
        torch_decode_out = torch_attn(
            decode_input,
            position_embeddings=(decode_cos, decode_sin),
            past_key_values=dynamic_cache,
            attention_mask=None,
        )

    # TTNN decode
    ttnn_decode_input = TorchTTNNTensor(decode_input)
    ttnn_decode_input.ttnn_tensor = ttnn.to_device(ttnn_decode_input.to_ttnn, device)
    decode_cache_pos = torch.tensor([[prefill_len]])

    ttnn_decode_out = ttnn_attn(
        ttnn_decode_input.ttnn_tensor,
        position_embeddings=(decode_cos, decode_sin),
        past_key_values=paged_cache,
        cache_position=decode_cache_pos,
    )

    torch_decode_attn = torch_decode_out[0] if isinstance(torch_decode_out, (tuple, list)) else torch_decode_out
    ttnn_decode_attn = ttnn_decode_out[0] if isinstance(ttnn_decode_out, (tuple, list)) else ttnn_decode_out

    compare_fn_outputs(torch_decode_attn, ttnn_decode_attn, "TTNNQwen3FullAttention_decode")
    pcc = assert_with_pcc(torch_decode_attn, ttnn_decode_attn, PCC_FULL_ATTN_DECODE)
    print(f"  Full attention decode PCC = {pcc:.6f}")
    print("PASS: test_full_attention_decode")


# ──────────────────────────────────────────────────────────────────────
# Test 10: Decoder layer — linear attention end-to-end
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_decoder_layer_linear_attention(device):
    """Test end-to-end decoder layer with linear attention.

    Loads layer 0 (linear attention), captures PyTorch output BEFORE module
    replacement, then replaces the linear_attn submodule with TTNN and
    compares outputs.
    PCC >= 0.90 expected.
    """
    # Set env BEFORE layer creation (flag read at __init__)
    original_env = os.environ.get("TTNN_LINEAR_ATTN_PROJECTIONS", "0")
    os.environ["TTNN_LINEAR_ATTN_PROJECTIONS"] = "1"

    try:
        model, config = load_model(num_hidden_layers=1)
        decoder_layer = model.model.layers[0]

        batch_size, seq_len = 1, 32
        hidden_states = torch.randn(batch_size, seq_len, get_config_attr(config, "hidden_size"), dtype=torch.bfloat16)

        # position_embeddings required by Qwen3_5DecoderLayer.forward()
        pos = torch.arange(seq_len).unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden_states, pos)

        # Capture PyTorch reference BEFORE module replacement
        with torch.no_grad():
            torch_out = decoder_layer(
                hidden_states,
                position_embeddings=(cos, sin),
            )

        # Extract output tensor from tuple
        torch_out_tensor = torch_out[0] if isinstance(torch_out, (tuple, list)) else torch_out

        # Now replace linear_attn with TTNN version
        torch_linear_attn = decoder_layer.linear_attn
        linear_attn_class = torch_linear_attn.__class__

        nn_to_ttnn = {linear_attn_class: TTNNQwen3LinearAttention}
        modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
        set_device(model, device)

        for k, v in modules.items():
            v.preprocess_weights()
            v.move_weights_to_device()

        # Run with TTNN replacement
        ttnn_input = TorchTTNNTensor(hidden_states)
        ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)
        ttnn_out = decoder_layer(
            ttnn_input,
            position_embeddings=(cos, sin),
        )

        ttnn_out_tensor = ttnn_out[0] if isinstance(ttnn_out, (tuple, list)) else ttnn_out

        compare_fn_outputs(torch_out_tensor, ttnn_out_tensor, "DecoderLayer_LinearAttn")
        pcc = assert_with_pcc(torch_out_tensor, ttnn_out_tensor, PCC_DECODER_LAYER)
        print(f"  Decoder layer (linear attn) PCC = {pcc:.6f}")
    finally:
        os.environ["TTNN_LINEAR_ATTN_PROJECTIONS"] = original_env

    print("PASS: test_decoder_layer_linear_attention")


# ──────────────────────────────────────────────────────────────────────
# Test 11: Decoder layer — full attention end-to-end
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_decoder_layer_full_attention(device):
    """Test end-to-end decoder layer with full attention.

    Loads 4 layers to get layer 3 (full attention). Captures PyTorch output
    BEFORE module replacement, then replaces self_attn with TTNN and compares.
    PCC >= 0.90 expected.
    """
    model, config = load_model(num_hidden_layers=4)
    decoder_layer = model.model.layers[3]

    batch_size, seq_len = 1, 32
    hidden_states = torch.randn(batch_size, seq_len, get_config_attr(config, "hidden_size"), dtype=torch.bfloat16)

    # Position embeddings (required for full attention)
    pos = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden_states, pos)

    # Capture PyTorch reference BEFORE module replacement
    from transformers.cache_utils import DynamicCache

    dynamic_cache = DynamicCache()

    with torch.no_grad():
        torch_out = decoder_layer(
            hidden_states,
            position_embeddings=(cos, sin),
            past_key_values=dynamic_cache,
            attention_mask=None,
        )

    torch_out_tensor = torch_out[0] if isinstance(torch_out, (tuple, list)) else torch_out

    # Replace self_attn with TTNN version
    torch_full_attn = decoder_layer.self_attn
    full_attn_class = torch_full_attn.__class__

    nn_to_ttnn = {full_attn_class: TTNNQwen3FullAttention}
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)

    for k, v in modules.items():
        v.preprocess_weights()
        v.move_weights_to_device()

    # Create paged KV cache for TTNN
    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=32, batch_size=1)
    paged_cache = TTNNQwenPagedAttentionKVCache(
        num_layers=1,
        num_kv_heads=get_config_attr(config, "num_key_value_heads"),
        head_dim=get_config_attr(config, "head_dim"),
        config=paged_config,
        device=None,
        layer_indices=[3],
    ).to_device(device)

    cache_position = torch.arange(seq_len).unsqueeze(0)

    # Run with TTNN replacement
    ttnn_input = TorchTTNNTensor(hidden_states)
    ttnn_input.ttnn_tensor = ttnn.to_device(ttnn_input.to_ttnn, device)

    ttnn_out = decoder_layer(
        ttnn_input,
        position_embeddings=(cos, sin),
        past_key_values=paged_cache,
        cache_position=cache_position,
    )

    ttnn_out_tensor = ttnn_out[0] if isinstance(ttnn_out, (tuple, list)) else ttnn_out

    compare_fn_outputs(torch_out_tensor, ttnn_out_tensor, "DecoderLayer_FullAttn")
    pcc = assert_with_pcc(torch_out_tensor, ttnn_out_tensor, PCC_DECODER_LAYER)
    print(f"  Decoder layer (full attn) PCC = {pcc:.6f}")
    print("PASS: test_decoder_layer_full_attention")
