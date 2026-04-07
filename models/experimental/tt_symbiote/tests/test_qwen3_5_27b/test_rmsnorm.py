# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm accuracy tests for Qwen3.5-27B.

Qwen3_5RMSNorm uses (1 + weight) in forward, so TTNN weight must be
pre-adjusted to (1.0 + weight) before passing to ttnn.rms_norm.
"""

import pytest
import torch

from .conftest import (
    assert_with_pcc,
    get_config_attr,
    skip_no_ttnn,
    skip_no_transformers,
    skip_no_symbiote,
)

PCC_RMSNORM = 0.98


# ──────────────────────────────────────────────────────────────────────
# Test: input_layernorm from layer 0
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_input_layernorm_pcc(device, model_1_layer):
    """Test input_layernorm from layer 0 vs TTNN rms_norm with (1+weight) adjustment."""
    import ttnn

    model, config = model_1_layer
    torch_norm = model.model.layers[0].input_layernorm

    batch_size, seq_len = 1, 32
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # PyTorch reference
    with torch.no_grad():
        torch_out = torch_norm(x)

    # TTNN: adjust weight to (1 + weight) as Qwen3_5RMSNorm does
    adjusted_weight = (1.0 + torch_norm.weight.float()).to(torch.bfloat16)
    # Expand weight to tile-compatible shape (expand to 32 rows for TILE_LAYOUT)
    tt_weight = ttnn.from_torch(
        adjusted_weight.unsqueeze(0).expand(32, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_weight = ttnn.to_device(tt_weight, device)

    tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device)

    tt_out = ttnn.rms_norm(tt_input, weight=tt_weight, epsilon=torch_norm.eps)

    pcc = assert_with_pcc(torch_out, tt_out, PCC_RMSNORM)
    print(f"  input_layernorm PCC = {pcc:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Test: post_attention_layernorm from layer 0
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_post_attention_layernorm_pcc(device, model_1_layer):
    """Test post_attention_layernorm from layer 0 vs TTNN rms_norm."""
    import ttnn

    model, config = model_1_layer
    torch_norm = model.model.layers[0].post_attention_layernorm

    batch_size, seq_len = 1, 32
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_norm(x)

    adjusted_weight = (1.0 + torch_norm.weight.float()).to(torch.bfloat16)
    tt_weight = ttnn.from_torch(
        adjusted_weight.unsqueeze(0).expand(32, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_weight = ttnn.to_device(tt_weight, device)

    tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device)

    tt_out = ttnn.rms_norm(tt_input, weight=tt_weight, epsilon=torch_norm.eps)

    pcc = assert_with_pcc(torch_out, tt_out, PCC_RMSNORM)
    print(f"  post_attention_layernorm PCC = {pcc:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Test: final norm (model.norm) before LM head
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_final_norm_pcc(device, model_4_layers):
    """Test model.norm (final norm before LM head) vs TTNN rms_norm."""
    import ttnn

    model, config = model_4_layers
    torch_norm = model.model.norm

    batch_size, seq_len = 1, 32
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_norm(x)

    adjusted_weight = (1.0 + torch_norm.weight.float()).to(torch.bfloat16)
    tt_weight = ttnn.from_torch(
        adjusted_weight.unsqueeze(0).expand(32, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_weight = ttnn.to_device(tt_weight, device)

    tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device)

    tt_out = ttnn.rms_norm(tt_input, weight=tt_weight, epsilon=torch_norm.eps)

    pcc = assert_with_pcc(torch_out, tt_out, PCC_RMSNORM)
    print(f"  final_norm PCC = {pcc:.6f}")


# ──────────────────────────────────────────────────────────────────────
# Test: RMSNorm at various sequence lengths
# ──────────────────────────────────────────────────────────────────────


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 32, 128])
def test_rmsnorm_various_seq_lens(device, model_1_layer, seq_len):
    """Test input_layernorm at various sequence lengths."""
    import ttnn

    model, config = model_1_layer
    torch_norm = model.model.layers[0].input_layernorm

    batch_size = 1
    hidden_size = get_config_attr(config, "hidden_size")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_norm(x)

    adjusted_weight = (1.0 + torch_norm.weight.float()).to(torch.bfloat16)
    tt_weight = ttnn.from_torch(
        adjusted_weight.unsqueeze(0).expand(32, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_weight = ttnn.to_device(tt_weight, device)

    tt_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_device(tt_input, device)

    tt_out = ttnn.rms_norm(tt_input, weight=tt_weight, epsilon=torch_norm.eps)

    pcc = assert_with_pcc(torch_out, tt_out, PCC_RMSNORM)
    print(f"  rmsnorm seq_len={seq_len} PCC = {pcc:.6f}")
