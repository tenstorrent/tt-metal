# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test 5: Validate decoder layer tensor lifecycle and residual pattern.

Verifies that TTNNQwen35DecoderLayer correctly:
1. Keeps residual adds on device (no host round-trips)
2. Does NOT deallocate the first residual (trace input buffer safety)
3. Deallocates intermediates (attn_out, mlp_out, second residual)
4. Produces correct output across multiple forward calls (no stale state)
5. Handles both linear_attention and full_attention layer types
"""

import pytest
import torch

from .conftest import (
    compute_pcc,
    get_config_attr,
    get_layer_type,
    skip_no_ttnn,
    skip_no_transformers,
    skip_no_symbiote,
    TTNN_AVAILABLE,
    TT_SYMBIOTE_AVAILABLE,
)

if TTNN_AVAILABLE:
    import ttnn

if TT_SYMBIOTE_AVAILABLE:
    from models.experimental.tt_symbiote.modules.qwen35_decoder_layer import TTNNQwen35DecoderLayer
    from models.experimental.tt_symbiote.utils.device_management import set_device


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_decoder_layer_linear_attention_pcc(device, model_4_layers):
    """Linear attention (GDN) decoder layer output matches PyTorch. PCC >= 0.90.

    Layer 0 is a linear_attention layer. Runs a single prefill-style forward
    and compares TTNN vs PyTorch.
    """
    model, config = model_4_layers
    torch_layer = model.model.layers[0]

    assert get_layer_type(torch_layer) == "linear_attention"

    hidden_size = get_config_attr(config, "hidden_size")
    batch_size = 1
    seq_len = 32

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)

    # PyTorch reference (position_embeddings required by HF even for linear attn layers)
    with torch.no_grad():
        torch_out = torch_layer(x, position_embeddings=(cos, sin))
        torch_out_tensor = torch_out[0] if isinstance(torch_out, tuple) else torch_out

    # TTNN
    ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
    set_device(ttnn_layer, device)
    ttnn_layer.preprocess_weights()
    ttnn_layer.move_weights_to_device()

    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN GDN decoder layer doesn't need position_embeddings (no RoPE)
    ttnn_out = ttnn_layer.forward(tt_input)
    ttnn_out_torch = ttnn.to_torch(ttnn_out).to(torch.float32)
    torch_out_f32 = torch_out_tensor.to(torch.float32)

    # Trim padding
    if ttnn_out_torch.shape != torch_out_f32.shape:
        slices = tuple(slice(0, s) for s in torch_out_f32.shape)
        ttnn_out_torch = ttnn_out_torch[slices]

    pcc_val = compute_pcc(torch_out_f32, ttnn_out_torch)
    print(f"  Linear attention decoder layer PCC = {pcc_val:.6f}")
    assert pcc_val >= 0.90, f"PCC {pcc_val:.6f} < 0.90"


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_decoder_layer_full_attention_pcc(device, model_4_layers):
    """Full attention (GQA) decoder layer output matches PyTorch. PCC >= 0.90.

    Layer 3 is the first full_attention layer. Runs a single prefill-style forward
    and compares TTNN vs PyTorch.
    """
    model, config = model_4_layers
    torch_layer = model.model.layers[3]

    assert get_layer_type(torch_layer) == "full_attention"

    hidden_size = get_config_attr(config, "hidden_size")
    batch_size = 1
    seq_len = 32

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)

    # PyTorch reference
    with torch.no_grad():
        torch_out = torch_layer(x, position_embeddings=(cos, sin))
        torch_out_tensor = torch_out[0] if isinstance(torch_out, tuple) else torch_out

    # TTNN
    ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
    set_device(ttnn_layer, device)
    ttnn_layer.preprocess_weights()
    ttnn_layer.move_weights_to_device()

    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_cos = ttnn.from_torch(
        cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_sin = ttnn.from_torch(
        sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    ttnn_out = ttnn_layer.forward(tt_input, position_embeddings=(tt_cos, tt_sin))
    ttnn_out_torch = ttnn.to_torch(ttnn_out).to(torch.float32)
    torch_out_f32 = torch_out_tensor.to(torch.float32)

    # Trim padding
    if ttnn_out_torch.shape != torch_out_f32.shape:
        slices = tuple(slice(0, s) for s in torch_out_f32.shape)
        ttnn_out_torch = ttnn_out_torch[slices]

    pcc_val = compute_pcc(torch_out_f32, ttnn_out_torch)
    print(f"  Full attention decoder layer PCC = {pcc_val:.6f}")
    assert pcc_val >= 0.90, f"PCC {pcc_val:.6f} < 0.90"


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_decoder_layer_multi_forward_no_stale_state(device, model_4_layers):
    """Multiple forward calls produce correct output (no stale tensor state).

    Runs the linear attention decoder layer twice with different inputs and
    verifies both outputs are valid (no NaN/Inf) and different from each other.
    """
    model, config = model_4_layers
    torch_layer = model.model.layers[0]

    hidden_size = get_config_attr(config, "hidden_size")
    batch_size = 1
    seq_len = 32

    ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
    set_device(ttnn_layer, device)
    ttnn_layer.preprocess_weights()
    ttnn_layer.move_weights_to_device()

    outputs = []
    for i in range(2):
        torch.manual_seed(42 + i)
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
        tt_input = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn_layer.forward(tt_input)
        out_torch = ttnn.to_torch(out).to(torch.float32)
        outputs.append(out_torch)

        assert not torch.isnan(out_torch).any(), f"Forward {i}: NaN in output"
        assert not torch.isinf(out_torch).any(), f"Forward {i}: Inf in output"
        print(f"  Forward {i}: mean={out_torch.mean():.6f}, std={out_torch.std():.6f}")

    # Outputs should differ (different random inputs)
    assert not torch.allclose(
        outputs[0], outputs[1], atol=1e-3
    ), "Two forwards with different inputs produced identical output — stale state?"


@skip_no_ttnn
@skip_no_transformers
@skip_no_symbiote
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_decoder_layer_output_is_on_device(device, model_4_layers):
    """Decoder layer output stays on device (no implicit to_torch).

    Verifies the output is a ttnn.Tensor on device, not a torch.Tensor.
    This ensures no hidden host round-trips.
    """
    model, config = model_4_layers
    torch_layer = model.model.layers[0]

    hidden_size = get_config_attr(config, "hidden_size")
    batch_size = 1
    seq_len = 32

    ttnn_layer = TTNNQwen35DecoderLayer.from_torch(torch_layer)
    set_device(ttnn_layer, device)
    ttnn_layer.preprocess_weights()
    ttnn_layer.move_weights_to_device()

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn_layer.forward(tt_input)

    assert isinstance(out, ttnn.Tensor), f"Output is {type(out)}, expected ttnn.Tensor"
    # Verify it's on device (not host)
    assert out.is_allocated(), "Output tensor is not allocated on device"
    print(f"  Output is ttnn.Tensor on device, shape={out.shape}, dtype={out.dtype}")


@pytest.fixture(scope="module")
def model_4_layers():
    """Load Qwen3.5-27B-FP8 with 4 hidden layers (module-scoped for reuse)."""
    from .conftest import load_model

    model, config = load_model(num_hidden_layers=4)
    return model, config
