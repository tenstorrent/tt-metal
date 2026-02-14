# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC Tests for YOLOS-small TTNN Implementation

Tests each component against HuggingFace reference outputs.
"""

import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.yolos.common import load_torch_model
from models.demos.yolos.tt import ttnn_functional_yolos
from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_random(shape, low, high, dtype=torch.float32):
    """Generate random tensor for testing."""
    return torch.rand(shape, dtype=dtype) * (high - low) + low


@pytest.mark.parametrize("batch_size", [1])
def test_yolos_patch_embeddings(device, batch_size):
    """Test patch embeddings component."""
    torch.manual_seed(0)
    
    model = load_torch_model()
    config = model.config
    
    image_height, image_width = config.image_size
    # Use float32 for PyTorch reference
    torch_pixel_values = torch_random(
        (batch_size, 3, image_height, image_width), -1, 1, dtype=torch.float32
    )
    
    # Get HuggingFace patch embeddings output
    with torch.no_grad():
        torch_output = model.vit.embeddings.patch_embeddings(torch_pixel_values)
    
    print(f"HF patch embeddings shape: {torch_output.shape}")
    
    # Convert model to bfloat16 for TTNN parameter preprocessing
    model_bf16 = model.to(torch.bfloat16)
    
    # Preprocess parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model_bf16,
        device=device,
        custom_preprocessor=ttnn_functional_yolos.custom_preprocessor,
    )
    
    # Prepare TTNN input (use bfloat16)
    torch_pixel_values_bf16 = torch_pixel_values.to(torch.bfloat16)
    pixel_values_nhwc = torch.permute(torch_pixel_values_bf16, (0, 2, 3, 1))
    pixel_values_padded = torch.nn.functional.pad(pixel_values_nhwc, (0, 1, 0, 0, 0, 0, 0, 0))
    pixel_values = ttnn.from_torch(
        pixel_values_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    
    # Run TTNN patch embeddings
    ttnn_output = ttnn_functional_yolos.yolos_patch_embeddings(
        config, pixel_values, parameters=parameters, unittest_check=True
    )
    ttnn_output = ttnn.to_torch(ttnn_output)
    
    print(f"TTNN patch embeddings shape: {ttnn_output.shape}")
    
    # Compare (convert HF output to bfloat16 for comparison)
    assert_with_pcc(torch_output.to(torch.bfloat16), ttnn_output, 0.99)
    print("Patch embeddings test PASSED!")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [197])
def test_yolos_attention(device, batch_size, sequence_size):
    """Test attention layer."""
    torch.manual_seed(0)
    
    model = load_torch_model()
    config = model.config
    
    # Get first encoder layer attention
    attention_module = model.vit.encoder.layer[0].attention
    
    torch_hidden_states = torch_random(
        (batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32
    )
    
    with torch.no_grad():
        torch_output = attention_module(torch_hidden_states)[0]
    
    print(f"HF attention output shape: {torch_output.shape}")
    
    # Convert to bfloat16 for TTNN
    attention_module_bf16 = attention_module.to(torch.bfloat16)
    
    # Preprocess parameters for just the attention module
    parameters = preprocess_model_parameters(
        initialize_model=lambda: attention_module_bf16,
        device=device,
        custom_preprocessor=ttnn_functional_yolos.custom_preprocessor,
    )
    
    hidden_states = ttnn.from_torch(
        torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    
    ttnn_output = ttnn_functional_yolos.yolos_attention(
        config, hidden_states, attention_mask=None, parameters=parameters
    )
    ttnn_output = ttnn.to_torch(ttnn_output)
    
    print(f"TTNN attention output shape: {ttnn_output.shape}")
    
    assert_with_pcc(torch_output.to(torch.bfloat16), ttnn_output, 0.99)
    print("Attention test PASSED!")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [197])
def test_yolos_encoder_layer(device, batch_size, sequence_size):
    """Test single encoder layer."""
    torch.manual_seed(0)
    
    model = load_torch_model()
    config = model.config
    
    # Get first encoder layer
    layer_module = model.vit.encoder.layer[0]
    
    torch_hidden_states = torch_random(
        (batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32
    )
    
    with torch.no_grad():
        torch_output = layer_module(torch_hidden_states)[0]
    
    print(f"HF layer output shape: {torch_output.shape}")
    
    layer_module_bf16 = layer_module.to(torch.bfloat16)
    
    parameters = preprocess_model_parameters(
        initialize_model=lambda: layer_module_bf16,
        device=device,
        custom_preprocessor=ttnn_functional_yolos.custom_preprocessor,
    )
    
    hidden_states = ttnn.from_torch(
        torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )
    
    ttnn_output = ttnn_functional_yolos.yolos_layer(
        config, hidden_states, attention_mask=None, parameters=parameters
    )
    ttnn_output = ttnn.to_torch(ttnn_output)
    
    print(f"TTNN layer output shape: {ttnn_output.shape}")
    
    assert_with_pcc(torch_output.to(torch.bfloat16), ttnn_output, 0.99)
    print("Encoder layer test PASSED!")


@pytest.mark.parametrize("batch_size", [1])
def test_yolos_full_model(device, batch_size):
    """Test full YOLOS model against HuggingFace reference."""
    torch.manual_seed(0)
    
    # Load HuggingFace model
    model = load_torch_model()
    config = model.config
    
    # Create random input image (in HuggingFace format: NCHW)
    image_height, image_width = config.image_size
    torch_pixel_values = torch_random(
        (batch_size, 3, image_height, image_width), -1, 1, dtype=torch.float32
    )
    
    # Get HuggingFace output
    with torch.no_grad():
        hf_outputs = model(torch_pixel_values)
        torch_logits = hf_outputs.logits
        torch_boxes = hf_outputs.pred_boxes
    
    print(f"HF logits shape: {torch_logits.shape}")
    print(f"HF boxes shape: {torch_boxes.shape}")
    
    # Convert model to bfloat16
    model_bf16 = model.to(torch.bfloat16)
    
    # Preprocess parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model_bf16,
        device=device,
        custom_preprocessor=ttnn_functional_yolos.custom_preprocessor,
    )
    
    # Prepare input for TTNN (convert NCHW to NHWC and pad channels)
    torch_pixel_values_bf16 = torch_pixel_values.to(torch.bfloat16)
    pixel_values_nhwc = torch.permute(torch_pixel_values_bf16, (0, 2, 3, 1))
    pixel_values_padded = torch.nn.functional.pad(pixel_values_nhwc, (0, 1, 0, 0, 0, 0, 0, 0))
    pixel_values = ttnn.from_torch(
        pixel_values_padded, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    
    # Get embeddings for TTNN model
    state_dict = model_bf16.state_dict()
    
    # CLS token: expand to batch size
    torch_cls_token = state_dict["vit.embeddings.cls_token"]
    if batch_size > 1:
        torch_cls_token = torch_cls_token.expand(batch_size, -1, -1)
    cls_token = ttnn.from_torch(
        torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    
    # Detection tokens: expand to batch size
    torch_detection_tokens = state_dict["vit.embeddings.detection_tokens"]
    if batch_size > 1:
        torch_detection_tokens = torch_detection_tokens.expand(batch_size, -1, -1)
    detection_tokens = ttnn.from_torch(
        torch_detection_tokens, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    
    # Position embeddings: expand to batch size
    torch_position_embeddings = state_dict["vit.embeddings.position_embeddings"]
    if batch_size > 1:
        torch_position_embeddings = torch_position_embeddings.expand(batch_size, -1, -1)
    position_embeddings = ttnn.from_torch(
        torch_position_embeddings, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    
    # Run TTNN model
    ttnn_logits, ttnn_boxes = ttnn_functional_yolos.yolos_for_object_detection(
        config,
        pixel_values,
        cls_token,
        detection_tokens,
        position_embeddings,
        attention_mask=None,
        parameters=parameters,
    )
    
    # Convert back to torch
    ttnn_logits = ttnn.to_torch(ttnn_logits)
    ttnn_boxes = ttnn.to_torch(ttnn_boxes)
    
    print(f"TTNN logits shape: {ttnn_logits.shape}")
    print(f"TTNN boxes shape: {ttnn_boxes.shape}")
    
    # Compare with PCC
    assert_with_pcc(torch_logits.to(torch.bfloat16), ttnn_logits, 0.95)
    assert_with_pcc(torch_boxes.to(torch.bfloat16), ttnn_boxes, 0.93)
    
    print("Full model test PASSED!")


if __name__ == "__main__":
    # Simple test runner for development
    import ttnn
    
    device = ttnn.open_device(device_id=0)
    
    try:
        print("\n" + "="*60)
        print("Testing YOLOS-small TTNN Implementation")
        print("="*60 + "\n")
        
        # Run tests
        test_yolos_patch_embeddings(device, batch_size=1)
        print()
        
        test_yolos_attention(device, batch_size=1, sequence_size=197)
        print()
        
        test_yolos_encoder_layer(device, batch_size=1, sequence_size=197)
        print()
        
        test_yolos_full_model(device, batch_size=1)
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        
    finally:
        ttnn.close_device(device)
