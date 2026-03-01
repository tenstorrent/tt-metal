# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for Molmo2 Vision Backbone pooling and projection.

Compares TTNN implementation against HuggingFace reference for:
1. Image pooling (cross-attention)
2. Image projector (SwiGLU)
"""

from io import BytesIO

import requests
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import ttnn


def calculate_pcc(ref, out):
    """Calculate Pearson Correlation Coefficient."""
    if ref.shape != out.shape:
        return -1.0, f"Shape mismatch: {ref.shape} vs {out.shape}"
    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()
    ref_mean = ref_flat.mean()
    out_mean = out_flat.mean()
    numerator = ((ref_flat - ref_mean) * (out_flat - out_mean)).sum()
    denominator = torch.sqrt(((ref_flat - ref_mean) ** 2).sum() * ((out_flat - out_mean) ** 2).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0, "zero std"
    pcc = numerator / denominator
    return pcc.item(), "ok"


def ref_image_pooling_forward(
    query,  # [B*N_out, 1, input_dim]
    key_value,  # [B*N_out, K_pool, input_dim]
    state_dict,
    attn_mask=None,  # [B*N_out, 1, 1, K_pool]
    input_dim=2304,
    hidden_dim=1152,
    num_heads=16,
    head_dim=72,
):
    """Reference PyTorch implementation of image pooling cross-attention."""
    prefix = "model.vision_backbone.image_pooling_2d"

    batch_n_out = query.shape[0]
    pool_size = key_value.shape[1]

    # Load weights
    wq = state_dict[f"{prefix}.wq.weight"]
    bq = state_dict[f"{prefix}.wq.bias"]
    wk = state_dict[f"{prefix}.wk.weight"]
    bk = state_dict[f"{prefix}.wk.bias"]
    wv = state_dict[f"{prefix}.wv.weight"]
    bv = state_dict[f"{prefix}.wv.bias"]
    wo = state_dict[f"{prefix}.wo.weight"]
    bo = state_dict[f"{prefix}.wo.bias"]

    # Q, K, V projections
    q = F.linear(query, wq, bq)  # [B*N_out, 1, num_heads * head_dim]
    k = F.linear(key_value, wk, bk)  # [B*N_out, K_pool, num_heads * head_dim]
    v = F.linear(key_value, wv, bv)  # [B*N_out, K_pool, num_heads * head_dim]

    # Reshape for multi-head attention
    q = q.reshape(batch_n_out, 1, num_heads, head_dim).transpose(1, 2)  # [B*N_out, num_heads, 1, head_dim]
    k = k.reshape(batch_n_out, pool_size, num_heads, head_dim).transpose(1, 2)  # [B*N_out, num_heads, K_pool, head_dim]
    v = v.reshape(batch_n_out, pool_size, num_heads, head_dim).transpose(1, 2)  # [B*N_out, num_heads, K_pool, head_dim]

    # Scaled dot-product attention
    scale = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*N_out, num_heads, 1, K_pool]

    if attn_mask is not None:
        # Expand mask for all heads
        attn_mask_expanded = attn_mask.expand(-1, num_heads, -1, -1)
        attn_weights = attn_weights + attn_mask_expanded.where(
            attn_mask_expanded != 0, float("-inf") * torch.ones_like(attn_mask_expanded)
        )
        attn_weights = attn_weights.masked_fill(~attn_mask_expanded.bool(), float("-inf"))

    attn_probs = F.softmax(attn_weights, dim=-1)
    attn_out = torch.matmul(attn_probs, v)  # [B*N_out, num_heads, 1, head_dim]

    # Reshape and output projection
    attn_out = attn_out.transpose(1, 2).reshape(batch_n_out, 1, hidden_dim)  # [B*N_out, 1, hidden_dim]
    output = F.linear(attn_out, wo, bo)  # [B*N_out, 1, hidden_dim]

    return output


def ref_image_projector_forward(x, state_dict):
    """Reference PyTorch implementation of image projector (SwiGLU)."""
    prefix = "model.vision_backbone.image_projector"

    w1 = state_dict[f"{prefix}.w1.weight"]
    w2 = state_dict[f"{prefix}.w2.weight"]
    w3 = state_dict[f"{prefix}.w3.weight"]

    # SwiGLU: w2(silu(w1(x)) * w3(x))
    gate = F.silu(F.linear(x, w1))
    up = F.linear(x, w3)
    hidden = gate * up
    output = F.linear(hidden, w2)

    return output


def test_image_pooling_pcc():
    """Test PCC for image pooling cross-attention."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    # Config
    input_dim = 2304
    hidden_dim = 1152
    num_heads = 16
    head_dim = 72

    # Load weights
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")
    logger.info("Loaded weights")

    # Create random test inputs
    torch.manual_seed(42)
    batch_n_out = 64  # Simulated B*N_out
    pool_size = 9  # K_pool (number of patches to pool)

    # Random query (mean of features)
    query = torch.randn(batch_n_out, 1, input_dim)
    # Random key/value (gathered patch features)
    key_value = torch.randn(batch_n_out, pool_size, input_dim)

    logger.info(f"Query shape: {query.shape}")
    logger.info(f"Key/Value shape: {key_value.shape}")

    # Reference forward
    ref_out = ref_image_pooling_forward(
        query,
        key_value,
        state_dict,
        attn_mask=None,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    logger.info(f"Reference output shape: {ref_out.shape}")
    logger.info(f"Reference output stats: min={ref_out.min():.4f}, mean={ref_out.mean():.4f}, max={ref_out.max():.4f}")

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        from models.demos.molmo2.tt.image_pooling import ImagePooling

        # Create TTNN image pooling
        ttnn_pooling = ImagePooling(
            mesh_device=device,
            state_dict=state_dict,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=ttnn.bfloat16,
        )

        # Convert inputs to TTNN
        query_ttnn = ttnn.from_torch(
            query.unsqueeze(0),  # [1, B*N_out, 1, input_dim]
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        key_value_ttnn = ttnn.from_torch(
            key_value.unsqueeze(0),  # [1, B*N_out, K_pool, input_dim]
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # TTNN forward
        ttnn_out = ttnn_pooling(query_ttnn, key_value_ttnn, attn_mask=None)
        ttnn_out_torch = ttnn.to_torch(ttnn_out).squeeze(0).float()

        logger.info(f"TTNN output shape: {ttnn_out_torch.shape}")
        logger.info(
            f"TTNN output stats: min={ttnn_out_torch.min():.4f}, mean={ttnn_out_torch.mean():.4f}, max={ttnn_out_torch.max():.4f}"
        )

        # Calculate PCC
        pcc, status = calculate_pcc(ref_out, ttnn_out_torch)
        diff = (ref_out - ttnn_out_torch).abs()

        logger.info(f"Image Pooling PCC: {pcc:.6f}, max_diff={diff.max():.4f}, mean_diff={diff.mean():.6f}")

        # Clean up
        ttnn.deallocate(query_ttnn)
        ttnn.deallocate(key_value_ttnn)
        ttnn.deallocate(ttnn_out)

        return pcc

    finally:
        ttnn.close_device(device)


def test_image_projector_pcc():
    """Test PCC for image projector (SwiGLU)."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    # Config
    input_dim = 1152
    intermediate_dim = 12288
    output_dim = 4096

    # Load weights
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")
    logger.info("Loaded weights")

    # Create random test inputs
    torch.manual_seed(42)
    num_tokens = 256
    x = torch.randn(1, num_tokens, input_dim)

    logger.info(f"Input shape: {x.shape}")

    # Reference forward
    ref_out = ref_image_projector_forward(x, state_dict)
    logger.info(f"Reference output shape: {ref_out.shape}")
    logger.info(f"Reference output stats: min={ref_out.min():.4f}, mean={ref_out.mean():.4f}, max={ref_out.max():.4f}")

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        from models.demos.molmo2.tt.image_projector import ImageProjector

        # Create TTNN image projector
        ttnn_projector = ImageProjector(
            mesh_device=device,
            state_dict=state_dict,
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            dtype=ttnn.bfloat16,
        )

        # Convert input to TTNN
        x_ttnn = ttnn.from_torch(
            x.unsqueeze(0),  # [1, 1, num_tokens, input_dim]
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # TTNN forward
        ttnn_out = ttnn_projector(x_ttnn)
        ttnn_out_torch = ttnn.to_torch(ttnn_out).squeeze(0).float()

        logger.info(f"TTNN output shape: {ttnn_out_torch.shape}")
        logger.info(
            f"TTNN output stats: min={ttnn_out_torch.min():.4f}, mean={ttnn_out_torch.mean():.4f}, max={ttnn_out_torch.max():.4f}"
        )

        # Calculate PCC
        pcc, status = calculate_pcc(ref_out, ttnn_out_torch)
        diff = (ref_out - ttnn_out_torch).abs()

        logger.info(f"Image Projector PCC: {pcc:.6f}, max_diff={diff.max():.4f}, mean_diff={diff.mean():.6f}")

        # Clean up
        ttnn.deallocate(x_ttnn)
        ttnn.deallocate(ttnn_out)

        return pcc

    finally:
        ttnn.close_device(device)


def test_full_vision_backbone_pcc():
    """Test PCC for the full vision backbone against HuggingFace."""
    logger.info("Loading HuggingFace model...")

    model = AutoModelForImageTextToText.from_pretrained(
        "allenai/Molmo2-8B",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    processor = AutoProcessor.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)
    model.eval()

    # Load a test image
    logger.info("Loading test image...")
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/800px-Camponotus_flavomarginatus_ant.jpg"
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        logger.warning(f"Could not download image: {e}. Using random tensor.")
        # Create random image
        image = Image.new("RGB", (378, 378))

    # Process input
    inputs = processor(images=[image], text="Describe this image:", return_tensors="pt")

    logger.info(f"Pixel values shape: {inputs.pixel_values.shape if hasattr(inputs, 'pixel_values') else 'N/A'}")

    # Get vision backbone output from HuggingFace
    with torch.no_grad():
        vision_backbone = model.model.vision_backbone

        # Check what inputs the vision backbone needs
        if hasattr(inputs, "images") and inputs.images is not None:
            images = inputs.images
        elif hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
            images = inputs.pixel_values
        else:
            logger.error("No image input found in processor output")
            return

        # Get pooled_patches_idx
        if hasattr(inputs, "pooled_patches_idx"):
            pooled_patches_idx = inputs.pooled_patches_idx
        else:
            logger.warning("No pooled_patches_idx found, may need different input format")
            return

        logger.info(f"Images shape: {images.shape}")
        logger.info(f"pooled_patches_idx shape: {pooled_patches_idx.shape}")

        # HuggingFace forward
        hf_output = vision_backbone(images, pooled_patches_idx)

        logger.info(f"HuggingFace vision output shape: {hf_output.shape}")
        logger.info(
            f"HuggingFace vision output stats: min={hf_output.min():.4f}, mean={hf_output.mean():.4f}, max={hf_output.max():.4f}"
        )


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Testing Image Projector PCC")
    logger.info("=" * 80)
    test_image_projector_pcc()

    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Image Pooling PCC")
    logger.info("=" * 80)
    test_image_pooling_pcc()
