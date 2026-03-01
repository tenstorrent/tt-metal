# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Compare first ViT block output between TTNN and PyTorch.
"""

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn


def calculate_pcc(ref, out):
    """Calculate Pearson Correlation Coefficient."""
    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()
    ref_mean = ref_flat.mean()
    out_mean = out_flat.mean()
    numerator = ((ref_flat - ref_mean) * (out_flat - out_mean)).sum()
    denominator = torch.sqrt(((ref_flat - ref_mean) ** 2).sum() * ((out_flat - out_mean) ** 2).sum())
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return (numerator / denominator).item()


def main():
    """Compare first ViT block."""
    from models.demos.molmo2.demo.demo import preprocess_image_molmo2
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    logger.info("Loading weights...")
    state_dict = load_state_dict_from_safetensors("allenai/Molmo2-8B")

    # Load image
    image_path = "models/demos/molmo2/demo/dog.jpg"
    logger.info(f"Loading image: {image_path}")
    image_inputs = preprocess_image_molmo2(image_path)

    pixel_values = image_inputs["pixel_values"]

    # Get embedded input
    patch_w = state_dict["model.vision_backbone.image_vit.patch_embedding.weight"]
    patch_b = state_dict["model.vision_backbone.image_vit.patch_embedding.bias"]
    pos_embed = state_dict["model.vision_backbone.image_vit.positional_embedding"]

    patch_size = 14
    B, C, H, W = pixel_values.shape
    patches_h = H // patch_size
    patches_w = W // patch_size

    x = pixel_values.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, patches_h * patches_w, C * patch_size * patch_size)
    embedded = F.linear(x, patch_w, patch_b)
    x_input = embedded + pos_embed.unsqueeze(0)

    logger.info(f"Input shape: {x_input.shape}")
    logger.info(f"Input range: [{x_input.min():.4f}, {x_input.max():.4f}]")

    # ========= PyTorch Reference =========
    logger.info("\n=== PyTorch Reference ===")
    hidden_dim = 1152
    num_heads = 16
    head_dim = 72
    eps = 1e-6

    prefix = "model.vision_backbone.image_vit.transformer.resblocks.0"

    ln1_w = state_dict[f"{prefix}.attention_norm.weight"]
    ln1_b = state_dict[f"{prefix}.attention_norm.bias"]
    wq = state_dict[f"{prefix}.attention.wq.weight"]
    bq = state_dict[f"{prefix}.attention.wq.bias"]
    wk = state_dict[f"{prefix}.attention.wk.weight"]
    bk = state_dict[f"{prefix}.attention.wk.bias"]
    wv = state_dict[f"{prefix}.attention.wv.weight"]
    bv = state_dict[f"{prefix}.attention.wv.bias"]
    wo = state_dict[f"{prefix}.attention.wo.weight"]
    bo = state_dict[f"{prefix}.attention.wo.bias"]
    ln2_w = state_dict[f"{prefix}.ffn_norm.weight"]
    ln2_b = state_dict[f"{prefix}.ffn_norm.bias"]
    ff_w1 = state_dict[f"{prefix}.feed_forward.w1.weight"]
    ff_b1 = state_dict[f"{prefix}.feed_forward.w1.bias"]
    ff_w2 = state_dict[f"{prefix}.feed_forward.w2.weight"]
    ff_b2 = state_dict[f"{prefix}.feed_forward.w2.bias"]

    x = x_input.clone()
    residual = x

    # Pre-attention LayerNorm
    x = F.layer_norm(x, (hidden_dim,), ln1_w, ln1_b, eps)

    # Attention
    B, N, D = x.shape
    q = F.linear(x, wq, bq).reshape(B, N, num_heads, head_dim)
    k = F.linear(x, wk, bk).reshape(B, N, num_heads, head_dim)
    v = F.linear(x, wv, bv).reshape(B, N, num_heads, head_dim)

    q = q.float()
    k = k.float()
    scale = head_dim**-0.5
    attn_weights = torch.einsum("bnhd,bmhd->bhnm", q * scale, k)
    attn_probs = F.softmax(attn_weights, dim=-1)
    attn_out = torch.einsum("bhnm,bmhd->bnhd", attn_probs, v.float()).to(x.dtype)
    attn_out = attn_out.reshape(B, N, hidden_dim)
    attn_out = F.linear(attn_out, wo, bo)

    x = residual + attn_out

    # FFN
    residual = x
    x = F.layer_norm(x, (hidden_dim,), ln2_w, ln2_b, eps)
    hidden = F.gelu(F.linear(x, ff_w1, ff_b1), approximate="tanh")
    ffn_out = F.linear(hidden, ff_w2, ff_b2)
    x = residual + ffn_out

    pytorch_output = x
    logger.info(
        f"PyTorch output: range=[{pytorch_output.min():.4f}, {pytorch_output.max():.4f}], std={pytorch_output.std():.4f}"
    )

    # ========= TTNN =========
    logger.info("\n=== TTNN ===")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, 8)
    device = ttnn.open_mesh_device(mesh_shape)

    try:
        from models.demos.molmo2.tt.vision_block import VisionBlock

        # Create TTNN block
        block = VisionBlock(
            mesh_device=device,
            state_dict=state_dict,
            layer_num=0,
            hidden_dim=1152,
            intermediate_dim=4304,
            num_heads=16,
            head_dim=72,
            dtype=ttnn.bfloat16,  # Use bfloat16 for better precision
        )

        # Convert input to TTNN
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)

        x_ttnn = ttnn.from_torch(
            x_input.unsqueeze(0),  # [1, 1, 729, 1152]
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        # Run TTNN block
        ttnn_output = block(x_ttnn)
        ttnn_output_torch = ttnn.to_torch(ttnn_output, mesh_composer=mesh_composer)[0].squeeze()

        logger.info(
            f"TTNN output: range=[{ttnn_output_torch.min():.4f}, {ttnn_output_torch.max():.4f}], std={ttnn_output_torch.std():.4f}"
        )

        # Compare
        pcc = calculate_pcc(pytorch_output.squeeze(), ttnn_output_torch)
        diff = (pytorch_output.squeeze() - ttnn_output_torch).abs()
        logger.info(f"PCC: {pcc:.6f}")
        logger.info(f"Max diff: {diff.max():.4f}")
        logger.info(f"Mean diff: {diff.mean():.6f}")

        ttnn.deallocate(x_ttnn)
        ttnn.deallocate(ttnn_output)

    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
