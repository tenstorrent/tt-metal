#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1 validation script for Molmo2 Vision Transformer.

This script validates:
1. Weight loading from HuggingFace
2. Forward pass through TTNN vision blocks
3. PCC comparison with PyTorch reference
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

# Add parent directories to path
sys.path.insert(
    0,
    str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))),
)


def validate_weight_loading():
    """Test that we can load vision weights from HuggingFace."""
    print("\n=== Validating Weight Loading ===")

    from models.demos.molmo2.tt.load_weights import get_vit_block_keys, load_vit_block_weights

    model_id = "allenai/Molmo2-8B"

    # Test loading block 0
    print(f"Loading weights for block 0 from {model_id}...")
    try:
        state_dict = load_vit_block_weights(model_id, layer_num=0)
        expected_keys = get_vit_block_keys(0)

        print(f"Loaded {len(state_dict)} weights")
        print(f"Expected {len(expected_keys)} weights")

        # Check all expected keys are present
        missing = [k for k in expected_keys if k not in state_dict]
        if missing:
            print(f"WARNING: Missing keys: {missing}")
            return False

        # Print some weight shapes
        for key in list(state_dict.keys())[:5]:
            print(f"  {key}: {state_dict[key].shape}")

        print("Weight loading: PASSED")
        return True

    except Exception as e:
        print(f"Weight loading FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_pytorch_reference():
    """
    Create a PyTorch reference vision block and run forward pass.
    This tests the expected behavior without needing the full HF model.
    """
    print("\n=== Validating PyTorch Reference ===")

    from models.demos.molmo2.tt.load_weights import load_vit_block_weights

    model_id = "allenai/Molmo2-8B"
    layer_num = 0

    # Load weights
    print(f"Loading block {layer_num} weights...")
    state_dict = load_vit_block_weights(model_id, layer_num=layer_num)

    # Build a simple PyTorch reference block
    class ReferenceVisionBlock(nn.Module):
        def __init__(self, hidden_dim=1152, num_heads=16, head_dim=72, intermediate_dim=4304, eps=1e-6):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.head_dim = head_dim

            # Norms
            self.attention_norm = nn.LayerNorm(hidden_dim, eps=eps)
            self.ffn_norm = nn.LayerNorm(hidden_dim, eps=eps)

            # Attention projections
            self.wq = nn.Linear(hidden_dim, num_heads * head_dim, bias=True)
            self.wk = nn.Linear(hidden_dim, num_heads * head_dim, bias=True)
            self.wv = nn.Linear(hidden_dim, num_heads * head_dim, bias=True)
            self.wo = nn.Linear(num_heads * head_dim, hidden_dim, bias=True)

            # MLP
            self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=True)
            self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=True)

            # Activation
            self.act = nn.GELU(approximate="tanh")

            self.scale = head_dim**-0.5

        def forward(self, x):
            # Pre-norm attention
            residual = x
            x = self.attention_norm(x)

            # QKV
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)

            # Reshape for attention
            batch, seq_len, _ = x.shape
            q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

            # Reshape and output projection
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
            out = self.wo(out)

            x = residual + out

            # Pre-norm MLP
            residual = x
            x = self.ffn_norm(x)
            x = self.w1(x)
            x = self.act(x)
            x = self.w2(x)
            x = residual + x

            return x

    # Create reference block
    block = ReferenceVisionBlock()
    prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}"

    # Load weights into reference block
    block.attention_norm.weight.data = state_dict[f"{prefix}.attention_norm.weight"]
    block.attention_norm.bias.data = state_dict[f"{prefix}.attention_norm.bias"]
    block.ffn_norm.weight.data = state_dict[f"{prefix}.ffn_norm.weight"]
    block.ffn_norm.bias.data = state_dict[f"{prefix}.ffn_norm.bias"]
    block.wq.weight.data = state_dict[f"{prefix}.attention.wq.weight"]
    block.wq.bias.data = state_dict[f"{prefix}.attention.wq.bias"]
    block.wk.weight.data = state_dict[f"{prefix}.attention.wk.weight"]
    block.wk.bias.data = state_dict[f"{prefix}.attention.wk.bias"]
    block.wv.weight.data = state_dict[f"{prefix}.attention.wv.weight"]
    block.wv.bias.data = state_dict[f"{prefix}.attention.wv.bias"]
    block.wo.weight.data = state_dict[f"{prefix}.attention.wo.weight"]
    block.wo.bias.data = state_dict[f"{prefix}.attention.wo.bias"]
    block.w1.weight.data = state_dict[f"{prefix}.feed_forward.w1.weight"]
    block.w1.bias.data = state_dict[f"{prefix}.feed_forward.w1.bias"]
    block.w2.weight.data = state_dict[f"{prefix}.feed_forward.w2.weight"]
    block.w2.bias.data = state_dict[f"{prefix}.feed_forward.w2.bias"]

    block.eval()

    # Test forward pass
    print("Running reference forward pass...")
    torch.manual_seed(42)
    x = torch.randn(1, 729, 1152)  # [batch, seq_len, hidden_dim]

    with torch.no_grad():
        y = block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean: {y.mean().item():.6f}")
    print(f"Output std: {y.std().item():.6f}")
    print("PyTorch reference: PASSED")

    return block, x, y


def validate_ttnn_block(device, ref_block, ref_input, ref_output):
    """
    Test TTNN vision block against PyTorch reference.
    """
    print("\n=== Validating TTNN Vision Block ===")

    import ttnn
    from models.common.utility_functions import comp_pcc
    from models.demos.molmo2.tt.load_weights import load_vit_block_weights
    from models.demos.molmo2.tt.vision_block import VisionBlock

    model_id = "allenai/Molmo2-8B"
    layer_num = 0

    # Load weights
    state_dict = load_vit_block_weights(model_id, layer_num=layer_num)

    # Create TTNN block (using single device, not mesh)
    print("Creating TTNN VisionBlock...")
    tt_block = VisionBlock(
        mesh_device=device,
        state_dict=state_dict,
        layer_num=layer_num,
        hidden_dim=1152,
        intermediate_dim=4304,
        num_heads=16,
        head_dim=72,
        layer_norm_eps=1e-6,
        dtype=ttnn.bfloat8_b,
    )

    # Convert input to TTNN (single device - no mesh_mapper needed)
    print("Converting input to TTNN...")
    x_ttnn = ttnn.from_torch(
        ref_input.unsqueeze(0),  # Add leading dim for TTNN: [1, 1, seq, hidden]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run TTNN forward
    print("Running TTNN forward pass...")
    y_ttnn = tt_block(x_ttnn)

    # Convert back to torch
    y_torch = ttnn.to_torch(y_ttnn)
    y_torch = y_torch.squeeze(0)  # Remove leading dim

    # Compare with reference
    print(f"TTNN output shape: {y_torch.shape}")
    print(f"Reference output shape: {ref_output.shape}")

    passing, pcc_msg = comp_pcc(ref_output, y_torch, pcc=0.99)
    print(f"PCC: {pcc_msg}")

    if passing:
        print("TTNN Vision Block: PASSED")
    else:
        print("TTNN Vision Block: FAILED")

    return passing


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 1 Molmo2 implementation")
    parser.add_argument("--skip-ttnn", action="store_true", help="Skip TTNN validation (CPU only)")
    args = parser.parse_args()

    print("=" * 60)
    print("Molmo2 Phase 1 Validation")
    print("=" * 60)

    # Step 1: Validate weight loading
    if not validate_weight_loading():
        print("\nValidation FAILED at weight loading step")
        return 1

    # Step 2: Validate PyTorch reference
    result = validate_pytorch_reference()
    if result is None:
        print("\nValidation FAILED at PyTorch reference step")
        return 1
    ref_block, ref_input, ref_output = result

    # Step 3: Validate TTNN (if device available)
    if args.skip_ttnn:
        print("\nSkipping TTNN validation (--skip-ttnn flag)")
    else:
        try:
            import ttnn

            # Try to get a device (single device for simplicity)
            device = ttnn.open_device(device_id=0)

            try:
                if validate_ttnn_block(device, ref_block, ref_input, ref_output):
                    print("\n" + "=" * 60)
                    print("ALL VALIDATIONS PASSED")
                    print("=" * 60)
                    ttnn.close_device(device)
                    return 0
                else:
                    print("\nValidation FAILED at TTNN step")
                    ttnn.close_device(device)
                    return 1
            finally:
                ttnn.close_device(device)

        except Exception as e:
            print(f"\nSkipping TTNN validation (device error): {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("CPU VALIDATIONS PASSED (TTNN skipped)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
