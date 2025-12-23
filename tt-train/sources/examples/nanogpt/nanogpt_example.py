# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Example program demonstrating NanoGPT usage with ttml.

This example shows how to:
1. Create a NanoGPT model with a custom configuration
2. Perform a forward pass with token indices
3. Access model parameters
4. Set up a simple training loop with loss computation
"""

import numpy as np

import ttnn
import ttml
from ttml.models import NanoGPT, NanoGPTConfig, create_nanogpt


def main():
    """Main example function."""
    print("=" * 60)
    print("NanoGPT Example with ttml")
    print("=" * 60)

    # Create a smaller configuration for demonstration
    # (smaller than default to make it run faster)
    config = NanoGPTConfig(
        vocab_size=1000,  # Smaller vocabulary
        block_size=128,  # Shorter sequence length
        n_embd=256,  # Smaller embedding dimension
        n_layer=4,  # Fewer transformer blocks
        n_head=4,  # Fewer attention heads
        dropout=0.1,  # 10% dropout
        bias=True,  # Use bias in layers
    )

    print("\n1. Creating NanoGPT model...")
    print(f"   Configuration:")
    print(f"   - Vocab size: {config.vocab_size}")
    print(f"   - Block size: {config.block_size}")
    print(f"   - Embedding dim: {config.n_embd}")
    print(f"   - Number of layers: {config.n_layer}")
    print(f"   - Number of heads: {config.n_head}")
    print(f"   - Dropout: {config.dropout}")

    # Create the model
    model = create_nanogpt(config)
    print(f"\n   Model created: {model.__class__.__name__}")

    # Print model information
    print("\n2. Model structure:")
    print(f"   - Token embeddings: {model.wte.__class__.__name__}")
    print(f"   - Position embeddings: {model.wpe.__class__.__name__}")
    print(f"   - Number of transformer blocks: {len(model.blocks)}")
    print(f"   - Final layer norm: present")
    print(f"   - Language model head: present (weight-tied with token embeddings)")

    # Access model parameters
    print("\n3. Model parameters:")
    params = model.parameters()
    print(f"   - Total number of parameter groups: {len(params)}")

    # Count total parameters (approximate)
    total_params = 0
    for name, param in params.items():
        param_shape = param.to_numpy(ttnn.DataType.FLOAT32).shape
        param_count = np.prod(param_shape)
        total_params += param_count
        if (
            "weight" in name.lower()
            or "gamma" in name.lower()
            or "beta" in name.lower()
        ):
            # Only print a few key parameters to avoid clutter
            if "wte" in name or "ln_f" in name or "block_0" in name:
                print(f"   - {name}: shape {param_shape}, params: {param_count:,}")

    print(f"\n   - Total parameters (approx): {total_params:,}")

    # Create dummy input data
    print("\n4. Creating input data...")
    batch_size = 2
    seq_len = 32  # Shorter sequence for example

    # Generate random token indices (integers in range [0, vocab_size))
    # Embedding operation requires UINT32 for token indices
    token_indices = np.random.randint(
        0, config.vocab_size, size=(batch_size, 1, 1, seq_len), dtype=np.uint32
    )
    print(f"   - Batch size: {batch_size}")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Token indices shape: {token_indices.shape}")
    print(f"   - Token indices range: [{token_indices.min()}, {token_indices.max()}]")

    # Convert to ttml tensor with UINT32 dtype (required by embedding operation)
    input_tensor = ttml.autograd.Tensor.from_numpy(
        token_indices,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )
    print(f"   - Input tensor created: {type(input_tensor).__name__}")
    print(f"   - Input tensor dtype: UINT32 (required for embedding)")

    # Forward pass
    print("\n5. Performing forward pass...")
    try:
        model.eval()  # Set to evaluation mode
        logits = model(input_tensor)

        # Get output shape (convert to float32 for NumPy operations)
        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
        print(f"   - Forward pass successful!")
        print(f"   - Output logits shape: {logits_np.shape}")
        print(
            f"   - Expected shape: ({batch_size}, 1, 1, {seq_len}, {config.vocab_size})"
        )
        print(f"   - Logits range: [{logits_np.min():.4f}, {logits_np.max():.4f}]")
        print(f"   - Logits mean: {logits_np.mean():.4f}")
        print(f"   - Logits std: {logits_np.std():.4f}")
    except Exception as e:
        print(f"   - Forward pass failed: {e}")
        print(f"   - Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return

    # Demonstrate loss computation
    print("\n6. Computing loss...")
    try:
        # Create dummy target tokens (next token prediction)
        # Cross-entropy expects: prediction [B, 1, seq_len, vocab_size], target [B, seq_len]
        print(f"   - Logits shape: {logits_np.shape}")

        # Reshape logits to match cross_entropy_loss requirements
        # cross_entropy_loss expects: prediction [B, 1, seq_len, vocab_size], target [B, seq_len]
        if len(logits_np.shape) == 5:
            # [B, 1, 1, seq_len, vocab_size] -> [B, 1, seq_len, vocab_size]
            logits_reshaped = logits_np.reshape(
                logits_np.shape[0], 1, logits_np.shape[3], logits_np.shape[4]
            )
        elif len(logits_np.shape) == 4:
            # Already 4D, might just need to ensure correct shape
            logits_reshaped = logits_np
        else:
            raise ValueError(f"Unexpected logits shape: {logits_np.shape}")

        print(f"   - Reshaped logits shape: {logits_reshaped.shape}")
        print(f"   - Expected shape: ({batch_size}, 1, {seq_len}, {config.vocab_size})")

        target_tokens = np.random.randint(
            0, config.vocab_size, size=(batch_size, seq_len), dtype=np.uint32
        )
        target_tensor = ttml.autograd.Tensor.from_numpy(
            target_tokens,
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
        )

        # Convert to ttml tensors for loss computation
        logits_tensor = ttml.autograd.Tensor.from_numpy(
            logits_reshaped.astype(np.float32)
        )

        # Compute cross-entropy loss
        # cross_entropy_loss expects: prediction [B, 1, seq_len, vocab_size], target [B, seq_len]
        loss = ttml.ops.loss.cross_entropy_loss(
            logits_tensor, target_tensor, reduce=ttml.ops.ReduceType.MEAN
        )

        loss_np = loss.to_numpy(ttnn.DataType.FLOAT32)
        print(f"   - Loss computed successfully!")
        print(f"   - Cross-entropy loss: {loss_np.item():.4f}")
        print(
            f"   - Expected range: ~[0, log(vocab_size)] = [0, {np.log(config.vocab_size):.2f}]"
        )
    except Exception as e:
        print(f"   - Loss computation failed: {e}")
        print(f"   - Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()

    # Demonstrate training mode
    print("\n7. Training mode...")
    model.train()  # Set to training mode
    print(f"   - Model set to training mode")
    print(f"   - Dropout will be active during forward pass")

    # Show parameter access
    print("\n8. Accessing specific parameters...")
    try:
        # Access token embedding weight
        wte_weight = model.wte.weight.tensor
        wte_weight_np = wte_weight.to_numpy(ttnn.DataType.FLOAT32)
        print(f"   - Token embedding weight shape: {wte_weight_np.shape}")
        print(
            f"   - Token embedding weight stats: mean={wte_weight_np.mean():.4f}, std={wte_weight_np.std():.4f}"
        )

        # Access first block's attention weights
        first_block = model.blocks[0]
        if hasattr(first_block.attention, "qkv"):
            qkv_weight = first_block.attention.qkv.tensor
            qkv_weight_np = qkv_weight.to_numpy(ttnn.DataType.FLOAT32)
            print(f"   - First block QKV weight shape: {qkv_weight_np.shape}")
    except Exception as e:
        print(f"   - Parameter access note: {e}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nFor a complete training example with optimizer, training loop,")
    print("and data loading, see: nanogpt_training.py")


if __name__ == "__main__":
    main()
