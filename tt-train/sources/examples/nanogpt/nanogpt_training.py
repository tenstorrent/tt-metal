# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Complete training example for NanoGPT with ttml.

This example demonstrates:
1. Setting up an optimizer (AdamW, matching C++ implementation)
2. Creating a training loop with gradient computation
3. Loading and tokenizing text data (Shakespeare dataset)
4. Implementing proper data batching and sequence handling

This implementation follows the structure of the C++ nano_gpt example.
"""

import os
import numpy as np
from typing import Tuple, Optional

import ttnn
import ttml
from ttml.models import NanoGPT, NanoGPTConfig, create_nanogpt

# Try to import datasets for loading Shakespeare text
try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class CharTokenizer:
    """Simple character-level tokenizer for text data."""

    def __init__(self, text: str):
        """Initialize tokenizer from text corpus.

        Args:
            text: Input text to build vocabulary from
        """
        # Create sorted stable vocabulary for reproducibility
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = chars

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        return "".join(self.itos[i] for i in ids)


def load_shakespeare_text() -> str:
    """Load Shakespeare text dataset (matching C++ implementation).

    Returns:
        Text content as string
    """
    if HAS_DATASETS:
        try:
            tt_metal_home = os.environ.get("TT_METAL_HOME", "")
            shakespeare_path = os.path.join(
                tt_metal_home, "tt-train/data/shakespeare.txt"
            )
            if os.path.exists(shakespeare_path):
                ds = load_dataset("text", data_files={"train": shakespeare_path})
                return "\n".join(ds["train"]["text"])
        except Exception as e:
            print(f"Warning: Could not load Shakespeare dataset: {e}")
            print("Falling back to example text...")

    # Fallback to example text
    return (
        """
    The quick brown fox jumps over the lazy dog.
    To be or not to be, that is the question.
    In the beginning was the Word, and the Word was with God.
    All that glitters is not gold.
    A journey of a thousand miles begins with a single step.
    """
        * 200
    )  # Repeat to have enough data


def prepare_data(
    text: str, train_split: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, int, CharTokenizer]:
    """Prepare training and validation data splits.

    Args:
        text: Full text corpus
        train_split: Fraction of data for training (default: 0.9)

    Returns:
        Tuple of (train_ids, val_ids, vocab_size, tokenizer)
    """
    tokenizer = CharTokenizer(text)
    ids = np.array(tokenizer.encode(text), dtype=np.uint32)

    # Pad vocab size to multiple of 32 (required by ttml)
    vocab_size = ((tokenizer.vocab_size + 31) // 32) * 32

    # Split into train/val
    n = len(ids)
    n_train = int(n * train_split)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    return train_ids, val_ids, vocab_size, tokenizer


def get_batch(
    ids: np.ndarray, seq_len: int, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Get a random batch of sequences from tokenized data.

    Args:
        ids: Array of token IDs
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        Tuple of (input_tokens, target_tokens)
        - input_tokens: [batch_size, seq_len] - input sequence
        - target_tokens: [batch_size, seq_len] - target sequence (shifted by 1)
    """
    n = len(ids) - seq_len - 1
    if n <= 0:
        raise ValueError(
            f"Data too short: need at least {seq_len + 1} tokens, got {len(ids)}"
        )

    # Random starting positions
    ix = np.random.randint(0, n, size=(batch_size,))

    # Extract sequences
    x = np.stack([ids[i : i + seq_len] for i in ix], axis=0)  # [B, T]
    y = np.stack(
        [ids[i + 1 : i + seq_len + 1] for i in ix], axis=0
    )  # [B, T] - next token targets

    return x.astype(np.uint32), y.astype(np.uint32)


def train_step(
    model: NanoGPT,
    optimizer: ttml.optimizers.AdamW,
    input_tokens: np.ndarray,
    target_tokens: np.ndarray,
    seq_len: int,
) -> float:
    """Perform a single training step (matching C++ implementation).

    Args:
        model: NanoGPT model
        optimizer: AdamW optimizer (matching C++ implementation)
        input_tokens: Input token indices [batch_size, seq_len]
        target_tokens: Target token indices [batch_size, seq_len]
        seq_len: Sequence length

    Returns:
        Loss value (float)
    """
    batch_size = input_tokens.shape[0]

    # Convert to ttml tensors
    # Input: [B, seq_len] -> [B, 1, 1, seq_len] (UINT32)
    input_tensor = ttml.autograd.Tensor.from_numpy(
        input_tokens.reshape(batch_size, 1, 1, seq_len),
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    # Forward pass
    logits = model(
        input_tensor
    )  # [B, 1, 1, seq_len, vocab_size] or [B, 1, seq_len, vocab_size]

    # Reshape logits for loss computation if needed
    # C++ implementation expects: logits [B, 1, seq_len, vocab_size], targets [B, seq_len]
    logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
    if len(logits_np.shape) == 5:
        # Reshape from [B, 1, 1, seq_len, vocab_size] to [B, 1, seq_len, vocab_size]
        logits_reshaped = logits_np.reshape(batch_size, 1, seq_len, logits_np.shape[-1])
    elif len(logits_np.shape) == 4:
        logits_reshaped = logits_np
    else:
        raise ValueError(f"Unexpected logits shape: {logits_np.shape}")
    logits_tensor = ttml.autograd.Tensor.from_numpy(logits_reshaped.astype(np.float32))

    # Target: [B, seq_len] (UINT32) - matching C++ implementation
    target_tensor = ttml.autograd.Tensor.from_numpy(
        target_tokens,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    # Compute loss (matching C++: ttml::ops::cross_entropy_loss(output, target))
    loss = ttml.ops.loss.cross_entropy_loss(
        logits_tensor, target_tensor, reduce=ttml.ops.ReduceType.MEAN
    )

    # Backward pass
    loss.backward(False)

    # Optimizer step
    optimizer.step()

    # Reset computation graph
    ttml.autograd.AutoContext.get_instance().reset_graph()

    # Get loss value
    loss_val = float(loss.to_numpy(ttnn.DataType.FLOAT32))
    return loss_val


def train(
    model: NanoGPT,
    optimizer: ttml.optimizers.AdamW,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    seq_len: int,
    batch_size: int,
    num_steps: int,
    eval_every: int = 10,
) -> Tuple[list[float], list[float]]:
    """Training loop for NanoGPT (matching C++ implementation structure).

    Args:
        model: NanoGPT model
        optimizer: AdamW optimizer (matching C++ implementation)
        train_ids: Training token IDs
        val_ids: Validation token IDs
        seq_len: Sequence length
        batch_size: Batch size
        num_steps: Number of training steps
        eval_every: Evaluate every N steps

    Returns:
        Tuple of (train_losses, val_losses) lists
    """
    model.train()
    train_losses = []
    val_losses = []

    print(f"\nStarting training for {num_steps} steps...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Training data: {len(train_ids)} tokens")
    print(f"  - Validation data: {len(val_ids)} tokens")

    for step in range(1, num_steps + 1):
        # Zero gradients
        optimizer.zero_grad()

        # Get training batch
        input_tokens, target_tokens = get_batch(train_ids, seq_len, batch_size)

        # Training step
        loss = train_step(model, optimizer, input_tokens, target_tokens, seq_len)
        train_losses.append(loss)

        # Print progress
        if step % 10 == 0 or step == 1:
            print(f"Step {step:4d}/{num_steps}: loss = {loss:.4f}")

        # Validation
        if step % eval_every == 0:
            model.eval()
            # Get validation batch
            val_input, val_target = get_batch(val_ids, seq_len, batch_size)

            # Forward pass only (no gradients)
            batch_size_val = val_input.shape[0]
            val_input_tensor = ttml.autograd.Tensor.from_numpy(
                val_input.reshape(batch_size_val, 1, 1, seq_len),
                layout=ttnn.Layout.ROW_MAJOR,
                new_type=ttnn.DataType.UINT32,
            )

            val_logits = model(val_input_tensor)
            val_logits_np = val_logits.to_numpy(ttnn.DataType.FLOAT32)
            val_logits_reshaped = val_logits_np.reshape(
                batch_size_val, 1, seq_len, val_logits_np.shape[-1]
            )
            val_logits_tensor = ttml.autograd.Tensor.from_numpy(
                val_logits_reshaped.astype(np.float32)
            )

            val_target_tensor = ttml.autograd.Tensor.from_numpy(
                val_target,
                layout=ttnn.Layout.ROW_MAJOR,
                new_type=ttnn.DataType.UINT32,
            )

            val_loss = ttml.ops.loss.cross_entropy_loss(
                val_logits_tensor, val_target_tensor, reduce=ttml.ops.ReduceType.MEAN
            )
            val_loss_val = float(val_loss.to_numpy(ttnn.DataType.FLOAT32))
            val_losses.append(val_loss_val)

            print(f"  Validation loss: {val_loss_val:.4f}")
            model.train()

    return train_losses, val_losses


def main():
    """Main training function."""
    print("=" * 70)
    print("NanoGPT Training Example with ttml")
    print("=" * 70)

    # Configuration
    config = NanoGPTConfig(
        vocab_size=1000,  # Will be updated from tokenizer
        block_size=128,
        n_embd=256,
        n_layer=4,
        n_head=4,
        dropout=0.1,
        bias=True,
    )

    # Load and prepare data (matching C++ implementation)
    print("\n1. Loading and preparing data...")
    text = load_shakespeare_text()

    train_ids, val_ids, vocab_size, tokenizer = prepare_data(text, train_split=0.9)
    print(f"   - Vocabulary size: {tokenizer.vocab_size} (padded to {vocab_size})")
    print(f"   - Training tokens: {len(train_ids)}")
    print(f"   - Validation tokens: {len(val_ids)}")

    # Update config with actual vocab size
    config.vocab_size = vocab_size

    # Create model
    print("\n2. Creating model...")
    model = create_nanogpt(config)
    print(f"   - Model created: {model.__class__.__name__}")

    # Count parameters
    params = model.parameters()
    total_params = sum(
        np.prod(p.to_numpy(ttnn.DataType.FLOAT32).shape) for p in params.values()
    )
    print(f"   - Total parameters: {total_params:,}")

    # Create optimizer (matching C++ implementation: AdamW)
    print("\n3. Setting up optimizer...")
    learning_rate = 3e-4  # Default from C++ training config
    weight_decay = 1e-2  # Default from C++ training config
    adamw_config = ttml.optimizers.AdamWConfig.make(
        lr=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=weight_decay,
    )
    optimizer = ttml.optimizers.AdamW(model.parameters(), adamw_config)
    print(f"   - Optimizer: AdamW (matching C++ implementation)")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Weight decay: {weight_decay}")

    # Training configuration
    seq_len = 32
    batch_size = 2
    num_steps = 50

    # Train
    print("\n4. Training...")
    train_losses, val_losses = train(
        model=model,
        optimizer=optimizer,
        train_ids=train_ids,
        val_ids=val_ids,
        seq_len=seq_len,
        batch_size=batch_size,
        num_steps=num_steps,
        eval_every=10,
    )

    # Summary
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"\nFinal training loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"\nLoss trend:")
    print(f"  - Initial loss: {train_losses[0]:.4f}")
    print(f"  - Final loss: {train_losses[-1]:.4f}")
    print(f"  - Improvement: {train_losses[0] - train_losses[-1]:.4f}")

    print("\nNext steps:")
    print("  - Increase num_steps for longer training")
    print("  - Adjust learning rate and other hyperparameters")
    print("  - Use larger models (more layers, larger embedding dim)")
    print("  - Load real text datasets (Shakespeare, Wikipedia, etc.)")
    print("  - Implement learning rate scheduling")
    print("  - Add model checkpointing")


if __name__ == "__main__":
    main()
