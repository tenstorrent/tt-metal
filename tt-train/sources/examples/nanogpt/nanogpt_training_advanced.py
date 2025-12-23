# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Advanced training example for NanoGPT with ttml.

This example includes:
1. Configurable hyperparameters and model sizes
2. Learning rate scheduling (Python implementation)
3. Model checkpointing using Python pickle
4. Support for real text datasets (Shakespeare, etc.)
5. Extended training with proper configuration
"""

import os
import pickle
import argparse
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.models import NanoGPT, NanoGPTConfig, create_nanogpt
from ttml.modules import Parameter

# Try to import datasets for loading text data
try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model configuration
    vocab_size: int = 1000  # Will be updated from tokenizer
    block_size: int = 128
    n_embd: int = 384  # Embedding dimension
    n_layer: int = 6  # Number of transformer blocks
    n_head: int = 6  # Number of attention heads
    dropout: float = 0.1
    bias: bool = True

    # Training configuration
    batch_size: int = 64
    seq_len: int = 128
    num_steps: int = 5000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    eval_every: int = 100
    save_every: int = 500

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "warmup_linear"  # "identity", "warmup_linear", "cosine"
    warmup_steps: int = 100  # Reduced from 500 - too long warmup can prevent learning
    min_lr_factor: float = 0.1  # Minimum LR as fraction of base LR

    # Data configuration
    # Path to text file, or None for Shakespeare
    data_path: Optional[str] = None
    train_split: float = 0.9

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    save_best: bool = True  # Save checkpoint when validation loss improves

    # Model size preset (overrides n_embd, n_layer, n_head if set)
    model_size: Optional[str] = None  # "tiny", "small", "medium", "large"

    def apply_model_preset(self):
        """Apply model size preset if specified."""
        if self.model_size is None:
            return

        presets = {
            "tiny": {"n_embd": 128, "n_layer": 2, "n_head": 2},
            "small": {"n_embd": 256, "n_layer": 4, "n_head": 4},
            # NanoGPT default
            "medium": {"n_embd": 384, "n_layer": 6, "n_head": 6},
            # GPT-2 small
            "large": {"n_embd": 768, "n_layer": 12, "n_head": 12},
        }

        if self.model_size not in presets:
            raise ValueError(
                f"Unknown model size: {self.model_size}. Choose from {list(presets.keys())}"
            )

        preset = presets[self.model_size]
        self.n_embd = preset["n_embd"]
        self.n_layer = preset["n_layer"]
        self.n_head = preset["n_head"]


class CharTokenizer:
    """Simple character-level tokenizer for text data."""

    def __init__(self, text: str):
        """Initialize tokenizer from text corpus."""
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = chars

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return "".join(self.itos[i] for i in ids)


def load_shakespeare_text() -> str:
    """Load Shakespeare text dataset."""
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
    )


def load_text_from_file(file_path: str) -> str:
    """Load text from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_data(
    text: str, train_split: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, int, CharTokenizer]:
    """Prepare training and validation data splits."""
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
    """Get a random batch of sequences from tokenized data."""
    n = len(ids) - seq_len - 1
    if n <= 0:
        raise ValueError(
            f"Data too short: need at least {seq_len + 1} tokens, got {len(ids)}"
        )

    ix = np.random.randint(0, n, size=(batch_size,))
    x = np.stack([ids[i : i + seq_len] for i in ix], axis=0)  # [B, T]
    y = np.stack([ids[i + 1 : i + seq_len + 1] for i in ix], axis=0)  # [B, T]

    return x.astype(np.uint32), y.astype(np.uint32)


class LRScheduler:
    """Python-based learning rate scheduler."""

    def __init__(
        self,
        optimizer: ttml.optimizers.AdamW,
        scheduler_type: str,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_factor: float = 0.1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            scheduler_type: "identity", "warmup_linear", or "cosine"
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps (for warmup_linear)
            min_lr_factor: Minimum LR as fraction of base LR
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_factor = min_lr_factor
        self.base_lr = optimizer.get_lr()
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        if self.scheduler_type == "identity":
            lr = self.base_lr
        elif self.scheduler_type == "warmup_linear":
            if self.current_step <= self.warmup_steps:
                # Warmup: linear from 0 to base_lr
                lr = self.base_lr * (self.current_step / self.warmup_steps)
            else:
                # Linear decay from base_lr to min_lr
                decay_steps = self.total_steps - self.warmup_steps
                progress = (self.current_step - self.warmup_steps) / decay_steps
                progress = min(progress, 1.0)
                # Decay from base_lr to min_lr_factor * base_lr
                lr = self.base_lr * (
                    self.min_lr_factor + (1.0 - self.min_lr_factor) * (1.0 - progress)
                )
        elif self.scheduler_type == "cosine":
            if self.current_step <= self.warmup_steps:
                # Warmup
                lr = self.base_lr * (self.current_step / self.warmup_steps)
            else:
                # Cosine decay
                decay_steps = self.total_steps - self.warmup_steps
                progress = (self.current_step - self.warmup_steps) / decay_steps
                progress = min(progress, 1.0)
                lr = self.base_lr * (
                    self.min_lr_factor
                    + (1.0 - self.min_lr_factor)
                    * 0.5
                    * (1.0 + np.cos(np.pi * progress))
                )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        self.optimizer.set_lr(lr)

    def get_last_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.get_lr()


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: NanoGPT,
    optimizer: ttml.optimizers.AdamW,
    scheduler: Optional[LRScheduler],
    train_losses: list[float],
    val_losses: list[float],
    best_val_loss: Optional[float],
    config: TrainingConfig,
    tokenizer: CharTokenizer,
):
    """Save training checkpoint using Python pickle."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model parameters with metadata
    model_state = {}
    for name, param in model.parameters().items():
        # Handle both Parameter objects and direct Tensor objects
        if isinstance(param, Parameter):
            tensor = param.tensor
        else:
            # param is already a Tensor
            tensor = param

        # Get tensor metadata
        try:
            layout = tensor.get_layout()
        except:
            layout = ttnn.Layout.TILE  # Default for weights

        # Convert tensor to numpy for serialization
        numpy_array = tensor.to_numpy()
        model_state[name] = {
            "data": numpy_array,
            "layout": layout.value if hasattr(layout, "value") else str(layout),
            "shape": numpy_array.shape,
        }

    # Save optimizer state - only save simple values (step count, etc.)
    # NamedParameters can't be pickled, so we skip the full state
    try:
        optimizer_state_raw = optimizer.get_state_dict()
        # Extract only pickleable values (like step count)
        optimizer_state = {}
        for key, value in optimizer_state_raw.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                optimizer_state[key] = value
            # Try to get step count if available
            if hasattr(optimizer, "get_steps"):
                optimizer_state["steps"] = optimizer.get_steps()
    except Exception as e:
        print(f"  Warning: Could not save optimizer state: {e}")
        optimizer_state = {}
        if hasattr(optimizer, "get_steps"):
            optimizer_state["steps"] = optimizer.get_steps()

    # Save scheduler state
    scheduler_state = None
    if scheduler is not None:
        scheduler_state = {
            "current_step": scheduler.current_step,
            "scheduler_type": scheduler.scheduler_type,
        }

    checkpoint = {
        "step": step,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "config": config,
        "tokenizer": tokenizer,  # Save tokenizer for decoding
    }

    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    # Save latest checkpoint
    latest_path = checkpoint_dir / "checkpoint_latest.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"  Saved checkpoint to {checkpoint_path}")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: NanoGPT,
    optimizer: ttml.optimizers.AdamW,
    scheduler: Optional[LRScheduler],
) -> Dict[str, Any]:
    """Load training checkpoint from pickle file."""
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Load model parameters
    model_state = checkpoint["model_state"]
    model_params = model.parameters()

    print("  Loading model parameters...")
    for name, param_data in model_state.items():
        if name in model_params:
            # Handle both old format (just numpy array) and new format (dict with metadata)
            if isinstance(param_data, dict):
                numpy_array = param_data["data"]
                layout_str = param_data.get("layout", "TILE")
            else:
                # Old format: just numpy array
                numpy_array = param_data
                layout_str = "TILE"  # Default

            # Determine layout
            if layout_str == "ROW_MAJOR" or "ROW_MAJOR" in str(layout_str):
                layout = ttnn.Layout.ROW_MAJOR
            else:
                layout = ttnn.Layout.TILE  # Default for weights

            # Convert numpy back to ttml tensor
            # Use bfloat16 for weights (matching initialization)
            numpy_bfloat16 = numpy_array.astype(ml_dtypes.bfloat16)
            restored_tensor = ttml.autograd.Tensor.from_numpy(
                numpy_bfloat16, layout=layout
            )

            # Create new Parameter and assign to model
            new_param = Parameter(restored_tensor)

            # Update the parameter in the model
            # We need to access the model's internal structure
            parts = name.split(".")
            if len(parts) == 1:
                setattr(model, name, new_param)
            else:
                # Nested parameter
                module = model
                for part in parts[:-1]:
                    module = getattr(module, part)
                setattr(module, parts[-1], new_param)

            print(f"    Loaded: {name} (shape: {numpy_array.shape})")
        else:
            print(f"    Warning: Parameter {name} not found in model")

    # Load optimizer state
    print("  Loading optimizer state...")
    # Note: We can't fully restore optimizer state from pickle since NamedParameters
    # can't be reconstructed. We'll skip optimizer state restoration for now.
    # The optimizer will continue from its current state, which is acceptable
    # for most use cases (the model parameters are what matter most).
    # optimizer.set_state_dict(checkpoint["optimizer_state"])  # Skipped - see comment above
    print("    Note: Optimizer state not restored (NamedParameters not pickleable)")

    # Load scheduler state
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        print("  Loading scheduler state...")
        scheduler.current_step = checkpoint["scheduler_state"]["current_step"]

    return checkpoint


def train_step(
    model: NanoGPT,
    optimizer: ttml.optimizers.AdamW,
    scheduler: Optional[LRScheduler],
    input_tokens: np.ndarray,
    target_tokens: np.ndarray,
    seq_len: int,
) -> float:
    """Perform a single training step."""
    batch_size = input_tokens.shape[0]

    # Convert to ttml tensors
    input_tensor = ttml.autograd.Tensor.from_numpy(
        input_tokens.reshape(batch_size, 1, 1, seq_len),
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    # Forward pass
    logits = model(input_tensor)

    # Model should already output correct 4D shape [B, 1, seq_len, vocab_size]
    # Use logits directly - model.forward already handles reshaping
    logits_tensor = logits

    # Target tensor - cross_entropy_loss expects 2D [B, seq_len]
    # target_tokens is already [B, seq_len] from get_batch, convert to ttml tensor
    target_tensor = ttml.autograd.Tensor.from_numpy(
        target_tokens,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    # Compute loss
    loss = ttml.ops.loss.cross_entropy_loss(
        logits_tensor, target_tensor, reduce=ttml.ops.ReduceType.MEAN
    )

    # Backward pass
    loss.backward(False)

    # Optimizer step
    optimizer.step()

    # Scheduler step
    if scheduler is not None:
        scheduler.step()

    # Reset computation graph
    ttml.autograd.AutoContext.get_instance().reset_graph()

    # Get loss value - extract scalar to avoid deprecation warning
    loss_np = loss.to_numpy()
    loss_val = float(loss_np.item() if hasattr(loss_np, "item") else loss_np)
    return loss_val


def train(
    model: NanoGPT,
    optimizer: ttml.optimizers.AdamW,
    scheduler: Optional[LRScheduler],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    config: TrainingConfig,
    tokenizer: CharTokenizer,
    start_step: int = 0,
    best_val_loss: Optional[float] = None,
) -> Tuple[list[float], list[float]]:
    """Training loop for NanoGPT."""
    model.train()
    train_losses = []
    val_losses = []

    print(
        f"\nStarting training for {config.num_steps} steps (starting from step {start_step})..."
    )
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Sequence length: {config.seq_len}")
    print(f"  - Training data: {len(train_ids)} tokens")
    print(f"  - Validation data: {len(val_ids)} tokens")
    if scheduler:
        print(f"  - Scheduler: {scheduler.scheduler_type}")

    for step in range(start_step + 1, start_step + config.num_steps + 1):
        # Zero gradients
        optimizer.zero_grad()

        # Get training batch
        input_tokens, target_tokens = get_batch(
            train_ids, config.seq_len, config.batch_size
        )

        # Training step
        loss = train_step(
            model, optimizer, scheduler, input_tokens, target_tokens, config.seq_len
        )
        train_losses.append(loss)

        # Print progress
        if step % 10 == 0 or step == 1:
            lr = optimizer.get_lr()
            print(
                f"Step {step:5d}/{start_step + config.num_steps}: loss = {loss:.4f}, lr = {lr:.2e}"
            )

        # Validation
        if step % config.eval_every == 0:
            model.eval()
            val_input, val_target = get_batch(
                val_ids, config.seq_len, config.batch_size
            )

            batch_size_val = val_input.shape[0]
            val_input_tensor = ttml.autograd.Tensor.from_numpy(
                val_input.reshape(batch_size_val, 1, 1, config.seq_len),
                layout=ttnn.Layout.ROW_MAJOR,
                new_type=ttnn.DataType.UINT32,
            )

            val_logits = model(val_input_tensor)
            # Reshape if needed using ttml's reshape operation (preserves gradients)
            val_logits_shape = val_logits.shape()
            if len(val_logits_shape) == 5:
                # [B, 1, 1, seq_len, vocab_size] -> [B, 1, seq_len, vocab_size]
                new_shape = [batch_size_val, 1, config.seq_len, val_logits_shape[4]]
                val_logits_tensor = ttml.ops.reshape.reshape(val_logits, new_shape)
            else:
                val_logits_tensor = val_logits

            val_target_tensor = ttml.autograd.Tensor.from_numpy(
                val_target,
                layout=ttnn.Layout.ROW_MAJOR,
                new_type=ttnn.DataType.UINT32,
            )

            val_loss = ttml.ops.loss.cross_entropy_loss(
                val_logits_tensor, val_target_tensor, reduce=ttml.ops.ReduceType.MEAN
            )
            # Extract scalar value from numpy array to avoid deprecation warning
            loss_np = val_loss.to_numpy()
            val_loss_val = float(
                loss_np.item() if hasattr(loss_np, "item") else loss_np
            )
            val_losses.append(val_loss_val)

            print(f"  Validation loss: {val_loss_val:.4f}")

            # Save best checkpoint
            if config.save_best and (
                best_val_loss is None or val_loss_val < best_val_loss
            ):
                best_val_loss = val_loss_val
                save_checkpoint(
                    config.checkpoint_dir,
                    step,
                    model,
                    optimizer,
                    scheduler,
                    train_losses,
                    val_losses,
                    best_val_loss,
                    config,
                    tokenizer,
                )

            model.train()

        # Save checkpoint periodically
        if step % config.save_every == 0:
            save_checkpoint(
                config.checkpoint_dir,
                step,
                model,
                optimizer,
                scheduler,
                train_losses,
                val_losses,
                best_val_loss,
                config,
                tokenizer,
            )

    return train_losses, val_losses


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train NanoGPT with ttml")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "small", "medium", "large"],
        help="Model size preset",
    )
    parser.add_argument(
        "--num-steps", type=int, default=5000, help="Number of training steps"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to text data file"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["identity", "warmup_linear", "cosine"],
        default="warmup_linear",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--no-scheduler", action="store_true", help="Disable learning rate scheduler"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NanoGPT Advanced Training Example with ttml")
    print("=" * 70)

    # Create configuration
    config = TrainingConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        model_size=args.model_size,
        scheduler_type=args.scheduler if not args.no_scheduler else "identity",
        use_scheduler=not args.no_scheduler,
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
    )
    config.apply_model_preset()

    # Load and prepare data
    print("\n1. Loading and preparing data...")
    if config.data_path and os.path.exists(config.data_path):
        text = load_text_from_file(config.data_path)
        print(f"   - Loaded text from: {config.data_path}")
    else:
        text = load_shakespeare_text()
        print(f"   - Using Shakespeare dataset (or fallback)")

    train_ids, val_ids, vocab_size, tokenizer = prepare_data(
        text, train_split=config.train_split
    )
    print(f"   - Vocabulary size: {tokenizer.vocab_size} (padded to {vocab_size})")
    print(f"   - Training tokens: {len(train_ids)}")
    print(f"   - Validation tokens: {len(val_ids)}")

    # Update config with actual vocab size
    config.vocab_size = vocab_size

    # Create model
    print("\n2. Creating model...")
    model_config = NanoGPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        dropout=config.dropout,
        bias=config.bias,
    )
    model = create_nanogpt(model_config)
    print(
        f"   - Model: {config.n_layer} layers, {config.n_embd} embd, {config.n_head} heads"
    )

    # Count parameters
    params = model.parameters()
    total_params = sum(np.prod(p.to_numpy().shape) for p in params.values())
    print(f"   - Total parameters: {total_params:,}")

    # Create optimizer
    print("\n3. Setting up optimizer...")
    adamw_config = ttml.optimizers.AdamWConfig.make(
        lr=config.learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=config.weight_decay,
    )
    optimizer = ttml.optimizers.AdamW(model.parameters(), adamw_config)
    print(f"   - Optimizer: AdamW")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Weight decay: {config.weight_decay}")

    # Create scheduler
    scheduler = None
    if config.use_scheduler:
        print("\n4. Setting up learning rate scheduler...")
        scheduler = LRScheduler(
            optimizer,
            config.scheduler_type,
            config.num_steps,
            warmup_steps=config.warmup_steps,
            min_lr_factor=config.min_lr_factor,
        )
        print(f"   - Scheduler: {config.scheduler_type}")
        print(f"   - Warmup steps: {config.warmup_steps}")

    # Resume from checkpoint if specified
    start_step = 0
    best_val_loss = None
    if config.resume_from:
        print(f"\n5. Resuming from checkpoint: {config.resume_from}")
        checkpoint = load_checkpoint(config.resume_from, model, optimizer, scheduler)
        start_step = checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss")
        print(f"   - Resuming from step {start_step}")
        print(f"   - Best validation loss: {best_val_loss}")

    # Train
    print("\n6. Training...")
    train_losses, val_losses = train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_ids=train_ids,
        val_ids=val_ids,
        config=config,
        tokenizer=tokenizer,
        start_step=start_step,
        best_val_loss=best_val_loss,
    )

    # Final checkpoint
    final_step = start_step + config.num_steps
    save_checkpoint(
        config.checkpoint_dir,
        final_step,
        model,
        optimizer,
        scheduler,
        train_losses,
        val_losses,
        best_val_loss,
        config,
        tokenizer,
    )

    # Summary
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"\nFinal training loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")
        print(f"Best validation loss: {min(val_losses):.4f}")
    print(f"\nLoss trend:")
    print(f"  - Initial loss: {train_losses[0]:.4f}")
    print(f"  - Final loss: {train_losses[-1]:.4f}")
    print(f"  - Improvement: {train_losses[0] - train_losses[-1]:.4f}")
    print(f"\nCheckpoints saved to: {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
