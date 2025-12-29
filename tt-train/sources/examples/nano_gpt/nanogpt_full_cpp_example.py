# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Full-featured NanoGPT training example matching the C++ implementation.

This example provides a comprehensive Python implementation that mirrors the C++ nano_gpt example,
including:
- Full model training with GPT2/NanoGPT architectures
- Gradient accumulation
- Learning rate scheduling (identity, warmup_linear)
- Optimizers (AdamW, MorehAdamW)
- Model checkpointing and resuming
- Loss tracking and averaging
- Configurable training parameters
- Support for character and BPE tokenizers
- Proper tensor shapes matching C++ implementation
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import pickle

import numpy as np
import ml_dtypes
import yaml

import ttnn
import ttml
from ttml.models.nanogpt import NanoGPT, NanoGPTConfig, create_nanogpt
from ttml.modules import Parameter


@dataclass
class TrainingConfig:
    """Training configuration matching C++ TrainingConfig."""

    project_name: str = "tt_train_nano_gpt"
    seed: int = 5489
    model_save_interval: int = 500
    batch_size: int = 64
    num_epochs: int = 1
    max_steps: int = 5000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    use_no_op: bool = False
    use_moreh_adamw: bool = False
    use_kahan_summation: bool = False
    gradient_accumulation_steps: int = 1
    model_config: str = ""
    data_path: str = ""
    scheduler_type: str = "identity"  # "identity" or "warmup_linear"
    tokenizer_type: str = "char"  # "char" or "bpe"
    use_clip_grad_norm: bool = False
    clip_grad_norm_max_norm: float = 1.0


@dataclass
class ModelConfig:
    """Model configuration matching C++ ModelConfig."""

    model_type: str = "gpt2"  # "gpt2" or "llama"
    model_path: str = ""
    vocab_size: int = 50304
    block_size: int = 128  # Reduced from 1024 to avoid memory issues
    n_embd: int = 384  # NanoGPT default (reduced from 768)
    n_layer: int = 6  # NanoGPT default (reduced from 12)
    n_head: int = 6  # NanoGPT default (reduced from 12)
    dropout: float = 0.2  # Match C++ default: float dropout_prob = 0.2F
    bias: bool = True
    max_sequence_length: int = 128  # Reduced from 1024 to avoid memory issues


class LossAverageMeter:
    """Loss averaging meter matching C++ LossAverageMeter."""

    def __init__(self):
        self.m_sum = 0.0
        self.m_count = 0

    def update(self, loss: float, count: int = 1):
        """Update with a loss value."""
        self.m_sum += loss * count
        self.m_count += count

    def average(self) -> float:
        """Get average loss."""
        if self.m_count == 0:
            return 0.0
        return self.m_sum / self.m_count

    def reset(self):
        """Reset the meter."""
        self.m_sum = 0.0
        self.m_count = 0


class GradientAccumulator:
    """Gradient accumulator matching C++ GradientAccumulator."""

    def __init__(self, accumulation_steps: int):
        self.m_accumulation_steps = accumulation_steps
        self.m_steps = 0
        self.m_total_loss = 0.0
        self.m_total_samples = 0

    def should_zero_grad(self) -> bool:
        """Check if gradients should be zeroed."""
        return self.m_steps % self.m_accumulation_steps == 0

    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return self.m_steps % self.m_accumulation_steps == 0

    def scale(self, loss: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Scale loss by accumulation steps (matching C++: ttml::ops::mul(tensor, 1.0F / accumulation_steps))."""
        if self.m_accumulation_steps > 1:
            scale_factor = 1.0 / float(self.m_accumulation_steps)
            # Use float overload directly (matching C++ implementation)
            # This avoids creating an intermediate tensor and potential materialization issues
            return ttml.ops.binary.mul(loss, scale_factor)
        return loss

    def update(self, loss: float, samples: int = 1):
        """Update accumulator with loss."""
        self.m_total_loss += loss * samples * float(self.m_accumulation_steps)
        self.m_total_samples += samples
        self.m_steps += 1

    def reset(self):
        """Reset accumulator."""
        self.m_total_loss = 0.0
        self.m_total_samples = 0
        self.m_steps = 0

    def average_loss(self) -> float:
        """Get average loss."""
        if self.m_total_samples == 0:
            return 0.0
        return self.m_total_loss / float(self.m_total_samples)


class CharTokenizer:
    """Character-level tokenizer matching C++ CharTokenizer."""

    def __init__(self, text: str):
        """Initialize tokenizer from text."""
        # Get unique characters and sort them
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text."""
        # Handle out-of-range token IDs (can happen if model vocab_size > tokenizer vocab_size)
        result = []
        for t in tokens:
            if t in self.itos:
                result.append(self.itos[t])
            else:
                # Use a fallback character (space) for out-of-range tokens
                result.append(" ")
        return "".join(result)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size


def read_file_to_str(file_path: str) -> str:
    """Read file to string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def round_up_to_tile(value: int, tile_size: int = 32) -> int:
    """Round up value to nearest multiple of tile_size."""
    return ((value + tile_size - 1) // tile_size) * tile_size


def create_identity_scheduler(optimizer, total_steps: int):
    """Create identity (constant) learning rate scheduler."""
    # For identity scheduler, we don't need to do anything
    # The learning rate stays constant
    return None


def create_warmup_linear_scheduler(optimizer, total_steps: int):
    """Create warmup + linear decay scheduler matching C++ implementation."""
    warmup_factor = 0.1
    warmup_steps = int(total_steps * warmup_factor)
    linear_decay_steps = total_steps - warmup_steps

    def scheduler_fn(step: int) -> float:
        if step < warmup_steps:
            # Warmup: linear from 0.0 to 1.0
            return float(step) / float(warmup_steps)
        else:
            # Linear decay: from 1.0 to 0.01
            decay_step = step - warmup_steps
            return 1.0 - (0.99 * float(decay_step) / float(linear_decay_steps))

    return scheduler_fn, warmup_steps, linear_decay_steps


def create_dataset_from_text(
    text: str, sequence_length: int, tokenizer_type: str = "char"
) -> Tuple[list, CharTokenizer]:
    """Create dataset from text matching C++ create_dataset."""
    if tokenizer_type == "char":
        tokenizer = CharTokenizer(text)
        tokens = tokenizer.encode(text)

        # Create sequences
        dataset = []
        for i in range(0, len(tokens) - sequence_length, sequence_length):
            seq = tokens[i : i + sequence_length]
            target = tokens[i + 1 : i + sequence_length + 1]
            if len(seq) == sequence_length and len(target) == sequence_length:
                dataset.append((seq, target))

        return dataset, tokenizer
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def create_mask(sequence_length: int, device) -> ttml.autograd.Tensor:
    """Create causal attention mask matching C++ implementation."""
    mask = np.zeros((sequence_length, sequence_length), dtype=np.float32)
    for i in range(sequence_length):
        for j in range(sequence_length):
            mask[i, j] = 1.0 if i >= j else 0.0

    # Reshape to (1, 1, seq_len, seq_len) to match C++ shape
    # Use TILE layout and BFLOAT16 to match C++ implementation
    mask = mask.reshape(1, 1, sequence_length, sequence_length)
    mask_tensor = ttml.autograd.Tensor.from_numpy(
        mask, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )
    return mask_tensor


def collate_fn(
    samples: list, batch_size: int, sequence_length: int, device
) -> Tuple[ttml.autograd.Tensor, ttml.autograd.Tensor, ttml.autograd.Tensor]:
    """Collate function matching C++ collate_fn."""
    # Flatten samples into data and targets
    data = []
    targets = []
    for seq, target in samples[:batch_size]:
        data.extend(seq)
        targets.extend(target)

    # Create tensors matching C++ shapes:
    # data: (batch_size, 1, 1, sequence_length)
    # targets: (batch_size, sequence_length)
    data_np = np.array(data, dtype=np.uint32).reshape(batch_size, 1, 1, sequence_length)
    targets_np = np.array(targets, dtype=np.uint32).reshape(batch_size, sequence_length)

    data_tensor = ttml.autograd.Tensor.from_numpy(
        data_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )
    targets_tensor = ttml.autograd.Tensor.from_numpy(
        targets_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )

    return data_tensor, targets_tensor


def get_loss_value(loss: ttml.autograd.Tensor) -> float:
    """Extract loss value from tensor.

    This function extracts the loss value AFTER backward() and reset_graph() have been called.
    The working nanogpt_training.py example follows this pattern - extracting loss after
    the computation graph has been reset ensures the tensor data is properly materialized.

    Args:
        loss: Loss tensor from cross_entropy_loss

    Returns:
        Loss value as float
    """
    loss_np = loss.to_numpy(ttnn.DataType.FLOAT32)

    # Handle scalar or multi-element tensor
    if loss_np.size == 1:
        return float(loss_np.item() if hasattr(loss_np, "item") else float(loss_np))
    elif loss_np.size > 1:
        return float(np.mean(loss_np))
    else:
        raise RuntimeError("Loss tensor is empty!")


def train_step(
    model: NanoGPT,
    optimizer: ttml.optimizers.OptimizerBase,
    scheduler_fn: Optional[callable],
    scheduler_step: int,
    input_tokens: ttml.autograd.Tensor,
    target_tokens: ttml.autograd.Tensor,
    mask: ttml.autograd.Tensor,
    gradient_accumulator: GradientAccumulator,
    use_clip_grad_norm: bool,
    clip_grad_norm_max_norm: float,
) -> tuple:
    """Single training step matching C++ implementation with proper gradient accumulation.

    Returns:
        Tuple of (loss_float, step_time_ms, should_step)
    """
    start_time = time.time()

    # Zero gradients only when accumulator says to (matching C++ should_zero_grad)
    if gradient_accumulator.should_zero_grad():
        optimizer.zero_grad()

    # Forward pass with causal mask (matching C++: run_model(model, features, masks))
    logits = model(input_tokens, mask)

    # Compute loss
    loss = ttml.ops.loss.cross_entropy_loss(
        logits, target_tokens, reduce=ttml.ops.ReduceType.MEAN
    )

    # Scale loss for gradient accumulation (matching C++: gradient_accumulator_helper.scale(loss))
    loss = gradient_accumulator.scale(loss)

    # Extract loss value BEFORE backward/reset
    loss_float = get_loss_value(loss)

    # Backward pass
    loss.backward(False)

    # Reset computation graph after backward (matching C++: ttml::autograd::ctx().reset_graph())
    ttml.autograd.AutoContext.get_instance().reset_graph()

    # Get number of samples for accumulator update
    samples = input_tokens.shape()[0]

    # Update accumulator (matching C++: gradient_accumulator_helper.update(loss_float, samples))
    gradient_accumulator.update(loss_float, samples)

    # Check if we should step the optimizer
    should_step = gradient_accumulator.should_step()

    if should_step:
        # Gradient clipping (matching C++: clip_grad_norm)
        if use_clip_grad_norm:
            # Use ttml.core.clip_grad_norm which works with model parameters directly
            ttml.core.clip_grad_norm(
                model.parameters(),
                clip_grad_norm_max_norm,
                2.0,  # p_norm_type (L2 norm)
                False,  # error_if_nonfinite - set False to avoid errors on NaN
            )

        # Optimizer step
        optimizer.step()

        # Apply learning rate scheduler if provided (matching C++: scheduler->step())
        if scheduler_fn is not None:
            scheduler_fn(scheduler_step)

    step_time = (time.time() - start_time) * 1000  # Convert to ms
    return loss_float, step_time, should_step


def parse_training_config(yaml_config: dict) -> TrainingConfig:
    """Parse training config from YAML matching C++ parse_config."""
    training_config = yaml_config.get("training_config", {})
    config = TrainingConfig()

    # Parse all fields with defaults matching C++ implementation
    config.project_name = training_config.get("project_name", config.project_name)
    config.seed = training_config.get("seed", config.seed)
    config.model_save_interval = training_config.get(
        "model_save_interval", config.model_save_interval
    )
    config.batch_size = training_config.get("batch_size", config.batch_size)
    config.num_epochs = training_config.get("num_epochs", config.num_epochs)
    config.max_steps = training_config.get("max_steps", config.max_steps)
    config.learning_rate = training_config.get("learning_rate", config.learning_rate)
    config.weight_decay = training_config.get("weight_decay", config.weight_decay)
    config.use_no_op = training_config.get("use_no_op", config.use_no_op)
    config.use_moreh_adamw = training_config.get(
        "use_moreh_adamw", config.use_moreh_adamw
    )
    config.use_kahan_summation = training_config.get(
        "use_kahan_summation", config.use_kahan_summation
    )
    config.gradient_accumulation_steps = training_config.get(
        "gradient_accumulation_steps", config.gradient_accumulation_steps
    )
    config.model_config = training_config.get("model_config", config.model_config)

    # Data path with default matching C++ (DATA_FOLDER + "/shakespeare.txt")
    default_data_path = "data/shakespeare.txt"
    config.data_path = training_config.get("data_path", default_data_path)

    # Scheduler type defaults to "identity" if not specified
    config.scheduler_type = training_config.get("scheduler_type", config.scheduler_type)

    # Tokenizer type defaults to "char" if not specified
    config.tokenizer_type = training_config.get("tokenizer_type", config.tokenizer_type)

    config.use_clip_grad_norm = training_config.get(
        "use_clip_grad_norm", config.use_clip_grad_norm
    )
    config.clip_grad_norm_max_norm = training_config.get(
        "clip_grad_norm_max_norm", config.clip_grad_norm_max_norm
    )

    return config


def parse_model_config(yaml_config: dict) -> ModelConfig:
    """Parse model config from YAML matching C++ parse_model_config."""
    # The YAML has a "transformer_config" top-level key
    transformer_config = yaml_config.get("transformer_config", {})
    config = ModelConfig()

    config.model_type = transformer_config.get("model_type", config.model_type)
    config.model_path = transformer_config.get("model_path", config.model_path)

    if config.model_type == "gpt2":
        # GPT2 config fields are directly under transformer_config
        config.vocab_size = transformer_config.get("vocab_size", config.vocab_size)
        config.block_size = transformer_config.get("block_size", config.block_size)
        config.n_embd = transformer_config.get("embedding_dim", config.n_embd)
        config.n_layer = transformer_config.get("num_blocks", config.n_layer)
        config.n_head = transformer_config.get("num_heads", config.n_head)
        config.dropout = transformer_config.get(
            "dropout_prob", transformer_config.get("dropout", config.dropout)
        )
        config.bias = transformer_config.get("bias", config.bias)
        config.max_sequence_length = transformer_config.get(
            "max_sequence_length", config.block_size
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    return config


def sample_greedy(
    model: NanoGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    sequence_length: int,
    mask: ttml.autograd.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> str:
    """Generate text from a prompt using sampling with temperature.

    Args:
        model: Trained NanoGPT model
        tokenizer: Character tokenizer
        prompt: Starting prompt text
        max_new_tokens: Maximum number of tokens to generate
        sequence_length: Sequence length for the model
        mask: Causal attention mask
        temperature: Sampling temperature (1.0 = normal, <1.0 = more deterministic, >1.0 = more random)
        top_k: If > 0, only sample from top k tokens

    Returns:
        Generated text
    """
    # Set model to eval mode (matching C++: model_to_eval - disables dropout)
    model.eval()

    # Encode prompt
    if len(prompt) == 0:
        prompt = " "  # Default to space if empty

    # Encode prompt to token IDs
    try:
        prompt_ids = tokenizer.encode(prompt)
    except Exception as e:
        raise ValueError(f"Failed to encode prompt '{prompt}': {e}")

    # Initialize running context with prompt
    # For simplicity, we'll use the prompt tokens and pad/truncate to sequence_length
    # If prompt is shorter than sequence_length, pad with the first token (usually space/newline)
    running = list(prompt_ids[:sequence_length])
    if len(running) < sequence_length:
        # Pad with the first token in vocabulary (usually a common character like space)
        # Get the first token ID from the tokenizer's vocabulary
        if tokenizer.stoi:
            # Get the token ID for space character, or first token if space not found
            space_token_id = tokenizer.stoi.get(" ", None)
            if space_token_id is None:
                space_token_id = list(tokenizer.stoi.values())[0]
            padding = [space_token_id] * (sequence_length - len(running))
        else:
            # Fallback: use 0 if tokenizer has no vocabulary
            padding = [0] * (sequence_length - len(running))
        running = padding + running

    generated_tokens = []

    print(f"\nGenerating text from prompt: '{prompt}'")
    print("=" * 70)

    for step in range(max_new_tokens):
        # Prepare input: take last sequence_length tokens
        inp = np.array(running[-sequence_length:], dtype=np.uint32).reshape(
            1, 1, 1, sequence_length
        )
        input_tensor = ttml.autograd.Tensor.from_numpy(
            inp, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )

        # Forward pass with causal mask (matching C++ model call)
        logits = model(input_tensor, mask)

        # Get logits for last position
        # Model returns shape [B, 1, seq_len, vocab_size] or [B, 1, 1, seq_len, vocab_size]
        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
        logits_shape = logits_np.shape

        # Handle different possible shapes
        if len(logits_shape) == 5:
            # [B, 1, 1, seq_len, vocab_size] -> [B, seq_len, vocab_size]
            last_logits = logits_np.reshape(
                logits_shape[0], logits_shape[3], logits_shape[4]
            )[:, -1, :]
        elif len(logits_shape) == 4:
            # [B, 1, seq_len, vocab_size] -> [B, seq_len, vocab_size]
            last_logits = logits_np.reshape(
                logits_shape[0], logits_shape[2], logits_shape[3]
            )[:, -1, :]
        else:
            # Fallback: assume last dimension is vocab_size
            last_logits = logits_np.reshape(-1, logits_np.shape[-1])[-1:]

        # Get vocabulary size (model may have rounded up, but tokenizer has actual size)
        vocab_size = tokenizer.get_vocab_size()

        # Truncate logits to valid vocabulary
        last_logits = last_logits[:, :vocab_size]

        # Handle NaN/Inf in logits BEFORE any operations
        # This can happen due to numerical precision issues in bfloat16
        if not np.all(np.isfinite(last_logits)):
            last_logits = np.nan_to_num(last_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # Apply temperature (with clipping to prevent overflow)
        if temperature != 1.0 and temperature > 0:
            # Clip logits to reasonable range before dividing
            last_logits = np.clip(last_logits, -100, 100)
            last_logits = last_logits / temperature

        # Apply top_k filtering
        if top_k > 0:
            # Get top k values and indices
            top_k_val = min(top_k, vocab_size)
            threshold = np.partition(last_logits, -top_k_val, axis=-1)[:, -top_k_val:][
                :, 0:1
            ]
            indices_to_remove = last_logits < threshold
            # Use large negative instead of -inf
            last_logits[indices_to_remove] = -1e9

        # Convert to probabilities using softmax (numerically stable)
        max_logits = np.max(last_logits, axis=-1, keepdims=True)
        logits_exp = np.exp(last_logits - max_logits)
        probs = logits_exp / (np.sum(logits_exp, axis=-1, keepdims=True) + 1e-10)

        # Ensure probs sum to 1 and are valid
        probs = np.clip(probs, 0, 1)
        probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-10)

        # Sample from distribution (or use argmax if temperature is very low)
        if temperature < 0.01:
            next_id = int(np.argmax(probs, axis=-1)[0])
        else:
            # Sample from categorical distribution
            try:
                next_id = int(np.random.choice(vocab_size, p=probs[0]))
            except ValueError:
                # Fallback to argmax if probs are invalid
                next_id = int(np.argmax(last_logits, axis=-1)[0])

        # Append to running context
        running.append(next_id)
        generated_tokens.append(next_id)

        # Reset graph for next iteration
        ttml.autograd.AutoContext.get_instance().reset_graph()

        # Print progress every 50 tokens
        if (step + 1) % 50 == 0:
            current_text = tokenizer.decode(generated_tokens)
            print(f"[{step + 1}/{max_new_tokens}] {current_text[-100:]}...")

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)

    print("\n" + "=" * 70)
    print("Generated text:")
    print("=" * 70)
    print(generated_text)
    print("=" * 70)

    return generated_text


def save_checkpoint(
    checkpoint_path: str,
    step: int,
    model: NanoGPT,
    tokenizer: CharTokenizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> str:
    """Save model checkpoint to pickle file.

    Args:
        checkpoint_path: Path to save checkpoint (will add .pkl if not present)
        step: Training step number
        model: NanoGPT model to save
        tokenizer: Tokenizer to save
        model_config: Model configuration
        training_config: Training configuration

    Returns:
        Path to saved checkpoint file
    """
    # Ensure .pkl extension
    if not checkpoint_path.endswith(".pkl"):
        checkpoint_path = f"{checkpoint_path}.pkl"

    # Save model parameters
    model_state = {}
    for name, param in model.parameters().items():
        # Handle both Parameter objects and direct Tensor objects
        if isinstance(param, Parameter):
            tensor = param.tensor
        else:
            tensor = param

        # Get tensor metadata
        layout = tensor.get_value().get_layout()

        # Convert tensor to numpy for serialization
        numpy_array = tensor.to_numpy(ttnn.DataType.FLOAT32)
        model_state[name] = {
            "data": numpy_array,
            "layout": layout.value if hasattr(layout, "value") else str(layout),
            "shape": numpy_array.shape,
        }

    checkpoint = {
        "step": step,
        "model_state": model_state,
        "tokenizer": tokenizer,
        "model_config": model_config,
        "training_config": training_config,
    }

    # Save checkpoint
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"  Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_model_from_checkpoint(
    checkpoint_path: str,
) -> Tuple[NanoGPT, CharTokenizer, ModelConfig, TrainingConfig, int]:
    """Load model from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file (.pkl)

    Returns:
        Tuple of (model, tokenizer, model_config, training_config, step)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Extract components
    model_state = checkpoint["model_state"]
    tokenizer = checkpoint["tokenizer"]
    model_config = checkpoint["model_config"]
    training_config = checkpoint.get("training_config", None)
    step = checkpoint.get("step", 0)

    # Create model config
    nanogpt_config = NanoGPTConfig(
        vocab_size=model_config.vocab_size,
        block_size=model_config.block_size,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        dropout=model_config.dropout,
        bias=model_config.bias,
    )

    # Create model
    model = create_nanogpt(nanogpt_config)

    # Load model parameters
    print("  Loading model parameters...")
    model_params = model.parameters()

    for name, param_data in model_state.items():
        if name not in model_params:
            print(f"    Warning: Parameter {name} not found in model, skipping")
            continue

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
        if numpy_array.dtype != np.dtype("float32"):
            numpy_array = numpy_array.astype(np.float32)
        numpy_bfloat16 = numpy_array.astype(ml_dtypes.bfloat16)
        restored_tensor = ttml.autograd.Tensor.from_numpy(
            numpy_bfloat16, layout=layout, new_type=ttnn.DataType.BFLOAT16
        )

        # Update the parameter in the model
        # Parameter names use "/" separator and include model name prefix (e.g., "NanoGPT/block_0/attention/qkv")
        parts = name.split("/")

        # Skip the first part if it's the model name
        if len(parts) > 1 and parts[0] == model.get_name():
            parts = parts[1:]

        # Navigate to the correct module and update the parameter
        if len(parts) == 1:
            # Direct parameter of the model (e.g., "lm_head_weight")
            param_name = parts[0]
            if hasattr(model, param_name):
                param = getattr(model, param_name)
                if isinstance(param, Parameter):
                    # Update the tensor inside the existing Parameter (preserves weight tying)
                    param.tensor = restored_tensor
                else:
                    setattr(model, param_name, Parameter(restored_tensor))
            else:
                setattr(model, param_name, Parameter(restored_tensor))
        else:
            # Nested parameter (e.g., "block_0/attention/qkv" -> ["block_0", "attention", "qkv"])
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            param_name = parts[-1]
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                if isinstance(param, Parameter):
                    # Update the tensor inside the existing Parameter
                    param.tensor = restored_tensor
                else:
                    setattr(module, param_name, Parameter(restored_tensor))
            else:
                setattr(module, param_name, Parameter(restored_tensor))

        print(f"    Loaded: {name} (shape: {numpy_array.shape})")

    print(f"  Checkpoint loaded from step {step}")

    return model, tokenizer, model_config, training_config, step


def main():
    """Main training function matching C++ main."""
    parser = argparse.ArgumentParser(description="NanoGPT Full C++ Example (Python)")

    # Default config path matching C++ example
    default_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs",
        "training_configs",
        "training_shakespeare_nanogpt.yaml",
    )
    # Fallback to relative path if absolute doesn't exist
    if not os.path.exists(default_config_path):
        default_config_path = (
            "configs/training_configs/training_shakespeare_nanogpt.yaml"
        )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=default_config_path,
        help=f"Path to training config YAML file (default: {default_config_path})",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Path to training data (text file) - overrides config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size - overrides config",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps - overrides config",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs - overrides config",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate - overrides config",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=None,
        help="Enable gradient clipping with specified max norm (e.g., 1.0)",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=None,
        help="Sequence length - overrides config",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="",
        help="Path to save model checkpoints",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt for text generation (if provided, runs inference instead of training)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Maximum number of tokens to generate (default: 300)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8, lower=more deterministic)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40, 0=disabled)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to load model for inference (required if --prompt is provided)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NanoGPT Full C++ Example (Python Implementation)")
    print("=" * 70)
    print()

    # Set TT_METAL_RUNTIME_ROOT if not set and TT_METAL_HOME is available
    # This is needed for the runtime to find kernel files like moreh_mean
    if "TT_METAL_RUNTIME_ROOT" not in os.environ:
        tt_metal_home = os.environ.get("TT_METAL_HOME", "")
        if tt_metal_home and os.path.exists(tt_metal_home):
            os.environ["TT_METAL_RUNTIME_ROOT"] = tt_metal_home
            print(f"Set TT_METAL_RUNTIME_ROOT={tt_metal_home} (from TT_METAL_HOME)")
        else:
            # Try to auto-detect from current directory
            current_dir = os.getcwd()
            # Check if we're in the repo root (has tt_metal/ subdirectory)
            if os.path.exists(os.path.join(current_dir, "tt_metal")):
                os.environ["TT_METAL_RUNTIME_ROOT"] = current_dir
                print(
                    f"Set TT_METAL_RUNTIME_ROOT={current_dir} (auto-detected from current directory)"
                )
            else:
                # Try parent directories
                parent_dir = os.path.dirname(current_dir)
                if os.path.exists(os.path.join(parent_dir, "tt_metal")):
                    os.environ["TT_METAL_RUNTIME_ROOT"] = parent_dir
                    print(
                        f"Set TT_METAL_RUNTIME_ROOT={parent_dir} (auto-detected from parent directory)"
                    )
                else:
                    print(
                        "Warning: TT_METAL_RUNTIME_ROOT not set and could not be auto-detected."
                    )
                    print(
                        "  Kernel files may not be found. Set TT_METAL_RUNTIME_ROOT environment variable"
                    )
                    print("  to point to the tt-metal repository root directory.")
    else:
        print(f"Using TT_METAL_RUNTIME_ROOT={os.environ.get('TT_METAL_RUNTIME_ROOT')}")
    print()

    # Load configs matching C++ structure
    if args.config and os.path.exists(args.config):
        print(f"Loading training config from: {args.config}")
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        training_config = parse_training_config(yaml_config)

        # Load model config from separate file (matching C++ behavior)
        if training_config.model_config:
            model_config_path = training_config.model_config
            # Handle relative paths
            if not os.path.isabs(model_config_path):
                # Try relative to config file directory
                config_dir = os.path.dirname(os.path.abspath(args.config))
                model_config_path = os.path.join(config_dir, "..", model_config_path)
                model_config_path = os.path.normpath(model_config_path)
                # If still doesn't exist, try relative to current directory
                if not os.path.exists(model_config_path):
                    model_config_path = training_config.model_config

            if os.path.exists(model_config_path):
                print(f"Loading model config from: {model_config_path}")
                with open(model_config_path, "r") as f:
                    model_yaml = yaml.safe_load(f)
                model_config = parse_model_config(model_yaml)
            else:
                print(f"Warning: Model config file not found: {model_config_path}")
                print("Using default model config")
                model_config = ModelConfig()
        else:
            print("Warning: No model_config specified in training config")
            print("Using default model config")
            model_config = ModelConfig()
    else:
        if args.config:
            print(f"Warning: Config file not found: {args.config}")
        print("Using default configs")
        training_config = TrainingConfig()
        model_config = ModelConfig()

    # Override with command line args (only if provided)
    if args.data_path:
        training_config.data_path = args.data_path
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.max_steps is not None:
        training_config.max_steps = args.max_steps
    if args.num_epochs is not None:
        training_config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    if args.clip_grad_norm is not None:
        training_config.use_clip_grad_norm = True
        training_config.clip_grad_norm_max_norm = args.clip_grad_norm
    if args.sequence_length is not None:
        model_config.max_sequence_length = args.sequence_length
        model_config.block_size = args.sequence_length
    # Set checkpoint save path (separate from model_config YAML path)
    # If --model_save_path is provided, use it; otherwise use a default based on project name
    if args.model_save_path:
        checkpoint_save_path = args.model_save_path
    elif training_config.model_config:
        # Use model_config path as base (but this is the YAML path, so we'll extract a base name)
        # For now, use a default checkpoint directory
        checkpoint_save_path = f"checkpoints/{training_config.project_name}"
    else:
        checkpoint_save_path = f"checkpoints/{training_config.project_name}"

    # Create checkpoint directory if it doesn't exist
    if checkpoint_save_path:
        checkpoint_dir = (
            os.path.dirname(checkpoint_save_path)
            if os.path.dirname(checkpoint_save_path)
            else "checkpoints"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_save_path}_step_*.pkl")

    # Check if we're in inference-only mode (prompt + model_path)
    inference_only = args.prompt and (args.model_path or checkpoint_save_path)

    # Initialize device early (needed for both training and inference)
    ttml.autograd.AutoContext.get_instance().open_device()

    # Handle inference-only mode: load model from checkpoint
    if inference_only:
        model_path = args.model_path if args.model_path else checkpoint_save_path
        print("1. Loading model from checkpoint...")
        print(f"   - Model path: {model_path}")

        try:
            (
                model,
                tokenizer,
                model_config,
                training_config,
                loaded_step,
            ) = load_model_from_checkpoint(
                model_path,
            )
            sequence_length = model_config.max_sequence_length
            dataset = []  # Not needed for inference

            print(f"   - Model loaded from step {loaded_step}")
            print(f"   - Vocabulary size: {model_config.vocab_size}")
            print(f"   - Sequence length: {sequence_length}")
            print(
                f"   - Model: {model_config.n_layer} layers, {model_config.n_embd} embd, {model_config.n_head} heads"
            )

        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            ttml.autograd.AutoContext.get_instance().close_device()
            return
    else:
        # Training mode: load data and create model
        # Set default data path if not provided (matching C++ behavior)
        if not training_config.data_path:
            # Try to find Shakespeare dataset
            possible_paths = [
                "data/shakespeare.txt",
                "tt-train/data/shakespeare.txt",
                "../data/shakespeare.txt",
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    ),
                    "data",
                    "shakespeare.txt",
                ),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    training_config.data_path = path
                    break
            if not training_config.data_path:
                print(
                    "Warning: No data path specified and Shakespeare dataset not found."
                )
                print("Please specify --data_path or place shakespeare.txt in data/")
                print(f"  Searched paths: {possible_paths}")
                ttml.autograd.AutoContext.get_instance().close_device()
                return

        print("1. Loading and preparing data...")
        print(f"   - Data path: {training_config.data_path}")
        print(f"   - Tokenizer: {training_config.tokenizer_type}")

        # Load data
        text = read_file_to_str(training_config.data_path)
        sequence_length = model_config.max_sequence_length

        # Create dataset
        dataset, tokenizer = create_dataset_from_text(
            text, sequence_length, training_config.tokenizer_type
        )
        model_config.vocab_size = tokenizer.get_vocab_size()

        print(f"   - Vocabulary size: {model_config.vocab_size}")
        print(f"   - Dataset size: {len(dataset)} samples")
        print(f"   - Sequence length: {sequence_length}")

        print("\n2. Creating model...")
        # Round vocab size to tile boundary (matching C++ behavior)
        model_config.vocab_size = round_up_to_tile(model_config.vocab_size, 32)

        # Create model config
        nanogpt_config = NanoGPTConfig(
            vocab_size=model_config.vocab_size,
            block_size=model_config.block_size,
            n_embd=model_config.n_embd,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            dropout=model_config.dropout,
            bias=model_config.bias,
        )

        # Create model
        model = create_nanogpt(nanogpt_config)

        # Count parameters
        total_params = sum(
            p.tensor.to_numpy(ttnn.DataType.FLOAT32).size
            for p in model.parameters().values()
            if isinstance(p, Parameter) and hasattr(p, "tensor")
        )
        print(
            f"   - Model: {model_config.n_layer} layers, {model_config.n_embd} embd, {model_config.n_head} heads"
        )
        print(f"   - Total parameters: {total_params:,}")

    # Check if we're in inference mode
    if args.prompt:
        # Inference mode: skip optimizer setup
        optimizer = None
        scheduler_fn = None
        print("\n3. Inference mode - skipping optimizer setup")
    else:
        print("\n3. Setting up optimizer...")
        # Set seed
        ttml.autograd.AutoContext.get_instance().set_seed(training_config.seed)

        # Create optimizer config
        if training_config.use_no_op:
            print("   WARNING: Using NoOp optimizer - parameters will NOT be updated.")
            optimizer = None  # NoOp not available in Python API yet
        else:
            # AdamWConfig.make() requires: lr, beta1, beta2, epsilon, weight_decay
            # Default values matching C++ implementation
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8

            adamw_config = ttml.optimizers.AdamWConfig.make(
                training_config.learning_rate,  # lr
                beta1,  # beta1
                beta2,  # beta2
                epsilon,  # epsilon
                training_config.weight_decay,  # weight_decay
            )

            # Note: use_kahan_summation is not exposed in Python API yet
            # It's a property of AdamWConfig but can't be set via make()
            # For now, we'll skip it (defaults to False)

            # Get model parameters
            parameters = model.parameters()

            if training_config.use_moreh_adamw:
                optimizer = ttml.optimizers.MorehAdamW(parameters, adamw_config)
                print("   - Optimizer: MorehAdamW")
            else:
                optimizer = ttml.optimizers.AdamW(parameters, adamw_config)
                print("   - Optimizer: AdamW")

            print(f"   - Learning rate: {training_config.learning_rate}")
            print(f"   - Weight decay: {training_config.weight_decay}")
            print(f"   - Beta1: {beta1}, Beta2: {beta2}, Epsilon: {epsilon}")
            if training_config.use_kahan_summation:
                print(
                    "   - Note: Kahan summation requested but not available in Python API"
                )

        print("\n4. Setting up learning rate scheduler...")
        scheduler_fn = None
        if training_config.scheduler_type == "warmup_linear":
            scheduler_fn, warmup_steps, decay_steps = create_warmup_linear_scheduler(
                optimizer, training_config.max_steps
            )
            print(f"   - Scheduler: warmup_linear")
            print(f"   - Warmup steps: {warmup_steps}")
            print(f"   - Decay steps: {decay_steps}")
        else:
            print(f"   - Scheduler: identity (constant LR)")

    # Create attention mask (needed for both training and inference)
    if inference_only:
        print("\n2. Creating attention mask...")
    else:
        print("\n5. Creating attention mask...")
    mask = create_mask(
        sequence_length, ttml.autograd.AutoContext.get_instance().get_device()
    )

    # Training or inference mode
    if args.prompt:
        # Inference mode: skip training, go straight to inference
        print("\n6. Inference mode - skipping training")
    else:
        print("\n6. Training...")
        print()
        print(
            f"Starting training for {training_config.max_steps} steps (starting from step 0)..."
        )
        print(f"  - Batch size: {training_config.batch_size}")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Training data: {len(dataset)} samples")
        print(f"  - Scheduler: {training_config.scheduler_type}")
        print(
            f"  - Gradient accumulation steps: {training_config.gradient_accumulation_steps}"
        )
        print(f"  - Dropout: {model_config.dropout}")
        if training_config.use_clip_grad_norm:
            print(
                f"  - Gradient clipping: max_norm={training_config.clip_grad_norm_max_norm}"
            )
        print()

        # Set model to training mode (matching C++: model_to_train)
        model.train()

        # Training setup
        loss_meter = LossAverageMeter()
        gradient_accumulator = GradientAccumulator(
            training_config.gradient_accumulation_steps
        )
        global_step = 0

        # Training loop (matching C++ structure)
        start_time = time.time()
        for epoch in range(training_config.num_epochs):
            # Shuffle dataset
            np.random.shuffle(dataset)

            # Create batches
            for batch_start in range(0, len(dataset), training_config.batch_size):
                batch_samples = dataset[
                    batch_start : batch_start + training_config.batch_size
                ]
                if len(batch_samples) < training_config.batch_size:
                    continue  # Skip incomplete batches

                # Collate batch (matching C++ collate_fn)
                input_tokens, target_tokens = collate_fn(
                    batch_samples,
                    training_config.batch_size,
                    sequence_length,
                    ttml.autograd.AutoContext.get_instance().get_device(),
                )

                # Training step with mask (matching C++: run_model(model, features, masks))
                loss_float, step_time, should_step = train_step(
                    model,
                    optimizer,
                    scheduler_fn,
                    global_step,
                    input_tokens,
                    target_tokens,
                    mask,  # Pass the causal mask to model
                    gradient_accumulator,
                    training_config.use_clip_grad_norm,
                    training_config.clip_grad_norm_max_norm,
                )

                # Only update counters and print when we've stepped (matching C++: if should_step())
                if should_step:
                    avg_loss = gradient_accumulator.average_loss()
                    loss_meter.update(avg_loss)
                    print(
                        f"Step: {global_step}, Loss: {avg_loss:.6f}, Time: {step_time:.2f} ms"
                    )

                    # Save checkpoint if needed (matching C++: model_save_interval)
                    if (
                        checkpoint_save_path
                        and global_step % training_config.model_save_interval == 0
                    ):
                        checkpoint_path = (
                            f"{checkpoint_save_path}_step_{global_step}.pkl"
                        )
                        save_checkpoint(
                            checkpoint_path,
                            global_step,
                            model,
                            tokenizer,
                            model_config,
                            training_config,
                        )

                    # Check if max steps reached (matching C++: if (global_step >= training_config.max_steps))
                    if global_step >= training_config.max_steps:
                        break

                    # Reset gradient accumulator after stepping (matching C++: gradient_accumulator_helper.reset())
                    gradient_accumulator.reset()

                    # Increment global step AFTER all processing
                    global_step += 1

            if global_step >= training_config.max_steps:
                break

        # Save final checkpoint after training
        if checkpoint_save_path:
            final_checkpoint_path = f"{checkpoint_save_path}_final.pkl"
            save_checkpoint(
                final_checkpoint_path,
                global_step,
                model,
                tokenizer,
                model_config,
                training_config,
            )

        # Final summary
        total_time = time.time() - start_time
        print()
        print("=" * 70)
        print(f"Training completed!")
        print(f"  - Total steps: {global_step}")
        print(f"  - Total time: {total_time:.2f} s")
        print(f"  - Average loss: {loss_meter.average():.6f}")
        print("=" * 70)

    # Handle inference mode if prompt is provided
    if args.prompt:
        print("\n" + "=" * 70)
        print("Running Inference Mode")
        print("=" * 70)
        print(f"  Prompt: '{args.prompt}'")
        print(f"  Max new tokens: {args.max_new_tokens}")
        print()

        # Generate text (model and tokenizer already loaded above)
        sample_greedy(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            sequence_length,
            mask,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # Close device
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
