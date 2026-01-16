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
- Character tokenizer (via ttml.common.data.CharTokenizer)
- Proper tensor shapes matching C++ implementation
"""

import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import pickle

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.models.nanogpt import NanoGPT, NanoGPTConfig, create_nanogpt
from ttml.modules import Parameter
from ttml.common.utils import round_up_to_tile, get_tt_metal_home
from ttml.common.config import load_config, TrainingConfig as BaseTrainingConfig
from ttml.common.data import CharTokenizer, build_causal_mask

_autograd_ctx = None


def get_autograd_ctx():
    """Get cached AutoContext singleton instance."""
    global _autograd_ctx
    if _autograd_ctx is None:
        _autograd_ctx = ttml.autograd.AutoContext.get_instance()
    return _autograd_ctx


class TrainingConfig(BaseTrainingConfig):
    """Extended training config with NanoGPT-specific fields.

    Inherits from ttml.common.config.TrainingConfig and adds fields needed
    for the full NanoGPT training example.
    """

    def __init__(self, yaml_config=None):
        """Initialize training config, optionally from YAML.

        Args:
            yaml_config: Dictionary or path to YAML config. If None, uses defaults.
        """
        # Initialize base class (requires yaml_config, so pass empty dict for defaults)
        super().__init__(yaml_config if yaml_config is not None else {})

        # Get training_config section for additional fields
        tc = {}
        if isinstance(yaml_config, dict):
            tc = yaml_config.get("training_config", {})

        # Extended fields not in base TrainingConfig
        self.project_name = tc.get("project_name", "tt_train_nano_gpt")
        self.data_path = tc.get("data_path", "")
        self.scheduler_type = tc.get("scheduler_type", "identity")
        self.use_no_op = tc.get("use_no_op", False)
        self.use_moreh_adamw = tc.get("use_moreh_adamw", False)
        self.use_kahan_summation = tc.get("use_kahan_summation", False)
        self.use_clip_grad_norm = tc.get("use_clip_grad_norm", False)
        self.clip_grad_norm_max_norm = float(tc.get("clip_grad_norm_max_norm", 1.0))

        # Aliases to match expected field names in this example
        self.max_steps = self.steps
        self.num_epochs = self.epochs
        self.model_save_interval = self.save_every
        self.learning_rate = self.lr


@dataclass
class ModelConfig:
    """Model configuration aligned with ttml.common.config.TransformerConfig naming."""

    model_type: str = "gpt2"  # "gpt2" or "llama"
    model_path: str = ""
    vocab_size: int = 50304
    embedding_dim: int = 384  # NanoGPT default (reduced from 768)
    num_blocks: int = 6  # NanoGPT default (reduced from 12)
    num_heads: int = 6  # NanoGPT default (reduced from 12)
    dropout_prob: float = 0.2  # Match C++ default: float dropout_prob = 0.2F
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


def read_file_to_str(file_path: str) -> str:
    """Read file to string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


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
    text: str,
    sequence_length: int,
) -> Tuple[list, CharTokenizer]:
    """Create dataset from text using CharTokenizer from ttml.common.data.

    Args:
        text: Text corpus to create dataset from.
        sequence_length: Length of each sequence.

    Returns:
        Tuple of (dataset, tokenizer) where dataset is list of (seq, target) tuples.
    """
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


def collate_fn(
    samples: list, batch_size: int, sequence_length: int
) -> Tuple[ttml.autograd.Tensor, ttml.autograd.Tensor]:
    """Collate function matching C++ collate_fn.

    Args:
        samples: List of (sequence, target) tuples
        batch_size: Batch size
        sequence_length: Sequence length
    """
    # Flatten samples into data and targets
    data = []
    targets = []
    for seq, target in samples[:batch_size]:
        data.extend(seq)
        targets.extend(target)

    # Create NumPy arrays directly with the correct final shape
    # This avoids device reshape operations which add overhead
    data_np = np.array(data, dtype=np.uint32).reshape(batch_size, 1, 1, sequence_length)
    targets_np = np.array(targets, dtype=np.uint32).reshape(batch_size, sequence_length)

    # Create tensors directly from NumPy with correct shape (single host-to-device transfer)
    data_tensor = ttml.autograd.Tensor.from_numpy(
        data_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )
    targets_tensor = ttml.autograd.Tensor.from_numpy(
        targets_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )

    return data_tensor, targets_tensor


def get_loss_value(loss: ttml.autograd.Tensor) -> float:
    """Extract loss value from tensor without using NumPy.

    Uses ttnn.Tensor.item() which directly extracts scalar via to_vector<T>() without NumPy conversion.

    Args:
        loss: Loss tensor from cross_entropy_loss (should already be reduced to scalar)

    Returns:
        Loss value as float
    """
    # Extract scalar value directly using ttnn.Tensor.item() - avoids NumPy conversion
    # This uses to_vector<T>() internally which is more efficient than to_numpy()
    return float(loss.get_value().item())


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
    batch_size=None,
) -> tuple:
    """Single training step matching C++ implementation with proper gradient accumulation.

    Args:
        batch_size: Optional cached batch size (if None, will extract from input_tokens)

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

    loss_float = get_loss_value(loss)

    # Backward pass
    loss.backward(False)

    # Reset computation graph after backward (matching C++: ttml::autograd::ctx().reset_graph())
    get_autograd_ctx().reset_graph()

    # Get number of samples for accumulator update
    # Use cached batch_size if provided to avoid shape() call
    samples = batch_size if batch_size is not None else input_tokens.shape()[0]

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
        config.embedding_dim = transformer_config.get(
            "embedding_dim", config.embedding_dim
        )
        config.num_blocks = transformer_config.get("num_blocks", config.num_blocks)
        config.num_heads = transformer_config.get("num_heads", config.num_heads)
        config.dropout_prob = transformer_config.get(
            "dropout_prob", config.dropout_prob
        )
        config.bias = transformer_config.get("bias", config.bias)
        config.max_sequence_length = transformer_config.get(
            "max_sequence_length", config.max_sequence_length
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

    # Reset graph before inference to ensure clean state
    get_autograd_ctx().reset_graph()

    # Cache device to avoid repeated lookups
    device = get_autograd_ctx().get_device()

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

    # Create initial input tensor on device once, then update in-place
    # This avoids CPU->Device transfer every iteration
    inp_list = running[-sequence_length:]
    input_ttnn = ttnn.from_buffer(
        buffer=inp_list,
        shape=[1, 1, 1, sequence_length],
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    for step in range(max_new_tokens):
        # Wrap current input tensor for model (no data transfer)
        input_tensor = ttml.autograd.Tensor(input_ttnn, False)

        # Forward pass with causal mask (matching C++ model call)
        # Clone mask before each use to avoid TTNN memory reuse corrupting the original
        mask_for_model = ttml.autograd.Tensor(ttnn.clone(mask.get_value()), False)

        logits = model(input_tensor, mask_for_model)

        # Get logits for last position using ttml/ttnn operations
        # Model returns shape [B, 1, seq_len, vocab_size] or [B, 1, 1, seq_len, vocab_size]
        logits_shape = logits.shape()

        # Extract last position logits using ttnn operations (no autograd needed for inference)
        # Handle different possible shapes
        if len(logits_shape) == 5:
            # [B, 1, 1, seq_len, vocab_size] -> extract last position: [B, 1, 1, 1, vocab_size]
            seq_len = logits_shape[3]
            last_pos = seq_len - 1
            sliced_tensor = ttnn.slice(
                logits.get_value(),
                [0, 0, 0, last_pos, 0],
                [
                    logits_shape[0],
                    logits_shape[1],
                    logits_shape[2],
                    seq_len,
                    logits_shape[4],
                ],
            )
            # Reshape to [B, 1, 1, vocab_size] using ttnn (no autograd needed for inference)
            reshaped = ttnn.reshape(
                sliced_tensor, [logits_shape[0], 1, 1, logits_shape[4]]
            )
            last_logits = ttml.autograd.Tensor(reshaped, False)
        elif len(logits_shape) == 4:
            # [B, 1, seq_len, vocab_size] -> extract last position: [B, 1, 1, vocab_size]
            seq_len = logits_shape[2]
            last_pos = seq_len - 1
            sliced_tensor = ttnn.slice(
                logits.get_value(),
                [0, 0, last_pos, 0],
                [logits_shape[0], logits_shape[1], seq_len, logits_shape[3]],
            )
            # Reshape to [B, 1, 1, vocab_size] using ttnn (no autograd needed for inference)
            reshaped = ttnn.reshape(
                sliced_tensor, [logits_shape[0], 1, 1, logits_shape[3]]
            )
            last_logits = ttml.autograd.Tensor(reshaped, False)
        else:
            # Fallback: use reshape and take last element
            # This case should be rare
            reshaped = ttnn.reshape(logits.get_value(), [-1, logits_shape[-1]])
            reshaped_shape = reshaped.shape
            if reshaped_shape[0] > 1:
                sliced_tensor = ttnn.slice(
                    reshaped,
                    [reshaped_shape[0] - 1, 0],
                    [reshaped_shape[0], reshaped_shape[1]],
                )
                reshaped = ttnn.reshape(sliced_tensor, [1, 1, 1, reshaped_shape[1]])
            last_logits = ttml.autograd.Tensor(reshaped, False)

        # Get vocabulary size (model may have rounded up, but tokenizer has actual size)
        vocab_size = tokenizer.vocab_size

        # Truncate logits to valid vocabulary if needed
        # Note: If vocab_size matches the last dimension, no truncation needed
        # Otherwise, we'd need to slice, but for now we'll let the sampling handle it

        # Sample using ttml operations
        # For greedy sampling (very low temperature), use argmax directly
        if temperature < 0.01:
            # Use ttnn.argmax for greedy sampling
            argmax_result = ttnn.argmax(last_logits.get_value(), dim=3, keepdim=True)
            # Extract scalar value directly from ttnn.Tensor - avoids unnecessary wrapper
            next_id = int(argmax_result.item())
            # Clamp to valid vocabulary
            next_id = min(next_id, vocab_size - 1)
        else:
            # Use ttml.ops.sample.sample_op() for temperature-based sampling
            # Note: top_k filtering is not yet supported in ttml.ops.sample.sample_op,
            # so we'll skip it for now (can be added later if needed)
            seed = random.randint(0, 2**32 - 1)

            # If top_k is requested, apply on-device top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, vocab_size)
                if top_k_val < vocab_size:
                    # Get top-k values on device (keeps everything on-device)
                    last_logits_ttnn = last_logits.get_value()
                    topk_values, topk_indices = ttnn.topk(
                        last_logits_ttnn,
                        k=top_k_val,
                        dim=-1,  # Last dimension (vocab_size)
                        largest=True,
                        sorted=True,
                    )

                    # Extract threshold (k-th largest = last element of topk_values)
                    # topk_values shape: [1, 1, 1, top_k_val]
                    # Get the last element which is the smallest of top-k (our threshold)
                    threshold_tensor = ttnn.slice(
                        topk_values, [0, 0, 0, top_k_val - 1], [1, 1, 1, top_k_val]
                    )
                    # threshold_tensor shape: [1, 1, 1, 1]
                    # Use threshold_tensor directly - ttnn.lt() will automatically broadcast
                    # This avoids extracting scalar and recreating tensor with full_like

                    # Create mask: values below threshold should be masked
                    # Broadcasting happens automatically: [1,1,1,1] vs [1,1,1,vocab_size]
                    topk_mask = ttnn.lt(last_logits_ttnn, threshold_tensor)

                    # Apply mask: set values below threshold to -1e9
                    filter_value_tensor = ttnn.full_like(
                        last_logits_ttnn, -1e9, dtype=ttnn.bfloat16
                    )
                    filtered_logits_ttnn = ttnn.where(
                        topk_mask, filter_value_tensor, last_logits_ttnn
                    )

                    # Cleanup intermediate tensors
                    ttnn.deallocate(topk_values)
                    ttnn.deallocate(topk_indices)
                    ttnn.deallocate(threshold_tensor)
                    ttnn.deallocate(topk_mask)
                    ttnn.deallocate(filter_value_tensor)

                    # Convert back to ttml.autograd.Tensor
                    last_logits = ttml.autograd.Tensor(filtered_logits_ttnn, False)

            # Use ttml sampling operation
            sampled_tensor = ttml.ops.sample.sample_op(
                last_logits, temperature, seed, None  # logits_padding_mask
            )

            # Extract the sampled token ID directly using .item() - avoids NumPy conversion
            next_id = int(sampled_tensor.get_value().item())
            # Clamp to valid vocabulary
            next_id = min(next_id, vocab_size - 1)

        # Append to running context (for final decode)
        running.append(next_id)
        generated_tokens.append(next_id)

        # Update input tensor on device: shift left and append new token
        # Roll tensor left by 1 position: [t0, t1, t2, ...] -> [t1, t2, ..., t0]
        # Then overwrite last position with new token
        # This keeps everything on device, avoiding CPU->Device transfer

        # Shift left: slice [1:] and concat with placeholder, then overwrite last
        # More efficient: use ttnn.roll if available, otherwise slice and concat
        shifted = ttnn.slice(input_ttnn, [0, 0, 0, 1], [1, 1, 1, sequence_length])
        # Create single-element tensor with new token
        new_token_tensor = ttnn.from_buffer(
            buffer=[next_id],
            shape=[1, 1, 1, 1],
            dtype=ttnn.uint32,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # Concatenate: shifted [1:seq_len-1] + new_token -> full sequence
        input_ttnn = ttnn.concat([shifted, new_token_tensor], dim=3)

        # Cleanup intermediate tensors
        ttnn.deallocate(shifted)
        ttnn.deallocate(new_token_tensor)

        # Reset graph for next iteration
        get_autograd_ctx().reset_graph()

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


def find_latest_checkpoint(base_path: str) -> Optional[str]:
    """Find the latest checkpoint file matching the base path pattern.

    Searches for files matching {base_path}_step_*.pkl and {base_path}_final.pkl,
    returning the one with the highest step number.

    Args:
        base_path: Base path for checkpoints (e.g., "checkpoints/nano_gpt")

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    import glob
    import re

    # Look for step checkpoints and final checkpoint
    pattern = f"{base_path}_step_*.pkl"
    step_files = glob.glob(pattern)

    final_path = f"{base_path}_final.pkl"
    if os.path.exists(final_path):
        step_files.append(final_path)

    if not step_files:
        return None

    # Extract step numbers and find the maximum
    def get_step(path: str) -> int:
        if path.endswith("_final.pkl"):
            return float("inf")  # Final checkpoint is always "latest"
        match = re.search(r"_step_(\d+)\.pkl$", path)
        return int(match.group(1)) if match else -1

    latest = max(step_files, key=get_step)
    return latest


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

    # Create model config (map aligned names to NanoGPTConfig fields)
    nanogpt_config = NanoGPTConfig(
        vocab_size=model_config.vocab_size,
        block_size=model_config.max_sequence_length,
        n_embd=model_config.embedding_dim,
        n_layer=model_config.num_blocks,
        n_head=model_config.num_heads,
        dropout=model_config.dropout_prob,
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

        # Update the parameter using assign() - works with both C++ and Python modules
        # The model_params dict contains references to the actual parameter tensors
        model_params[name].assign(restored_tensor)
    print(f"  Checkpoint loaded from step {step}")

    return model, tokenizer, model_config, training_config, step


def main():
    """Main training function matching C++ main."""
    parser = argparse.ArgumentParser(description="NanoGPT Full C++ Example (Python)")

    # Default config path matching C++ example (relative to configs root)
    default_config_path = "training_shakespeare_nanogpt.yaml"

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
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume from (auto-detects latest if not specified)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh training, ignoring any existing checkpoints",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NanoGPT Full C++ Example (Python Implementation)")
    print("=" * 70)
    print()

    # Set TT_METAL_RUNTIME_ROOT if not set and TT_METAL_HOME is available
    # This is needed for the runtime to find kernel files like moreh_mean
    if "TT_METAL_RUNTIME_ROOT" not in os.environ:
        tt_metal_home = get_tt_metal_home()
        if tt_metal_home and os.path.exists(tt_metal_home):
            os.environ["TT_METAL_RUNTIME_ROOT"] = tt_metal_home
            print(f"Set TT_METAL_RUNTIME_ROOT={tt_metal_home} (from get_tt_metal_home)")
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

    # Load configs matching C++ structure using ttml.common.config utilities
    tt_train_root = f"{get_tt_metal_home()}/tt-train"
    configs_root = f"{tt_train_root}/configs"
    try:
        print(f"Loading training config from: {args.config}")
        yaml_config = load_config(args.config, f"{configs_root}/training_configs")
        training_config = TrainingConfig(yaml_config)

        # Load model config from separate file (matching C++ behavior)
        # Use tt_train_root as base since C++ uses paths like "configs/model_configs/..."
        if training_config.model_config:
            print(f"Loading model config from: {training_config.model_config}")
            model_yaml = load_config(training_config.model_config, tt_train_root)
            model_config = parse_model_config(model_yaml)
        else:
            print("Warning: No model_config specified in training config")
            print("Using default model config")
            model_config = ModelConfig()
    except FileNotFoundError as e:
        print(f"Warning: Config file not found: {e}")
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
    instance = get_autograd_ctx()
    instance.open_device()
    instance.get_device()

    instance.set_seed(training_config.seed)
    np.random.seed(training_config.seed)

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
                f"   - Model: {model_config.num_blocks} layers, {model_config.embedding_dim} embd, {model_config.num_heads} heads"
            )

        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            instance.close_device()
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
                instance.close_device()
                return

        print("1. Loading and preparing data...")
        print(f"   - Data path: {training_config.data_path}")

        # Load data
        text = read_file_to_str(training_config.data_path)
        sequence_length = model_config.max_sequence_length

        # Create dataset
        dataset, tokenizer = create_dataset_from_text(text, sequence_length)
        model_config.vocab_size = tokenizer.vocab_size

        print(f"   - Vocabulary size: {model_config.vocab_size}")
        print(f"   - Dataset size: {len(dataset)} samples")
        print(f"   - Sequence length: {sequence_length}")

        # Check if resuming from checkpoint (auto-resume by default)
        start_step = 0
        resume_path = None

        if not args.fresh:
            # Auto-detect or use specified checkpoint
            if args.resume:
                resume_path = args.resume
            else:
                # Try to find the latest checkpoint
                resume_path = find_latest_checkpoint(checkpoint_save_path)
                if resume_path:
                    print(f"\n   Found existing checkpoint: {resume_path}")

        if resume_path:
            print(f"\n2. Resuming from checkpoint: {resume_path}")
            try:
                (
                    model,
                    loaded_tokenizer,
                    model_config,
                    _,  # training_config from checkpoint (we use CLI config instead)
                    start_step,
                ) = load_model_from_checkpoint(resume_path)
                # Use tokenizer from checkpoint to ensure vocab consistency
                tokenizer = loaded_tokenizer
                sequence_length = model_config.max_sequence_length
                print(f"   - Resumed from step {start_step}")
                print(
                    f"   - Model: {model_config.num_blocks} layers, {model_config.embedding_dim} embd, {model_config.num_heads} heads"
                )
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting fresh training instead...")
                resume_path = None  # Fall through to create new model

        if not resume_path:
            print("\n2. Creating model...")
            # Round vocab size to tile boundary (matching C++ behavior)
            model_config.vocab_size = round_up_to_tile(model_config.vocab_size, 32)

            # Create model config (map aligned names to NanoGPTConfig fields)
            nanogpt_config = NanoGPTConfig(
                vocab_size=model_config.vocab_size,
                block_size=model_config.max_sequence_length,
                n_embd=model_config.embedding_dim,
                n_layer=model_config.num_blocks,
                n_head=model_config.num_heads,
                dropout=model_config.dropout_prob,
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
                f"   - Model: {model_config.num_blocks} layers, {model_config.embedding_dim} embd, {model_config.num_heads} heads"
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
    mask_np = build_causal_mask(sequence_length)
    mask = ttml.autograd.Tensor.from_numpy(
        mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )

    # Training or inference mode
    if args.prompt:
        # Inference mode: skip training, go straight to inference
        print("\n6. Inference mode - skipping training")
    else:
        print("\n6. Training...")
        print()
        remaining_steps = training_config.max_steps - start_step
        print(
            f"Training for {remaining_steps} steps (step {start_step} to {training_config.max_steps})..."
        )
        print(f"  - Batch size: {training_config.batch_size}")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Training data: {len(dataset)} samples")
        print(f"  - Scheduler: {training_config.scheduler_type}")
        print(
            f"  - Gradient accumulation steps: {training_config.gradient_accumulation_steps}"
        )
        print(f"  - Dropout: {model_config.dropout_prob}")
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
        global_step = start_step

        # Training loop (matching C++ structure)
        start_time = time.time()
        # Cache values used in hot path
        batch_size = training_config.batch_size
        max_steps = training_config.max_steps
        dataset_len = len(dataset)

        for epoch in range(training_config.num_epochs):
            np.random.shuffle(dataset)

            for batch_start in range(0, dataset_len, batch_size):
                batch_end = batch_start + batch_size
                if batch_end > dataset_len:
                    continue  # Skip incomplete batches

                batch_samples = dataset[batch_start:batch_end]
                input_tokens, target_tokens = collate_fn(
                    batch_samples, batch_size, sequence_length
                )

                loss_float, step_time, should_step = train_step(
                    model,
                    optimizer,
                    scheduler_fn,
                    global_step,
                    input_tokens,
                    target_tokens,
                    mask,
                    gradient_accumulator,
                    training_config.use_clip_grad_norm,
                    training_config.clip_grad_norm_max_norm,
                    batch_size=batch_size,
                )

                if should_step:
                    global_step += 1
                    avg_loss = gradient_accumulator.average_loss()
                    loss_meter.update(avg_loss)
                    print(
                        f"Step: {global_step}, Loss: {avg_loss:.6f}, Time: {step_time:.2f} ms"
                    )

                    if (
                        checkpoint_save_path
                        and global_step % training_config.model_save_interval == 0
                    ):
                        save_checkpoint(
                            f"{checkpoint_save_path}_step_{global_step}.pkl",
                            global_step,
                            model,
                            tokenizer,
                            model_config,
                            training_config,
                        )

                    gradient_accumulator.reset()

                    if global_step >= max_steps:
                        break

            if global_step >= max_steps:
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
    instance.close_device()


if __name__ == "__main__":
    main()
