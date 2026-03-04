# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT example using ttml/ttnn primitives only.

This script implements the NanoGPT model in Python using ttml ops and ttnn
primitives, without relying on nanobind model wrappers (e.g., LinearLayer).
All modules inherit from ttml's ModuleBase via AbstractModuleBase.
"""

import argparse
import os
import time
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, ModuleList, Parameter, RunMode
from ttml.common.utils import round_up_to_tile, get_tt_metal_home
from ttml.common.config import load_config, TrainingConfig as BaseTrainingConfig
from ttml.common.data import CharTokenizer, build_causal_mask

# Module-level cache for AutoContext singleton (initialized on first use)
_autograd_ctx = None


def get_autograd_ctx():
    """Get cached AutoContext singleton instance."""
    global _autograd_ctx
    if _autograd_ctx is None:
        _autograd_ctx = ttml.autograd.AutoContext.get_instance()
    return _autograd_ctx


class TrainingConfig(BaseTrainingConfig):
    """Extended training config with NanoGPT-specific fields."""

    def __init__(self, yaml_config=None):
        super().__init__(yaml_config if yaml_config is not None else {})
        tc = {}
        if isinstance(yaml_config, dict):
            tc = yaml_config.get("training_config", {})

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

    model_type: str = "gpt2"
    model_path: str = ""
    vocab_size: int = 50304
    embedding_dim: int = 384
    num_blocks: int = 6
    num_heads: int = 6
    dropout_prob: float = 0.2
    bias: bool = True
    max_sequence_length: int = 128


@dataclass
class PrimitiveNanoGPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_layer: int
    n_head: int
    dropout: float
    bias: bool


class LossAverageMeter:
    """Loss averaging meter matching C++ LossAverageMeter."""

    def __init__(self):
        self.m_sum = 0.0
        self.m_count = 0

    def update(self, loss: float, count: int = 1):
        self.m_sum += loss * count
        self.m_count += count

    def average(self) -> float:
        if self.m_count == 0:
            return 0.0
        return self.m_sum / self.m_count

    def reset(self):
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
        return self.m_steps % self.m_accumulation_steps == 0

    def should_step(self) -> bool:
        return self.m_steps % self.m_accumulation_steps == 0

    def scale(self, loss: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        if self.m_accumulation_steps > 1:
            scale_factor = 1.0 / float(self.m_accumulation_steps)
            return ttml.ops.binary.mul(loss, scale_factor)
        return loss

    def update(self, loss: float, samples: int = 1):
        self.m_total_loss += loss * samples * float(self.m_accumulation_steps)
        self.m_total_samples += samples
        self.m_steps += 1

    def reset(self):
        self.m_total_loss = 0.0
        self.m_total_samples = 0
        self.m_steps = 0

    def average_loss(self) -> float:
        if self.m_total_samples == 0:
            return 0.0
        return self.m_total_loss / float(self.m_total_samples)


class PrimitiveEmbedding(AbstractModuleBase):
    """Embedding layer using ttml ops."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        weight_shape = (1, 1, num_embeddings, embedding_dim)
        weight_np = np.random.normal(0.0, 0.02, size=weight_shape).astype(
            ml_dtypes.bfloat16
        )
        weight_tensor = ttml.autograd.Tensor.from_numpy(
            weight_np, layout=ttnn.Layout.TILE
        )
        self.weight = Parameter(weight_tensor)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return ttml.ops.embedding.embedding(x, self.weight.tensor)

    def __call__(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return self.forward(x)


class PrimitiveLinear(AbstractModuleBase):
    """Linear layer matching C++ LinearLayer parameter naming (weight, bias)."""

    def __init__(
        self, in_features: int, out_features: int, has_bias: bool = True
    ) -> None:
        super().__init__()
        # Match C++ LinearLayer naming: creates "linear" as module name
        self.create_name("linear")

        init_k = np.sqrt(1.0 / in_features)
        weight_shape = (1, 1, out_features, in_features)
        weight_np = np.random.uniform(-init_k, init_k, size=weight_shape).astype(
            ml_dtypes.bfloat16
        )
        weight_tensor = ttml.autograd.Tensor.from_numpy(
            weight_np, layout=ttnn.Layout.TILE
        )
        self.weight = Parameter(weight_tensor)

        if has_bias:
            bias_shape = (1, 1, 1, out_features)
            bias_np = np.random.uniform(-init_k, init_k, size=bias_shape).astype(
                ml_dtypes.bfloat16
            )
            bias_tensor = ttml.autograd.Tensor.from_numpy(
                bias_np, layout=ttnn.Layout.TILE
            )
            self.bias = Parameter(bias_tensor)
        else:
            self.bias = None

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        bias_tensor = self.bias.tensor if self.bias is not None else None
        return ttml.ops.linear.linear(x, self.weight.tensor, bias_tensor)

    def __call__(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return self.forward(x)


class PrimitiveMLP(AbstractModuleBase):
    """GPT-style MLP layer matching full model parameter naming."""

    def __init__(
        self, embedding_dim: int, dropout: float = 0.0, bias: bool = True
    ) -> None:
        super().__init__()
        self.dropout_prob = dropout

        # Use PrimitiveLinear to get fc1/weight, fc1/bias naming
        self.fc1 = PrimitiveLinear(embedding_dim, embedding_dim * 4, has_bias=bias)
        self.fc2 = PrimitiveLinear(embedding_dim * 4, embedding_dim, has_bias=bias)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        x = self.fc1(x)
        x = ttml.ops.unary.gelu(x)
        x = self.fc2(x)
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            x = ttml.ops.dropout.dropout(x, self.dropout_prob)
        return x

    def __call__(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return self.forward(x)


class PrimitiveMultiHeadAttention(AbstractModuleBase):
    """Multi-head attention matching full model parameter naming."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.dropout_prob = dropout

        # Use PrimitiveLinear to get qkv_linear/weight, qkv_linear/bias naming
        self.qkv_linear = PrimitiveLinear(
            embedding_dim, embedding_dim * 3, has_bias=bias
        )
        self.out_linear = PrimitiveLinear(embedding_dim, embedding_dim, has_bias=bias)

    def forward(
        self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        qkv = self.qkv_linear(x)
        query, key, value = ttml.ops.multi_head_utils.heads_creation(
            qkv, self.num_heads
        )
        attn_out = ttml.ops.attention.scaled_dot_product_attention(
            query, key, value, mask
        )
        fused = ttml.ops.multi_head_utils.heads_fusion(attn_out)
        out = self.out_linear(fused)
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            out = ttml.ops.dropout.dropout(out, self.dropout_prob)
        return out

    def __call__(
        self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        return self.forward(x, mask)


class PrimitiveGPTBlock(AbstractModuleBase):
    """GPT transformer block using ttml ops only."""

    def __init__(
        self, embedding_dim: int, num_heads: int, dropout: float, bias: bool
    ) -> None:
        super().__init__()
        ln_shape = (1, 1, 1, embedding_dim)

        gamma1_np = np.ones(ln_shape, dtype=ml_dtypes.bfloat16)
        gamma1_tensor = ttml.autograd.Tensor.from_numpy(
            gamma1_np, layout=ttnn.Layout.TILE
        )
        self.ln1_gamma = Parameter(gamma1_tensor)
        if bias:
            beta1_np = np.zeros(ln_shape, dtype=ml_dtypes.bfloat16)
            beta1_tensor = ttml.autograd.Tensor.from_numpy(
                beta1_np, layout=ttnn.Layout.TILE
            )
            self.ln1_beta = Parameter(beta1_tensor)
        else:
            self.ln1_beta = None

        gamma2_np = np.ones(ln_shape, dtype=ml_dtypes.bfloat16)
        gamma2_tensor = ttml.autograd.Tensor.from_numpy(
            gamma2_np, layout=ttnn.Layout.TILE
        )
        self.ln2_gamma = Parameter(gamma2_tensor)
        if bias:
            beta2_np = np.zeros(ln_shape, dtype=ml_dtypes.bfloat16)
            beta2_tensor = ttml.autograd.Tensor.from_numpy(
                beta2_np, layout=ttnn.Layout.TILE
            )
            self.ln2_beta = Parameter(beta2_tensor)
        else:
            self.ln2_beta = None

        self.attention = PrimitiveMultiHeadAttention(
            embedding_dim, num_heads, dropout, bias
        )
        self.mlp = PrimitiveMLP(embedding_dim, dropout, bias)

    def forward(
        self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        residual = x
        x = ttml.ops.layernorm.composite_layernorm(
            x, self.ln1_gamma.tensor, self.ln1_beta.tensor if self.ln1_beta else None
        )
        x = self.attention(x, mask)
        x = ttml.ops.binary.add(x, residual)

        residual = x
        x = ttml.ops.layernorm.composite_layernorm(
            x, self.ln2_gamma.tensor, self.ln2_beta.tensor if self.ln2_beta else None
        )
        x = self.mlp(x)
        x = ttml.ops.binary.add(x, residual)
        return x

    def __call__(
        self, x: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        return self.forward(x, mask)


class PrimitiveNanoGPT(AbstractModuleBase):
    """NanoGPT model implemented with ttml/ttnn primitives.

    Parameter names match the full NanoGPT model exactly for checkpoint compatibility.
    """

    def __init__(self, config: PrimitiveNanoGPTConfig) -> None:
        super().__init__()
        # Use "NanoGPT" as module name to match full model's parameter paths
        self.create_name("NanoGPT")
        self.config = config

        # Match full model naming: tok_emb, pos_emb
        vocab_size_divisible_by_32 = (config.vocab_size + 31) // 32 * 32
        self.tok_emb = PrimitiveEmbedding(vocab_size_divisible_by_32, config.n_embd)
        self.pos_emb = PrimitiveEmbedding(config.block_size, config.n_embd)

        # LM head (fc) - no bias, matching full model
        self.fc = PrimitiveLinear(config.n_embd, config.vocab_size, has_bias=False)

        # Transformer blocks (ModuleList auto-registers all blocks)
        self.blocks = ModuleList(
            [
                PrimitiveGPTBlock(
                    config.n_embd, config.n_head, config.dropout, config.bias
                )
                for _ in range(config.n_layer)
            ]
        )

        ln_f_shape = (1, 1, 1, config.n_embd)
        gamma_f_np = np.ones(ln_f_shape, dtype=ml_dtypes.bfloat16)
        gamma_f_tensor = ttml.autograd.Tensor.from_numpy(
            gamma_f_np, layout=ttnn.Layout.TILE
        )
        self.ln_f_gamma = Parameter(gamma_f_tensor)

        if config.bias:
            beta_f_np = np.zeros(ln_f_shape, dtype=ml_dtypes.bfloat16)
            beta_f_tensor = ttml.autograd.Tensor.from_numpy(
                beta_f_np, layout=ttnn.Layout.TILE
            )
            self.ln_f_beta = Parameter(beta_f_tensor)
        else:
            self.ln_f_beta = None

        # Cache for position tensor (avoids recreating every forward pass)
        self._cached_pos_tensor = None
        self._cached_pos_seq_len = None

    def forward(
        self, idx: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        tok_emb = self.tok_emb(idx)

        # Create position indices (cached for constant sequence length during training)
        idx_shape = idx.shape()
        seq_len = idx_shape[-1]

        # Use cached position tensor if sequence length matches
        if self._cached_pos_seq_len != seq_len:
            pos_np = np.arange(seq_len, dtype=np.uint32).reshape(1, 1, 1, seq_len)
            self._cached_pos_tensor = ttml.autograd.Tensor.from_numpy(
                pos_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
            )
            self._cached_pos_seq_len = seq_len

        pos_emb = self.pos_emb(self._cached_pos_tensor)

        x = ttml.ops.binary.add(tok_emb, pos_emb)
        if self.get_run_mode() == RunMode.TRAIN and self.config.dropout > 0.0:
            x = ttml.ops.dropout.dropout(x, self.config.dropout)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = ttml.ops.layernorm.composite_layernorm(
            x, self.ln_f_gamma.tensor, self.ln_f_beta.tensor if self.ln_f_beta else None
        )
        logits = self.fc(x)
        logits_shape = logits.shape()
        if len(logits_shape) == 5:
            new_shape = [logits_shape[0], 1, logits_shape[3], logits_shape[4]]
            logits = ttml.ops.reshape.reshape(logits, new_shape)
        return logits

    def __call__(
        self, idx: ttml.autograd.Tensor, mask: Optional[ttml.autograd.Tensor] = None
    ) -> ttml.autograd.Tensor:
        return self.forward(idx, mask)


def read_file_to_str(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def create_warmup_linear_scheduler(optimizer, total_steps: int):
    warmup_factor = 0.1
    warmup_steps = int(total_steps * warmup_factor)
    linear_decay_steps = total_steps - warmup_steps

    def scheduler_fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
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


def create_causal_mask_tensor(sequence_length: int) -> ttml.autograd.Tensor:
    mask_np = build_causal_mask(sequence_length)
    return ttml.autograd.Tensor.from_numpy(
        mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )


def collate_fn(
    samples: list, batch_size: int, sequence_length: int
) -> Tuple[ttml.autograd.Tensor, ttml.autograd.Tensor]:
    data = []
    targets = []
    for seq, target in samples[:batch_size]:
        data.extend(seq)
        targets.extend(target)
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
    """Extract loss value from tensor efficiently.

    Uses ttnn.Tensor.item() which directly extracts scalar via to_vector<T>()
    without NumPy conversion overhead.
    """
    return float(loss.get_value().item())


def train_step(
    model: PrimitiveNanoGPT,
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
    """Single training step with proper gradient accumulation.

    Args:
        batch_size: Optional cached batch size (if None, will extract from input_tokens)

    Returns:
        Tuple of (loss_float, step_time_ms, should_step)
    """
    start_time = time.time()

    if gradient_accumulator.should_zero_grad():
        optimizer.zero_grad()

    logits = model(input_tokens, mask)
    loss = ttml.ops.loss.cross_entropy_loss(
        logits, target_tokens, reduce=ttml.ops.ReduceType.MEAN
    )
    loss = gradient_accumulator.scale(loss)
    loss_float = get_loss_value(loss)
    loss.backward(False)

    get_autograd_ctx().reset_graph()

    # Use cached batch_size if provided to avoid shape() call
    samples = batch_size if batch_size is not None else input_tokens.shape()[0]
    gradient_accumulator.update(loss_float, samples)

    should_step = gradient_accumulator.should_step()
    if should_step:
        if use_clip_grad_norm:
            ttml.core.clip_grad_norm(
                model.parameters(),
                clip_grad_norm_max_norm,
                2.0,
                False,
            )
        optimizer.step()
        if scheduler_fn is not None:
            scheduler_fn(scheduler_step)

    step_time = (time.time() - start_time) * 1000
    return loss_float, step_time, should_step


def parse_model_config(yaml_config: dict) -> ModelConfig:
    transformer_config = yaml_config.get("transformer_config", {})
    config = ModelConfig()
    config.model_type = transformer_config.get("model_type", config.model_type)
    config.model_path = transformer_config.get("model_path", config.model_path)

    if config.model_type == "gpt2":
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
    model: PrimitiveNanoGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    sequence_length: int,
    mask: ttml.autograd.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> str:
    model.eval()

    # Reset graph before inference to ensure clean state
    get_autograd_ctx().reset_graph()

    if len(prompt) == 0:
        prompt = " "

    try:
        prompt_ids = tokenizer.encode(prompt)
    except Exception as e:
        raise ValueError(f"Failed to encode prompt '{prompt}': {e}")

    running = list(prompt_ids[:sequence_length])
    if len(running) < sequence_length:
        if tokenizer.stoi:
            space_token_id = tokenizer.stoi.get(" ", None)
            if space_token_id is None:
                space_token_id = list(tokenizer.stoi.values())[0]
            padding = [space_token_id] * (sequence_length - len(running))
        else:
            padding = [0] * (sequence_length - len(running))
        running = padding + running

    generated_tokens = []

    print(f"\nGenerating text from prompt: '{prompt}'")
    print("=" * 70)

    for step in range(max_new_tokens):
        inp = np.array(running[-sequence_length:], dtype=np.uint32).reshape(
            1, 1, 1, sequence_length
        )
        input_tensor = ttml.autograd.Tensor.from_numpy(
            inp, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
        )
        # Clone mask before each use to avoid TTNN memory reuse corrupting the original
        mask_for_model = ttml.autograd.Tensor(ttnn.clone(mask.get_value()), False)
        logits = model(input_tensor, mask_for_model)
        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)
        logits_shape = logits_np.shape
        if len(logits_shape) == 5:
            last_logits = logits_np.reshape(
                logits_shape[0], logits_shape[3], logits_shape[4]
            )[:, -1, :]
        elif len(logits_shape) == 4:
            last_logits = logits_np.reshape(
                logits_shape[0], logits_shape[2], logits_shape[3]
            )[:, -1, :]
        else:
            last_logits = logits_np.reshape(-1, logits_np.shape[-1])[-1:]

        vocab_size = tokenizer.vocab_size
        last_logits = last_logits[:, :vocab_size]

        if not np.all(np.isfinite(last_logits)):
            last_logits = np.nan_to_num(last_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        if temperature != 1.0 and temperature > 0:
            last_logits = np.clip(last_logits, -100, 100)
            last_logits = last_logits / temperature

        if top_k > 0:
            top_k_val = min(top_k, vocab_size)
            threshold = np.partition(last_logits, -top_k_val, axis=-1)[:, -top_k_val:][
                :, 0:1
            ]
            indices_to_remove = last_logits < threshold
            last_logits[indices_to_remove] = -1e9

        max_logits = np.max(last_logits, axis=-1, keepdims=True)
        logits_exp = np.exp(last_logits - max_logits)
        probs = logits_exp / (np.sum(logits_exp, axis=-1, keepdims=True) + 1e-10)
        probs = np.clip(probs, 0, 1)
        probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-10)

        if temperature < 0.01:
            next_id = int(np.argmax(probs, axis=-1)[0])
        else:
            try:
                next_id = int(np.random.choice(vocab_size, p=probs[0]))
            except ValueError:
                next_id = int(np.argmax(last_logits, axis=-1)[0])

        running.append(next_id)
        generated_tokens.append(next_id)

        get_autograd_ctx().reset_graph()

        if (step + 1) % 50 == 0:
            current_text = tokenizer.decode(generated_tokens)
            print(f"[{step + 1}/{max_new_tokens}] {current_text[-100:]}...")

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
    model: PrimitiveNanoGPT,
    tokenizer: CharTokenizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> str:
    if not checkpoint_path.endswith(".pkl"):
        checkpoint_path = f"{checkpoint_path}.pkl"

    model_state = {}
    for name, param in model.parameters().items():
        if isinstance(param, Parameter):
            tensor = param.tensor
        else:
            tensor = param
        layout = tensor.get_value().get_layout()
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

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"  Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def find_latest_checkpoint(base_path: str) -> Optional[str]:
    import glob
    import re

    pattern = f"{base_path}_step_*.pkl"
    step_files = glob.glob(pattern)
    final_path = f"{base_path}_final.pkl"
    if os.path.exists(final_path):
        step_files.append(final_path)
    if not step_files:
        return None

    def get_step(path: str) -> int:
        if path.endswith("_final.pkl"):
            return float("inf")
        match = re.search(r"_step_(\d+)\.pkl$", path)
        return int(match.group(1)) if match else -1

    latest = max(step_files, key=get_step)
    return latest


def load_model_from_checkpoint(
    checkpoint_path: str,
) -> Tuple[PrimitiveNanoGPT, CharTokenizer, ModelConfig, TrainingConfig, int]:
    """Load model from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    model_state = checkpoint["model_state"]
    tokenizer = checkpoint["tokenizer"]
    model_config = checkpoint["model_config"]
    training_config = checkpoint.get("training_config", None)
    step = checkpoint.get("step", 0)

    nanogpt_config = PrimitiveNanoGPTConfig(
        vocab_size=model_config.vocab_size,
        block_size=model_config.max_sequence_length,
        n_embd=model_config.embedding_dim,
        n_layer=model_config.num_blocks,
        n_head=model_config.num_heads,
        dropout=model_config.dropout_prob,
        bias=model_config.bias,
    )
    model = PrimitiveNanoGPT(nanogpt_config)

    print("  Loading model parameters...")
    model_params = model.parameters()

    loaded_count = 0
    skipped_count = 0
    for name, param_data in model_state.items():
        if name not in model_params:
            print(f"    Warning: Parameter {name} not found in model, skipping")
            skipped_count += 1
            continue

        if isinstance(param_data, dict):
            numpy_array = param_data["data"]
            layout_str = param_data.get("layout", "TILE")
        else:
            numpy_array = param_data
            layout_str = "TILE"

        if layout_str == "ROW_MAJOR" or "ROW_MAJOR" in str(layout_str):
            layout = ttnn.Layout.ROW_MAJOR
        else:
            layout = ttnn.Layout.TILE

        if numpy_array.dtype != np.dtype("float32"):
            numpy_array = numpy_array.astype(np.float32)
        numpy_bfloat16 = numpy_array.astype(ml_dtypes.bfloat16)
        restored_tensor = ttml.autograd.Tensor.from_numpy(
            numpy_bfloat16, layout=layout, new_type=ttnn.DataType.BFLOAT16
        )
        model_params[name].assign(restored_tensor)
        loaded_count += 1

    print(f"  Loaded {loaded_count} parameters, skipped {skipped_count}")
    print(f"  Checkpoint loaded from step {step}")
    return model, tokenizer, model_config, training_config, step


def main():
    parser = argparse.ArgumentParser(description="NanoGPT Primitives Example (Python)")
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
    print("NanoGPT Primitives Example (Python Implementation)")
    print("=" * 70)
    print()

    if "TT_METAL_RUNTIME_ROOT" not in os.environ:
        tt_metal_home = get_tt_metal_home()
        if tt_metal_home and os.path.exists(tt_metal_home):
            os.environ["TT_METAL_RUNTIME_ROOT"] = tt_metal_home
            print(f"Set TT_METAL_RUNTIME_ROOT={tt_metal_home} (from get_tt_metal_home)")
        else:
            current_dir = os.getcwd()
            if os.path.exists(os.path.join(current_dir, "tt_metal")):
                os.environ["TT_METAL_RUNTIME_ROOT"] = current_dir
                print(
                    f"Set TT_METAL_RUNTIME_ROOT={current_dir} (auto-detected from current directory)"
                )
            else:
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

    tt_train_root = f"{get_tt_metal_home()}/tt-train"
    configs_root = f"{tt_train_root}/configs"
    try:
        print(f"Loading training config from: {args.config}")
        yaml_config = load_config(args.config, f"{configs_root}/training_configs")
        training_config = TrainingConfig(yaml_config)
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

    if args.model_save_path:
        checkpoint_save_path = args.model_save_path
    elif training_config.model_config:
        checkpoint_save_path = f"checkpoints/{training_config.project_name}"
    else:
        checkpoint_save_path = f"checkpoints/{training_config.project_name}"

    if checkpoint_save_path:
        checkpoint_dir = (
            os.path.dirname(checkpoint_save_path)
            if os.path.dirname(checkpoint_save_path)
            else "checkpoints"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {checkpoint_save_path}_step_*.pkl")

    inference_only = args.prompt and (args.model_path or checkpoint_save_path)

    instance = ttml.autograd.AutoContext.get_instance()
    instance.open_device()
    instance.get_device()

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
        if not training_config.data_path:
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

        text = read_file_to_str(training_config.data_path)
        sequence_length = model_config.max_sequence_length
        dataset, tokenizer = create_dataset_from_text(text, sequence_length)
        model_config.vocab_size = tokenizer.vocab_size

        print(f"   - Vocabulary size: {model_config.vocab_size}")
        print(f"   - Dataset size: {len(dataset)} samples")
        print(f"   - Sequence length: {sequence_length}")

        start_step = 0
        resume_path = None
        if not args.fresh:
            if args.resume:
                resume_path = args.resume
            else:
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
                    _,
                    start_step,
                ) = load_model_from_checkpoint(resume_path)
                tokenizer = loaded_tokenizer
                sequence_length = model_config.max_sequence_length
                print(f"   - Resumed from step {start_step}")
                print(
                    f"   - Model: {model_config.num_blocks} layers, {model_config.embedding_dim} embd, {model_config.num_heads} heads"
                )
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting fresh training instead...")
                resume_path = None

        if not resume_path:
            print("\n2. Creating model...")
            model_config.vocab_size = round_up_to_tile(model_config.vocab_size, 32)
            nanogpt_config = PrimitiveNanoGPTConfig(
                vocab_size=model_config.vocab_size,
                block_size=model_config.max_sequence_length,
                n_embd=model_config.embedding_dim,
                n_layer=model_config.num_blocks,
                n_head=model_config.num_heads,
                dropout=model_config.dropout_prob,
                bias=model_config.bias,
            )
            model = PrimitiveNanoGPT(nanogpt_config)
            total_params = sum(
                p.to_numpy(ttnn.DataType.FLOAT32).size
                for p in model.parameters().values()
            )
            print(
                f"   - Model: {model_config.num_blocks} layers, {model_config.embedding_dim} embd, {model_config.num_heads} heads"
            )
            print(f"   - Total parameters: {total_params:,}")

    if args.prompt:
        optimizer = None
        scheduler_fn = None
        print("\n3. Inference mode - skipping optimizer setup")
    else:
        print("\n3. Setting up optimizer...")
        instance.set_seed(training_config.seed)
        if training_config.use_no_op:
            print("   WARNING: Using NoOp optimizer - parameters will NOT be updated.")
            optimizer = None
        else:
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            adamw_config = ttml.optimizers.AdamWConfig.make(
                training_config.learning_rate,
                beta1,
                beta2,
                epsilon,
                training_config.weight_decay,
            )

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

    if inference_only:
        print("\n2. Creating attention mask...")
    else:
        print("\n5. Creating attention mask...")
    mask = create_causal_mask_tensor(sequence_length)

    if args.prompt:
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

        model.train()
        loss_meter = LossAverageMeter()
        gradient_accumulator = GradientAccumulator(
            training_config.gradient_accumulation_steps
        )
        global_step = start_step

        # Cache batch_size to avoid repeated lookups in hot path
        batch_size = training_config.batch_size

        start_time = time.time()
        for epoch in range(training_config.num_epochs):
            np.random.shuffle(dataset)
            dataset_len = len(dataset)
            for batch_start in range(0, dataset_len, batch_size):
                batch_end = batch_start + batch_size
                if batch_end > dataset_len:
                    continue
                batch_samples = dataset[batch_start:batch_end]

                input_tokens, target_tokens = collate_fn(
                    batch_samples,
                    batch_size,
                    sequence_length,
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
                    avg_loss = gradient_accumulator.average_loss()
                    loss_meter.update(avg_loss)
                    print(
                        f"Step: {global_step}, Loss: {avg_loss:.6f}, Time: {step_time:.2f} ms"
                    )
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
                    if global_step >= training_config.max_steps:
                        break

                    gradient_accumulator.reset()
                    global_step += 1

            if global_step >= training_config.max_steps:
                break

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

        total_time = time.time() - start_time
        print()
        print("=" * 70)
        print("Training completed!")
        print(f"  - Total steps: {global_step}")
        print(f"  - Total time: {total_time:.2f} s")
        print(f"  - Average loss: {loss_meter.average():.6f}")
        print("=" * 70)

    if args.prompt:
        print("\n" + "=" * 70)
        print("Running Inference Mode")
        print("=" * 70)
        print(f"  Prompt: '{args.prompt}'")
        print(f"  Max new tokens: {args.max_new_tokens}")
        print()

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

    instance.close_device()


if __name__ == "__main__":
    main()
