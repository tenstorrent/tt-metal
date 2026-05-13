# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-featured NanoGPT training example.

This example provides a comprehensive Python implementation that mirrors the C++ nano_gpt example,
including:
- Full model training with GPT2/NanoGPT and Llama architectures
- Gradient accumulation
- Learning rate scheduling (identity, warmup_linear)
- Optimizers (via create_optimizer from config)
- Model checkpointing and resuming
- Loss tracking and averaging
- Configurable training parameters
- Character tokenizer (via ttml.common.data.CharTokenizer)
- Proper tensor shapes
"""

import argparse
import glob
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, Union
import time
import pickle

import numpy as np
import ml_dtypes

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

import ttnn
import ttml
import ttml.common.muon_optimizer

ttml.common.muon_optimizer.register()
from ttml.models.nanogpt import (
    NanoGPT,
    NanoGPTConfig,
    NanoGPTExperimentalConfig,
    create_nanogpt,
)
from ttml.models.llama import (
    Llama,
    LlamaConfig,
    LlamaRopeScalingConfig,
)
from ttml.models.deepseek import (
    DeepSeek,
    DeepSeekConfig,
)
from ttml.modules import Parameter
from ttml.common.utils import round_up_to_tile, get_tt_metal_runtime_root, create_optimizer, summary
from ttml.common.config import load_config, TrainingConfig as BaseTrainingConfig, DeviceConfig
from ttml.common.data import CharTokenizer, build_causal_mask
from ttml.common.profiler_utils import profiler_marker
from ttml.common.schedulers import (
    CosineAnnealingScheduler,
    LinearScheduler,
    SequentialScheduler,
)

# Union type for models that share the same forward(input, mask) interface
Model = Union[NanoGPT, Llama, DeepSeek]

# Memory tracking utilities
MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker


def get_device_peak_tflops_bf16() -> float:
    """Get theoretical peak BF16 TFLOPS for the current TT device.

    Wormhole: 1.0 TFLOPS/core, Blackhole: 1.35 TFLOPS/core.
    Returns total peak TFLOPS across all compute cores.
    """
    from ttnn.device import is_blackhole, is_wormhole_b0

    device = ttml.autograd.AutoContext.get_instance().get_device()
    grid_size = device.compute_with_storage_grid_size()
    num_cores = grid_size.x * grid_size.y

    if is_wormhole_b0(device):
        tflops_per_core = 1.0
    elif is_blackhole(device):
        tflops_per_core = 1.35
    else:
        raise ValueError(f"Unknown device: {device.arch()}")
    return num_cores * tflops_per_core


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
        self.use_clip_grad_norm = tc.get("use_clip_grad_norm", False)
        self.clip_grad_norm_max_norm = float(tc.get("clip_grad_norm_max_norm", 1.0))

        # Cosine-with-warmup scheduler hyperparameters (TinyLlama paper-style).
        # Only consumed when ``scheduler_type == "cosine_with_warmup"``. Defaults
        # mirror the TinyLlama paper (warmup=2000 steps, min_lr=4e-5 with
        # peak_lr=4e-4). ``lr_decay_steps`` <= 0 means "fill the rest of training":
        # decay_steps = max_steps - lr_warmup_steps so cosine lands exactly at
        # ``lr_min`` at ``max_steps``.
        self.lr_warmup_steps = int(tc.get("lr_warmup_steps", 2000))
        self.lr_min = float(tc.get("lr_min", 0.0))
        self.lr_decay_steps = int(tc.get("lr_decay_steps", 0))

        # Pre-tokenized dataset fields (TinyLlama / lit_gpt PackedDataset semantics).
        # Empty tokenized_data_dir keeps the legacy CharTokenizer + text-file path active.
        self.tokenized_data_dir = tc.get("tokenized_data_dir", "")
        self.train_split = tc.get("train_split", "train")
        self.val_split = tc.get("val_split", "validation")
        self.n_chunks = int(tc.get("n_chunks", 8))
        self.eval_interval = int(tc.get("eval_interval", 0))
        self.eval_iters = int(tc.get("eval_iters", 100))

        # Re-read with defaults (override BaseTrainingConfig defaults)
        self.seed = int(tc.get("seed", 5489))
        self.max_steps = int(tc.get("max_steps", 5000))

        # Aliases to match expected field names in this example
        self.num_epochs = self.epochs
        self.model_save_interval = self.save_every


@dataclass
class ModelExperimentalConfig:
    use_composite_layernorm: bool = False  # Use composite vs fused layernorm


@dataclass
class ModelConfig:
    """Model configuration aligned with ttml.common.config.TransformerConfig naming.

    Field names follow the universal project format (YAML conventions).
    Conversion to model-specific config (e.g. LlamaConfig) happens at model creation time.
    """

    model_type: str = "gpt2"  # "gpt2", "llama", or "deepseek"
    model_path: str = ""
    vocab_size: int = 256
    embedding_dim: int = 384
    num_blocks: int = 6
    num_heads: int = 6
    dropout_prob: float = 0.2
    bias: bool = True
    max_sequence_length: int = 256
    runner_type: ttml.models.RunnerType = ttml.models.RunnerType.Default
    weight_tying: ttml.models.WeightTyingType = ttml.models.WeightTyingType.Disabled
    positional_embedding_type: Literal["trainable", "fixed"] = "trainable"
    experimental: ModelExperimentalConfig = field(default_factory=ModelExperimentalConfig)
    # Llama-specific fields (universal naming YAML conventions)
    num_groups: int = 3  # GQA: num_key_value_heads
    theta: float = 500000.0  # RoPE theta parameter
    intermediate_dim: Optional[int] = None  # MLP intermediate dimension
    # RoPE NTK-aware scaling (nested under rope_scaling in YAML)
    scaling_factor: float = 0.0  # 0.0 means no scaling
    high_freq_factor: float = 4.0
    low_freq_factor: float = 1.0
    original_context_length: int = 0  # 0 means no scaling
    # DeepSeek-specific fields
    inter_dim: Optional[int] = None
    moe_inter_dim: int = 256
    n_dense_layers: int = 2
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 2
    n_limited_groups: int = 1
    score_func: str = "sigmoid"
    route_scale: float = 2.5
    q_lora_rank: int = 256
    kv_lora_rank: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64


class LossAverageMeter:
    """Loss averaging meter."""

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
    """Gradient accumulator."""

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
        """Scale loss by accumulation steps"""
        if self.m_accumulation_steps > 1:
            scale_factor = 1.0 / float(self.m_accumulation_steps)
            # Use float overload directly
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


def build_lr_scheduler(optimizer, training_config: "TrainingConfig", yaml_config: dict):
    """Create the LR scheduler for the configured ``scheduler_type``.

    Returns ``(lr_scheduler, info_dict)`` where ``info_dict`` carries the
    resolved hyperparameters for logging. Returns ``(None, {"type": ...})`` for
    ``scheduler_type == "identity"`` so the optimizer keeps its constant LR.

    Both supported schedule types are built as
    ``SequentialScheduler([linear_warmup, decay], milestones=[warmup, decay])``
    and produce objects with the same interface
    (``.step()`` / ``.get_state_dict()`` / ``.set_state_dict()``):

    - ``cosine_with_warmup``: linear warmup ``0 -> peak_lr`` over
      ``lr_warmup_steps``, then ``CosineAnnealingScheduler`` from ``peak_lr``
      down to ``lr_min`` over ``lr_decay_steps`` (defaults to
      ``max_steps - lr_warmup_steps`` so cosine lands exactly on ``lr_min`` at
      ``max_steps``).
    - ``warmup_linear``: linear warmup ``0 -> peak_lr`` over the first 10% of
      ``max_steps``, then a linear decay from ``peak_lr`` down to
      ``0.01 * peak_lr`` over the remaining steps. Hyperparameters are
      hard-coded to preserve exact backward compatibility with the original
      ``create_warmup_linear_scheduler`` closure (``warmup_factor=0.1``,
      end-of-decay factor ``0.01``).

    Both paths read ``optimizer.get_lr()`` as the schedulers' ``_base_lr`` at
    construction time. On a fresh start the optimizer is at the configured
    peak LR (from ``create_optimizer``) so the snapshot is correct. On
    resume the optimizer state is restored *before* this call, which means
    ``optimizer.get_lr()`` is the decayed checkpoint LR — the snapshot here
    is wrong, and the caller relies on ``lr_scheduler.set_state_dict(...)``
    to subsequently overwrite ``_base_lr`` from the saved peak LR. The
    upstream ``ttml.common.schedulers._SchedulerBase`` persists ``_base_lr``
    for exactly this reason.
    """
    scheduler_type = training_config.scheduler_type
    peak_lr = float(yaml_config["training_config"]["optimizer"]["lr"])

    if scheduler_type == "cosine_with_warmup":
        warmup_steps = max(1, int(training_config.lr_warmup_steps))
        decay_steps = int(training_config.lr_decay_steps)
        if decay_steps <= 0:
            decay_steps = max(1, int(training_config.max_steps) - warmup_steps)
        eta_min = float(training_config.lr_min)

        warmup = LinearScheduler(optimizer, start_factor=0.0, end_factor=1.0, total_steps=warmup_steps)
        decay = CosineAnnealingScheduler(optimizer, T_max=decay_steps, eta_min=eta_min)
        scheduler = SequentialScheduler(optimizer, [warmup, decay], milestones=[warmup_steps, decay_steps])
        info = {
            "type": "cosine_with_warmup",
            "peak_lr": peak_lr,
            "eta_min": eta_min,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
        }
        return scheduler, info

    if scheduler_type == "warmup_linear":
        # Legacy schedule: 10% warmup then linear decay from peak_lr to
        # 0.01 * peak_lr. Mirrors the original ``create_warmup_linear_scheduler``
        # closure exactly (``LinearScheduler(start, end, T).step()`` produces
        # ``factor = start + (end - start) * k/T`` after the k-th call, which
        # for ``(1.0, 0.01, T)`` is ``1 - 0.99 * k/T`` — matching the
        # hand-rolled formula).
        total_steps = max(1, int(training_config.max_steps))
        warmup_steps = max(1, int(total_steps * 0.1))
        decay_steps = max(1, total_steps - warmup_steps)
        end_factor = 0.01

        warmup = LinearScheduler(optimizer, start_factor=0.0, end_factor=1.0, total_steps=warmup_steps)
        decay = LinearScheduler(optimizer, start_factor=1.0, end_factor=end_factor, total_steps=decay_steps)
        scheduler = SequentialScheduler(optimizer, [warmup, decay], milestones=[warmup_steps, decay_steps])
        info = {
            "type": "warmup_linear",
            "peak_lr": peak_lr,
            "end_lr": peak_lr * end_factor,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
        }
        return scheduler, info

    return None, {"type": scheduler_type}


class InMemoryTokenDataset:
    """Lazy token dataset.

    Stores tokens once and generates (input, target) pairs on the fly
    using a sliding window with stride=1.
    Size = len(tokens) - seq_length (every token offset is a valid sample).
    """

    def __init__(self, tokens: np.ndarray, seq_length: int):
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self) -> int:
        if len(self.tokens) <= self.seq_length:
            return 0
        return len(self.tokens) - self.seq_length

    def __getitem__(self, index: int):
        return (
            self.tokens[index : index + self.seq_length],
            self.tokens[index + 1 : index + self.seq_length + 1],
        )


def create_dataset_from_text(
    text: str,
    sequence_length: int,
) -> Tuple["InMemoryTokenDataset", CharTokenizer]:
    """Create dataset from text using CharTokenizer from ttml.common.data.

    Stores tokens once and uses a sliding
    window with stride=1 so every token offset is a valid sample.

    Args:
        text: Text corpus to create dataset from.
        sequence_length: Length of each sequence.

    Returns:
        Tuple of (dataset, tokenizer).
    """
    tokenizer = CharTokenizer(text)
    tokens = np.array(tokenizer.encode(text), dtype=np.uint32)
    return InMemoryTokenDataset(tokens, sequence_length), tokenizer


def collate_fn(samples: list, sequence_length: int) -> Tuple[ttml.autograd.Tensor, ttml.autograd.Tensor]:
    """Collate function.

    Args:
        samples: List of (sequence, target) tuples
        sequence_length: Sequence length
    """
    actual_batch_size = len(samples)
    data = []
    targets = []
    for seq, target in samples:
        data.extend(seq)
        targets.extend(target)

    data_np = np.array(data, dtype=np.uint32).reshape(actual_batch_size, 1, 1, sequence_length)
    targets_np = np.array(targets, dtype=np.uint32).reshape(actual_batch_size, sequence_length)

    # Create tensors directly from NumPy with correct shape (single host-to-device transfer)
    data_tensor = ttml.autograd.Tensor.from_numpy(data_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32)
    targets_tensor = ttml.autograd.Tensor.from_numpy(
        targets_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )

    return data_tensor, targets_tensor


@dataclass
class _PackedTokenizerStub:
    """Lightweight tokenizer placeholder for pre-tokenized datasets.

    The pre-tokenized SlimPajama path skips host-side tokenization entirely
    (the parquet -> uint16 conversion is done up-front by
    dataset_preprocessing.py). We still need an object in this slot so the
    existing checkpoint/W&B code paths can pickle and serialize without
    branching. ``encode``/``decode`` deliberately raise so that ``--prompt``
    inference fails loudly and points at the missing HF tokenizer adapter
    instead of silently producing garbage.
    """

    name: str = ""
    vocab_size: int = 0
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    # Kept for compatibility with sample_greedy's tokenizer.stoi access path.
    stoi: dict = field(default_factory=dict)

    def encode(self, text: str):
        raise NotImplementedError(
            "Pre-tokenized datasets do not ship a runtime tokenizer. "
            "Add an HF tokenizer adapter for tokenizer={!r} to use --prompt.".format(self.name)
        )

    def decode(self, ids):
        raise NotImplementedError(
            "Pre-tokenized datasets do not ship a runtime tokenizer. "
            "Add an HF tokenizer adapter for tokenizer={!r} to use --prompt.".format(self.name)
        )


class _HFTokenizerAdapter:
    """Wraps a HuggingFace tokenizer to satisfy sample_greedy's interface.

    sample_greedy expects ``encode(str) -> list[int]``, ``decode(list[int]) -> str``,
    and a ``stoi`` dict it consults for prompt padding. HF tokenizers don't
    expose ``stoi`` directly, so we build one from ``get_vocab()``.
    """

    def __init__(self, hf_name: str):
        from transformers import AutoTokenizer  # lazy import; only needed for --prompt

        self._hf = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        self.name = hf_name
        self.vocab_size = self._hf.vocab_size
        self.bos_id = self._hf.bos_token_id
        self.eos_id = self._hf.eos_token_id
        # sample_greedy reads stoi[" "] for padding; HF's get_vocab maps str -> id.
        self.stoi = self._hf.get_vocab()

    def encode(self, text: str):
        # add_special_tokens=False matches the preprocessing pipeline (which
        # injects BOS/EOS manually around each document, not via the tokenizer).
        return self._hf(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self._hf.decode(list(ids), skip_special_tokens=True)


class PackedTokenDataset:
    """lit_gpt.PackedDataset-faithful iterator over raw uint16 .bin shards.

    Mirrors the sampling semantics of ``lit_gpt.packed_dataset.PackedDatasetIterator``
    used by TinyLlama: mmap ``n_chunks`` shards at a time, view each as
    ``n_blocks = shard_tokens // (block_size + 1)`` non-overlapping aligned
    blocks of ``block_size + 1`` uint16 tokens, draw a random permutation over
    the union of those blocks, and yield slices in that permuted order. When
    the buffer is exhausted, advance by ``n_chunks`` files and re-permute.

    With ``wrap=True`` (train), cycles through the shard list indefinitely.
    With ``wrap=False`` (val), runs through every shard once and stops.

    Each yielded sample is a numpy uint16 array of length ``block_size + 1``;
    callers pass it through ``collate_packed`` to build batches.
    """

    def __init__(
        self,
        shard_paths: list,
        block_size: int,
        n_chunks: int = 8,
        seed: int = 12345,
        shuffle: bool = True,
        wrap: bool = True,
    ):
        if not shard_paths:
            raise ValueError("PackedTokenDataset got an empty shard list")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._shard_paths = list(shard_paths)
        self._block_size = int(block_size)
        self._n_chunks = max(1, int(n_chunks))
        self._seed = int(seed)
        self._shuffle = bool(shuffle)
        self._wrap = bool(wrap)
        # Cursor describing the *next* block to serve, as
        # (file_idx_at_start_of_buffer, position_within_buffer).
        # Updated each yield; checkpointed by callers via state_dict().
        self._cursor: Tuple[int, int] = (0, 0)
        # Optional one-shot resume target consumed at the next __iter__ call.
        self._start_state: Optional[Tuple[int, int]] = None

    def state_dict(self) -> dict:
        """Return a serializable snapshot of the iterator position.

        The snapshot describes the *next* block that would be served, so calling
        ``load_state_dict`` on a fresh dataset and then iterating yields exactly
        the same sequence as continuing the original iterator from this point
        (assuming the same shard files and seed).
        """
        file_idx, buffer_pos = self._cursor
        return {
            "file_idx": int(file_idx),
            "buffer_pos": int(buffer_pos),
            "seed": int(self._seed),
            "n_chunks": int(self._n_chunks),
            "n_shards": len(self._shard_paths),
            "shuffle": bool(self._shuffle),
            "wrap": bool(self._wrap),
        }

    def load_state_dict(self, state: dict) -> None:
        """Set up the dataset to resume from ``state`` on the next ``iter()``."""
        if state is None:
            self._start_state = None
            self._cursor = (0, 0)
            return
        # Sanity-check that the dataset shape matches what was checkpointed.
        # Mismatches don't necessarily prevent resume, but they mean the RNG
        # fast-forward will land on a different permutation than the original
        # run, so we surface a clear warning.
        for key, want in (
            ("seed", self._seed),
            ("n_chunks", self._n_chunks),
            ("shuffle", self._shuffle),
            ("wrap", self._wrap),
        ):
            if key in state and state[key] != want:
                print(
                    f"  [packed] WARNING: resume state {key}={state[key]} differs from "
                    f"current {key}={want}; data ordering after resume will diverge."
                )
        if "n_shards" in state and state["n_shards"] != len(self._shard_paths):
            print(
                f"  [packed] WARNING: resume state n_shards={state['n_shards']} differs from "
                f"current {len(self._shard_paths)}; data ordering after resume will diverge."
            )
        file_idx = int(state.get("file_idx", 0))
        buffer_pos = int(state.get("buffer_pos", 0))
        self._start_state = (file_idx, buffer_pos)
        self._cursor = (file_idx, buffer_pos)

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        record_size = self._block_size + 1
        n_files = len(self._shard_paths)

        # One-shot consume of the resume target.
        if self._start_state is not None:
            start_file_idx, start_buffer_pos = self._start_state
            self._start_state = None
        else:
            start_file_idx, start_buffer_pos = 0, 0

        # Fast-forward the RNG to the saved buffer's starting state by
        # replaying one ``rng.permutation(n_all)`` per skipped buffer. This is
        # cheap relative to actual training (each call is O(n_all) host work).
        fwd_idx = 0
        while fwd_idx < start_file_idx:
            buffer_end_fwd = min(fwd_idx + self._n_chunks, n_files)
            if buffer_end_fwd <= fwd_idx:
                break
            paths = self._shard_paths[fwd_idx:buffer_end_fwd]
            mmaps = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
            blocks_per = [m.size // record_size for m in mmaps]
            n_all_fwd = int(sum(blocks_per))
            del mmaps
            if n_all_fwd > 0 and self._shuffle:
                rng.permutation(n_all_fwd)
            fwd_idx = buffer_end_fwd

        file_idx = start_file_idx
        self._cursor = (file_idx, start_buffer_pos)

        while True:
            if file_idx >= n_files:
                if not self._wrap:
                    return
                file_idx = 0

            buffer_end = min(file_idx + self._n_chunks, n_files)
            paths = self._shard_paths[file_idx:buffer_end]
            mmaps = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
            blocks_per = [m.size // record_size for m in mmaps]
            n_all = int(sum(blocks_per))
            if n_all == 0:
                file_idx = buffer_end
                self._cursor = (file_idx, 0)
                continue

            order = rng.permutation(n_all) if self._shuffle else np.arange(n_all)
            cum = np.cumsum([0] + blocks_per[:-1]) if blocks_per else np.array([0])

            for pos in range(start_buffer_pos, n_all):
                bi = int(order[pos])
                cid = int(np.searchsorted(cum, bi, side="right") - 1)
                local = bi - int(cum[cid])
                start = local * record_size
                # Cursor reflects the next block to serve, so a checkpoint
                # taken after the consumer commits this block resumes at
                # pos+1.
                self._cursor = (file_idx, pos + 1)
                yield np.asarray(mmaps[cid][start : start + record_size])

            start_buffer_pos = 0
            file_idx = buffer_end
            self._cursor = (file_idx, 0)


def _load_packed_split(
    tokenized_dir: str,
    split: str,
    block_size: int,
    n_chunks: int,
    seed: int,
    shuffle: bool,
    wrap: bool,
) -> Tuple[PackedTokenDataset, Optional[dict]]:
    """Discover shard files for a split and build a PackedTokenDataset.

    Prefers the per-split ``meta.json`` written by dataset_preprocessing.py for
    deterministic shard ordering and tokenizer metadata; falls back to globbing
    ``{split}_*.bin`` if meta is missing.
    """
    split_dir = os.path.join(tokenized_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Pre-tokenized split directory not found: {split_dir}")

    meta_path = os.path.join(split_dir, "meta.json")
    meta: Optional[dict] = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        shard_paths = [os.path.join(split_dir, s["file"]) for s in meta.get("shards", [])]
    else:
        shard_paths = sorted(glob.glob(os.path.join(split_dir, f"{split}_*.bin")))

    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {split_dir} (expected meta.json or {split}_*.bin)")

    pds = PackedTokenDataset(
        shard_paths=shard_paths,
        block_size=block_size,
        n_chunks=n_chunks,
        seed=seed,
        shuffle=shuffle,
        wrap=wrap,
    )
    return pds, meta


def collate_packed(blocks: list, sequence_length: int) -> Tuple[ttml.autograd.Tensor, ttml.autograd.Tensor]:
    """Build (input, target) tensors from a list of (sequence_length+1) uint16 blocks.

    Splits each block into ``inputs[:seq]`` and ``targets[1:seq+1]`` (causal LM
    teacher-forcing layout). Casts uint16 -> uint32 since ttnn token tensors
    use UINT32 indices.

    Args:
        blocks: list of uint16 numpy arrays, each of length ``sequence_length + 1``.
        sequence_length: target sequence length L.

    Returns:
        (input_tensor [B,1,1,L] UINT32, target_tensor [B,L] UINT32)
    """
    if not blocks:
        raise ValueError("collate_packed got an empty block list")
    actual_batch_size = len(blocks)
    expected = sequence_length + 1
    arr = np.empty((actual_batch_size, expected), dtype=np.uint32)
    for i, b in enumerate(blocks):
        if b.size != expected:
            raise ValueError(f"collate_packed: block {i} has length {b.size}, expected {expected}")
        arr[i] = b.astype(np.uint32, copy=False)

    inputs_np = arr[:, :sequence_length].reshape(actual_batch_size, 1, 1, sequence_length)
    targets_np = arr[:, 1 : sequence_length + 1]

    data_tensor = ttml.autograd.Tensor.from_numpy(
        inputs_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
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


def _run_eval(
    model: "Model",
    val_pds: "PackedTokenDataset",
    eval_iters: int,
    batch_size: int,
    seq_len: int,
    attn_mask: Optional["ttml.autograd.Tensor"],
    log_step: int,
    wandb_enabled: bool,
) -> float:
    """Evaluate validation loss over up to ``eval_iters`` batches from ``val_pds``.

    Mirrors TinyLlama's ``validate(...)`` (see lit_gpt) without backprop: switches
    the model to eval mode, draws ``eval_iters`` batches, averages the per-batch
    cross-entropy, logs to W&B as ``val/loss`` and ``val/ppl`` (when enabled),
    and switches back to train mode.

    Returns the average validation loss (NaN if no batches could be drawn).
    """
    model.eval()
    val_iter = iter(val_pds)
    losses = []
    eval_t0 = time.time()
    for _ in range(eval_iters):
        blocks = []
        try:
            for _b in range(batch_size):
                blocks.append(next(val_iter))
        except StopIteration:
            break
        if not blocks:
            break

        inp, tgt = collate_packed(blocks, seq_len)
        logits = model(inp, attn_mask)
        loss = ttml.ops.loss.cross_entropy_loss(logits, tgt, reduce=ttml.ops.ReduceType.MEAN)
        losses.append(get_loss_value(loss))
        ttml.autograd.AutoContext.get_instance().reset_graph()

    eval_time = time.time() - eval_t0
    if losses:
        val_loss = float(np.mean(losses))
        val_ppl = math.exp(val_loss) if val_loss < 50 else float("inf")
        print(
            f"  [eval] step {log_step}: val_loss={val_loss:.4f} "
            f"val_ppl={val_ppl:.2f} ({len(losses)} batches, {eval_time:.1f}s)"
        )
        if wandb_enabled and _WANDB_AVAILABLE:
            wandb.log(
                {"val/loss": val_loss, "val/ppl": val_ppl, "val/eval_time_sec": eval_time},
                step=log_step,
            )
    else:
        val_loss = float("nan")
        print(f"  [eval] step {log_step}: no validation batches available")

    model.train()
    return val_loss


def train_step(
    model: Model,
    optimizer: ttml.optimizers.OptimizerBase,
    lr_scheduler: Optional[object],
    input_tokens: ttml.autograd.Tensor,
    target_tokens: ttml.autograd.Tensor,
    mask: Optional[ttml.autograd.Tensor],
    gradient_accumulator: GradientAccumulator,
    use_clip_grad_norm: bool,
    clip_grad_norm_max_norm: float,
    batch_size=None,
    memory_snapshot_fn=None,
) -> tuple:
    """Single training step with proper gradient accumulation.

    Args:
        lr_scheduler: Optional stateful LR scheduler with a ``.step()`` method
            (e.g. ``ttml.common.schedulers.SequentialScheduler``). Stepped once
            per real optimizer update. The scheduler is the source of truth for
            the LR: ``step()`` advances its internal step counter, computes the
            new LR from its own state (``_base_lr`` snapshotted at construction
            plus child hyperparameters), and pushes that value into the
            optimizer via ``optimizer.set_lr``. The caller never has to read
            from or write to ``optimizer.lr`` here.
        mask: Optional attention mask. Pass None to let the SDPA kernel use its
              native causal mask path.
        batch_size: Optional cached batch size (if None, will extract from input_tokens)
        memory_snapshot_fn: Optional callback function to take memory snapshots.
                           Should accept a name string as argument.

    Returns:
        Tuple of (loss_float, step_time_ms, should_step)
    """
    start_time = time.time()

    # Zero gradients only when accumulator says to
    if gradient_accumulator.should_zero_grad():
        optimizer.zero_grad()

    # Forward pass
    # When mask is None, SDPA kernel uses native causal masking (AttentionMaskType::Causal)
    logits = model(input_tokens, mask)

    # Compute loss
    loss = ttml.ops.loss.cross_entropy_loss(logits, target_tokens, reduce=ttml.ops.ReduceType.MEAN)

    # Scale loss for gradient accumulation
    loss = gradient_accumulator.scale(loss)

    loss_float = get_loss_value(loss)

    profiler_marker(None, "forward_pass_done")

    # Memory snapshot after forward pass
    if memory_snapshot_fn:
        memory_snapshot_fn("FORWARD_PASS")

    # Backward pass
    loss.backward(False)

    profiler_marker(None, "backward_pass_done")

    # Memory snapshot after backward pass
    if memory_snapshot_fn:
        memory_snapshot_fn("BACKWARD_PASS")

    # Reset computation graph after backward
    ttml.autograd.AutoContext.get_instance().reset_graph()

    # Get number of samples for accumulator update
    # Use cached batch_size if provided to avoid shape() call
    samples = batch_size if batch_size is not None else input_tokens.shape()[0]

    # Update accumulator
    gradient_accumulator.update(loss_float, samples)

    # Check if we should step the optimizer
    should_step = gradient_accumulator.should_step()

    if should_step:
        # Gradient clipping
        if use_clip_grad_norm:
            # Use ttml.core.clip_grad_norm which works with model parameters directly
            ttml.core.clip_grad_norm(
                model.parameters(),
                clip_grad_norm_max_norm,
                2.0,  # p_norm_type (L2 norm)
                False,  # error_if_nonfinite - set False to avoid errors on NaN
            )

        profiler_marker(None, "gradient_sync_done")

        # Optimizer step
        optimizer.step()

        profiler_marker(None, "optimizer_step_done")

        # Step the LR scheduler (it calls optimizer.set_lr internally).
        if lr_scheduler is not None:
            lr_scheduler.step()

    step_time = (time.time() - start_time) * 1000  # Convert to ms
    return loss_float, step_time, should_step


def parse_model_config(yaml_config: dict) -> ModelConfig:
    """Parse model config from YAML"""
    # The YAML has a "transformer_config" top-level key
    transformer_config = yaml_config.get("transformer_config", {})
    config = ModelConfig()

    config.model_type = transformer_config.get("model_type", config.model_type)
    config.model_path = transformer_config.get("model_path", config.model_path)

    # Common fields shared between model types
    config.vocab_size = transformer_config.get("vocab_size", config.vocab_size)
    config.embedding_dim = transformer_config.get("embedding_dim", config.embedding_dim)
    config.num_blocks = transformer_config.get("num_blocks", config.num_blocks)
    config.num_heads = transformer_config.get("num_heads", config.num_heads)
    config.dropout_prob = transformer_config.get("dropout_prob", config.dropout_prob)
    config.max_sequence_length = transformer_config.get("max_sequence_length", config.max_sequence_length)

    if "runner_type" in transformer_config:
        config.runner_type = ttml.models.RunnerType.from_string(transformer_config["runner_type"])

    if "weight_tying" in transformer_config:
        config.weight_tying = ttml.models.WeightTyingType.from_string(transformer_config["weight_tying"])

    if config.model_type == "gpt2":
        # GPT2-specific fields
        config.bias = transformer_config.get("bias", config.bias)
        config.positional_embedding_type = transformer_config.get(
            "positional_embedding_type", config.positional_embedding_type
        )

        if "experimental" in transformer_config:
            experimental = transformer_config["experimental"]
            config.experimental.use_composite_layernorm = experimental.get(
                "use_composite_layernorm", config.experimental.use_composite_layernorm
            )
    elif config.model_type == "llama":
        # Llama-specific fields
        config.num_groups = transformer_config.get("num_groups", config.num_groups)
        config.theta = transformer_config.get("theta", config.theta)
        config.intermediate_dim = transformer_config.get("intermediate_dim", config.intermediate_dim)

        # RoPE NTK-aware scaling parameters (nested under rope_scaling in YAML)
        if "rope_scaling" in transformer_config:
            rope_scaling = transformer_config["rope_scaling"]
            config.scaling_factor = rope_scaling.get("scaling_factor", config.scaling_factor)
            config.high_freq_factor = rope_scaling.get("high_freq_factor", config.high_freq_factor)
            config.low_freq_factor = rope_scaling.get("low_freq_factor", config.low_freq_factor)
            config.original_context_length = rope_scaling.get("original_context_length", config.original_context_length)
    elif config.model_type == "deepseek":
        config.theta = transformer_config.get("theta", 10000.0)
        config.inter_dim = transformer_config.get("inter_dim", config.inter_dim)
        config.moe_inter_dim = transformer_config.get("moe_inter_dim", config.moe_inter_dim)
        config.n_dense_layers = transformer_config.get("n_dense_layers", config.n_dense_layers)
        config.n_routed_experts = transformer_config.get("n_routed_experts", config.n_routed_experts)
        config.n_shared_experts = transformer_config.get("n_shared_experts", config.n_shared_experts)
        config.n_activated_experts = transformer_config.get("n_activated_experts", config.n_activated_experts)
        config.n_expert_groups = transformer_config.get("n_expert_groups", config.n_expert_groups)
        config.n_limited_groups = transformer_config.get("n_limited_groups", config.n_limited_groups)
        config.score_func = transformer_config.get("score_func", config.score_func)
        config.route_scale = transformer_config.get("route_scale", config.route_scale)
        config.q_lora_rank = transformer_config.get("q_lora_rank", config.q_lora_rank)
        config.kv_lora_rank = transformer_config.get("kv_lora_rank", config.kv_lora_rank)
        config.qk_nope_head_dim = transformer_config.get("qk_nope_head_dim", config.qk_nope_head_dim)
        config.qk_rope_head_dim = transformer_config.get("qk_rope_head_dim", config.qk_rope_head_dim)
        config.v_head_dim = transformer_config.get("v_head_dim", config.v_head_dim)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    return config


def sample_greedy(
    model: Model,
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
        model: Trained model (NanoGPT or Llama)
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
    # Set model to eval mode
    model.eval()

    # Reset graph before inference to ensure clean state
    ttml.autograd.AutoContext.get_instance().reset_graph()

    # Cache device to avoid repeated lookups
    device = ttml.autograd.AutoContext.get_instance().get_device()

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

        # Forward pass with causal mask
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
            reshaped = ttnn.reshape(sliced_tensor, [logits_shape[0], 1, 1, logits_shape[4]])
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
            reshaped = ttnn.reshape(sliced_tensor, [logits_shape[0], 1, 1, logits_shape[3]])
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
                    threshold_tensor = ttnn.slice(topk_values, [0, 0, 0, top_k_val - 1], [1, 1, 1, top_k_val])
                    # threshold_tensor shape: [1, 1, 1, 1]
                    # Use threshold_tensor directly - ttnn.lt() will automatically broadcast
                    # This avoids extracting scalar and recreating tensor with full_like

                    # Create mask: values below threshold should be masked
                    # Broadcasting happens automatically: [1,1,1,1] vs [1,1,1,vocab_size]
                    topk_mask = ttnn.lt(last_logits_ttnn, threshold_tensor)

                    # Apply mask: set values below threshold to -1e9
                    filter_value_tensor = ttnn.full_like(last_logits_ttnn, -1e9, dtype=ttnn.bfloat16)
                    filtered_logits_ttnn = ttnn.where(topk_mask, filter_value_tensor, last_logits_ttnn)

                    # Cleanup intermediate tensors
                    ttnn.deallocate(topk_values)
                    ttnn.deallocate(topk_indices)
                    ttnn.deallocate(threshold_tensor)
                    ttnn.deallocate(topk_mask)
                    ttnn.deallocate(filter_value_tensor)

                    # Convert back to ttml.autograd.Tensor
                    last_logits = ttml.autograd.Tensor(filtered_logits_ttnn, False)

            # Use ttml sampling operation
            sampled_tensor = ttml.ops.sample.sample_op(last_logits, temperature, seed, None)  # logits_padding_mask

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


def create_model_from_config(model_config: ModelConfig) -> Model:
    """Create a model from ModelConfig, dispatching on model_type.

    Converts universal ModelConfig field names to model-specific config fields.

    Args:
        model_config: Universal model configuration

    Returns:
        A NanoGPT, Llama, or DeepSeek model instance
    """
    if model_config.model_type == "gpt2":
        nanogpt_exp_config = NanoGPTExperimentalConfig(
            use_composite_layernorm=model_config.experimental.use_composite_layernorm,
        )
        nanogpt_config = NanoGPTConfig(
            vocab_size=model_config.vocab_size,
            block_size=model_config.max_sequence_length,
            n_embd=model_config.embedding_dim,
            n_layer=model_config.num_blocks,
            n_head=model_config.num_heads,
            dropout=model_config.dropout_prob,
            bias=model_config.bias,
            runner_type=model_config.runner_type,
            weight_tying=model_config.weight_tying,
            positional_embedding_type=model_config.positional_embedding_type,
            experimental=nanogpt_exp_config,
        )
        return create_nanogpt(nanogpt_config)
    elif model_config.model_type == "llama":
        if model_config.num_groups <= 0:
            raise ValueError("model_config.num_groups must be a positive integer.")
        if model_config.num_heads % model_config.num_groups != 0:
            raise ValueError("model_config.num_heads must be divisible by model_config.num_groups.")
        rope_scaling_config = LlamaRopeScalingConfig(
            scaling_factor=model_config.scaling_factor,
            high_freq_factor=model_config.high_freq_factor,
            low_freq_factor=model_config.low_freq_factor,
            original_context_length=model_config.original_context_length,
        )
        llama_config = LlamaConfig(
            hidden_size=model_config.embedding_dim,
            intermediate_size=model_config.intermediate_dim,
            num_hidden_layers=model_config.num_blocks,
            num_attention_heads=model_config.num_heads,
            num_key_value_heads=model_config.num_groups,
            vocab_size=model_config.vocab_size,
            max_position_embeddings=model_config.max_sequence_length,
            rope_theta=model_config.theta,
            attention_dropout=model_config.dropout_prob,
            mlp_dropout=model_config.dropout_prob,
            runner_type=model_config.runner_type,
            weight_tying=model_config.weight_tying,
            rope_scaling=rope_scaling_config,
        )
        return Llama(llama_config)
    elif model_config.model_type == "deepseek":
        inter_dim = model_config.inter_dim
        if inter_dim is None:
            inter_dim = ((4 * model_config.embedding_dim * 2) // 3 + 255) // 256 * 256
        deepseek_config = DeepSeekConfig(
            vocab_size=model_config.vocab_size,
            dim=model_config.embedding_dim,
            inter_dim=inter_dim,
            moe_inter_dim=model_config.moe_inter_dim,
            n_layers=model_config.num_blocks,
            n_dense_layers=model_config.n_dense_layers,
            n_heads=model_config.num_heads,
            n_routed_experts=model_config.n_routed_experts,
            n_shared_experts=model_config.n_shared_experts,
            n_activated_experts=model_config.n_activated_experts,
            n_expert_groups=model_config.n_expert_groups,
            n_limited_groups=model_config.n_limited_groups,
            score_func=model_config.score_func,
            route_scale=model_config.route_scale,
            q_lora_rank=model_config.q_lora_rank,
            kv_lora_rank=model_config.kv_lora_rank,
            qk_nope_head_dim=model_config.qk_nope_head_dim,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            v_head_dim=model_config.v_head_dim,
            max_seq_len=model_config.max_sequence_length,
            rope_theta=model_config.theta,
            runner_type=model_config.runner_type,
        )
        return DeepSeek(deepseek_config)
    else:
        raise ValueError(f"Unsupported model type: {model_config.model_type}")


def _tensor_to_numpy_entry(tensor: "ttml.autograd.Tensor") -> dict:
    """Serialize a single ttml/ttnn tensor to a dict of numpy + metadata.

    Mirrors the format used for ``model_state`` so model and optimizer tensors
    share the same on-disk layout.
    """
    # ttml.autograd.Tensor and ttnn.Tensor expose the same get_value()/get_layout()
    # surface through this code path; both are accepted here.
    inner = tensor.get_value() if hasattr(tensor, "get_value") else tensor
    layout = inner.get_layout()
    # prefer_half=True reads bf16 storage directly, avoiding a device-side float32 typecast
    # (and its persistent cache in AutocastTensor); the cast to float32 is done on the host.
    numpy_array = tensor.to_numpy(cast_on_device=False)
    return {
        "data": numpy_array,
        "layout": layout.value if hasattr(layout, "value") else str(layout),
        "shape": numpy_array.shape,
    }


def _numpy_entry_to_tensor(entry) -> "ttml.autograd.Tensor":
    """Inverse of ``_tensor_to_numpy_entry``: rebuild a bf16 ttml tensor."""
    if isinstance(entry, dict):
        numpy_array = entry["data"]
        layout_str = entry.get("layout", "TILE")
    else:
        numpy_array = entry
        layout_str = "TILE"
    if numpy_array.dtype != np.dtype("float32"):
        numpy_array = numpy_array.astype(np.float32)
    if layout_str == "ROW_MAJOR" or "ROW_MAJOR" in str(layout_str):
        layout = ttnn.Layout.ROW_MAJOR
    else:
        layout = ttnn.Layout.TILE
    numpy_bfloat16 = numpy_array.astype(ml_dtypes.bfloat16)
    return ttml.autograd.Tensor.from_numpy(numpy_bfloat16, layout=layout, new_type=ttnn.DataType.BFLOAT16)


# Marker keys used to round-trip non-primitive optimizer state values through
# pickle. Plain primitives (bool/int/float/str) are stored as-is.
_OPT_KIND_TENSOR = "__ttml_tensor__"
_OPT_KIND_NAMED_PARAMS = "__ttml_named_parameters__"


def _serialize_optimizer_state(
    state: dict,
) -> dict:
    """Convert an optimizer ``get_state_dict()`` into a pickle-safe dict.

    AdamW returns ``{"steps": int, "exp_avg": NamedParameters, "exp_avg_sq":
    NamedParameters, "amsgrad": bool, ...}``. NamedParameters values can't be
    pickled directly, so we walk the dict and convert any tensor / NamedParameters
    leaves to a tagged numpy form that ``_deserialize_optimizer_state`` reverses.
    """
    out = {}
    for key, value in state.items():
        # NamedParameters (dict-like of name -> tensor)
        if isinstance(value, ttml.NamedParameters):
            out[key] = {
                "kind": _OPT_KIND_NAMED_PARAMS,
                "params": {k: _tensor_to_numpy_entry(v) for k, v in value.items()},
            }
        elif isinstance(value, (ttml.autograd.Tensor, ttnn.Tensor)):
            out[key] = {
                "kind": _OPT_KIND_TENSOR,
                "tensor": _tensor_to_numpy_entry(value),
            }
        else:
            # Primitive (or something pickle handles natively).
            out[key] = value
    return out


def _deserialize_optimizer_state(
    state: dict,
) -> dict:
    """Inverse of ``_serialize_optimizer_state``."""
    out = {}
    for key, value in state.items():
        if isinstance(value, dict) and "kind" in value:
            if value["kind"] == _OPT_KIND_NAMED_PARAMS:
                rebuilt = {k: _numpy_entry_to_tensor(v) for k, v in value["params"].items()}
                out[key] = ttml.NamedParameters(rebuilt)
            elif value["kind"] == _OPT_KIND_TENSOR:
                out[key] = _numpy_entry_to_tensor(value["tensor"])
            else:
                out[key] = value
        else:
            out[key] = value
    return out


def save_checkpoint(
    checkpoint_path: str,
    step: int,
    model: Model,
    tokenizer: CharTokenizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    optimizer: Optional["ttml.optimizers.OptimizerBase"] = None,
    lr_scheduler: Optional[object] = None,
    train_iter_state: Optional[dict] = None,
) -> str:
    """Save model checkpoint to pickle file.

    Args:
        checkpoint_path: Path to save checkpoint (will add .pkl if not present)
        step: Training step number
        model: Model to save (NanoGPT or Llama)
        tokenizer: Tokenizer to save
        model_config: Model configuration
        training_config: Training configuration
        optimizer: Optional optimizer whose state (Adam moments, step counter,
            etc.) should be persisted. If omitted, only the model is saved and
            resume will start optimizer state from scratch.
        lr_scheduler: Optional LR scheduler (any object with
            ``get_state_dict()``). Persisting this lets resume reproduce the
            exact LR trajectory of an interrupted run; without it the schedule
            is fast-forwarded by ``step`` calls (close but not bit-exact).
        train_iter_state: Optional packed-token iterator cursor (from
            ``PackedTokenDataset.state_dict()``) so resume continues from the
            same data position.

    Returns:
        Path to saved checkpoint file
    """
    # Ensure .pkl extension
    if not checkpoint_path.endswith(".pkl"):
        checkpoint_path = f"{checkpoint_path}.pkl"

    # Save model parameters
    model_state = {}
    for name, param in model.parameters().items():
        model_state[name] = _tensor_to_numpy_entry(param.tensor)

    optimizer_state = None
    if optimizer is not None:
        optimizer_state = _serialize_optimizer_state(optimizer.get_state_dict())

    lr_scheduler_state = None
    if lr_scheduler is not None:
        try:
            lr_scheduler_state = lr_scheduler.get_state_dict()
        except Exception as e:
            print(f"  WARNING: failed to serialize LR scheduler state ({e}); skipping")

    saved_scheduler_type = training_config.scheduler_type if training_config is not None else None

    checkpoint = {
        "step": step,
        "model_state": model_state,
        "tokenizer": tokenizer,
        "model_config": model_config,
        "training_config": training_config,
        "optimizer_state": optimizer_state,
        "lr_scheduler_state": lr_scheduler_state,
        "scheduler_type": saved_scheduler_type,
        "train_iter_state": train_iter_state,
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
) -> Tuple[Model, CharTokenizer, ModelConfig, TrainingConfig, int, Optional[dict], Optional[dict], Optional[dict],]:
    """Load model from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file (.pkl)

    Returns:
        Tuple of (model, tokenizer, model_config, training_config, step,
        optimizer_state, lr_scheduler_state, train_iter_state). The
        ``optimizer_state`` and ``lr_scheduler_state`` slots are ``None`` when
        the checkpoint predates their persistence (legacy format).
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
    optimizer_state = checkpoint.get("optimizer_state", None)
    lr_scheduler_state = checkpoint.get("lr_scheduler_state", None)
    train_iter_state = checkpoint.get("train_iter_state", None)

    # Create model from config
    model = create_model_from_config(model_config)

    # Load model parameters
    print("  Loading model parameters...")
    model_params = model.parameters()

    for name, param_data in model_state.items():
        if name not in model_params:
            print(f"    Warning: Parameter {name} not found in model, skipping")
            continue
        restored_tensor = _numpy_entry_to_tensor(param_data)
        # Update the parameter using assign() - works with both C++ and Python modules
        # The model_params dict contains references to the actual parameter tensors
        model_params[name].assign(restored_tensor)
    print(f"  Checkpoint loaded from step {step}")

    return (
        model,
        tokenizer,
        model_config,
        training_config,
        step,
        optimizer_state,
        lr_scheduler_state,
        train_iter_state,
    )


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="NanoGPT Example")

    # Default config path (relative to configs root)
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
        "--tokenized_data_dir",
        type=str,
        default="",
        help=(
            "Path to a pre-tokenized dataset root produced by dataset_preprocessing.py "
            "(e.g. /data/awliu/datasets/SlimPajama-6B-tokenized). When set, activates "
            "the TinyLlama / lit_gpt PackedDataset-style packed-token training path "
            "and overrides --data_path. Expects {tokenized_data_dir}/{train_split}/ "
            "and (optionally) {tokenized_data_dir}/{val_split}/ subdirectories."
        ),
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default=None,
        help="Name of the train split subdirectory under --tokenized_data_dir (default: train)",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default=None,
        help="Name of the validation split subdirectory under --tokenized_data_dir (default: validation)",
    )
    parser.add_argument(
        "--n_chunks",
        type=int,
        default=None,
        help=(
            "Number of shard files to mmap and shuffle together at a time in packed mode "
            "(matches lit_gpt PackedDataset n_chunks; default: 8)"
        ),
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=None,
        help=(
            "Run validation loss every N optimizer steps in packed mode "
            "(0 disables validation eval; default: 0 / from config)"
        ),
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=None,
        help="Number of validation batches per eval pass (default: 100)",
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
    parser.add_argument(
        "--track_memory",
        action="store_true",
        help="Enable memory usage tracking (prints memory stats after first iteration)",
    )
    parser.add_argument(
        "--print_summary",
        action="store_true",
        help="Print model layer-by-layer summary after creation",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging of training metrics",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (default: training_config.project_name)",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated by wandb)",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help=(
            "Stable W&B run id used for resume. If unset and --wandb_run_name is set, "
            "the run name is reused as the id so resubmits attach to the same run."
        ),
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity / team",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="W&B mode (online/offline/disabled). If unset, uses WANDB_MODE env var or wandb default.",
    )
    parser.add_argument(
        "--wandb_log_interval",
        type=int,
        default=1,
        help="Log to W&B every N optimizer steps (default: 1)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NanoGPT Example")
    print("=" * 70)
    print()

    # Require TT_METAL_RUNTIME_ROOT to be explicitly set.
    # This is needed for the runtime to find kernel files like moreh_mean.
    tt_metal_root = get_tt_metal_runtime_root()
    print(f"Using TT_METAL_RUNTIME_ROOT={tt_metal_root}")
    print()

    # Load configs using ttml.common.config utilities
    tt_train_root = f"{tt_metal_root}/tt-train"
    configs_root = f"{tt_train_root}/configs"
    try:
        print(f"Loading training config from: {args.config}")
        yaml_config = load_config(args.config, f"{configs_root}/training_configs")
        training_config = TrainingConfig(yaml_config)

        # Load model config from separate file
        # Use tt_train_root as base
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
        yaml_config = {}
        training_config = TrainingConfig()
        model_config = ModelConfig()

    # Override with command line args (only if provided)
    if args.data_path:
        training_config.data_path = args.data_path
    if args.tokenized_data_dir:
        training_config.tokenized_data_dir = args.tokenized_data_dir
    if args.train_split is not None:
        training_config.train_split = args.train_split
    if args.val_split is not None:
        training_config.val_split = args.val_split
    if args.n_chunks is not None:
        training_config.n_chunks = args.n_chunks
    if args.eval_interval is not None:
        training_config.eval_interval = args.eval_interval
    if args.eval_iters is not None:
        training_config.eval_iters = args.eval_iters
    if args.batch_size is not None:
        training_config.batch_size = args.batch_size
    if args.max_steps is not None:
        training_config.max_steps = args.max_steps
    if args.num_epochs is not None:
        training_config.num_epochs = args.num_epochs
    if args.clip_grad_norm is not None:
        training_config.use_clip_grad_norm = True
        training_config.clip_grad_norm_max_norm = args.clip_grad_norm
    if args.sequence_length is not None:
        model_config.max_sequence_length = args.sequence_length

    # Only checkpoint when explicitly requested via --model_save_path.
    # No model_path in YAML -> no checkpointing.
    if args.model_save_path:
        checkpoint_dir = os.path.dirname(args.model_save_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {args.model_save_path}_step_*.pkl")

    # Check if we're in inference-only mode (prompt + explicit model_path).
    inference_only = args.prompt and args.model_path

    # Initialize device early (needed for both training and inference)
    ttml.autograd.AutoContext.get_instance().open_device()
    ttml.autograd.AutoContext.get_instance().get_device()

    # Start memory tracking if enabled
    # Pass RunMode.NO_DISPATCH to measure memory usage of models that don't fit in device memory
    memory_guard = None
    if args.track_memory:
        print("\nMemory tracking enabled")
        memory_guard = MemoryUsageTracker.begin_capture()

    ttml.autograd.AutoContext.get_instance().set_seed(training_config.seed)
    np.random.seed(training_config.seed)

    # Handle inference-only mode: load model from checkpoint
    if inference_only:
        print("1. Loading model from checkpoint...")
        print(f"   - Model path: {args.model_path}")

        try:
            (
                model,
                tokenizer,
                model_config,
                training_config,
                loaded_step,
                _opt_state,
                _lr_scheduler_state,
                _train_iter_state,
            ) = load_model_from_checkpoint(
                args.model_path,
            )
            seq_len = model_config.max_sequence_length
            dataset = []  # Not needed for inference

            # Pre-tokenized-dataset checkpoints store a stub tokenizer (no
            # encode/decode). For --prompt we need a real one; load the HF
            # tokenizer named in the stub and swap it in.
            if isinstance(tokenizer, _PackedTokenizerStub) and tokenizer.name:
                print(f"   - Loading HF tokenizer for inference: {tokenizer.name}")
                tokenizer = _HFTokenizerAdapter(tokenizer.name)

            print(f"   - Model loaded from step {loaded_step}")
            print(f"   - Vocabulary size: {model_config.vocab_size}")
            print(f"   - Sequence length: {seq_len}")
            print(
                f"   - Model: {model_config.num_blocks} layers, {model_config.embedding_dim} embd, {model_config.num_heads} heads"
            )

        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            ttml.autograd.AutoContext.get_instance().close_device()
            return
    else:
        # Training mode: load data and create model
        # Two data paths:
        # 1) Pre-tokenized SlimPajama-style packed dataset (training_config.tokenized_data_dir).
        # 2) Legacy CharTokenizer + raw text file (training_config.data_path).
        # The tokenized path takes precedence when both are configured.
        packed_train_pds: Optional[PackedTokenDataset] = None
        packed_val_pds: Optional[PackedTokenDataset] = None
        packed_meta: Optional[dict] = None

        if training_config.tokenized_data_dir:
            print("1. Loading pre-tokenized dataset...")
            print(f"   - Tokenized data dir: {training_config.tokenized_data_dir}")
            print(f"   - Train split: {training_config.train_split}")
            print(f"   - Val split: {training_config.val_split}")
            print(f"   - n_chunks: {training_config.n_chunks}")

            seq_len = model_config.max_sequence_length

            try:
                packed_train_pds, packed_meta = _load_packed_split(
                    tokenized_dir=training_config.tokenized_data_dir,
                    split=training_config.train_split,
                    block_size=seq_len,
                    n_chunks=training_config.n_chunks,
                    seed=training_config.seed,
                    shuffle=True,
                    wrap=True,
                )
            except FileNotFoundError as e:
                print(f"Error loading tokenized train split: {e}")
                ttml.autograd.AutoContext.get_instance().close_device()
                return

            if training_config.eval_interval > 0:
                try:
                    packed_val_pds, _ = _load_packed_split(
                        tokenized_dir=training_config.tokenized_data_dir,
                        split=training_config.val_split,
                        block_size=seq_len,
                        n_chunks=training_config.n_chunks,
                        # Different seed so train and val do not co-vary if the same
                        # rng state would otherwise produce correlated permutations.
                        seed=training_config.seed + 1,
                        shuffle=False,
                        wrap=False,
                    )
                except FileNotFoundError as e:
                    print(f"Warning: validation split unavailable, disabling eval ({e})")
                    packed_val_pds = None
                    training_config.eval_interval = 0

            if packed_meta is not None:
                tok_name = packed_meta.get("tokenizer", "")
                vocab_size = int(packed_meta.get("vocab_size", model_config.vocab_size))
                bos_id = packed_meta.get("bos_id")
                eos_id = packed_meta.get("eos_id")
                total_tokens = int(packed_meta.get("total_tokens", 0))
                num_documents = int(packed_meta.get("num_documents", 0))
                print(f"   - Tokenizer: {tok_name}")
                print(f"   - Vocab size: {vocab_size}")
                print(f"   - Train tokens: {total_tokens:,}  Documents: {num_documents:,}")
                model_config.vocab_size = vocab_size
                tokenizer = _PackedTokenizerStub(name=tok_name, vocab_size=vocab_size, bos_id=bos_id, eos_id=eos_id)
            else:
                # No meta.json — keep model_config.vocab_size as-is and stub the tokenizer.
                tokenizer = _PackedTokenizerStub(name="unknown", vocab_size=model_config.vocab_size)
                print(
                    f"   - Warning: no meta.json found; relying on model_config.vocab_size="
                    f"{model_config.vocab_size}"
                )

            # ``dataset`` is referenced downstream for logging and len() reporting.
            # In tokenized mode the training loop uses next(train_iter) directly,
            # so a placeholder list is fine here.
            dataset = []
            print(f"   - Sequence length: {seq_len}")
        else:
            if not training_config.data_path:
                # Try to find Shakespeare dataset
                possible_paths = [
                    "data/shakespeare.txt",
                    "tt-train/data/shakespeare.txt",
                    "../data/shakespeare.txt",
                    os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data",
                        "shakespeare.txt",
                    ),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        training_config.data_path = path
                        break
                if not training_config.data_path:
                    print("Warning: No data path specified and Shakespeare dataset not found.")
                    print("Please specify --data_path or place shakespeare.txt in data/")
                    print(f"  Searched paths: {possible_paths}")
                    ttml.autograd.AutoContext.get_instance().close_device()
                    return

            print("1. Loading and preparing data...")
            print(f"   - Data path: {training_config.data_path}")

            # Load data
            text = read_file_to_str(training_config.data_path)
            seq_len = model_config.max_sequence_length

            # Create dataset
            dataset, tokenizer = create_dataset_from_text(text, seq_len)
            model_config.vocab_size = tokenizer.vocab_size

            print(f"   - Vocabulary size: {model_config.vocab_size}")
            print(f"   - Dataset size: {len(dataset)} samples")
            print(f"   - Sequence length: {seq_len}")

        # Check if resuming from checkpoint (auto-resume by default)
        start_step = 0
        resume_path = None
        resume_optimizer_state: Optional[dict] = None
        resume_lr_scheduler_state: Optional[dict] = None
        resume_train_iter_state: Optional[dict] = None

        if not args.fresh:
            # Auto-detect or use specified checkpoint
            if args.resume:
                resume_path = args.resume
            elif args.model_save_path:
                # Only auto-detect if a save path is configured — otherwise there's
                # nowhere to look and find_latest_checkpoint("") would be a no-op anyway.
                resume_path = find_latest_checkpoint(args.model_save_path)
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
                    resume_optimizer_state,
                    resume_lr_scheduler_state,
                    resume_train_iter_state,
                ) = load_model_from_checkpoint(resume_path)
                # Use tokenizer from checkpoint to ensure vocab consistency
                tokenizer = loaded_tokenizer
                seq_len = model_config.max_sequence_length
                print(f"   - Resumed from step {start_step}")
                if resume_optimizer_state is not None:
                    print("   - Optimizer state available (will restore after optimizer creation)")
                else:
                    print("   - WARNING: checkpoint has no optimizer state; AdamW moments " "will restart from zero")
                if resume_lr_scheduler_state is not None:
                    print("   - LR scheduler state available (will restore after scheduler creation)")
                else:
                    print(
                        "   - INFO: checkpoint has no LR scheduler state; scheduler will be "
                        "fast-forwarded by start_step calls instead"
                    )
                if resume_train_iter_state is not None:
                    print(
                        f"   - Train iterator cursor: file_idx={resume_train_iter_state.get('file_idx')}, "
                        f"buffer_pos={resume_train_iter_state.get('buffer_pos')}"
                    )
                else:
                    print(
                        "   - WARNING: checkpoint has no iterator cursor; data ordering "
                        "will restart from the beginning of the shard list"
                    )
                print(
                    f"   - Model: {model_config.num_blocks} layers, {model_config.embedding_dim} embd, {model_config.num_heads} heads"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint from {resume_path}") from e

        if not resume_path:
            print("\n2. Creating model...")
            # Round vocab size to tile boundary
            print("Overriding vocab size to be divisible by 32")
            model_config.vocab_size = round_up_to_tile(model_config.vocab_size, 32)

            # Print transformer configuration
            runner_type_str = str(model_config.runner_type).split(".")[-1]
            weight_tying_str = str(model_config.weight_tying).split(".")[-1]
            print("Transformer configuration:")
            print(f"    Vocab size: {model_config.vocab_size}")
            print(f"    Max sequence length: {model_config.max_sequence_length}")
            print(f"    Runner type: {runner_type_str}")
            print(f"    Weight tying: {weight_tying_str}")

            # Create model
            model = create_model_from_config(model_config)
            if args.print_summary:
                summary(model)

            # Count parameters
            total_params = sum(math.prod(p.shape()) for p in model.parameters().values())
            print(
                f"   - Model: {model_config.num_blocks} layers, {model_config.embedding_dim} embd, {model_config.num_heads} heads"
            )
            print(f"   - Total parameters: {total_params:,}")

        # Compute FLOPs per token for throughput reporting
        flops_per_token = 0
        if model_config.model_type == "deepseek":
            from ttml.models.deepseek import DeepSeekConfig, calculate_flops_per_token

            ds_cfg = model.config if hasattr(model, "config") and isinstance(model.config, DeepSeekConfig) else None
            if ds_cfg is not None:
                flops_per_token = calculate_flops_per_token(ds_cfg, model_config.max_sequence_length)
                print(f"   - FLOPs per token: {flops_per_token:,} ({flops_per_token/1e9:.2f}G)")

        # Memory snapshot after model creation
        if args.track_memory:
            MemoryUsageTracker.snapshot("MODEL_CREATION")

    # Check if we're in inference mode
    if args.prompt:
        # Inference mode: skip optimizer setup
        optimizer = None
        lr_scheduler = None
        print("\n3. Inference mode - skipping optimizer setup")
    else:
        print("\n3. Setting up optimizer...")
        optimizer = create_optimizer(model, yaml_config)
        print(f"   - Optimizer: {optimizer.get_name()}")
        print(f"   - Learning rate: {optimizer.get_lr()}")

        # Restore optimizer state (Adam moments, step counter, LR, betas, ...)
        # from checkpoint so resume continues with bias-correct gradient
        # statistics and at the right LR. ``AdamW::set_state_dict`` restores
        # ALL hyperparameters including ``lr`` (see
        # tt-train/sources/ttml/optimizers/adamw.cpp), so after this call
        # ``optimizer.get_lr()`` is the decayed checkpoint LR, not peak LR.
        if resume_optimizer_state is not None:
            try:
                optimizer.set_state_dict(_deserialize_optimizer_state(resume_optimizer_state))
                print(f"   - Restored optimizer state from checkpoint (LR={optimizer.get_lr():.6e})")
            except Exception as e:
                print(f"   - WARNING: failed to restore optimizer state ({e}); continuing with fresh moments")

        # Memory snapshot after optimizer creation
        if args.track_memory:
            MemoryUsageTracker.snapshot("OPTIMIZER_CREATION")

        # Build the LR scheduler. ``LinearScheduler`` / ``CosineAnnealingScheduler``
        # snapshot ``optimizer.get_lr()`` as ``_base_lr`` at construction time,
        # so on resume after the optimizer-state restore above this snapshot is
        # the decayed checkpoint LR (not peak LR). That's fine for new-format
        # checkpoints because ``lr_scheduler.set_state_dict`` (below) restores
        # the saved ``_base_lr`` (= peak LR) and overwrites the wrong snapshot.
        # Old-format checkpoints (no ``lr_scheduler_state``, fast-forward path)
        # would capture the wrong ``_base_lr`` here; if that becomes a real
        # concern, add a one-time ``optimizer.set_lr(peak_lr)`` around this
        # call before falling back to the fast-forward branch.
        print("\n4. Setting up learning rate scheduler...")
        lr_scheduler, sched_info = build_lr_scheduler(optimizer, training_config, yaml_config)
        if sched_info.get("type") == "cosine_with_warmup":
            print(f"   - Scheduler: cosine_with_warmup")
            print(f"   - Peak LR: {sched_info['peak_lr']}")
            print(f"   - Min LR (eta_min): {sched_info['eta_min']}")
            print(f"   - Warmup steps (linear 0 -> peak_lr): {sched_info['warmup_steps']}")
            print(f"   - Cosine decay steps (peak_lr -> eta_min): {sched_info['decay_steps']}")
        elif sched_info.get("type") == "warmup_linear":
            print(f"   - Scheduler: warmup_linear")
            print(f"   - Peak LR: {sched_info['peak_lr']}")
            print(f"   - End LR (1% of peak): {sched_info['end_lr']}")
            print(f"   - Warmup steps (linear 0 -> peak_lr): {sched_info['warmup_steps']}")
            print(f"   - Linear decay steps (peak_lr -> end_lr): {sched_info['decay_steps']}")
        else:
            print(f"   - Scheduler: {sched_info.get('type', training_config.scheduler_type)} (constant LR)")

        # Restore LR scheduler state from checkpoint so resume picks up exactly
        # where the interrupted run left off. The upstream scheduler base
        # class persists ``_base_lr`` in its state dict, so this overwrites
        # the (possibly-wrong) construction-time snapshot. If no scheduler
        # state is in the checkpoint (legacy format) but we are resuming
        # mid-training, fast-forward by ``start_step`` calls to land on the
        # same LR.
        #
        # Defensive optimizer-LR sync: ``_SchedulerBase.set_state_dict`` only
        # restores the scheduler's own bookkeeping; it does not push the saved
        # ``last_lr`` back into the optimizer. Normally that's fine because
        # ``optimizer.set_state_dict`` (above) already restored the same LR,
        # but we sync explicitly so a partial restore (e.g. fresh moments due
        # to deserialization failure) cannot leave optimizer.lr and
        # scheduler.last_lr disagreeing on the LR for the very next step.
        if lr_scheduler is not None:
            if resume_lr_scheduler_state is not None:
                try:
                    lr_scheduler.set_state_dict(resume_lr_scheduler_state)
                    # optimizer.set_lr(float(lr_scheduler.get_last_lr()))
                    print(f"   - Restored LR scheduler state from checkpoint " f"(LR={lr_scheduler.get_last_lr():.6e})")
                except Exception as e:
                    print(
                        f"   - WARNING: failed to restore LR scheduler state ({e}); "
                        f"fast-forwarding by {start_step} steps instead"
                    )
                    for _ in range(start_step):
                        lr_scheduler.step()
            elif start_step > 0:
                print(
                    f"   - No saved LR scheduler state; fast-forwarding scheduler by "
                    f"{start_step} steps to match resumed training position"
                )
                for _ in range(start_step):
                    lr_scheduler.step()

    # Create attention mask (needed for both training and inference)
    if inference_only:
        print("\n2. Creating attention mask...")
    else:
        print("\n5. Creating attention mask...")
    mask_np = build_causal_mask(seq_len)
    mask = ttml.autograd.Tensor.from_numpy(mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

    # Initialize W&B (training mode only)
    wandb_enabled = False
    if not args.prompt and args.wandb:
        if not _WANDB_AVAILABLE:
            print("Warning: --wandb specified but the 'wandb' package is not installed; skipping W&B logging.")
        else:
            total_params = int(sum(math.prod(p.shape()) for p in model.parameters().values()))
            wandb_config = {
                "model_type": model_config.model_type,
                "vocab_size": model_config.vocab_size,
                "embedding_dim": model_config.embedding_dim,
                "num_blocks": model_config.num_blocks,
                "num_heads": model_config.num_heads,
                "max_sequence_length": model_config.max_sequence_length,
                "dropout_prob": model_config.dropout_prob,
                "batch_size": training_config.batch_size,
                "max_steps": training_config.max_steps,
                "num_epochs": training_config.num_epochs,
                "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
                "scheduler_type": training_config.scheduler_type,
                "use_clip_grad_norm": training_config.use_clip_grad_norm,
                "clip_grad_norm_max_norm": training_config.clip_grad_norm_max_norm,
                "seed": training_config.seed,
                "optimizer": optimizer.get_name() if optimizer is not None else None,
                "learning_rate": optimizer.get_lr() if optimizer is not None else None,
                "total_params": total_params,
                "data_path": training_config.data_path,
                "resume_from": resume_path,
                "start_step": start_step,
            }
            # Stable run id so a resubmitted SLURM job attaches to the previous
            # W&B run instead of starting a new one. We prefer an explicit
            # --wandb_run_id; otherwise fall back to --wandb_run_name (which is
            # typically stable across resubmits in the sbatch wrapper).
            wandb_run_id = args.wandb_run_id or args.wandb_run_name
            wandb.init(
                project=args.wandb_project or training_config.project_name,
                id=wandb_run_id,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                config=wandb_config,
                resume="allow" if (resume_path and not args.fresh) else None,
            )
            wandb_enabled = True
            print(
                f"   - W&B logging enabled "
                f"(project={args.wandb_project or training_config.project_name}, "
                f"id={wandb_run_id or '<auto>'}, "
                f"interval={args.wandb_log_interval})"
            )

    # Training or inference mode
    if args.prompt:
        # Inference mode: skip training, go straight to inference
        print("\n6. Inference mode - skipping training")
    else:
        print("\n6. Training...")
        print()
        remaining_steps = training_config.max_steps - start_step
        print(f"Training for {remaining_steps} steps (step {start_step} to {training_config.max_steps})...")
        print(f"  - Batch size: {training_config.batch_size}")
        print(f"  - Sequence length: {seq_len}")
        print(f"  - Training data: {len(dataset)} samples")
        print(f"  - Scheduler: {training_config.scheduler_type}")
        print(f"  - Gradient accumulation steps: {training_config.gradient_accumulation_steps}")
        print(f"  - Dropout: {model_config.dropout_prob}")
        if training_config.use_clip_grad_norm:
            print(f"  - Gradient clipping: max_norm={training_config.clip_grad_norm_max_norm}")
        print()

        # Set model to training mode
        model.train()

        # Training setup
        loss_meter = LossAverageMeter()
        gradient_accumulator = GradientAccumulator(training_config.gradient_accumulation_steps)
        global_step = start_step

        # Compute peak device TFLOPS for MFU calculation
        peak_tflops = 0.0
        if flops_per_token > 0:
            peak_tflops = get_device_peak_tflops_bf16()
            print(f"  - Device peak: {peak_tflops:.1f} TFLOPS (bf16)")

        # Training loop
        start_time = time.time()
        # Cache values used in hot path
        batch_size = training_config.batch_size
        max_steps = training_config.max_steps
        dataset_len = len(dataset)
        num_devices = DeviceConfig(yaml_config).total_devices()

        # Flag to track if first iteration is complete (for memory tracking)
        is_everything_compiled = False
        # Accumulates step_time across gradient-accumulation micro-steps; reset after each optimizer step.
        macro_batch_step_time = 0.0

        # Helper for memory snapshots (only takes snapshots during first iteration)
        def memory_snapshot(name: str):
            nonlocal is_everything_compiled
            if args.track_memory and not is_everything_compiled:
                MemoryUsageTracker.snapshot(name)

        # Composite SDPA (used by DeepSeek) has no built-in causal masking,
        # so we must pass an explicit mask. Fused SDPA (GPT-2/Llama) uses
        # its native causal mode when mask is None.
        attn_mask = mask if model_config.model_type == "deepseek" else None

        # Per-step body shared by the legacy CharTokenizer epoch loop and the
        # new packed-token step-driven loop. Returns True if the outer loop
        # should stop (max_steps reached after a real optimizer step).
        def _run_step(input_tokens, target_tokens, actual_batch_size, epoch_idx):
            nonlocal global_step, is_everything_compiled, macro_batch_step_time

            profiler_marker(None, "dataloader_step_done")

            loss_float, step_time, should_step = train_step(
                model,
                optimizer,
                lr_scheduler,
                input_tokens,
                target_tokens,
                attn_mask,
                gradient_accumulator,
                training_config.use_clip_grad_norm,
                training_config.clip_grad_norm_max_norm,
                batch_size=actual_batch_size,
                memory_snapshot_fn=memory_snapshot if args.track_memory else None,
            )

            macro_batch_step_time += step_time

            if not should_step:
                return False

            global_step += 1
            avg_loss = gradient_accumulator.average_loss()
            loss_meter.update(avg_loss)

            global_tokens = actual_batch_size * num_devices * seq_len * training_config.gradient_accumulation_steps
            tps = global_tokens / (macro_batch_step_time / 1000.0) if macro_batch_step_time > 0 else 0.0
            achieved_tflops = None
            mfu = None
            if flops_per_token > 0 and macro_batch_step_time > 0:
                achieved_tflops = tps * flops_per_token / 1e12
                mfu_str = ""
                if peak_tflops > 0:
                    mfu = achieved_tflops / peak_tflops * 100.0
                    mfu_str = f", MFU: {mfu:.1f}%"
                print(
                    f"Step: {global_step}, Loss: {avg_loss:.6f}, Time: {macro_batch_step_time:.2f} ms, "
                    f"TPS: {tps:.0f}, TFLOPS: {achieved_tflops:.2f}{mfu_str}"
                )
            else:
                print(
                    f"Step: {global_step}, Loss: {avg_loss:.6f}, Time: {macro_batch_step_time:.2f} ms, TPS: {tps:.0f}"
                )

            if wandb_enabled and (global_step % args.wandb_log_interval == 0 or global_step >= max_steps):
                log_data = {
                    "train/loss": avg_loss,
                    "train/loss_running_avg": loss_meter.average(),
                    "train/learning_rate": optimizer.get_lr(),
                    "perf/step_time_ms": macro_batch_step_time,
                    "perf/tokens_per_sec": tps,
                    "system/epoch": epoch_idx,
                }
                if achieved_tflops is not None:
                    log_data["perf/tflops"] = achieved_tflops
                if mfu is not None:
                    log_data["perf/mfu_pct"] = mfu
                wandb.log(log_data, step=global_step)

            if args.model_save_path and global_step % training_config.model_save_interval == 0:
                # Capture iterator cursor (packed mode only) so resume continues
                # from the next block instead of replaying from the start.
                iter_state = packed_train_pds.state_dict() if packed_train_pds is not None else None
                save_checkpoint(
                    f"{args.model_save_path}_step_{global_step}.pkl",
                    global_step,
                    model,
                    tokenizer,
                    model_config,
                    training_config,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_iter_state=iter_state,
                )

            macro_batch_step_time = 0.0
            gradient_accumulator.reset()

            if model_config.model_type == "deepseek" and hasattr(model, "get_moe_layers"):
                for moe_layer in model.get_moe_layers():
                    moe_layer.update_expert_bias()

            if args.track_memory and not is_everything_compiled:
                is_everything_compiled = True
                MemoryUsageTracker.end_capture("FIRST_ITERATION_COMPLETE")
                MemoryUsageTracker.print_memory_usage()
                MemoryUsageTracker.clear()
                if memory_guard:
                    memory_guard.release()

            profiler_marker(None, f"iteration_{global_step}", dump_results=True)
            if global_step == start_step + 1:
                profiler_marker(None, "compilation_finished")

            # Periodic validation eval (packed mode only).
            if (
                packed_val_pds is not None
                and training_config.eval_interval > 0
                and global_step % training_config.eval_interval == 0
            ):
                _run_eval(
                    model,
                    packed_val_pds,
                    eval_iters=training_config.eval_iters,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    attn_mask=attn_mask,
                    log_step=global_step,
                    wandb_enabled=wandb_enabled,
                )

            return global_step >= max_steps

        if packed_train_pds is not None:
            # Prime the iterator with the saved cursor (no-op on a fresh start).
            packed_train_pds.load_state_dict(resume_train_iter_state)
            # TinyLlama / lit_gpt PackedDataset-style step-driven loop.
            train_iter = iter(packed_train_pds)
            while global_step < max_steps:
                blocks = []
                try:
                    for _ in range(batch_size):
                        blocks.append(next(train_iter))
                except StopIteration:
                    if not blocks:
                        print("  [packed] training iterator exhausted; ending training")
                        break
                    # Partial trailing batch (only happens if wrap=False, which we
                    # don't use for train) - drop and exit cleanly.
                    print("  [packed] dropping trailing partial batch and ending training")
                    break

                input_tokens, target_tokens = collate_packed(blocks, seq_len)
                if _run_step(input_tokens, target_tokens, len(blocks), epoch_idx=0):
                    break
        else:
            for epoch in range(training_config.num_epochs):
                # Shuffle indices,
                # avoids copying token data unlike shuffling a list of tuples.
                indices = np.arange(dataset_len, dtype=np.int64)
                np.random.shuffle(indices)

                for batch_start in range(0, dataset_len, batch_size):
                    batch_end = min(batch_start + batch_size, dataset_len)

                    batch_samples = [dataset[i] for i in indices[batch_start:batch_end]]
                    input_tokens, target_tokens = collate_fn(batch_samples, seq_len)
                    actual_batch_size = batch_end - batch_start

                    if _run_step(input_tokens, target_tokens, actual_batch_size, epoch_idx=epoch):
                        break

                if global_step >= max_steps:
                    break

        # Final validation eval after training (packed mode only).
        if packed_val_pds is not None and training_config.eval_iters > 0:
            _run_eval(
                model,
                packed_val_pds,
                eval_iters=training_config.eval_iters,
                batch_size=batch_size,
                seq_len=seq_len,
                attn_mask=attn_mask,
                log_step=global_step,
                wandb_enabled=wandb_enabled,
            )

        # Save final checkpoint after training
        if args.model_save_path:
            final_checkpoint_path = f"{args.model_save_path}_final.pkl"
            iter_state = packed_train_pds.state_dict() if packed_train_pds is not None else None
            save_checkpoint(
                final_checkpoint_path,
                global_step,
                model,
                tokenizer,
                model_config,
                training_config,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_iter_state=iter_state,
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

        if wandb_enabled:
            wandb.summary["final_loss"] = loss_meter.average()
            wandb.summary["total_steps"] = global_step
            wandb.summary["total_time_sec"] = total_time
            wandb.finish()

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
            seq_len,
            mask,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # Close device
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
