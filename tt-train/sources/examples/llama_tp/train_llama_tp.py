# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP training script for the Python Llama model using SFTTrainer infrastructure.

Uses the new layout-aware dispatch layer (``ttml.distributed``) to distribute
the Python Llama model with tensor parallelism.  Model code is written once;
distribution is handled by module rules and op dispatch.

This script demonstrates how to use the SFTTrainer infrastructure for distributed
training with TP (tensor parallelism) and DP (data parallelism) support.

Example:
    python train_llama_tp.py -c training_shakespeare_tinyllama_tp_galaxy.yaml \
                             --data_path data/shakespeare.txt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from functools import partial
from typing import Any, Callable, Optional, Tuple

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig
from ttml.common.utils import round_up_to_tile, get_tt_metal_home
from ttml.common.config import load_config, DeviceConfig
from ttml.common.data import CharTokenizer, build_causal_mask

# SFT Trainer infrastructure
from ttml.trainers import SFTConfig, SFTTrainer, TrainerCallback
from ttml.datasets import Batch, InMemoryDataloader

# Layout-aware dispatch layer
import ttml.distributed
from ttml.distributed import (
    Layout,
    Shard,
    Replicate,
    distribute_module,
    sync_gradients,
    init_ops,
)
from ttml.distributed.debug import DispatchTracer, dispatch_trace

# Memory profiling
MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker


def print_memory_stats(label: str):
    """Print DRAM and L1 memory usage for a named snapshot.

    Note: Each snapshot shows memory for that SEGMENT only (since last snapshot).
    - peak: peak memory during this segment
    - total_allocations: memory allocated during this segment
    - total_deallocations: memory freed during this segment
    - net change: allocations - deallocations (memory retained)
    """
    try:
        dram = MemoryUsageTracker.get_dram_usage(label)
        l1 = MemoryUsageTracker.get_l1_usage(label)

        MB = 1024 * 1024
        alloc_mb = dram.total_allocations / MB
        dealloc_mb = dram.total_deallocations / MB
        net_mb = alloc_mb - dealloc_mb
        peak_mb = dram.peak / MB

        print(
            f"   [{label}] DRAM: alloc={alloc_mb:.1f} MB, dealloc={dealloc_mb:.1f} MB, "
            f"net={net_mb:+.1f} MB, segment_peak={peak_mb:.1f} MB"
        )
    except Exception as e:
        print(f"   [{label}] Memory stats unavailable: {e}")


# ---------------------------------------------------------------------------
# Config parsing (reuses the NanoGPT approach for Llama-specific fields)
# ---------------------------------------------------------------------------


def parse_training_config(yaml_config: dict) -> dict:
    return yaml_config.get("training_config", {})


def parse_model_config(yaml_config: dict) -> LlamaConfig:
    tc = yaml_config.get("transformer_config", {})

    rope_scaling = LlamaRopeScalingConfig(
        scaling_factor=tc.get("scaling_factor", 0.0),
        high_freq_factor=tc.get("high_freq_factor", 4.0),
        low_freq_factor=tc.get("low_freq_factor", 1.0),
        original_context_length=tc.get("original_context_length", 0),
    )

    runner_str = tc.get("runner_type", "default")
    if runner_str == "memory_efficient":
        runner_type = ttml.models.RunnerType.MemoryEfficient
    else:
        runner_type = ttml.models.RunnerType.Default

    wt_str = tc.get("weight_tying", "disabled")
    if wt_str == "enabled":
        weight_tying = ttml.models.WeightTyingType.Enabled
    else:
        weight_tying = ttml.models.WeightTyingType.Disabled

    vocab_size = round_up_to_tile(tc.get("vocab_size", 256), 32)

    return LlamaConfig(
        hidden_size=tc.get("embedding_dim", 384),
        intermediate_size=tc.get("intermediate_dim", None),
        num_hidden_layers=tc.get("num_blocks", 6),
        num_attention_heads=tc.get("num_heads", 6),
        num_key_value_heads=tc.get("num_groups", 2),
        vocab_size=vocab_size,
        max_position_embeddings=tc.get("max_sequence_length", 256),
        rope_theta=tc.get("theta", 10000.0),
        attention_dropout=tc.get("dropout_prob", 0.0),
        mlp_dropout=tc.get("dropout_prob", 0.0),
        runner_type=runner_type,
        weight_tying=weight_tying,
        rope_scaling=rope_scaling,
    )


# ---------------------------------------------------------------------------
# Data Parallel Model Wrapper
# ---------------------------------------------------------------------------


class _GradSyncFunction(ttml.autograd.Function):
    """Custom autograd function that syncs gradients in backward pass."""

    @staticmethod
    def forward(ctx, output, model, dp_axis):
        """Forward pass - just returns the output unchanged."""
        ctx.model = model
        ctx.dp_axis = dp_axis
        return output.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass - sync gradients across DP axis, then return grad unchanged."""
        sync_gradients(ctx.model, cluster_axes=[ctx.dp_axis])
        return grad_output


class DataParallelModel:
    """Wrapper that adds automatic gradient synchronization for data parallel training.

    This wrapper is orthogonal to the trainer - it wraps any model and adds
    automatic gradient synchronization across the DP axis at the end of backward.

    The gradient sync happens automatically when backward() is called on the loss,
    because the wrapper inserts a custom autograd function that syncs gradients
    in its backward pass.

    Example::

        model = Llama(config)
        model = distribute_module(model, mesh_device, tp_policy)  # TP sharding
        model = DataParallelModel(model, dp_axis=0)  # DP wrapper

        # Use with SFTTrainer - no special hooks needed
        trainer = SFTTrainer(model, ...)
    """

    def __init__(self, model: Any, dp_axis: int):
        """
        Args:
            model: The underlying model to wrap.
            dp_axis: The mesh axis for data parallelism (gradient sync axis).
        """
        self._model = model
        self._dp_axis = dp_axis

    def __call__(self, *args, **kwargs):
        """Forward pass with automatic gradient sync in backward."""
        output = self._model(*args, **kwargs)
        # Insert gradient sync node into the autograd graph
        return _GradSyncFunction.apply(output, self._model, self._dp_axis)

    def train(self):
        """Set model to training mode."""
        self._model.train()

    def eval(self):
        """Set model to evaluation mode."""
        self._model.eval()

    def parameters(self):
        """Return model parameters."""
        return self._model.parameters()

    @property
    def config(self):
        """Return model config (for gradient checkpointing support)."""
        return self._model.config

    @config.setter
    def config(self, value):
        """Set model config."""
        self._model.config = value

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying model."""
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Dataset helpers (same as NanoGPT example)
# ---------------------------------------------------------------------------


class InMemoryTokenDataset:
    """Simple dataset that yields (input_tokens, target_tokens) pairs."""

    def __init__(self, tokens: np.ndarray, seq_length: int):
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self) -> int:
        if len(self.tokens) <= self.seq_length:
            return 0
        return len(self.tokens) - self.seq_length

    def __getitem__(self, index: int):
        return {
            "input_ids": self.tokens[index : index + self.seq_length],
            "labels": self.tokens[index + 1 : index + self.seq_length + 1],
        }


def create_dataset(text: str, seq_length: int):
    tokenizer = CharTokenizer(text)
    tokens = np.array(tokenizer.encode(text), dtype=np.uint32)
    return InMemoryTokenDataset(tokens, seq_length), tokenizer


def distributed_collate_fn(
    examples: list,
    seq_length: int,
    mesh_device: Any = None,
    dp_axis: Optional[int] = None,
) -> Batch:
    """Collate samples into a Batch with distributed tensor support.

    This collate function creates Batch objects compatible with SFTTrainer,
    with optional DP sharding when dp_axis is provided.

    Use with functools.partial to bind seq_length, mesh_device, and dp_axis::

        collate = partial(distributed_collate_fn, seq_length=1024,
                          mesh_device=mesh_device, dp_axis=0)
        dataloader = InMemoryDataloader(dataset, collate, batch_size=8)

    Args:
        examples: List of dicts with "input_ids" and "labels" keys.
        seq_length: Sequence length for the batch.
        mesh_device: Mesh device for distributed tensors.
        dp_axis: Data parallel axis for batch sharding (None to disable).

    Returns:
        Batch object with input_ids, labels, and loss_mask tensors.
    """
    batch_size = len(examples)

    input_ids_np = np.zeros((batch_size, 1, 1, seq_length), dtype=np.uint32)
    labels_np = np.zeros((batch_size, seq_length), dtype=np.uint32)

    for i, ex in enumerate(examples):
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        n = min(len(input_ids), seq_length)
        input_ids_np[i, 0, 0, :n] = input_ids[:n]
        labels_np[i, :n] = labels[:n]

    # Shard batch across DP axis if enabled
    if dp_axis is not None and mesh_device is not None:
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
            mesh_device, 0, dp_axis
        )
        input_ids_t = ttml.autograd.Tensor.from_numpy(
            input_ids_np,
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
            mapper=mapper,
        )
        labels_t = ttml.autograd.Tensor.from_numpy(
            labels_np,
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
            mapper=mapper,
        )
    else:
        input_ids_t = ttml.autograd.Tensor.from_numpy(
            input_ids_np,
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
        )
        labels_t = ttml.autograd.Tensor.from_numpy(
            labels_np,
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
        )

    return Batch(
        input_ids=input_ids_t,
        labels=labels_t,
        loss_mask=None,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class MemoryTrackingCallback(TrainerCallback):
    """Callback for memory tracking on the first training step."""

    def __init__(self):
        self._memory_guard = None
        self._is_first_step = True

    def on_train_begin(self, trainer: "SFTTrainer") -> None:
        print("\n   Memory tracking enabled")
        self._memory_guard = MemoryUsageTracker.begin_capture()

    def on_step_end(
        self, trainer: "SFTTrainer", step: int, loss: float, lr: float
    ) -> None:
        if self._is_first_step:
            self._is_first_step = False
            MemoryUsageTracker.end_capture("FIRST_ITERATION_COMPLETE")
            print("\n" + "=" * 70)
            print("MEMORY USAGE REPORT")
            print("=" * 70)
            MemoryUsageTracker.print_memory_usage()
            MemoryUsageTracker.clear()
            if self._memory_guard:
                self._memory_guard.release()
                self._memory_guard = None
            print("=" * 70 + "\n")


class DispatchTraceCallback(TrainerCallback):
    """Callback for dispatch trace logging."""

    def on_train_begin(self, trainer: "SFTTrainer") -> None:
        dispatch_trace.enable()

    def on_train_end(self, trainer: "SFTTrainer") -> None:
        dispatch_trace.disable()
        print(f"\nDispatch trace: {len(dispatch_trace.entries)} recorded events")
        for entry in dispatch_trace.entries[:20]:
            print(f"  {entry}")
        if len(dispatch_trace.entries) > 20:
            print(f"  ... and {len(dispatch_trace.entries) - 20} more")


# ---------------------------------------------------------------------------
# TP policy for Llama
# ---------------------------------------------------------------------------


def build_llama_tp_policy(mesh_device, tp_axis: int) -> dict:
    """Build a TP sharding policy for the Python Llama model using regex patterns.

    Convention (matches C++ DistributedGroupedQueryAttention / linear):
      - q_linear, kv_linear: column-parallel  (weight sharded on out_features, dim -2)
      - out_linear:          row-parallel      (weight sharded on in_features,  dim -1)
      - MLP w1, w3:          column-parallel
      - MLP w2:              row-parallel
      - Everything else (embeddings, norms, fc head): replicated (no entry in policy)

    Policy keys are regex patterns matched with ``re.fullmatch`` against
    fully-qualified parameter names like ``layers.0.attention.q_linear.weight``.
    """
    mesh_shape = mesh_device.shape
    # MeshShape doesn't support len(), use dims() method
    ndim = mesh_shape.dims() if hasattr(mesh_shape, "dims") else len(mesh_shape)

    col_parallel = Layout(
        placements=tuple(
            Shard(-2) if i == tp_axis else Replicate() for i in range(ndim)
        )
    )
    row_parallel = Layout(
        placements=tuple(
            Shard(-1) if i == tp_axis else Replicate() for i in range(ndim)
        )
    )

    return {
        r".*\.(q_linear|kv_linear|w1|w3)\.weight": col_parallel,
        r".*\.(out_linear|w2)\.weight": row_parallel,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Llama TP Training with SFTTrainer")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="YAML config path"
    )
    parser.add_argument(
        "--data_path", type=str, default="", help="Path to training text file"
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--debug_dispatch", action="store_true", help="Enable dispatch trace logging"
    )
    # Mesh and parallelism overrides
    parser.add_argument(
        "--mesh_shape",
        type=str,
        default=None,
        help="Mesh shape as 'rows,cols' (e.g., '8,4' for 32 devices). Overrides config.",
    )
    parser.add_argument(
        "--tp_axis",
        type=int,
        default=None,
        help="Tensor parallel axis (0 or 1). Overrides config. Use -1 to disable TP.",
    )
    parser.add_argument(
        "--dp_axis",
        type=int,
        default=None,
        help="Data parallel axis (0 or 1). Overrides config. Use -1 to disable DP.",
    )
    # Memory profiling
    parser.add_argument(
        "--track_memory",
        action="store_true",
        help="Enable memory usage tracking (DRAM/L1)",
    )
    # SFTTrainer specific options
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=0,
        help="Evaluation interval (0 to disable)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Checkpoint save interval (0 to disable)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for LR schedule",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (memory efficient mode)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Llama TP Training with SFTTrainer (layout-aware dispatch)")
    print("=" * 70)

    # Env setup
    if "TT_METAL_RUNTIME_ROOT" not in os.environ:
        tt_metal_home = get_tt_metal_home()
        if tt_metal_home:
            os.environ["TT_METAL_RUNTIME_ROOT"] = tt_metal_home

    tt_train_root = f"{get_tt_metal_home()}/tt-train"
    configs_root = f"{tt_train_root}/configs"

    yaml_config = load_config(args.config, f"{configs_root}/training_configs")
    tc = parse_training_config(yaml_config)
    device_cfg = DeviceConfig(yaml_config)

    # Override mesh_shape from command line if provided
    if args.mesh_shape:
        parts = args.mesh_shape.split(",")
        if len(parts) != 2:
            raise ValueError(f"mesh_shape must be 'rows,cols', got: {args.mesh_shape}")
        device_cfg.mesh_shape = [int(parts[0]), int(parts[1])]

    model_config_path = tc.get("model_config", "")
    if model_config_path:
        model_yaml = load_config(model_config_path, tt_train_root)
        llama_config = parse_model_config(model_yaml)
    else:
        llama_config = LlamaConfig()

    batch_size = args.batch_size or int(tc.get("batch_size", 1))
    max_steps = args.max_steps or int(tc.get("max_steps", 5000))
    seed = int(tc.get("seed", 5489))
    grad_accum = int(tc.get("gradient_accumulation_steps", 1))
    use_clip = tc.get("use_clip_grad_norm", False)
    clip_norm = float(tc.get("clip_grad_norm_max_norm", 1.0))
    optimizer_cfg = tc.get("optimizer", {})
    learning_rate = float(optimizer_cfg.get("lr", tc.get("learning_rate", 3e-4)))

    # Data
    data_path = args.data_path or tc.get("data_path", "")
    if not data_path:
        for p in ["data/shakespeare.txt", "tt-train/data/shakespeare.txt"]:
            if os.path.exists(p):
                data_path = p
                break
    if not data_path:
        raise RuntimeError("No data_path specified and shakespeare.txt not found")

    print(f"\n1. Loading data from {data_path}")
    with open(data_path) as f:
        text = f.read()
    seq_len = llama_config.max_position_embeddings
    dataset, tokenizer = create_dataset(text, seq_len)
    llama_config = LlamaConfig(
        hidden_size=llama_config.hidden_size,
        intermediate_size=llama_config.intermediate_size,
        num_hidden_layers=llama_config.num_hidden_layers,
        num_attention_heads=llama_config.num_attention_heads,
        num_key_value_heads=llama_config.num_key_value_heads,
        vocab_size=round_up_to_tile(tokenizer.vocab_size, 32),
        max_position_embeddings=llama_config.max_position_embeddings,
        rope_theta=llama_config.rope_theta,
        attention_dropout=llama_config.attention_dropout,
        mlp_dropout=llama_config.mlp_dropout,
        runner_type=llama_config.runner_type,
        weight_tying=llama_config.weight_tying,
        rope_scaling=llama_config.rope_scaling,
    )
    print(
        f"   Dataset: {len(dataset)} samples, seq_len={seq_len}, vocab={llama_config.vocab_size}"
    )

    # Device
    print("\n2. Opening device mesh...")
    num_devices = device_cfg.mesh_shape[0] * device_cfg.mesh_shape[1]
    if num_devices > 1:
        print(f"   Enabling fabric for {num_devices} devices...")
        ttml.core.distributed.enable_fabric(num_devices)

    ttml.autograd.AutoContext.get_instance().open_device(
        device_cfg.mesh_shape, device_cfg.device_ids
    )
    mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
    ttml.autograd.AutoContext.get_instance().set_seed(seed)
    np.random.seed(seed)

    # Determine TP/DP axes - command line overrides config
    if args.tp_axis is not None:
        tp_axis = args.tp_axis if args.tp_axis >= 0 else None
    else:
        tp_axis = 1 if device_cfg.enable_tp else None

    if args.dp_axis is not None:
        dp_axis = args.dp_axis if args.dp_axis >= 0 else None
    else:
        dp_axis = 0 if device_cfg.enable_ddp else None

    # Validate axes
    mesh_dims = len(device_cfg.mesh_shape)
    if tp_axis is not None and tp_axis >= mesh_dims:
        raise ValueError(
            f"tp_axis={tp_axis} out of range for mesh with {mesh_dims} dims"
        )
    if dp_axis is not None and dp_axis >= mesh_dims:
        raise ValueError(
            f"dp_axis={dp_axis} out of range for mesh with {mesh_dims} dims"
        )
    if tp_axis is not None and dp_axis is not None and tp_axis == dp_axis:
        raise ValueError(f"tp_axis and dp_axis cannot be the same (both are {tp_axis})")

    tp_size = device_cfg.mesh_shape[tp_axis] if tp_axis is not None else 1
    dp_size = device_cfg.mesh_shape[dp_axis] if dp_axis is not None else 1
    print(f"   Mesh shape: {device_cfg.mesh_shape}")
    print(
        f"   TP axis: {tp_axis} (size={tp_size}), DP axis: {dp_axis} (size={dp_size})"
    )

    # Initialize dispatch layer for distributed ops
    print("\n   Initializing dispatch layer...")
    init_ops()

    # Model
    print("\n3. Creating Llama model...")

    # Validate TP size vs model config
    if tp_axis is not None:
        if llama_config.num_attention_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({llama_config.num_attention_heads}) must be divisible by "
                f"TP size ({tp_size}). Try a different mesh_shape or tp_axis."
            )
        if llama_config.num_key_value_heads % tp_size != 0:
            raise ValueError(
                f"num_key_value_heads ({llama_config.num_key_value_heads}) must be divisible by "
                f"TP size ({tp_size}). Try a different mesh_shape or tp_axis."
            )

    model = Llama(llama_config)
    total_params = sum(math.prod(p.shape()) for p in model.parameters().values())
    print(
        f"   Config: {llama_config.num_hidden_layers} layers, {llama_config.hidden_size} hidden, "
        f"{llama_config.num_attention_heads} heads, {llama_config.num_key_value_heads} kv_heads"
    )
    print(f"   Total parameters: {total_params:,}")

    # Distribute model with TP
    if tp_axis is not None:
        print("\n4. Distributing model with TP...")
        policy = build_llama_tp_policy(mesh_device, tp_axis)
        print(f"   Policy covers {len(policy)} parameter patterns")
        model = distribute_module(model, mesh_device, policy)
        print("   Model distributed.")
    else:
        print("\n4. TP not enabled, skipping distribution...")

    # Wrap model with DataParallelModel for DP gradient sync
    if dp_axis is not None:
        print(f"\n   Wrapping model with DataParallelModel (dp_axis={dp_axis})...")
        model = DataParallelModel(model, dp_axis)

    # Create dataloader with distributed collate function
    print("\n5. Creating dataloader...")
    collate_fn = partial(
        distributed_collate_fn,
        seq_length=seq_len,
        mesh_device=mesh_device,
        dp_axis=dp_axis,
    )
    train_dataloader = InMemoryDataloader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    print(f"   Dataloader: {len(train_dataloader)} batches, batch_size={batch_size}")

    # Create SFTConfig
    sft_config = SFTConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=grad_accum,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        seed=seed,
        max_seq_len=seq_len,
        learning_rate=learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=clip_norm if use_clip else 0.0,
        log_interval=1,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Build optimizer config from yaml (use optimizer_cfg directly, with defaults)
    optimizer_dict = (
        optimizer_cfg if optimizer_cfg else {"type": "AdamW", "lr": learning_rate}
    )

    # Setup callbacks
    callbacks = []
    if args.track_memory:
        callbacks.append(MemoryTrackingCallback())
    if args.debug_dispatch:
        callbacks.append(DispatchTraceCallback())

    # Create and run trainer
    print(f"\n6. Training for {max_steps} steps...")
    trainer = SFTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        config=sft_config,
        optimizer=optimizer_dict,
        callbacks=callbacks if callbacks else None,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining complete: {trainer.step} steps in {elapsed:.1f}s")

    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
