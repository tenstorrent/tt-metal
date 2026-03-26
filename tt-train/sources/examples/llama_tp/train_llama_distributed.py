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
import time
from functools import partial
from typing import Any, Optional

import numpy as np

import ttnn
import ttml
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig
from ttml.common.utils import round_up_to_tile, get_tt_metal_home
from ttml.common.config import load_config, DeviceConfig
from ttml.common.data import CharTokenizer

# SFT Trainer infrastructure
from ttml.trainers import SFTConfig, SFTTrainer
from ttml.datasets import Batch, InMemoryDataloader

# Layout-aware dispatch layer
from ttml.distributed import (
    parallelize_module,
    TpPlan,
    ColwiseParallel,
    RowwiseParallel,
    init_ops,
)
from ttml.distributed.debug import DispatchTraceCallback
from ttml.distributed.layout import Shard, get_layout


def _mesh_axis_sizes(mesh_device) -> list[int]:
    ms = mesh_device.shape
    if hasattr(ms, "dims"):
        return [int(ms[i]) for i in range(ms.dims())]
    return [int(x) for x in ms]


def _global_numel_from_shard_shape_and_layout(
    local_shape: tuple[int, ...],
    layout,
    mesh_device,
) -> int:
    """Invert ``Layout.shard_shape``: recover global element count from shard-local logical shape.

    For each ``Shard(dim)`` on mesh axis ``a``, multiplies ``local_shape[dim]`` by
    ``mesh_shape[a]``. Replicated axes leave sizes unchanged.
    """
    if layout is None or mesh_device is None:
        return math.prod(local_shape)
    mesh_sizes = _mesh_axis_sizes(mesh_device)
    rank = len(local_shape)
    g = list(local_shape)
    for mesh_axis, placement in enumerate(layout.placements):
        if mesh_axis >= len(mesh_sizes):
            break
        if isinstance(placement, Shard):
            dim = placement.dim if placement.dim >= 0 else rank + placement.dim
            n = mesh_sizes[mesh_axis]
            if n > 1:
                g[dim] *= n
    return math.prod(g)


def parameter_count_stats_from_sharding(model, mesh_device) -> tuple[int, int]:
    """Sum over ``model.parameters()``: (local_shard_numel, global_unique_numel).

    ``local_shard_numel`` is ``sum(prod(t.shape()))`` as reported by the runtime.
    ``global_unique_numel`` applies the mesh × ``Layout`` shard inverse (same convention
    as ``Layout.shard_shape``) so TP-sharded weights count once at full width.
    """
    local_total = 0
    global_total = 0
    for _name, t in model.parameters().items():
        sh = tuple(int(x) for x in t.shape())
        local_total += math.prod(sh)
        layout = get_layout(t)
        global_total += _global_numel_from_shard_shape_and_layout(
            sh, layout, mesh_device
        )
    return local_total, global_total

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


# ---------------------------------------------------------------------------
# Scheduler factories (mirror C++ create_identity_scheduler /
#                              create_warmup_with_linear_scheduler)
# ---------------------------------------------------------------------------


def create_identity_scheduler(optimizer, total_steps: int):
    """Constant LR — no change throughout training."""
    return LambdaScheduler(optimizer, lambda step: 1.0)


def create_warmup_linear_scheduler(optimizer, total_steps: int):
    """10 % linear warmup then linear decay to 1 % of peak lr."""
    warmup_steps = max(1, int(total_steps * 0.1))
    decay_steps = max(1, total_steps - warmup_steps)
    warmup = LinearScheduler(
        optimizer, start_factor=0.0, end_factor=1.0, total_steps=warmup_steps
    )
    decay = LinearScheduler(
        optimizer, start_factor=1.0, end_factor=0.01, total_steps=decay_steps
    )
    return SequentialScheduler(
        optimizer, [warmup, decay], milestones=[warmup_steps, decay_steps]
    )


_SCHEDULER_FACTORIES = {
    "identity": create_identity_scheduler,
    "warmup_linear": create_warmup_linear_scheduler,
}


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
    ddp_axis: Optional[int] = None,
    cp_axis: Optional[int] = None,
) -> Batch:
    """Collate samples into a Batch with distributed tensor support.

    This collate function creates Batch objects compatible with SFTTrainer,
    with optional DP sharding (batch dim) and CP sharding (sequence dim).

    Use with functools.partial to bind parameters::

        collate = partial(distributed_collate_fn, seq_length=1024,
                          mesh_device=mesh_device, ddp_axis=0, cp_axis=1)
        dataloader = InMemoryDataloader(dataset, collate, batch_size=8)

    Args:
        examples: List of dicts with "input_ids" and "labels" keys.
        seq_length: Sequence length for the batch.
        mesh_device: Mesh device for distributed tensors.
        ddp_axis: Data parallel axis for batch sharding (None to disable).
        cp_axis: Context parallel axis for sequence sharding (None to disable).

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

    # Determine sharding strategy
    # DP shards batch (dim 0), CP shards sequence (dim 3 for input_ids, dim 1 for labels)
    if mesh_device is not None and (ddp_axis is not None or cp_axis is not None):
        # Build placements list for each mesh axis
        # input_ids: (batch, 1, 1, seq_length) -> DP shards dim 0, CP shards dim 3
        # labels: (batch, seq_length) -> DP shards dim 0, CP shards dim 1
        mesh_shape = mesh_device.shape
        ndim = mesh_shape.dims() if hasattr(mesh_shape, "dims") else len(mesh_shape)

        # Build placements: None = replicate, int = shard on that dim
        input_placements = [None] * ndim
        labels_placements = [None] * ndim

        if ddp_axis is not None:
            input_placements[ddp_axis] = 0  # Shard batch dim
            labels_placements[ddp_axis] = 0  # Shard batch dim

        if cp_axis is not None:
            input_placements[cp_axis] = 3  # Shard sequence dim (dim 3 for input_ids)
            labels_placements[cp_axis] = 1  # Shard sequence dim (dim 1 for labels)

        mapper_input = ttml.core.distributed.create_tensor_to_mesh_mapper(
            mesh_device, input_placements
        )
        mapper_labels = ttml.core.distributed.create_tensor_to_mesh_mapper(
            mesh_device, labels_placements
        )

        input_ids_t = ttml.autograd.Tensor.from_numpy(
            input_ids_np,
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
            mapper=mapper_input,
        )
        labels_t = ttml.autograd.Tensor.from_numpy(
            labels_np,
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
            mapper=mapper_labels,
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
# TP plan for Llama (PyTorch-style ParallelStyle)
# ---------------------------------------------------------------------------

# Module name patterns (regex or exact) -> ParallelStyle.
# tp_axis is set at model creation time via TpPlan(..., tp_axis=...).
LLAMA_TP_STYLES = {
    r".*\.(q_linear|kv_linear|w1|w3)": ColwiseParallel(),
    r".*\.(out_linear|w2)": RowwiseParallel(),
    "fc": ColwiseParallel(gather_output=True),  # LM head
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
    parser.add_argument(
        "--debug_dispatch_first_step_only",
        action="store_true",
        help="Only trace the first training step (reduces memory)",
    )
    parser.add_argument(
        "--debug_dispatch_dump",
        type=str,
        default=None,
        help="Path to dump trace: .html for a viewable report (open in browser), otherwise JSONL",
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
        help="Tensor parallel axis (0 or 1). Overrides config tp_axis. Use -1 to disable.",
    )
    parser.add_argument(
        "--ddp_axis",
        type=int,
        default=None,
        help="Data parallel axis (0 or 1). Overrides config ddp_axis. Use -1 to disable.",
    )
    parser.add_argument(
        "--cp_axis",
        type=int,
        default=None,
        help="Context parallel axis (0 or 1). Overrides config cp_axis. Use -1 to disable.",
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
        "--scheduler_type",
        type=str,
        default=None,
        help="LR scheduler type: 'identity' (constant) or 'warmup_linear'. Overrides config.",
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
    scheduler_type = args.scheduler_type or tc.get("scheduler_type", "identity")

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

    # Vocab must be divisible by TILE_SIZE * TP_DEGREE for clean sharding
    # Otherwise slice ops produce non-tile-aligned outputs causing TILE->ROW_MAJOR conversions
    tp_degree = (
        device_cfg.mesh_shape[device_cfg.tp_axis]
        if device_cfg.tp_axis is not None
        else 1
    )
    vocab_alignment = 32 * tp_degree  # TILE_SIZE * TP_DEGREE

    llama_config = LlamaConfig(
        hidden_size=llama_config.hidden_size,
        intermediate_size=llama_config.intermediate_size,
        num_hidden_layers=llama_config.num_hidden_layers,
        num_attention_heads=llama_config.num_attention_heads,
        num_key_value_heads=llama_config.num_key_value_heads,
        vocab_size=round_up_to_tile(tokenizer.vocab_size, vocab_alignment),
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

    # Determine TP/DP/CP axes - command line overrides config
    if args.tp_axis is not None:
        tp_axis = args.tp_axis if args.tp_axis >= 0 else None
    else:
        tp_axis = device_cfg.tp_axis  # Uses config or legacy enable_tp

    if args.ddp_axis is not None:
        ddp_axis = args.ddp_axis if args.ddp_axis >= 0 else None
    else:
        ddp_axis = device_cfg.ddp_axis  # Uses config or legacy enable_ddp

    if args.cp_axis is not None:
        cp_axis = args.cp_axis if args.cp_axis >= 0 else None
    else:
        cp_axis = device_cfg.cp_axis  # Uses config

    # Validate axes
    mesh_dims = len(device_cfg.mesh_shape)
    if tp_axis is not None and tp_axis >= mesh_dims:
        raise ValueError(
            f"tp_axis={tp_axis} out of range for mesh with {mesh_dims} dims"
        )
    if ddp_axis is not None and ddp_axis >= mesh_dims:
        raise ValueError(
            f"ddp_axis={ddp_axis} out of range for mesh with {mesh_dims} dims"
        )
    if cp_axis is not None and cp_axis >= mesh_dims:
        raise ValueError(
            f"cp_axis={cp_axis} out of range for mesh with {mesh_dims} dims"
        )
    if tp_axis is not None and ddp_axis is not None and tp_axis == ddp_axis:
        raise ValueError(
            f"tp_axis and ddp_axis cannot be the same (both are {tp_axis})"
        )
    if tp_axis is not None and cp_axis is not None and tp_axis == cp_axis:
        raise ValueError(f"tp_axis and cp_axis cannot be the same (both are {tp_axis})")
    if ddp_axis is not None and cp_axis is not None and ddp_axis == cp_axis:
        raise ValueError(
            f"ddp_axis and cp_axis cannot be the same (both are {ddp_axis})"
        )

    tp_size = device_cfg.mesh_shape[tp_axis] if tp_axis is not None else 1
    dp_size = device_cfg.mesh_shape[ddp_axis] if ddp_axis is not None else 1
    cp_size = device_cfg.mesh_shape[cp_axis] if cp_axis is not None else 1
    if ddp_axis is not None and batch_size % dp_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by DP size ({dp_size}) "
            "so each data-parallel rank gets an equal batch shard (same check as C++ nano_gpt)."
        )
    print(f"   Mesh shape: {device_cfg.mesh_shape}")
    print(
        f"   TP axis: {tp_axis} (size={tp_size}), DP axis: {ddp_axis} (size={dp_size}), "
        f"CP axis: {cp_axis} (size={cp_size})"
    )

    # Initialize dispatch layer for distributed ops
    print("\n   Initializing dispatch layer...")

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

    # Track model creation memory
    if args.track_memory:
        model_guard = MemoryUsageTracker.begin_capture()

    # Build model — when TP is enabled, pass a TpPlan so TransformerBase
    # creates weights directly sharded on device (no allocate-full → scatter).
    tp_plan = TpPlan(LLAMA_TP_STYLES, tp_axis=tp_axis) if tp_axis is not None else None
    if tp_plan is not None:
        print(f"   Using lazy init with {len(tp_plan)} tp_plan patterns")
    model = Llama(llama_config, mesh_device=mesh_device, tp_plan=tp_plan)

    local_p, global_p = parameter_count_stats_from_sharding(model, mesh_device)
    print(
        f"   Config: {llama_config.num_hidden_layers} layers, {llama_config.hidden_size} hidden, "
        f"{llama_config.num_attention_heads} heads, {llama_config.num_key_value_heads} kv_heads"
    )
    print(
        f"   Parameters (local shard logical, sum of prod(shape) per weight): {local_p:,}"
    )
    print(f"   Parameters (global unique, from Layout × mesh sharding): {global_p:,}")

    if args.track_memory:
        MemoryUsageTracker.end_capture("MODEL_CREATION")
        print_memory_stats("MODEL_CREATION")

    # Install forward collective hooks (all_reduce / all_gather / ring_sdpa).
    # Weights are already sharded from lazy init; parallelize_module only patches
    # module.forward here (skips distribute_tensor for already-distributed tensors).
    if tp_axis is not None or cp_axis is not None:
        print("\n4. Installing distributed forward hooks...")
        if tp_axis is not None:
            print(f"   TP enabled on axis {tp_axis}")
            print(
                f"   Plan: {len(LLAMA_TP_STYLES)} module patterns (ColwiseParallel / RowwiseParallel)"
            )
        if cp_axis is not None:
            print(f"   CP enabled on axis {cp_axis}")

        if args.track_memory:
            dist_guard = MemoryUsageTracker.begin_capture()

        if tp_axis is not None:
            model = parallelize_module(
                model,
                mesh_device,
                LLAMA_TP_STYLES,
                tp_axis=tp_axis,
                cp_axis=cp_axis,
            )
        else:
            # CP only: parallelize_module with empty plan still runs GQA rule (cp_axis)
            model = parallelize_module(
                model, mesh_device, {}, tp_axis=None, cp_axis=cp_axis
            )

        if args.track_memory:
            MemoryUsageTracker.end_capture("MODEL_DISTRIBUTION")
            print_memory_stats("MODEL_DISTRIBUTION")

        print("   Hooks installed.")
    else:
        print("\n4. TP/CP not enabled, skipping distribution...")

    # Create dataloader with distributed collate function
    print("\n5. Creating dataloader...")
    collate_fn = partial(
        distributed_collate_fn,
        seq_length=seq_len,
        mesh_device=mesh_device,
        ddp_axis=ddp_axis,
        cp_axis=cp_axis,
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
        max_grad_norm=clip_norm if use_clip else 0.0,
        log_interval=1,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Build optimizer config from yaml (use optimizer_cfg directly, with defaults)
    optimizer_dict = (
        optimizer_cfg if optimizer_cfg else {"type": "AdamW", "lr": learning_rate}
    )
    print(f"   Optimizer: {optimizer_dict}")

    optimizer = ttml.optimizers.create_optimizer(optimizer_dict, model.parameters())

    print(f"   Scheduler: identity")

    # Setup callbacks
    callbacks = []
    if args.debug_dispatch:
        callbacks.append(
            DispatchTraceCallback(
                first_step_only=args.debug_dispatch_first_step_only,
                dump_path=args.debug_dispatch_dump,
            )
        )

    # Create and run trainer
    print(f"\n6. Training for {max_steps} steps...")
    trainer = SFTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        config=sft_config,
        optimizer=optimizer,
        callbacks=callbacks if callbacks else None,
    )

    # When CP is enabled, don't pass causal mask - ring_attention_sdpa handles it internally
    if cp_axis is not None:
        trainer._causal_mask = None

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining complete: {trainer.step} steps in {elapsed:.1f}s")

    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
