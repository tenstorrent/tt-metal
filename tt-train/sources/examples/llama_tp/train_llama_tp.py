# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP training script for the Python Llama model.

Uses the new layout-aware dispatch layer (``ttml.distributed``) to distribute
the Python Llama model with tensor parallelism.  Model code is written once;
distribution is handled by module rules and op dispatch.

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
from typing import Optional, Tuple

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig
from ttml.common.utils import round_up_to_tile, get_tt_metal_home, create_optimizer
from ttml.common.config import load_config, DeviceConfig
from ttml.common.data import CharTokenizer, build_causal_mask

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
        # Get memory usage for the specific snapshot name
        # DRAMUsage has .peak, .total_allocations, .total_deallocations fields
        # L1UsagePerCore has .peak_cb, .peak_l1, .peak_total fields
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
# Dataset helpers (same as NanoGPT example)
# ---------------------------------------------------------------------------


class InMemoryTokenDataset:
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


def create_dataset(text: str, seq_length: int):
    tokenizer = CharTokenizer(text)
    tokens = np.array(tokenizer.encode(text), dtype=np.uint32)
    return InMemoryTokenDataset(tokens, seq_length), tokenizer


def collate(samples, seq_length: int):
    batch_size = len(samples)
    data, targets = [], []
    for s, t in samples:
        data.extend(s)
        targets.extend(t)
    data_np = np.array(data, dtype=np.uint32).reshape(batch_size, 1, 1, seq_length)
    targets_np = np.array(targets, dtype=np.uint32).reshape(batch_size, seq_length)
    data_t = ttml.autograd.Tensor.from_numpy(
        data_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )
    targets_t = ttml.autograd.Tensor.from_numpy(
        targets_np, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32
    )
    return data_t, targets_t


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    model: Llama,
    optimizer,
    input_tokens,
    target_tokens,
    mask,
    mesh_device,
    dp_axis: int = None,
    grad_accum_steps: int = 1,
    use_clip_grad_norm: bool = False,
    clip_grad_norm_max_norm: float = 1.0,
    track_memory: bool = False,
) -> float:
    optimizer.zero_grad()

    logits = model(input_tokens, mask)
    loss = ttml.ops.loss.cross_entropy_loss(
        logits, target_tokens, reduce=ttml.ops.ReduceType.MEAN
    )

    if grad_accum_steps > 1:
        loss = ttml.ops.binary.mul(loss, 1.0 / float(grad_accum_steps))

    # Use composer to gather distributed loss tensor before getting scalar value
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)
    loss_numpy = loss.to_numpy(composer=composer)
    loss_float = float(loss_numpy.flatten()[0])

    if track_memory:
        MemoryUsageTracker.snapshot("FORWARD_PASS")
        print_memory_stats("FORWARD_PASS")

    loss.backward(False)

    if track_memory:
        MemoryUsageTracker.snapshot("BACKWARD_PASS")
        print_memory_stats("BACKWARD_PASS")

    ttml.autograd.AutoContext.get_instance().reset_graph()

    # Sync gradients across DP axis if enabled
    cluster_axes = [dp_axis] if dp_axis is not None else None
    sync_gradients(model, cluster_axes=cluster_axes)

    if use_clip_grad_norm:
        ttml.core.clip_grad_norm(
            model.parameters(),
            clip_grad_norm_max_norm,
            2.0,
            False,
        )

    optimizer.step()
    return loss_float


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
    parser = argparse.ArgumentParser(description="Llama TP Training")
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
    args = parser.parse_args()

    print("=" * 70)
    print("Llama TP Training (layout-aware dispatch)")
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

    # Memory tracking
    memory_guard = None
    if args.track_memory:
        print("\n   Memory tracking enabled")
        memory_guard = MemoryUsageTracker.begin_capture()

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

    if args.track_memory:
        MemoryUsageTracker.snapshot("MODEL_CREATION")
        print_memory_stats("MODEL_CREATION")

    # Distribute
    if tp_axis is not None:
        print("\n4. Distributing model with TP...")
        policy = build_llama_tp_policy(mesh_device, tp_axis)
        print(f"   Policy covers {len(policy)} parameter patterns")

        if args.debug_dispatch:
            dispatch_trace.enable()

        model = distribute_module(model, mesh_device, policy)
        print("   Model distributed.")

        if args.track_memory:
            MemoryUsageTracker.snapshot("MODEL_DISTRIBUTED")
            print_memory_stats("MODEL_DISTRIBUTED")
    else:
        print("\n4. TP not enabled, skipping distribution...")
        if args.debug_dispatch:
            dispatch_trace.enable()

    # Optimizer
    print("\n5. Creating optimizer...")
    optimizer = create_optimizer(model, yaml_config)
    print(f"   Optimizer: {optimizer.get_name()}, lr={optimizer.get_lr()}")

    if args.track_memory:
        MemoryUsageTracker.snapshot("OPTIMIZER_CREATION")
        print_memory_stats("OPTIMIZER_CREATION")

    # Mask
    mask_np = build_causal_mask(seq_len)
    mask = ttml.autograd.Tensor.from_numpy(
        mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )

    # Training loop
    print(f"\n6. Training for {max_steps} steps (batch_size={batch_size})...")
    model.train()
    dataset_len = len(dataset)
    indices = np.arange(dataset_len, dtype=np.int64)

    global_step = 0
    start_time = time.time()
    is_first_step = True

    for epoch in range(int(tc.get("num_epochs", 1))):
        np.random.shuffle(indices)
        for batch_start in range(0, dataset_len, batch_size):
            batch_end = min(batch_start + batch_size, dataset_len)
            samples = [dataset[i] for i in indices[batch_start:batch_end]]
            input_tokens, target_tokens = collate(samples, seq_len)

            # Track memory only on first step
            track_this_step = args.track_memory and is_first_step

            step_start = time.time()
            loss = train_step(
                model,
                optimizer,
                input_tokens,
                target_tokens,
                mask,
                mesh_device,
                dp_axis,
                grad_accum,
                use_clip,
                clip_norm,
                track_memory=track_this_step,
            )
            step_time = time.time() - step_start

            global_step += 1
            elapsed = time.time() - start_time
            print(
                f"  Step {global_step:5d} | loss={loss:.4f} | step_time={step_time:.3f}s | elapsed={elapsed:.1f}s"
            )

            # Print memory report after first step
            if track_this_step:
                is_first_step = False
                MemoryUsageTracker.end_capture("FIRST_ITERATION_COMPLETE")
                print("\n" + "=" * 70)
                print("MEMORY USAGE REPORT")
                print("=" * 70)
                MemoryUsageTracker.print_memory_usage()
                MemoryUsageTracker.clear()
                if memory_guard:
                    memory_guard.release()
                    memory_guard = None
                print("=" * 70 + "\n")

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    elapsed = time.time() - start_time
    print(f"\nTraining complete: {global_step} steps in {elapsed:.1f}s")

    if args.debug_dispatch:
        dispatch_trace.disable()
        print(f"\nDispatch trace: {len(dispatch_trace.entries)} recorded events")
        for entry in dispatch_trace.entries[:20]:
            print(f"  {entry}")
        if len(dispatch_trace.entries) > 20:
            print(f"  ... and {len(dispatch_trace.entries) - 20} more")

    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
