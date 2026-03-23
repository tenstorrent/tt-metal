# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama fine-tuning with LoRA on Shakespeare, with optional DDP."""

import argparse
import os
import re
import time

import numpy as np
import ttnn
import ttml

from ttml.common.config import load_config
from ttml.common.data import (
    CharTokenizer,
    load_shakespeare_text,
)
from ttml.common.utils import (
    set_seed,
    get_tt_metal_runtime_root,
    summary,
    get_loss_over_devices,
)
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import (
    Llama,
    LlamaConfig,
    LlamaRopeScalingConfig,
    load_from_safetensors,
)
from ttml.modules import LoraConfig, LoraModel

MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker

# ── Config ────────────────────────────────────────────────────────────────────

BATCH_SIZE_DEFAULT = 1
STEPS_DEFAULT = 500
LR = 3e-4
WEIGHT_DECAY = 0.01
PRINT_INTERVAL = 1

LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_linear", "kv_linear", "out_linear"]
LORA_IS_BIAS_TRAINABLE = False
LORA_TRAINABLE_MODULES: list[str] = []
LORA_DROPOUT = 0.05


def validate_mesh_graph_descriptor(mesh_shape: list[int]) -> None:
    """Validate that the MGD file's topology matches the requested mesh shape.

    Reads the TT_MESH_GRAPH_DESC_PATH env var, parses the textproto to extract
    device_topology dims and dim_types, then checks:
      1. dims match the requested mesh_shape
      2. the DDP axis (axis 1) uses RING topology
    """
    mgd_path = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if not mgd_path:
        print("WARNING: TT_MESH_GRAPH_DESC_PATH not set, skipping MGD validation")
        return

    if not os.path.isfile(mgd_path):
        print(f"WARNING: MGD file not found: {mgd_path}, skipping validation")
        return

    with open(mgd_path) as f:
        content = f.read()

    dims_match = re.search(r"device_topology\s*\{[^}]*dims\s*:\s*\[\s*([\d\s,]+)\]", content)
    if not dims_match:
        print(f"WARNING: Could not parse dims from MGD file: {mgd_path}")
        return

    mgd_dims = [int(d.strip()) for d in dims_match.group(1).split(",")]
    if list(mgd_dims) != list(mesh_shape):
        raise RuntimeError(
            f"Mesh shape mismatch!\n"
            f"  Requested mesh_shape: {mesh_shape}\n"
            f"  MGD device_topology dims: {mgd_dims}\n"
            f"Please ensure --ddp value matches the MGD file."
        )

    types_match = re.search(r"device_topology\s*\{[^}]*dim_types\s*:\s*\[\s*([A-Z_,\s]+)\]", content)
    if types_match:
        dim_types = [t.strip() for t in types_match.group(1).split(",")]
        ddp_axis = 1
        if ddp_axis < len(dim_types) and dim_types[ddp_axis] != "RING":
            raise RuntimeError(
                f"DDP axis (axis {ddp_axis}) expected RING topology  "
                f", but MGD has '{dim_types[ddp_axis]}'.\n"
                f"  MGD dim_types: {dim_types}\n"
                f"  MGD file: {mgd_path}"
            )

    print(f"MGD validated: dims={mgd_dims}, file={mgd_path}")


def llama_config_from_yaml(yaml_config: dict, vocab_size: int) -> LlamaConfig:
    """Build a LlamaConfig from a model YAML (transformer_config section)."""
    tc = yaml_config.get("transformer_config", {})

    rope_scaling = LlamaRopeScalingConfig()
    if "rope_scaling" in tc:
        rs = tc["rope_scaling"]
        rope_scaling = LlamaRopeScalingConfig(
            scaling_factor=rs.get("scaling_factor", rope_scaling.scaling_factor),
            high_freq_factor=rs.get("high_freq_factor", rope_scaling.high_freq_factor),
            low_freq_factor=rs.get("low_freq_factor", rope_scaling.low_freq_factor),
            original_context_length=rs.get("original_context_length", rope_scaling.original_context_length),
        )

    runner_type = RunnerType.Default
    if "runner_type" in tc:
        runner_type = RunnerType.from_string(tc["runner_type"])

    weight_tying = WeightTyingType.Disabled
    if "weight_tying" in tc:
        weight_tying = WeightTyingType.from_string(tc["weight_tying"])

    return LlamaConfig(
        hidden_size=tc.get("embedding_dim", 384),
        num_hidden_layers=tc.get("num_blocks", 6),
        num_attention_heads=tc.get("num_heads", 6),
        num_key_value_heads=tc.get("num_groups", 3),
        vocab_size=vocab_size,
        max_position_embeddings=tc.get("max_sequence_length", 256),
        rope_theta=tc.get("theta", 10000.0),
        attention_dropout=tc.get("dropout_prob", 0.0),
        mlp_dropout=tc.get("dropout_prob", 0.0),
        runner_type=runner_type,
        weight_tying=weight_tying,
        rope_scaling=rope_scaling,
    )


def save_lora_checkpoint(model: LoraModel, path: str, step: int) -> None:
    """Save only the trainable (LoRA) parameters to a safetensors file."""
    from safetensors.numpy import save_file

    tensors = {}
    for name, param in model.parameters().items():
        if param.get_requires_grad():
            tensors[name] = param.to_numpy(ttnn.DataType.FLOAT32)

    filepath = os.path.join(path, f"lora_step_{step}.safetensors")
    save_file(tensors, filepath)
    print(f"Saved {len(tensors)} LoRA parameters to {filepath}")


def load_lora_checkpoint(model: LoraModel, filepath: str) -> None:
    """Restore LoRA parameters from a previously saved safetensors checkpoint."""
    import ml_dtypes
    from safetensors.numpy import load_file

    saved = load_file(filepath)
    parameters = model.parameters()

    loaded, skipped = 0, []
    for name, arr in saved.items():
        if name not in parameters:
            skipped.append(name)
            continue
        restored = ttml.autograd.Tensor.from_numpy(arr.astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE)
        parameters[name].assign(restored)
        loaded += 1

    print(f"Resumed {loaded} LoRA parameters from {filepath}")
    if skipped:
        print(f"Warning: {len(skipped)} saved tensors not found in model:")
        for n in skipped:
            print(f"  - {n}")


def parse_args():
    parser = argparse.ArgumentParser(description="Llama LoRA fine-tuning on Shakespeare")
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default=None,
        help="Path to model config YAML (e.g. configs/model_configs/nanollama3.yaml). "
        "Resolved relative to tt-train/ if not absolute.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0) or "
        "local path to a directory with .safetensors files. "
        "When set, uses the HF BPE tokenizer and loads pretrained weights.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a LoRA checkpoint .safetensors file to resume training from.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save LoRA checkpoint every N steps (0 = disabled). Also saves at the final step.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory for LoRA checkpoints (default: checkpoints/).",
    )
    parser.add_argument(
        "--track_memory",
        action="store_true",
        help="Enable memory usage tracking (prints memory stats after first iteration)",
    )
    parser.add_argument(
        "--ddp",
        type=int,
        default=1,
        help="Number of devices for distributed data parallel (default: 1, no DDP).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=f"Global batch size (default: {BATCH_SIZE_DEFAULT}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS_DEFAULT,
        help=f"Number of training steps (default: {STEPS_DEFAULT}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    batch_size = args.batch
    steps = args.steps

    # ── Memory tracking ───────────────────────────────────────────────────────
    memory_guard = None
    if args.track_memory:
        print("\nMemory tracking enabled")
        memory_guard = MemoryUsageTracker.begin_capture()

    # ── Data ──────────────────────────────────────────────────────────────────
    set_seed(42)

    text = load_shakespeare_text()
    pretrained_path = None

    if args.pretrained:
        from pathlib import Path
        from transformers import AutoTokenizer

        src = args.pretrained
        if Path(src).is_dir():
            pretrained_path = src
        else:
            from huggingface_hub import snapshot_download

            print(f"Downloading weights from HuggingFace: {src}")
            pretrained_path = snapshot_download(src, allow_patterns=["*.safetensors"])
            print(f"Weights downloaded to: {pretrained_path}")

        hf_tokenizer = AutoTokenizer.from_pretrained(src)
        vocab_size = (hf_tokenizer.vocab_size + 31) // 32 * 32
        ids = np.array(hf_tokenizer.encode(text), dtype=np.uint32)
        print(f"Using HF BPE tokenizer, vocab_size={hf_tokenizer.vocab_size} (padded to {vocab_size})")
    else:
        tokenizer = CharTokenizer(text)
        vocab_size = (tokenizer.vocab_size + 31) // 32 * 32
        ids = np.array(tokenizer.encode(text), dtype=np.uint32)

    n_train = int(len(ids) * 0.9)
    train_ids = ids[:n_train]

    # ── Device ────────────────────────────────────────────────────────────────
    num_devices = args.ddp
    use_ddp = num_devices > 1
    mesh_shape = [1, num_devices]

    if use_ddp:
        if batch_size % num_devices != 0:
            raise ValueError(f"--batch ({batch_size}) must be divisible by --ddp ({num_devices})")
        ttml.core.distributed.enable_fabric(num_devices)
        validate_mesh_graph_descriptor(mesh_shape)

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    if use_ddp:
        autograd_ctx.open_device(mesh_shape)
        autograd_ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=True))
        print(f"DDP enabled: {num_devices} devices, mesh_shape={mesh_shape}")
    else:
        autograd_ctx.open_device([1, 1], [0])

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model_config is not None:
        tt_train_root = f"{get_tt_metal_runtime_root()}/tt-train"
        print(f"Loading model config from: {args.model_config}")
        yaml_config = load_config(args.model_config, tt_train_root)
        llama_cfg = llama_config_from_yaml(yaml_config, vocab_size)
    else:
        llama_cfg = LlamaConfig(
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            num_key_value_heads=3,
            vocab_size=vocab_size,
            max_position_embeddings=256,
            rope_theta=500000.0,
        )

    seq_len = llama_cfg.max_position_embeddings
    print(
        f"Model: hidden_size={llama_cfg.hidden_size}, layers={llama_cfg.num_hidden_layers}, "
        f"heads={llama_cfg.num_attention_heads}, kv_heads={llama_cfg.num_key_value_heads}, "
        f"seq_len={seq_len}"
    )

    model = Llama(llama_cfg)

    if pretrained_path is not None:
        print(f"Loading pretrained weights from: {pretrained_path}")
        load_from_safetensors(model, pretrained_path, llama_cfg)

    if args.track_memory:
        MemoryUsageTracker.snapshot("MODEL_CREATION")

    lora_config = LoraConfig(
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        is_bias_trainable=LORA_IS_BIAS_TRAINABLE,
        trainable_modules=LORA_TRAINABLE_MODULES,
        lora_dropout=LORA_DROPOUT,
    )
    model = LoraModel(model, lora_config)

    if args.resume:
        load_lora_checkpoint(model, args.resume)

    if args.track_memory:
        MemoryUsageTracker.snapshot("LORA_INJECTION")

    # ── Trainable parameters ──────────────────────────────────────────────────
    summary(model)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    adamw_cfg = ttml.optimizers.AdamWConfig.make(LR, 0.9, 0.999, 1e-8, WEIGHT_DECAY)
    optimizer = ttml.optimizers.AdamW(model.parameters(), adamw_cfg)

    if args.track_memory:
        MemoryUsageTracker.snapshot("OPTIMIZER_CREATION")

    # ── Checkpointing setup ────────────────────────────────────────────────────
    if args.save_every > 0:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── DDP mapper ─────────────────────────────────────────────────────────────
    mapper = None
    if use_ddp:
        device = autograd_ctx.get_device()
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)

    # ── Data chunks ─────────────────────────────────────────────────────────
    n_chunks = (len(train_ids) - 1) // seq_len
    _chunk_order = np.arange(n_chunks)
    np.random.shuffle(_chunk_order)
    _chunk_ptr = 0

    def get_chunk_batch():
        nonlocal _chunk_order, _chunk_ptr
        if _chunk_ptr + batch_size > len(_chunk_order):
            np.random.shuffle(_chunk_order)
            _chunk_ptr = 0
        sel = _chunk_order[_chunk_ptr : _chunk_ptr + batch_size]
        _chunk_ptr += batch_size
        x = np.stack([train_ids[i * seq_len : i * seq_len + seq_len] for i in sel])
        y = np.stack([train_ids[i * seq_len + 1 : i * seq_len + seq_len + 1] for i in sel])
        return x.astype(np.uint32), y.astype(np.uint32)

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    is_first_step = True

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        x_np, y_np = get_chunk_batch()

        tt_x = ttml.autograd.Tensor.from_numpy(
            x_np.reshape(batch_size, 1, 1, seq_len),
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
            mapper,
        )
        tt_y = ttml.autograd.Tensor.from_numpy(y_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper)

        optimizer.zero_grad()
        logits = model(tt_x, None)
        loss = ttml.ops.loss.cross_entropy_loss(logits, tt_y, ttml.ops.ReduceType.MEAN)

        if use_ddp:
            loss_val = float(get_loss_over_devices(loss))
        else:
            loss_val = float(loss.get_value().item())

        if args.track_memory and is_first_step:
            MemoryUsageTracker.snapshot("FORWARD_PASS")

        loss.backward(retain_graph=False)

        if args.track_memory and is_first_step:
            MemoryUsageTracker.snapshot("BACKWARD_PASS")

        autograd_ctx.reset_graph()

        if use_ddp:
            ttml.core.distributed.synchronize_gradients(model.parameters())

        optimizer.step()
        step_ms = (time.perf_counter() - t0) * 1000

        if step % PRINT_INTERVAL == 0 or step == 1:
            print(f"step {step:>4}/{steps}  loss={loss_val:.4f}  step_time={step_ms:.1f}ms")

        if args.track_memory and is_first_step:
            is_first_step = False
            MemoryUsageTracker.end_capture("FIRST_ITERATION_COMPLETE")
            MemoryUsageTracker.print_memory_usage()
            MemoryUsageTracker.clear()
            if memory_guard:
                memory_guard.release()

        if args.save_every > 0 and (step % args.save_every == 0 or step == steps):
            save_lora_checkpoint(model, args.save_dir, step)

    autograd_ctx.close_device()


if __name__ == "__main__":
    main()
