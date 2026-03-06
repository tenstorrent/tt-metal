# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal single-device Llama fine-tuning with LoRA on Shakespeare."""

import argparse
import time

import numpy as np
import ttnn
import ttml

from ttml.common.config import load_config
from ttml.common.data import (
    CharTokenizer,
    load_shakespeare_text,
    get_batch,
)
from ttml.common.utils import set_seed, get_tt_metal_home, summary
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

BATCH_SIZE = 13
STEPS = 500
LR = 3e-4
WEIGHT_DECAY = 0.01
PRINT_INTERVAL = 1

LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_linear", "kv_linear", "out_linear"]
LORA_IS_BIAS_TRAINABLE = False
LORA_TRAINABLE_MODULES: list[str] = []


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
            original_context_length=rs.get(
                "original_context_length", rope_scaling.original_context_length
            ),
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Llama LoRA fine-tuning on Shakespeare"
    )
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
        "--track_memory",
        action="store_true",
        help="Enable memory usage tracking (prints memory stats after first iteration)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
        print(
            f"Using HF BPE tokenizer, vocab_size={hf_tokenizer.vocab_size} (padded to {vocab_size})"
        )
    else:
        tokenizer = CharTokenizer(text)
        vocab_size = (tokenizer.vocab_size + 31) // 32 * 32
        ids = np.array(tokenizer.encode(text), dtype=np.uint32)

    n_train = int(len(ids) * 0.9)
    train_ids = ids[:n_train]

    # ── Device ────────────────────────────────────────────────────────────────
    ttml.autograd.AutoContext.get_instance().open_device([1, 1], [0])

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model_config is not None:
        tt_train_root = f"{get_tt_metal_home()}/tt-train"
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
    )
    model = LoraModel(model, lora_config)

    if args.track_memory:
        MemoryUsageTracker.snapshot("LORA_INJECTION")

    # ── Trainable parameters ──────────────────────────────────────────────────
    summary(model)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    adamw_cfg = ttml.optimizers.AdamWConfig.make(LR, 0.9, 0.999, 1e-8, WEIGHT_DECAY)
    optimizer = ttml.optimizers.AdamW(model.parameters(), adamw_cfg)

    if args.track_memory:
        MemoryUsageTracker.snapshot("OPTIMIZER_CREATION")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    is_first_step = True

    for step in range(1, STEPS + 1):
        t0 = time.perf_counter()
        x_np, y_np = get_batch(train_ids, seq_len, BATCH_SIZE)

        tt_x = ttml.autograd.Tensor.from_numpy(
            x_np.reshape(BATCH_SIZE, 1, 1, seq_len),
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
        )
        tt_y = ttml.autograd.Tensor.from_numpy(
            y_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
        )

        optimizer.zero_grad()
        logits = model(tt_x, None)
        loss = ttml.ops.loss.cross_entropy_loss(logits, tt_y, ttml.ops.ReduceType.MEAN)
        loss_val = float(loss.get_value().item())

        if args.track_memory and is_first_step:
            MemoryUsageTracker.snapshot("FORWARD_PASS")

        loss.backward(retain_graph=False)

        if args.track_memory and is_first_step:
            MemoryUsageTracker.snapshot("BACKWARD_PASS")

        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()
        step_ms = (time.perf_counter() - t0) * 1000

        if step % PRINT_INTERVAL == 0 or step == 1:
            print(
                f"step {step:>4}/{STEPS}  loss={loss_val:.4f}  step_time={step_ms:.1f}ms"
            )

        if args.track_memory and is_first_step:
            is_first_step = False
            MemoryUsageTracker.end_capture("FIRST_ITERATION_COMPLETE")
            MemoryUsageTracker.print_memory_usage()
            MemoryUsageTracker.clear()
            if memory_guard:
                memory_guard.release()

    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
