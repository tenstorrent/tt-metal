# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama LoRA fine-tuning on Shakespeare using SFTTrainer, with optional DDP and TP.

This is a reimplementation of train_lora_llama.py that delegates the training
loop to :class:`SFTTrainer`.  DDP is wired via a collate function that shards
batch tensors across the mesh; the trainer synchronises gradients automatically.
"""

import argparse
import os
import time
from functools import partial
from typing import Any

import numpy as np
import ttml
from ttnn.device import is_blackhole, is_wormhole_b0

from ttml.common.config import load_config
from ttml.common.data import CharTokenizer, load_shakespeare_text
from ttml.common.utils import get_tt_metal_runtime_root, set_seed, summary
from ttml.datasets import Batch, InMemoryDataloader, causal_lm_collate_fn
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama.flops import calculate_flops_per_token
from ttml.models.llama import (
    Llama,
    LlamaConfig,
    LlamaRopeScalingConfig,
    load_from_safetensors,
)
from ttml.modules import LoraConfig
from ttml.trainers import SFTConfig, SFTTrainer, TrainerCallback

# ── Defaults ──────────────────────────────────────────────────────────────────

BATCH_SIZE_DEFAULT = 1
STEPS_DEFAULT = 500
LR = 3e-4
WEIGHT_DECAY = 0.01

LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_linear", "kv_linear", "out_linear"]
LORA_DROPOUT = 0.05


class LossLogger(TrainerCallback):
    """Collect per-step losses and write them to a JSON file at the end."""

    def __init__(self, path: str):
        self._path = path
        self._records: list[dict] = []

    def on_step_end(self, trainer, step, loss, lr):
        self._records.append({"step": step, "loss": loss, "lr": lr})

    def on_train_end(self, trainer):
        import json

        # Ensure the parent directory exists before writing the loss log.
        dirpath = os.path.dirname(self._path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)
        print(f"Loss log saved to {self._path}")


def get_device_peak_tflops_bf16() -> float:
    """Per-device theoretical BF16 TFLOPS. Whole-mesh peak = this × num_devices.

    Ported from examples/train/train.py so this script logs MFU on the same basis.
    """
    device = ttml.autograd.AutoContext.get_instance().get_device()
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    if is_wormhole_b0(device):
        per_core = 1.0
    elif is_blackhole(device):
        per_core = 1.35
    else:
        raise ValueError(f"Unknown device: {device.arch()}")
    return num_cores * per_core


class ThroughputCallback(TrainerCallback):
    """Print wall-clock Loss / Time / TPS / TFLOPS / MFU every ``log_interval`` steps.

    Identical log-line format and metric math to examples/train/callbacks.py's
    ThroughputCallback (ported here so the LoRA-SFT path prints the same per-step
    metrics as the main training example).
    """

    def __init__(self, flops_per_token: float, peak_tflops: float, log_interval: int = 1) -> None:
        self._flops_per_token = flops_per_token
        self._peak_tflops = peak_tflops
        self._log_interval = max(1, int(log_interval))
        self._step_start: float | None = None
        self._tokens_in_step: int = 0
        self._dp_size: int = 1

    def on_train_begin(self, trainer: SFTTrainer) -> None:
        mesh = ttml.mesh()
        self._dp_size = mesh.axis_size("dp") if mesh.has_axis("dp") else 1
        self._step_start = time.time()
        self._tokens_in_step = 0

    def on_after_forward(self, trainer: SFTTrainer, batch: Batch, loss: float) -> None:
        shape = batch.input_ids.shape()
        # Per-rank micro-shard tokens × dp_size = global tokens processed this step.
        self._tokens_in_step += int(shape[0]) * int(shape[-1]) * self._dp_size

    def on_step_end(self, trainer: SFTTrainer, step: int, step_loss: float = 0.0, *args: Any, **kwargs: Any) -> None:
        if step % self._log_interval != 0 or self._step_start is None:
            self._step_start = time.time()
            self._tokens_in_step = 0
            return
        now = time.time()
        elapsed_ms = (now - self._step_start) * 1000.0
        tps = self._tokens_in_step / max(1e-6, elapsed_ms / 1000.0)
        line = f"Step: {step}, Loss: {step_loss:.6f}, Time: {elapsed_ms:.2f} ms, TPS: {tps:.0f}"
        if self._flops_per_token > 0 and elapsed_ms > 0:
            achieved = tps * self._flops_per_token / 1e12
            line += f", TFLOPS: {achieved:.3g}"
            if self._peak_tflops > 0:
                mfu = achieved / self._peak_tflops * 100.0
                line += f", MFU: {mfu:.3g}%"
        print(line)
        self._step_start = now
        self._tokens_in_step = 0


# ── Dataset & collate ─────────────────────────────────────────────────────────


class ShakespeareChunkDataset:
    """Indexable dataset of non-overlapping (seq_len+1)-token windows."""

    def __init__(self, token_ids: np.ndarray, seq_len: int):
        self._ids = token_ids
        self._seq_len = seq_len
        self._n = (len(token_ids) - 1) // seq_len

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        start = idx * self._seq_len
        return {
            "input_ids": self._ids[start : start + self._seq_len].tolist(),
            "labels": self._ids[start + 1 : start + self._seq_len + 1].tolist(),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def llama_config_from_yaml(yaml_config: dict, vocab_size: int, use_tp: bool = False) -> LlamaConfig:
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
        # Read the MLP intermediate size from the yaml. When absent, LlamaConfig
        # falls back to the SwiGLU default (~8/3*hidden), which does NOT match real
        # Llama-3 checkpoints (e.g. 8B uses 14336, not the derived 11008) and would
        # fail weight loading with a down_proj shape mismatch.
        intermediate_size=tc.get("intermediate_dim", None),
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
        use_tp=use_tp,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Llama LoRA fine-tuning on Shakespeare (SFTTrainer)")
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default=None,
        help="Path to model config YAML. Resolved relative to tt-train/ if not absolute.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="HuggingFace repo ID or local path to .safetensors weights.",
    )
    parser.add_argument(
        "--mesh_shape",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=(1, 1),
        help="Mesh shape as comma-separated integers, e.g. '2,4' (default: '1,1').",
    )
    parser.add_argument(
        "--dp_axis",
        type=int,
        default=-1,
        help="Index of the DP axis in --mesh_shape (default: -1, no DP).",
    )
    parser.add_argument(
        "--tp_axis",
        type=int,
        default=-1,
        help="Index of the TP axis in --mesh_shape (default: -1, no TP).",
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
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save checkpoint every N steps (0 = disabled).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints (default: checkpoints/).",
    )
    parser.add_argument(
        "--loss_log",
        type=str,
        default=None,
        help="Path to write per-step loss JSON (disabled if not set).",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    batch_size = args.batch

    set_seed(42)

    # ── Data ──────────────────────────────────────────────────────────────────

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
        print(f"Using HF BPE tokenizer, vocab_size={hf_tokenizer.vocab_size} " f"(padded to {vocab_size})")
    else:
        tokenizer = CharTokenizer(text)
        vocab_size = (tokenizer.vocab_size + 31) // 32 * 32
        ids = np.array(tokenizer.encode(text), dtype=np.uint32)

    n_train = int(len(ids) * 0.9)
    train_ids = ids[:n_train]

    # ── Device ────────────────────────────────────────────────────────────────

    shape = args.mesh_shape
    for name, value in (("dp_axis", args.dp_axis), ("tp_axis", args.tp_axis)):
        if value != -1 and not (0 <= value < len(shape)):
            raise ValueError(f"--{name} ({value}) is out of range for --mesh_shape of length {len(shape)}")
    if args.dp_axis != -1 and args.dp_axis == args.tp_axis:
        raise ValueError(f"--dp_axis and --tp_axis must differ (both set to {args.dp_axis})")
    axis_names_list = [f"_{i}" for i in range(len(shape))]
    if args.dp_axis != -1:
        axis_names_list[args.dp_axis] = "dp"
    if args.tp_axis != -1:
        axis_names_list[args.tp_axis] = "tp"
    mesh = ttml.Mesh(shape, tuple(axis_names_list))
    ttml.open_device_mesh(mesh)
    autograd_ctx = ttml.autograd.AutoContext.get_instance()

    dp_size = mesh.axis_size("dp") if mesh.has_axis("dp") else 1
    tp_size = mesh.axis_size("tp") if mesh.has_axis("tp") else 1
    use_ddp = dp_size > 1
    use_tp = tp_size > 1

    if use_ddp and batch_size % dp_size != 0:
        raise ValueError(f"--batch ({batch_size}) must be divisible by dp axis size ({dp_size})")

    if use_tp and args.save_every > 0:
        raise ValueError("Checkpointing (--save_every) is not supported with tensor parallelism (tp > 1)")

    if use_ddp or use_tp:
        mode = "+".join(filter(None, ["DP" if use_ddp else "", "TP" if use_tp else ""]))
        print(f"{mode} enabled: mesh={dict(zip(mesh.axis_names, mesh.shape))}")

    # ── Data mapper ───────────────────────────────────────────────────────────

    mapper = None
    if use_ddp:
        mapper = ttml.mesh().axis_mapper("dp", tdim=0)

    # ── Model ─────────────────────────────────────────────────────────────────

    if args.model_config is not None:
        tt_train_root = f"{get_tt_metal_runtime_root()}/tt-train"
        print(f"Loading model config from: {args.model_config}")
        yaml_config = load_config(args.model_config, tt_train_root)
        llama_cfg = llama_config_from_yaml(yaml_config, vocab_size, use_tp=use_tp)
    else:
        llama_cfg = LlamaConfig(
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            num_key_value_heads=3,
            vocab_size=vocab_size,
            max_position_embeddings=256,
            rope_theta=500000.0,
            use_tp=use_tp,
        )

    seq_len = llama_cfg.max_position_embeddings
    print(
        f"Model: hidden_size={llama_cfg.hidden_size}, "
        f"layers={llama_cfg.num_hidden_layers}, "
        f"heads={llama_cfg.num_attention_heads}, "
        f"kv_heads={llama_cfg.num_key_value_heads}, "
        f"seq_len={seq_len}"
    )

    model = Llama(llama_cfg)

    if pretrained_path is not None:
        print(f"Loading pretrained weights from: {pretrained_path}")
        load_from_safetensors(model, pretrained_path, llama_cfg)

    # ── Dataloader ────────────────────────────────────────────────────────────

    dataset = ShakespeareChunkDataset(train_ids, seq_len)
    collate = partial(causal_lm_collate_fn, seq_len=seq_len, mapper=mapper)
    train_loader = InMemoryDataloader(dataset, collate, batch_size=batch_size, shuffle=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    callbacks: list[TrainerCallback] = []
    if args.loss_log:
        callbacks.append(LossLogger(args.loss_log))

    # Per-step Loss / Time / TPS / TFLOPS / MFU logging, matching the main
    # training example. flops_per_token uses the Llama FLOPs model; peak_tflops
    # is the whole-mesh theoretical BF16 peak (per-device × num_devices).
    flops_per_token = calculate_flops_per_token(llama_cfg, seq_len)
    peak_tflops = get_device_peak_tflops_bf16() * mesh.num_devices() if flops_per_token > 0 else 0.0
    callbacks.append(ThroughputCallback(flops_per_token, peak_tflops, log_interval=1))
    print(
        f"Throughput logging: flops/token={flops_per_token / 1e9:.3g}G, "
        f"peak={peak_tflops:.1f} TFLOPS bf16 ({mesh.num_devices()} devices)"
    )

    # ── LoRA + SFTTrainer ─────────────────────────────────────────────────────

    peft_config = LoraConfig(
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
    )

    sft_config = SFTConfig(
        max_steps=args.steps,
        learning_rate=LR,
        max_seq_len=seq_len,
        save_interval=args.save_every,
        checkpoint_dir=args.save_dir,
        eval_interval=0,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=None,
        config=sft_config,
        peft_config=peft_config,
        optimizer={
            "type": "AdamW",
            "lr": LR,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": WEIGHT_DECAY,
        },
        callbacks=callbacks,
    )

    summary(trainer.model)
    trainer.train()

    autograd_ctx.close_device()


if __name__ == "__main__":
    main()
