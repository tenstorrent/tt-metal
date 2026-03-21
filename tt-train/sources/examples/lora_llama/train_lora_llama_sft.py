# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Llama LoRA fine-tuning on Shakespeare using SFTTrainer, with optional DDP.

This is a reimplementation of train_lora_llama.py that delegates the training
loop to :class:`SFTTrainer`.  DDP support is wired externally via:

* A collate function that shards batch tensors across the mesh.
* An ``on_before_optimizer_step`` callback that synchronises gradients.
"""

import argparse
import os
import re
from functools import partial

import ml_dtypes
import numpy as np
import ttnn
import ttml

from ttml.common.config import load_config
from ttml.common.data import CharTokenizer, load_shakespeare_text
from ttml.common.utils import get_tt_metal_runtime_root, set_seed, summary
from ttml.datasets import Batch, InMemoryDataloader
from ttml.models import RunnerType, WeightTyingType
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


# ── DDP callback ──────────────────────────────────────────────────────────────


class DDPCallback(TrainerCallback):
    """Synchronise gradients across all DDP devices before the optimiser step."""

    def on_before_optimizer_step(self, trainer):
        ttml.core.distributed.synchronize_gradients(trainer.model.parameters())


class LossLogger(TrainerCallback):
    """Collect per-step losses and write them to a JSON file at the end."""

    def __init__(self, path: str):
        self._path = path
        self._records: list[dict] = []

    def on_step_end(self, trainer, step, loss, lr):
        self._records.append({"step": step, "loss": loss, "lr": lr})

    def on_train_end(self, trainer):
        import json

        with open(self._path, "w") as f:
            json.dump(self._records, f, indent=2)
        print(f"Loss log saved to {self._path}")


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


def causal_lm_collate(
    examples: list,
    seq_len: int,
    mapper=None,
) -> Batch:
    """Collate for causal LM -- every token position contributes to the loss."""
    batch_size = len(examples)

    input_ids_np = np.zeros((batch_size, 1, 1, seq_len), dtype=np.uint32)
    labels_np = np.zeros((batch_size, seq_len), dtype=np.uint32)
    loss_mask_np = np.ones((batch_size, 1, seq_len, 1), dtype=np.float32)

    for i, ex in enumerate(examples):
        ids = ex["input_ids"][:seq_len]
        lbs = ex["labels"][:seq_len]
        n = len(ids)
        input_ids_np[i, 0, 0, :n] = ids
        labels_np[i, :n] = lbs
        if n < seq_len:
            loss_mask_np[i, 0, n:, 0] = 0.0

    total = loss_mask_np.sum()
    if total > 0:
        loss_mask_np *= (batch_size * seq_len) / total

    return Batch(
        input_ids=ttml.autograd.Tensor.from_numpy(
            input_ids_np,
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
            mapper,
        ),
        labels=ttml.autograd.Tensor.from_numpy(
            labels_np,
            ttnn.Layout.ROW_MAJOR,
            ttnn.DataType.UINT32,
            mapper,
        ),
        loss_mask=ttml.autograd.Tensor.from_numpy(
            loss_mask_np.astype(ml_dtypes.bfloat16),
            ttnn.Layout.TILE,
            ttnn.DataType.BFLOAT16,
            mapper,
        ),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def validate_mesh_graph_descriptor(mesh_shape: list[int]) -> None:
    """Validate that the MGD file topology matches the requested mesh shape."""
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
        "--ddp",
        type=int,
        default=1,
        help="Number of devices for distributed data parallel (default: 1).",
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
    num_devices = args.ddp
    use_ddp = num_devices > 1

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

    # ── DDP mapper ────────────────────────────────────────────────────────────

    mapper = None
    if use_ddp:
        device = autograd_ctx.get_device()
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)

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
    collate = partial(causal_lm_collate, seq_len=seq_len, mapper=mapper)
    train_loader = InMemoryDataloader(dataset, collate, batch_size=batch_size, shuffle=True)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    callbacks: list[TrainerCallback] = []
    if use_ddp:
        callbacks.append(DDPCallback())
    if args.loss_log:
        callbacks.append(LossLogger(args.loss_log))

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
