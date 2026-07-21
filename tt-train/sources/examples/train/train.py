# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT/Llama/DeepSeek/Qwen3 training entry point built on `SFTTrainer`.

Two modes, selected by CLI flags:

* Training (default): build dataset + model + trainer from a YAML config and CLI
  overrides, then run.
* Inference: with both `--prompt` and `--model-path` set, load a checkpoint and
  generate text.

Old `train_nanogpt.py` callers keep working: underscore-form flags are accepted
silently, and the three renamed flags (`--num_epochs`, `--clip_grad_norm`,
`--model_save_path`) are accepted with a deprecation warning.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

import ttnn
from ttnn.device import is_blackhole, is_wormhole_b0
import ttml
import ttml.common.muon_optimizer
from ttml.common.config import DeviceConfig, SpeedrunSchedulerConfig, TrainingConfig as BaseTrainingConfig, load_config
from ttml.common.data import CharTokenizer
from ttml.common.schedulers import SpeedrunScheduler
from ttml.common.utils import (
    build_causal_mask,
    build_mesh,
    create_optimizer,
    get_available_device_memory_in_bytes,
    get_tt_metal_runtime_root,
    round_up_to_tile,
    summary,
)
from ttml.datasets import InMemoryDataloader, causal_lm_collate_fn
from ttml.trainers import SFTConfig, SFTTrainer, TrainerCallback

from formatting import HEADER_WIDTH, print_footer, print_header, shorten_home
from model_builders import FLOPS_REGISTRY, Model, ModelConfig, instantiate_model_from_config, parse_model_config
from callbacks import (
    AverageLossCallback,
    DramFootprintCallback,
    MemoryTrackerCallback,
    MoECallback,
    ThroughputCallback,
)
from checkpointing import (
    build_checkpoint_io,
    find_latest_checkpoint,
    load_for_inference,
    peek_checkpoint,
    prefixed,
)
from cli import parse_args

# Side effect: registers the Muon optimizer with ttml's optimizer factory so
# YAML configs can name it.
ttml.common.muon_optimizer.register()

MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker


# ── Training config ───────────────────────────────────────────────────────────


class TrainingConfig(BaseTrainingConfig):
    """Base training config + NanoGPT-specific defaults + legacy field-name aliases."""

    def __init__(self, yaml_config: dict | None = None) -> None:
        yaml_config = yaml_config if yaml_config is not None else {}
        super().__init__(yaml_config)
        tc = yaml_config.get("training_config", {})

        self.project_name = tc.get("project_name", "tt_train_nano_gpt")
        self.data_path = tc.get("data_path", "")
        self.scheduler_type = tc.get("scheduler_type", "identity")
        self.use_clip_grad_norm = tc.get("use_clip_grad_norm", False)
        self.clip_grad_norm_max_norm = float(tc.get("clip_grad_norm_max_norm", 1.0))

        # Override base-class defaults with NanoGPT-specific values.
        self.seed = int(tc.get("seed", 5489))
        self.max_steps = int(tc.get("max_steps", 5000))

        # Legacy field names kept alive for callers that still read them.
        self.num_epochs = self.epochs
        self.model_save_interval = self.save_every


# ── Mesh / device ─────────────────────────────────────────────────────────────


def _device_arch_name() -> str:
    """Short identifier for the active device's architecture (used in the header table)."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    if is_wormhole_b0(device):
        return "wormhole_b0"
    if is_blackhole(device):
        return "blackhole"
    return "unknown"


def get_device_peak_tflops_bf16() -> float:
    """Per-device theoretical BF16 TFLOPS. Whole-mesh peak = this × num_devices."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    # Per-core BF16 TFLOPS for each supported TT architecture.
    if is_wormhole_b0(device):
        per_core = 1.0
    elif is_blackhole(device):
        per_core = 1.35
    else:
        raise ValueError(f"Unknown device: {device.arch()}")
    return num_cores * per_core


# ── Dataset ───────────────────────────────────────────────────────────────────


class CausalLMDataset:
    """Sliding-window dataset of `{input_ids, labels}` samples; stride 1, labels shifted by one."""

    def __init__(self, tokens: np.ndarray, seq_length: int) -> None:
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self) -> int:
        if len(self.tokens) <= self.seq_length:
            return 0
        return len(self.tokens) - self.seq_length

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {
            "input_ids": self.tokens[index : index + self.seq_length].tolist(),
            "labels": self.tokens[index + 1 : index + self.seq_length + 1].tolist(),
        }


def build_dataset(data_path: str, seq_len: int, vocab_size: int) -> tuple[CausalLMDataset, CharTokenizer | None]:
    """Build (dataset, tokenizer). `.yaml`/`.yml` paths = pre-tokenized; otherwise plain text + char tokenizer."""
    is_pretokenized = data_path.endswith((".yaml", ".yml"))

    if is_pretokenized:
        if vocab_size == 0:
            raise ValueError(f"Pre-tokenized data ({data_path}) requires vocab_size in the model config.")
        with open(data_path, "r") as f:
            token_data = yaml.safe_load(f)
        tokens = np.array(token_data["tokens"], dtype=np.uint32)
        data_vocab_size = int(token_data["tokenizer_vocab_size"])
        max_token_id = int(tokens.max())
        if max_token_id >= vocab_size:
            raise ValueError(
                f"Tokenized data contains token id {max_token_id} but model vocab_size is "
                f"{vocab_size} (data file reports tokenizer_vocab_size={data_vocab_size})."
            )
        return CausalLMDataset(tokens, seq_len), None

    text = Path(data_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    # Char tokenization auto-detects vocab_size, so the config should leave it 0. On resume the
    # config carries the checkpoint's (raw) vocab, which must match the freshly-detected one.
    if vocab_size not in (0, tokenizer.vocab_size):
        raise ValueError(
            f"Plain text data ({data_path}) auto-detects vocab_size={tokenizer.vocab_size}, "
            f"but the model config has vocab_size={vocab_size} (set it to 0 to auto-detect)."
        )
    tokens = np.array(tokenizer.encode(text), dtype=np.uint32)
    return CausalLMDataset(tokens, seq_len), tokenizer


# ── LR schedule ───────────────────────────────────────────────────────────────

_WARMUP_LINEAR_WARMUP_FRACTION = 0.1
_WARMUP_LINEAR_MIN_LR_FRACTION = 0.01


def build_lr_schedule(training_cfg: TrainingConfig, optimizer: Any, max_steps: int) -> Callable[[int], float]:
    """Return a `step -> lr` callable. `warmup_linear` runs warmup → linear decay; anything else is constant LR."""
    base_lr = optimizer.get_lr()

    if training_cfg.scheduler_type == "warmup_linear":
        warmup_steps = int(max_steps * _WARMUP_LINEAR_WARMUP_FRACTION)
        sched = SpeedrunScheduler(
            SpeedrunSchedulerConfig(
                max_lr=base_lr,
                min_lr=base_lr * _WARMUP_LINEAR_MIN_LR_FRACTION,
                warmup_steps=warmup_steps,
                hold_steps=0,
                total_steps=max_steps,
            )
        )
        return sched.lr_at

    return lambda _step: base_lr


# ── Inference ─────────────────────────────────────────────────────────────────

_MAX_RANDOM_SEED = 2**32 - 1


def _extract_last_step_logits(logits: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
    """Slice the final sequence position from `logits` regardless of tensor rank (handles 5D, 4D, and flat shapes)."""
    shape = logits.shape()
    if len(shape) == 5:
        seq = shape[3]
        sliced = ttnn.slice(
            logits.get_value(),
            [0, 0, 0, seq - 1, 0],
            [shape[0], shape[1], shape[2], seq, shape[4]],
        )
        return ttml.autograd.create_tensor(ttnn.reshape(sliced, [shape[0], 1, 1, shape[4]]), requires_grad=False)
    if len(shape) == 4:
        seq = shape[2]
        sliced = ttnn.slice(
            logits.get_value(),
            [0, 0, seq - 1, 0],
            [shape[0], shape[1], seq, shape[3]],
        )
        return ttml.autograd.create_tensor(ttnn.reshape(sliced, [shape[0], 1, 1, shape[3]]), requires_grad=False)
    # Fallback for any other rank: flatten to [positions, vocab] and keep the last position.
    # Assumes batch_size == 1 — with batch > 1 the batch dim folds into `positions`, so this
    # would keep only the last batch element's final position. Inference is single-prompt, so it holds.
    vocab = shape[-1]
    flat = ttnn.reshape(logits.get_value(), [-1, vocab])  # [positions, vocab]
    positions = flat.shape[0]
    if positions > 1:
        flat = ttnn.slice(flat, [positions - 1, 0], [positions, vocab])  # last position → [1, vocab]
    last = ttnn.reshape(flat, [1, 1, 1, vocab])  # canonicalize to [1, 1, 1, vocab]
    return ttml.autograd.create_tensor(last, requires_grad=False)


def _vocab_pad_mask(padded_vocab: int, vocab_size: int) -> Any:
    """Additive mask `[1, 1, 1, padded_vocab]` that drives the tile-padding columns `[vocab_size:]` to -inf.

    Constant for a given `(padded_vocab, vocab_size)`, so callers build it once and reuse it across
    sampling steps rather than re-transferring it to device every token.
    """
    bias_np = np.zeros((1, 1, 1, padded_vocab), dtype=np.float32)
    bias_np[..., vocab_size:] = -1e9
    return ttml.autograd.Tensor.from_numpy(
        bias_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    ).get_value()


def _sample_next_token(
    last_logits: ttml.autograd.Tensor,
    vocab_size: int,
    temperature: float,
    top_k: int,
    pad_mask: Any | None = None,
) -> int:
    """Pick the next token id. Argmax when `temperature ≈ 0`; otherwise top-k filter + temperature-scaled sample."""
    # Logits span the tile-padded vocab; columns >= the real vocab_size are padding slots with
    # arbitrary values. `pad_mask` (when the vocab is padded) biases them to -inf so neither argmax
    # nor sampling can select an out-of-range id. last_logits is always [1, 1, 1, vocab].
    if pad_mask is not None:
        last_logits = ttml.autograd.create_tensor(ttnn.add(last_logits.get_value(), pad_mask), requires_grad=False)

    if temperature < 0.01:
        argmax = ttnn.argmax(last_logits.get_value(), dim=3, keepdim=True)
        return int(argmax.item())
    if 0 < top_k < vocab_size:  # top_k >= vocab_size keeps everything → no filtering needed
        logits = last_logits.get_value()
        topk_values, topk_indices = ttnn.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        threshold = ttnn.slice(topk_values, [0, 0, 0, top_k - 1], [1, 1, 1, top_k])
        below_threshold = ttnn.lt(logits, threshold)
        neg_inf = ttnn.full_like(logits, -1e9, dtype=ttnn.bfloat16)
        filtered = ttnn.where(below_threshold, neg_inf, logits)
        for t in (topk_values, topk_indices, threshold, below_threshold, neg_inf):
            ttnn.deallocate(t)
        last_logits = ttml.autograd.create_tensor(filtered, requires_grad=False)
    seed = random.randint(0, _MAX_RANDOM_SEED)
    sampled = ttml.ops.sample.sample_op(last_logits, temperature, seed, None)
    return int(sampled.get_value().item())


def generate(
    model: Model,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    sequence_length: int,
    mask: ttml.autograd.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
) -> tuple[str, float, float]:
    """Generate up to `max_new_tokens` tokens from `prompt`; greedy when `temperature ≈ 0`, else `temperature`/`top_k` sampling.

    Does no printing. Returns `(text, elapsed_s, first_token_s)` — total generation time and the
    first token's time (compile-heavy), so callers can report total and steady-state speed.
    """
    model.eval()
    ttml.autograd.AutoContext.get_instance().reset_graph()
    device = ttml.autograd.AutoContext.get_instance().get_device()

    if not prompt:
        prompt = " "
    prompt_ids = tokenizer.encode(prompt)

    running = list(prompt_ids[:sequence_length])
    if len(running) < sequence_length:
        pad_id = tokenizer.stoi.get(" ", next(iter(tokenizer.stoi.values()), 0))
        running = [pad_id] * (sequence_length - len(running)) + running

    generated = []

    input_ttnn = ttnn.from_buffer(
        buffer=running[-sequence_length:],
        shape=[1, 1, 1, sequence_length],
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    vocab_size = tokenizer.vocab_size
    # Built lazily on the first step (the padded logit width is only known after a forward) and
    # reused across steps; it survives the per-step reset_graph below like `mask` does.
    pad_mask: Any | None = None
    start = time.time()
    first_token_s = 0.0
    for step in range(max_new_tokens):
        model_input = ttml.autograd.create_tensor(input_ttnn, requires_grad=False)
        # Clone the mask each step; the autograd graph reset below invalidates prior tensors.
        mask_for_model = ttml.autograd.create_tensor(ttnn.clone(mask.get_value()), requires_grad=False)
        logits = model(model_input, mask_for_model)

        last_logits = _extract_last_step_logits(logits)
        if pad_mask is None and last_logits.get_value().shape[3] > vocab_size:
            pad_mask = _vocab_pad_mask(last_logits.get_value().shape[3], vocab_size)
        next_id = _sample_next_token(last_logits, vocab_size, temperature, top_k, pad_mask)

        running.append(next_id)
        generated.append(next_id)

        shifted = ttnn.slice(input_ttnn, [0, 0, 0, 1], [1, 1, 1, sequence_length])
        new_token = ttnn.from_buffer(
            buffer=[next_id], shape=[1, 1, 1, 1], dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        input_ttnn = ttnn.concat([shifted, new_token], dim=3)
        ttnn.deallocate(shifted)
        ttnn.deallocate(new_token)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        if step == 0:
            first_token_s = time.time() - start

    elapsed = time.time() - start
    text = tokenizer.decode(generated)
    return text, elapsed, first_token_s


# ── Run modes ─────────────────────────────────────────────────────────────────


def _resolve_data_path(training_cfg: TrainingConfig) -> str:
    """Return explicit `training_cfg.data_path` if set; otherwise probe known shakespeare.txt locations."""
    if training_cfg.data_path:
        return training_cfg.data_path
    candidates = [
        Path("data/shakespeare.txt"),
        Path("tt-train/data/shakespeare.txt"),
        Path("../data/shakespeare.txt"),
        Path(__file__).resolve().parents[3] / "data" / "shakespeare.txt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No data_path set and shakespeare.txt not found in any of: {candidates}")


def run_training(
    args: argparse.Namespace,
    yaml_config: dict,
    training_cfg: TrainingConfig,
    model_cfg: ModelConfig,
    device_cfg: DeviceConfig,
) -> None:
    """Build dataset/model/optimizer/callbacks/trainer from configs + CLI overrides, then run training."""
    mem_guard = None
    if args.track_memory:
        # Open the capture here so the setup snapshots below land in one session; held for the
        # whole run (release ends the capture), and MemoryTrackerCallback closes it after step 1.
        mem_guard = MemoryUsageTracker.begin_capture()  # noqa: F841 -- held to keep the capture active
        MemoryUsageTracker.snapshot("ENTRY")

    # Build everything (printing phase progress for the slow steps), then emit the header block.
    data_path = _resolve_data_path(training_cfg)

    resume_path = None
    if not args.fresh:
        if args.resume:
            resume_path = args.resume
        elif args.checkpoint_dir:
            resume_path = find_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)

    resume_step: int | None = None
    if resume_path:
        try:
            resume_step, model_cfg = peek_checkpoint(resume_path)
        except Exception:
            resume_path = None

    # model_cfg is now final (the checkpoint's on resume), so build the dataset from its
    # seq_len — dataset windowing must match the model and collate_fn(seq_len=...) below.
    seq_len = model_cfg.max_sequence_length
    print("Loading data...", flush=True)
    dataset, tokenizer = build_dataset(data_path, seq_len, model_cfg.vocab_size)

    # Char data auto-detects its vocab (config leaves it 0); record the real size so the model
    # and checkpoint use it. Pre-tokenized data already has vocab_size from the config.
    if tokenizer is not None:
        model_cfg.vocab_size = tokenizer.vocab_size

    # Lazy alloc only helps when sharding, so default it on under FSDP (--no-lazy to disable).
    lazy_init = device_cfg.enable_fsdp and not args.no_lazy
    print("Building model...", flush=True)
    model = instantiate_model_from_config(model_cfg, lazy_init=lazy_init, use_tp=device_cfg.enable_tp)

    flops_per_token = 0
    flops_fn = FLOPS_REGISTRY.get(model_cfg.model_type)
    if flops_fn is not None:
        flops_per_token = flops_fn(model.config, seq_len)

    # Shard params before the optimizer is built so its state matches the sharded shapes (the caller
    # owns sharding, not the trainer).
    if device_cfg.enable_fsdp:
        print("Sharding model...", flush=True)
        for block in model.blocks:
            ttml.fsdp.fully_shard(block)
        ttml.fsdp.fully_shard(model)

    # Materialize after fully_shard rewrote the mappers, so weights allocate already-sharded.
    if lazy_init:
        print("Materializing parameters...", flush=True)
        ttml.materialize_module(model)

    if args.print_summary:
        summary(model)
    total_params = sum(math.prod(p.shape()) for p in model.parameters().values())
    if args.track_memory:
        MemoryUsageTracker.snapshot("MODEL_CREATION")

    optimizer = create_optimizer(model, yaml_config)
    if args.track_memory:
        MemoryUsageTracker.snapshot("OPTIMIZER_CREATION")

    # DeepSeek's composite SDPA needs the mask passed explicitly; gpt2/llama/qwen3 use a fused
    # SDPA that materializes its own causal mask internally, so only build it for DeepSeek.
    attn_mask_for_model = None
    if model_cfg.model_type == "deepseek":
        attn_mask_for_model = ttml.autograd.Tensor.from_numpy(
            build_causal_mask(seq_len), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
        )

    mesh = ttml.mesh()
    # Shard the batch across the data-parallel axes (dp and/or fsdp): one axis → shard along it;
    # HSDP (both) → each device gets a unique B/(D*F) slice; HSDP+TP is unsupported.
    data_axes = [a for a in ("dp", "fsdp") if mesh.has_axis(a) and mesh.axis_size(a) > 1]
    if len(data_axes) == 1:
        mapper = mesh.axis_mapper(data_axes[0], tdim=0)
    elif len(data_axes) >= 2:
        if mesh.has_axis("tp") and mesh.axis_size("tp") > 1:
            raise NotImplementedError("HSDP + TP batch sharding is not supported.")
        flat_size = int(np.prod([mesh.axis_size(a) for a in data_axes]))
        if training_cfg.batch_size % flat_size != 0:
            raise ValueError(
                f"HSDP batch sharding requires batch_size ({training_cfg.batch_size}) to be "
                f"divisible by D*F ({flat_size}); got data_axes={data_axes}."
            )
        device = ttml.autograd.AutoContext.get_instance().get_device()
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0, None)
    else:
        mapper = None
    collate = partial(causal_lm_collate_fn, seq_len=seq_len, mapper=mapper)
    dataloader = InMemoryDataloader(
        dataset, collate, batch_size=training_cfg.batch_size, shuffle=True, drop_last=True, seed=training_cfg.seed
    )

    peak_tflops = get_device_peak_tflops_bf16() * mesh.num_devices() if flops_per_token > 0 else 0.0

    callbacks: list[TrainerCallback] = []

    if model_cfg.model_type == "deepseek":
        callbacks.append(MoECallback(args.log_expert_activations))

    # Metrics.
    callbacks.append(ThroughputCallback(flops_per_token, peak_tflops, log_interval=1))
    avg_loss_cb = AverageLossCallback()
    callbacks.append(avg_loss_cb)

    # Peak DRAM footprint over the first few steps, which cover the peak. Near-zero overhead, always on;
    # prints the footprint during training once its window closes.
    callbacks.append(DramFootprintCallback())

    # Diagnostics.
    if args.track_memory:
        callbacks.append(MemoryTrackerCallback())

    if training_cfg.use_clip_grad_norm and (device_cfg.enable_tp or device_cfg.enable_fsdp):
        raise ValueError("Clip grad norm is not supported with TP or FSDP")

    saver, loader = build_checkpoint_io(tokenizer, model_cfg)

    # Match old train_nanogpt.py: training stops at min(max_steps, num_epochs × batches_per_epoch / grad_accum).
    batches_per_epoch = len(dataloader)
    grad_accum = max(1, training_cfg.gradient_accumulation_steps)
    epoch_bound = (training_cfg.num_epochs * batches_per_epoch) // grad_accum
    effective_max_steps = min(training_cfg.max_steps, epoch_bound) if epoch_bound > 0 else training_cfg.max_steps

    # Schedule over the steps actually run, so warmup + decay complete by the end of training.
    schedule = build_lr_schedule(training_cfg, optimizer, effective_max_steps)

    sft_cfg = SFTConfig(
        max_steps=effective_max_steps,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        save_interval=training_cfg.model_save_interval if args.checkpoint_dir else 0,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_prefix=args.checkpoint_prefix,
        max_grad_norm=(training_cfg.clip_grad_norm_max_norm if training_cfg.use_clip_grad_norm else 0.0),
        disable_progress_bar=True,
    )

    def _causal_lm_loss(logits, batch):
        if mesh.has_axis("tp") and mesh.axis_size("tp") > 1:
            return ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
                logits, batch.labels, cluster_axis=mesh.axis_index("tp"), reduce=ttml.ops.ReduceType.MEAN
            )
        return ttml.ops.loss.cross_entropy_loss(logits, batch.labels, ttml.ops.ReduceType.MEAN)

    trainer = SFTTrainer(
        model=model,
        train_dataloader=dataloader,
        eval_dataloader=None,
        config=sft_cfg,
        optimizer=optimizer,
        lr_schedule=schedule,
        callbacks=callbacks,
        attention_mask=attn_mask_for_model,
        checkpoint_saver=saver,
        checkpoint_loader=loader,
        compute_loss_func=_causal_lm_loss,
    )

    if resume_path:
        print(f"Resuming from {shorten_home(resume_path)}...", flush=True)
        trainer.load_checkpoint(resume_path)

    # ─ Build header sections ─
    padded_vocab = round_up_to_tile(model_cfg.vocab_size, 32)
    if model_cfg.vocab_size != padded_vocab:
        vocab_str = f"{model_cfg.vocab_size} -> {padded_vocab} padded"
    else:
        vocab_str = f"{model_cfg.vocab_size}"

    if training_cfg.scheduler_type == "warmup_linear":
        warmup = int(effective_max_steps * _WARMUP_LINEAR_WARMUP_FRACTION)
        schedule_str = f"warmup_linear ; {warmup:,} warmup ; {effective_max_steps - warmup:,} decay"
    else:
        schedule_str = "constant"

    model_fields: list[tuple[str, str]] = [
        (
            "arch",
            f"{model_cfg.num_blocks} layers ; {model_cfg.embedding_dim} dim ; {model_cfg.num_heads} heads ; seq {seq_len}",
        ),
        ("params", f"{total_params:,}"),
    ]
    if flops_per_token > 0:
        model_fields.append(("flops", f"{flops_per_token / 1e9:.3g}G per token"))
    grad_ckpt = "on" if model_cfg.runner_type == ttml.models.RunnerType.MemoryEfficient else "off"
    model_fields.append(("grad ckpt", grad_ckpt))

    optimizer_fields: list[tuple[str, str]] = [
        ("name", optimizer.get_name()),
        ("lr", f"{optimizer.get_lr():.2e}"),
        ("schedule", schedule_str),
    ]
    if training_cfg.use_clip_grad_norm:
        optimizer_fields.append(("clip", f"max-norm {training_cfg.clip_grad_norm_max_norm}"))

    # Training section: run length, data coverage, and seed.
    steps_str = f"{effective_max_steps:,}"
    if effective_max_steps < training_cfg.max_steps:
        steps_str += f" (epoch-capped from {training_cfg.max_steps:,})"
    passes = effective_max_steps * grad_accum / batches_per_epoch if batches_per_epoch else 0.0
    training_fields: list[tuple[str, str]] = [
        ("steps", steps_str),
        ("epochs", f"{passes:.3g} ({training_cfg.num_epochs} configured)"),
        ("seed", str(training_cfg.seed)),
    ]

    dp_size = mesh.axis_size("dp") if mesh.has_axis("dp") else 1
    global_batch = training_cfg.batch_size * grad_accum * dp_size
    batch_str = (
        f"size {training_cfg.batch_size} ; accum {training_cfg.gradient_accumulation_steps} "
        f"; global {global_batch:,} ; dropout {model_cfg.dropout_prob}"
    )

    hardware_fields: list[tuple[str, str]] = [
        ("chip", _device_arch_name()),
        ("mesh", "x".join(str(s) for s in mesh.shape)),
    ]
    if peak_tflops > 0:
        hardware_fields.append(("peak", f"{peak_tflops:.1f} TFLOPS bf16"))
    hardware_fields.append(("memory", f"{get_available_device_memory_in_bytes() / (1024 * 1024):,.0f} MB"))

    ckpt_lines: list[str] = []
    if args.checkpoint_dir:
        ckpt_pattern = os.path.join(args.checkpoint_dir, prefixed(args.checkpoint_prefix, "step_*.pkl"))
        ckpt_lines.append(f"save every {training_cfg.model_save_interval} -> {shorten_home(ckpt_pattern)}")
    if resume_path:
        ckpt_lines.append(f"resume {shorten_home(resume_path)} @ step {resume_step}")

    diag_lines: list[str] = []
    if args.track_memory:
        diag_lines.append("memory tracking enabled")

    sections: list[tuple[str, list[tuple[str, str]] | str]] = [
        ("config", args.config),
        ("training", training_fields),
        (
            "data",
            [
                ("path", shorten_home(data_path)),
                ("samples", f"{len(dataset):,}"),
                ("vocab", vocab_str),
            ],
        ),
        ("model", model_fields),
        ("optimizer", optimizer_fields),
        ("batch", batch_str),
        ("hardware", hardware_fields),
    ]
    if ckpt_lines:
        sections.append(("checkpoint", "\n".join(ckpt_lines)))
    if diag_lines:
        sections.append(("diagnostics", "\n".join(diag_lines)))

    print_header(f"{model_cfg.model_type} - training", sections)

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    if args.checkpoint_dir:
        final_path = os.path.join(args.checkpoint_dir, prefixed(args.checkpoint_prefix, "final.pkl"))
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        saver(trainer, final_path)

    print()
    print("=" * 70)
    print("Training completed!")
    print(f"  - Total steps: {trainer.step}")
    print(f"  - Total time: {elapsed:.2f} s")
    print(f"  - Average loss: {avg_loss_cb.average:.6f}")
    print("=" * 70)


def run_inference(args: argparse.Namespace, seed: int) -> None:
    """Load checkpoint via `--model-path` and generate text from `--prompt`."""
    model, tokenizer, model_cfg, loaded_step = load_for_inference(args.model_path)
    seq_len = model_cfg.max_sequence_length

    causal_mask = ttml.autograd.Tensor.from_numpy(
        build_causal_mask(seq_len), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )

    total_params = sum(math.prod(p.shape()) for p in model.parameters().values())
    padded_vocab = round_up_to_tile(model_cfg.vocab_size, 32)
    vocab_str = (
        f"{tokenizer.vocab_size} -> {padded_vocab} padded"
        if tokenizer.vocab_size != padded_vocab
        else str(model_cfg.vocab_size)
    )

    sections: list[tuple[str, list[tuple[str, str]] | str]] = [
        (
            "checkpoint",
            [
                ("path", shorten_home(args.model_path)),
                ("step", f"{loaded_step:,}"),
            ],
        ),
        (
            "model",
            [
                (
                    "arch",
                    f"{model_cfg.num_blocks} layers ; {model_cfg.embedding_dim} dim ; {model_cfg.num_heads} heads ; seq {seq_len}",
                ),
                ("params", f"{total_params:,}"),
                ("vocab", vocab_str),
            ],
        ),
        ("prompt", repr(args.prompt)),
        (
            "generate",
            [
                ("tokens", str(args.max_new_tokens)),
                ("temp", str(args.temperature)),
                ("top-k", str(args.top_k)),
                ("mode", "greedy" if args.temperature < 0.01 else "sampling"),
                ("seed", str(seed)),
            ],
        ),
        (
            "hardware",
            [
                ("chip", _device_arch_name()),
                ("mesh", "x".join(str(s) for s in ttml.mesh().shape)),
                ("memory", f"{get_available_device_memory_in_bytes() / (1024 * 1024):,.0f} MB"),
            ],
        ),
    ]
    print_header(f"{model_cfg.model_type} - inference", sections)

    text, elapsed, first_token_s = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        seq_len,
        causal_mask,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    output_divider = "[OUTPUT] "
    print(output_divider + "-" * (HEADER_WIDTH - len(output_divider)))
    print(text)

    n = args.max_new_tokens
    speed = f"{n / elapsed:.3g} tok/s" if elapsed > 0 else "n/a"
    if n > 1 and elapsed > first_token_s:
        speed += f"  ({(n - 1) / (elapsed - first_token_s):.3g} steady)"
    print_footer(
        f"{model_cfg.model_type} - generated",
        [("tokens", f"{n:,}"), ("time", f"{elapsed:.2f} s"), ("speed", speed)],
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse args, load configs + CLI overrides, open device mesh, dispatch to inference or training."""
    args = parse_args()

    tt_metal_root = get_tt_metal_runtime_root()
    tt_train_root = f"{tt_metal_root}/tt-train"
    configs_root = f"{tt_train_root}/configs"
    try:
        yaml_config = load_config(args.config, f"{configs_root}/training_configs")
        training_cfg = TrainingConfig(yaml_config)
        if training_cfg.model_config:
            model_yaml = load_config(training_cfg.model_config, tt_train_root)
            model_cfg = parse_model_config(model_yaml)
        else:
            print("Warning: no model_config specified; using defaults", file=sys.stderr)
            model_cfg = parse_model_config({})
    except FileNotFoundError as e:
        print(f"Error: config file not found: {e}", file=sys.stderr)
        raise

    if args.data_path:
        training_cfg.data_path = args.data_path
    if args.batch_size is not None:
        training_cfg.batch_size = args.batch_size
    if args.max_steps is not None:
        training_cfg.max_steps = args.max_steps
    if args.epochs is not None:
        training_cfg.num_epochs = args.epochs
    if args.max_grad_norm is not None:
        training_cfg.use_clip_grad_norm = True
        training_cfg.clip_grad_norm_max_norm = args.max_grad_norm
    if args.sequence_length is not None:
        model_cfg.max_sequence_length = args.sequence_length

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    device_cfg = DeviceConfig(yaml_config)

    mesh = build_mesh(device_cfg)
    ttml.open_device_mesh(mesh, tuple(device_cfg.device_ids) if device_cfg.device_ids else None)
    ttml.autograd.AutoContext.get_instance().get_device()
    ttml.manual_seed(training_cfg.seed)
    np.random.seed(training_cfg.seed)
    random.seed(training_cfg.seed)  # Python RNG drives the per-token sampling seed in inference

    inference_only = args.prompt and args.model_path
    try:
        if inference_only:
            run_inference(args, training_cfg.seed)
        else:
            run_training(args, yaml_config, training_cfg, model_cfg, device_cfg)
    finally:
        ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
