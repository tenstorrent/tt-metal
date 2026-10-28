#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA fine-tuning of Qwen3.8-27B on GovReport long-document summarization at seq_len 8192.

Each example is a government report (3k-7.5k tokens) followed by its summary, padded to
``--seq-len``. The loss is the mean cross-entropy over the summary tokens only, so lowering it
requires reading the report. Data comes from ``prepare_govreport.py``.

Runs on a 1x4 Blackhole mesh with TP=4 + sequence parallelism and full block recompute (the
fastest 8k configuration measured, see ../PERF_ANALYSIS.md). Logs to stdout and Weights & Biases.

    See README.md in this directory (run_govreport.sh sets up the environment).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

import ttnn
import ttml
from ttml.checkpointing import load_checkpoint, save_checkpoint
from ttml.common.performance import get_device_peak_tflops_bf16
from ttml.models.qwen38 import Qwen38Config, Qwen38Transformer
from ttml.models.qwen38.flops import flops_per_token
from ttml.models.qwen38.loading import load_from_safetensors
from ttml.models.qwen38.lora import apply_lora, build_lora_config, trainable_summary

DEFAULT_MODEL = os.environ.get("QWEN38_MODEL_PATH", "/localdev/aleksajovanovic/models/Qwen/Qwen3.8-27B")
DEFAULT_DATA = os.environ.get("GOVREPORT_DATA", "/localdev/umales/data/govreport/tok")
# Column-parallel LoRA targets: their lora_A is replicated over TP (see ../LORA_A_GRADIENT.md).
COLUMN_PARALLEL = ("q_proj", "k_proj", "v_proj", "in_proj_qkv", "in_proj_z")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--updates", type=int, default=400, help="optimizer steps")
    p.add_argument("--grad-accum", type=int, default=4, help="examples per optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine decays to lr * this")
    p.add_argument("--warmup", type=int, default=20, help="linear warmup, in optimizer steps")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--eval-every", type=int, default=50, help="optimizer steps between evals (0 = only start/end)")
    p.add_argument("--eval-examples", type=int, default=32)
    p.add_argument("--save-every", type=int, default=100, help="export LoRA tensors as .npz every N updates")
    p.add_argument(
        "--ckpt-every",
        type=int,
        default=0,
        help="write <out>/checkpoint.pkl (LoRA + AdamW state + data position) every N updates, for --resume "
        "(0 = never)",
    )
    p.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pkl if it exists")
    p.add_argument(
        "--start-update",
        type=int,
        default=1,
        help="debugging: begin at this update's data position and LR (fresh LoRA, no step-0 eval)",
    )
    p.add_argument("--out", default="/localdev/umales/runs/qwen38_govreport")
    p.add_argument("--layers", type=int, default=0, help="truncate the stack (smoke tests only)")
    p.add_argument("--no-sequence-parallel", action="store_true")
    p.add_argument("--check-lora-a", action="store_true", help="assert replicated lora_A stays identical over TP")
    p.add_argument("--debug-grads", action="store_true", help="print per-tensor grad/param magnitudes each step")
    p.add_argument("--wandb-project", default="qwen38-lora-govreport")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class Examples:
    """Tokenized examples from prepare_govreport.py, served as padded (inputs, targets, weights)."""

    def __init__(self, path: Path, seq_len: int, pad_id: int) -> None:
        d = np.load(path)
        self.tokens, self.offsets, self.prompt_lens = d["tokens"], d["offsets"], d["prompt_lens"]
        self.seq_len, self.pad_id = seq_len, pad_id

    def __len__(self) -> int:
        return len(self.prompt_lens)

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        ex = self.tokens[self.offsets[i] : self.offsets[i + 1]]
        n = len(ex)
        assert n <= self.seq_len + 1
        full = np.full(self.seq_len + 1, self.pad_id, dtype=np.uint32)
        full[:n] = ex
        inputs, targets = full[:-1], full[1:].copy()
        # Target j is token j+1: train on summary tokens (index >= prompt_len) up to and including eos.
        idx = np.arange(1, self.seq_len + 1)
        mask = (idx >= self.prompt_lens[i]) & (idx < n)
        targets[~mask] = 0
        count = int(mask.sum())
        # mean over all seq_len positions of loss * weight == mean over the summary tokens
        weights = mask.astype(np.float32) * (self.seq_len / count)
        return inputs, targets, weights, count


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.updates - args.warmup)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    tp = 4

    ttml.open_device_mesh(ttml.Mesh((1, tp), ("dp", "tp")))
    ttml.autograd.AutoContext.get_instance().initialize_parallelism_context(
        ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=True)
    )
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    mesh = ttml.mesh()
    tp_axis = mesh.axis_index("tp")
    composer = ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig([0, 1]))

    model_path = Path(args.model_path)
    config = Qwen38Config.from_hf_json(model_path / "config.json")
    if args.layers:
        config.num_hidden_layers = args.layers
        config.layer_types = config.layer_types[: args.layers]
    config.use_tp = True
    config.delta_rule_impl = "fused"
    config.conv_impl = "fused"
    config.recompute_layers = True
    config.sequence_parallel = not args.no_sequence_parallel

    from transformers import AutoTokenizer

    pad_id = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True).eos_token_id
    train = Examples(Path(args.data) / "train.npz", args.seq_len, pad_id)
    val = Examples(Path(args.data) / "validation.npz", args.seq_len, pad_id)
    eval_ids = rng.choice(len(val), size=min(args.eval_examples, len(val)), replace=False)
    order = rng.permutation(len(train))
    print(f"train {len(train)} examples, val {len(val)} (eval on {len(eval_ids)}), seq_len {args.seq_len}")

    t0 = time.perf_counter()
    model = Qwen38Transformer(config)
    stats = load_from_safetensors(model, model_path, config)
    print(f"built + loaded {stats['loaded']} tensors in {time.perf_counter() - t0:.0f}s")
    model = apply_lora(model, build_lora_config(rank=args.rank, alpha=args.alpha))
    summary = trainable_summary(model)
    print(f"LoRA rank {args.rank} alpha {args.alpha}: {summary['trainable']:,} trainable / {summary['total']:,}")

    params = model.parameters()
    trainable = {name: p for name, p in params.items() if p.get_requires_grad()}
    optimizer = ttml.optimizers.AdamW(
        trainable, ttml.optimizers.AdamWConfig.make(args.lr, 0.9, 0.95, 1e-8, args.weight_decay)
    )
    replicated_a = [
        n for n in trainable if "lora_A" in n and any(f"{c}/" in n or f"{c}." in n for c in COLUMN_PARALLEL)
    ]

    fpt = flops_per_token(config, 1, args.seq_len, train_base=False)
    peak = get_device_peak_tflops_bf16() * tp

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_cfg = {**vars(args), "train_examples": len(train), "val_examples": len(val), **summary}
    (out / "config.json").write_text(json.dumps(run_cfg, indent=2, default=str))
    ckpt_path = out / "checkpoint.pkl"
    trainable_np = ttml.NamedParameters()
    for name, p in trainable.items():
        trainable_np[name] = p
    start_step = args.start_update - 1
    cursor = start_step * args.grad_accum
    if args.resume and ckpt_path.exists():
        t0 = time.perf_counter()
        header = load_checkpoint(str(ckpt_path), model_params=trainable_np, optimizer=optimizer)
        start_step, cursor = header["step"], header["cursor"]
        print(f"resumed from {ckpt_path} at update {start_step} ({cursor} examples) in {time.perf_counter() - t0:.0f}s")

    wandb = None
    if not args.no_wandb:
        import wandb as _wandb

        wandb = _wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=run_cfg, dir=str(out))

    def to_device(inputs, targets, weights):
        x = ttml.autograd.Tensor.from_numpy(inputs.reshape(1, 1, 1, -1), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
        y = ttml.autograd.Tensor.from_numpy(targets.reshape(1, -1), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
        w = ttml.autograd.Tensor.from_numpy(weights.reshape(1, 1, -1, 1), ttnn.Layout.TILE, ttnn.DataType.FLOAT32)
        w.set_requires_grad(False)
        return x, y, w

    def summary_loss(x, y, w):
        per_token = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
            model(x, None), y, cluster_axis=tp_axis, reduce=ttml.ops.ReduceType.NONE
        )
        return ttml.ops.unary.mean(ttml.ops.binary.mul(per_token, w))

    def scalar(t) -> float:
        return float(t.to_numpy(composer=composer).reshape(-1)[0])

    def evaluate() -> float:
        model.eval()
        total, tokens = 0.0, 0
        for i in eval_ids:
            inputs, targets, weights, count = val.get(int(i))
            loss = summary_loss(*to_device(inputs, targets, weights))
            total += scalar(loss) * count
            tokens += count
            ctx.reset_graph()
        model.train()
        return total / tokens

    def check_lora_a(step: int) -> None:
        worst = 0.0
        for name in replicated_a:
            # One host copy per device, read device by device (a composer may collapse replicated tensors).
            copies = np.stack(
                [ttnn.to_torch(t).float().numpy() for t in ttnn.get_device_tensors(trainable[name].get_value())]
            )
            assert len(copies) == tp and np.isfinite(copies).all(), f"{name}: {len(copies)} copies, finite check"
            worst = max(worst, float(np.abs(copies - copies[:1]).max()))
        print(f"[check] step {step}: max |lora_A copy - device 0| over {len(replicated_a)} tensors = {worst:.3e}")
        assert worst == 0.0, "replicated lora_A diverged across TP devices"

    def debug_grads(step: int, when: str) -> None:
        rows = []
        for name, p in trainable.items():
            v = p.to_numpy(composer=composer)
            g = p.get_grad()
            g = ttml.autograd.create_tensor(g, False).to_numpy(composer=composer) if p.is_grad_initialized() else None
            rows.append(
                (
                    name,
                    float(np.abs(v).max()),
                    None if g is None else float(np.nanmax(np.abs(g))),
                    bool(np.isfinite(v).all()),
                    g is None or bool(np.isfinite(g).all()),
                )
            )
        rows.sort(key=lambda r: -(r[2] or 0.0))
        bad = [r for r in rows if not (r[3] and r[4])]
        print(f"[grads] step {step} {when}: {len(bad)} non-finite tensors; largest grads:")
        for name, vmax, gmax, vf, gf in (bad or rows)[:8]:
            print(f"    {name}: |param| {vmax:.3e} |grad| {gmax} finite param {vf} grad {gf}")

    def save(step: int) -> None:
        # Raw per-device copies, stacked on a new leading axis: [tp, ...]. Column-parallel lora_B and
        # row-parallel lora_A are shards; everything else is replicated.
        arrays = {
            name: np.stack([ttnn.to_torch(t).float().numpy() for t in ttnn.get_device_tensors(p.get_value())])
            for name, p in trainable.items()
        }
        np.savez(out / f"lora_step{step:05d}.npz", **arrays)
        print(f"saved {len(arrays)} LoRA tensors to {out / f'lora_step{step:05d}.npz'}")

    def log(data: dict, step: int) -> None:
        print(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in data.items()), flush=True)
        if wandb:
            wandb.log(data, step=step)

    def checkpoint(step: int) -> None:
        t0 = time.perf_counter()
        save_checkpoint(
            str(ckpt_path), header={"step": step, "cursor": cursor}, model_params=trainable_np, optimizer=optimizer
        )
        print(f"checkpoint at update {step} written in {time.perf_counter() - t0:.1f}s", flush=True)

    model.train()
    if start_step == 0:
        t_eval = time.perf_counter()
        log({"eval/loss": evaluate(), "eval/seconds": time.perf_counter() - t_eval}, step=0)

    for step in range(start_step, args.updates):
        lr = lr_at(step, args)
        optimizer.set_lr(lr)
        optimizer.zero_grad()
        t_step = time.perf_counter()
        losses, counts = [], []
        for _ in range(args.grad_accum):
            example = int(order[cursor % len(order)])
            if args.start_update > 1:
                print(f"update {step + 1}: training example {example}", flush=True)
            inputs, targets, weights, count = train.get(example)
            cursor += 1
            loss = summary_loss(*to_device(inputs, targets, weights))
            if args.grad_accum > 1:
                loss_scaled = ttml.ops.binary.mul(loss, 1.0 / args.grad_accum)
            else:
                loss_scaled = loss
            loss_scaled.backward(False)
            losses.append(scalar(loss))
            counts.append(count)
            ctx.reset_graph()
        if args.debug_grads:
            debug_grads(step + 1, "before optimizer step")
        optimizer.step()
        ttnn.synchronize_device(device)
        if args.debug_grads:
            debug_grads(step + 1, "after optimizer step")
        elapsed = time.perf_counter() - t_step
        if not all(math.isfinite(v) for v in losses):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {losses}")

        computed_tokens = args.grad_accum * args.seq_len
        log(
            {
                "train/loss": float(np.average(losses, weights=counts)),
                "train/lr": lr,
                "train/step_seconds": elapsed,
                "train/examples_seen": cursor,
                "train/epoch": cursor / len(train),
                "perf/tokens_per_s": computed_tokens / elapsed,
                "perf/mfu": computed_tokens / elapsed * fpt / 1e12 / peak,
            },
            step=step + 1,
        )
        if args.check_lora_a:
            check_lora_a(step + 1)
        done = step + 1 == args.updates
        if (args.eval_every and (step + 1) % args.eval_every == 0) or done:
            t_eval = time.perf_counter()
            log({"eval/loss": evaluate(), "eval/seconds": time.perf_counter() - t_eval}, step=step + 1)
        if (args.save_every and (step + 1) % args.save_every == 0) or done:
            save(step + 1)
        if args.ckpt_every and ((step + 1) % args.ckpt_every == 0 or done):
            checkpoint(step + 1)

    if wandb:
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
