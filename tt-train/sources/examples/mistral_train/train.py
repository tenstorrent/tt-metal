# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA fine-tuning of Mistral-Small-24B-Base-2501 on one chip (tt-train bring-up).

The full 40-layer model does not fit one Blackhole chip (43.9 GiB of bf16 weights), so ``--layers`` truncates
the stack; every run prints which depth it ran. Recipe and numbers: ``ttml/models/mistral/doc/plan.md``.

    python sources/examples/mistral_train/train.py --layers 4 --steps 6 --warmup-steps 2 --random-init --random-data
    python sources/examples/mistral_train/train.py --layers 8 --steps 20 --loss-log loss.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

import ttnn
import ttml
from ttml.common.performance import get_device_peak_tflops_bf16
from ttml.common.utils import set_seed
from ttml.common.utils import get_loss_over_devices
from ttml.models import EmbeddingPlacement
from ttml.models.mistral import Mistral, MistralConfig, load_from_safetensors, useful_flops_per_step
from ttml.modules import LoraConfig, LoraModel, transpose_frozen_weights

DEFAULT_MODEL = "mistralai/Mistral-Small-24B-Base-2501"
DEFAULT_REVISION = "b0a2e4ed093c26997495ae625528f81ea04b749f"
LORA_TARGETS = [r"attention\.qkv_linear$", r"attention\.out_linear$", r"mlp\.w_gate_up$", r"mlp\.w2$"]
MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL, help="HF repo id or a local directory with config.json")
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--data", default=None, help="plain-text corpus; default tt-train/data/shakespeare.txt")
    p.add_argument("--layers", type=int, default=0, help="truncate the stack (0 = all 40; needs --tp 4 on this box)")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size: 1 = one chip; 4 = a 1x4 mesh over the box")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=2, help="steps excluded from the step-time average")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--random-init", action="store_true", help="skip the checkpoint (perf runs)")
    p.add_argument("--random-data", action="store_true", help="uniform random tokens (perf runs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--track-memory", action="store_true", help="print device memory after the first step")
    p.add_argument("--mem-probe", action="store_true", help="print allocated DRAM after the forward and after the step")
    p.add_argument(
        "--no-transposed-weights",
        action="store_true",
        help="keep the frozen base weights as [out, in] with transpose_b matmuls (the composite-phase path)",
    )
    p.add_argument(
        "--host-loss",
        action="store_true",
        help="also compute the loss on the host in fp32 from the device logits (unquantised; slow, for loss comparisons)",
    )
    p.add_argument("--loss-log", default=None, help="write per-step losses and timings as JSON")
    p.add_argument("--data-log", default=None, help="write the token windows used per step (for the torch reference)")
    p.add_argument(
        "--dump-lora-init", default=None, help="save the LoRA parameters at init as .npz (for the torch reference)"
    )
    return p.parse_args()


def resolve_model_dir(src: str, revision: str) -> Path:
    if Path(src).is_dir():
        return Path(src)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(src, revision=revision, allow_patterns=["model-*.safetensors", "*.json", "tokenizer*"])
    )


def load_tokens(model_dir: Path, data_path: Path, vocab_size: int) -> np.ndarray:
    cache = data_path.with_suffix(".mistral.npy")
    if cache.exists():
        return np.load(cache)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokens = np.asarray(tokenizer(data_path.read_text())["input_ids"], dtype=np.uint32)
    if tokens.max() >= vocab_size:
        raise ValueError(f"token id {tokens.max()} exceeds vocab {vocab_size}")
    np.save(cache, tokens)
    return tokens


def host_cross_entropy(logits, targets: np.ndarray) -> float:
    """Mean next-token cross-entropy of the device logits, computed on the host in fp32 (no bf16 loss quantisation)."""
    z = logits.to_numpy(ttnn.DataType.FLOAT32).reshape(-1, logits.shape()[-1]).astype(np.float32)
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    picked = z[np.arange(z.shape[0]), targets.reshape(-1).astype(np.int64)]
    return float(np.mean(lse - picked))


def dram_allocated_gib() -> float:
    device = ttml.autograd.AutoContext.get_instance().get_device()
    view = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks / 2**30


class Batcher:
    """Random ``seq_len + 1`` windows; inputs are the first seq_len tokens, targets the last seq_len."""

    def __init__(self, tokens: np.ndarray, seq_len: int, seed: int) -> None:
        if len(tokens) <= seq_len + 1:
            raise ValueError(f"corpus of {len(tokens)} tokens is too short for seq_len={seq_len}")
        self.tokens, self.seq_len, self.rng = tokens, seq_len, np.random.default_rng(seed)

    def next(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len - 1, batch)
        window = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts])
        return window[:, :-1].astype(np.uint32), window[:, 1:].astype(np.uint32)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model_dir = resolve_model_dir(args.model_path, args.revision)

    overrides = {} if args.layers == 0 else {"num_hidden_layers": args.layers}
    config = MistralConfig.from_hf(model_dir, max_position_embeddings=args.seq_len, **overrides)
    print(
        f"Mistral: {config.num_hidden_layers} of 40 layers, hidden {config.hidden_size}, heads "
        f"{config.num_attention_heads}/{config.num_key_value_heads} x {config.attention_head_dim}, "
        f"intermediate {config.intermediate_size}, vocab {config.vocab_size}, seq {args.seq_len}, batch {args.batch}"
    )

    memory_guard = MemoryUsageTracker.begin_capture() if args.track_memory else None
    use_tp = args.tp > 1
    if use_tp:
        # Weights sharded over the "tp" axis (column/row-parallel linears, vocab-parallel embedding and LM head).
        mesh = ttml.Mesh((1, args.tp), ("dp", "tp"))
        ttml.open_device_mesh(mesh)
        config = dataclasses.replace(config, use_tp=True, embedding_placement=EmbeddingPlacement.VocabParallel)
        if args.host_loss:
            raise ValueError("--host-loss needs the full logits on one chip; not available with --tp > 1")
    else:
        mesh = None
        ttml.autograd.AutoContext.get_instance().open_device()
    model = Mistral(config)
    if not args.random_init:
        print(load_from_safetensors(model, model_dir))
    lora = LoraConfig(rank=args.rank, alpha=args.alpha, target_modules=LORA_TARGETS)
    model = LoraModel(model, lora)
    if not args.no_transposed_weights:
        converted = transpose_frozen_weights(model)
        print(f"frozen weights stored transposed: {len(converted)} linears")
    model.train()
    params = model.parameters()
    trainable = sum(int(np.prod(t.shape())) for t in params.values() if t.get_requires_grad())
    print(f"LoRA rank {args.rank} alpha {args.alpha}: {trainable / 1e6:.1f}M trainable parameters")

    if args.dump_lora_init:
        np.savez(
            args.dump_lora_init,
            # Under TP the adapters are sharded over the mesh: gather each one into its full host array.
            **{
                n: (ttml.Sharding.from_tensor(t).gather(t) if use_tp else t.to_numpy(ttnn.DataType.FLOAT32))
                for n, t in params.items()
                if t.get_requires_grad()
            },
        )

    opt_cfg = ttml.optimizers.AdamWConfig.make(args.lr, 0.9, 0.999, 1e-8, args.weight_decay)
    optimizer = ttml.optimizers.AdamW(params, opt_cfg)

    if args.random_data:
        tokens = np.random.default_rng(args.seed).integers(0, config.vocab_size, 200_000, dtype=np.uint32)
    else:
        data = (
            Path(args.data)
            if args.data
            else Path(os.environ.get("TT_METAL_HOME", ".")) / "tt-train/data/shakespeare.txt"
        )
        tokens = load_tokens(model_dir, data, config.vocab_size)
    batcher = Batcher(tokens, args.seq_len, args.seed)

    flops = useful_flops_per_step(config, args.batch * args.seq_len, args.seq_len, lora_rank=args.rank)
    peak = get_device_peak_tflops_bf16() * 1e12 * args.tp
    print(
        f"useful FLOPs/step {flops / 1e12:.3f} T ({config.num_hidden_layers} layers), peak {peak / 1e12:.1f} TFLOP/s "
        f"({args.tp} chip{'s' if args.tp > 1 else ''})"
    )

    ctx = ttml.autograd.AutoContext.get_instance()
    log: list[dict] = []
    steps_ms: list[float] = []
    for step in range(1, args.steps + 1):
        x_np, y_np = batcher.next(args.batch)
        t0 = time.perf_counter()
        x = ttml.autograd.Tensor.from_numpy(
            x_np.reshape(args.batch, 1, 1, args.seq_len), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
        )
        y = ttml.autograd.Tensor.from_numpy(y_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
        optimizer.zero_grad()
        logits = model(x, None)
        host_loss = host_cross_entropy(logits, y_np) if args.host_loss else None
        if use_tp:
            # The LM head output stays vocab-sharded; the loss reduces across the tp axis.
            loss = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(logits, y, cluster_axis=mesh.axis_index("tp"))
        else:
            loss = ttml.ops.loss.cross_entropy_loss(logits, y, ttml.ops.ReduceType.MEAN)
        if args.mem_probe and step <= 2:
            print(f"step {step}: DRAM allocated after forward {dram_allocated_gib():.2f} GiB")
        loss.backward(False)
        if args.mem_probe and step <= 2:
            print(f"step {step}: DRAM allocated after backward {dram_allocated_gib():.2f} GiB")
        ctx.reset_graph()
        optimizer.step()
        # The read-back is the step's host sync; under TP every chip holds the same value.
        loss_val = float(get_loss_over_devices(loss)) if use_tp else float(loss.to_numpy(ttnn.DataType.FLOAT32).item())
        if args.mem_probe and step <= 2:
            print(
                f"step {step}: DRAM allocated after the optimizer step {dram_allocated_gib():.2f} GiB (weights + state)"
            )
        ms = (time.perf_counter() - t0) * 1e3
        if step > args.warmup_steps:
            steps_ms.append(ms)
        entry = {"step": step, "loss": loss_val, "ms": ms}
        if host_loss is not None:
            entry["host_loss"] = host_loss
        if args.data_log:
            entry["inputs"] = x_np.tolist()
            entry["targets"] = y_np.tolist()
        log.append(entry)
        print(
            f"step {step:4d}/{args.steps}  loss {loss_val:.4f}"
            + (f"  host fp32 loss {host_loss:.4f}" if host_loss is not None else "")
            + f"  {ms:8.1f} ms"
        )
        if args.track_memory and step == 1:
            MemoryUsageTracker.end_capture("FIRST_STEP")
            MemoryUsageTracker.print_memory_usage()
            MemoryUsageTracker.clear()
            if memory_guard:
                memory_guard.release()

    if steps_ms:
        mean_ms = float(np.mean(steps_ms))
        mfu = flops / (mean_ms / 1e3) / peak * 100
        print(
            f"{config.num_hidden_layers} of 40 layers, tp {args.tp}: {mean_ms:.1f} ms/step over {len(steps_ms)} steps "
            f"(best {min(steps_ms):.1f}), {flops / (mean_ms / 1e3) / 1e12:.1f} TFLOP/s, MFU {mfu:.2f}%"
        )
    if args.loss_log:
        Path(args.loss_log).write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "layers": config.num_hidden_layers,
                    "flops_per_step": flops,
                    "peak_tflops": peak / 1e12,
                    "steps": log,
                },
                indent=1,
            )
        )
    if args.data_log:
        Path(args.data_log).write_text(
            json.dumps([{"step": e["step"], "inputs": e["inputs"], "targets": e["targets"]} for e in log])
        )
    if use_tp:
        ttml.close_device_mesh()
    else:
        ctx.close_device()


if __name__ == "__main__":
    main()
