#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA fine-tuning for the Qwen3.8 text backbone, with MFU reporting.

Runs on 8 chips as a ``[2, 4]`` mesh: 2-way data parallel x 4-way tensor
parallel.  TP is capped at 4 by the attention layers' 4 KV heads, so the spare
factor of 2 goes to data parallelism (see :mod:`ttml.models.qwen38.parallel`).

Only LoRA adapters train; the base weights are frozen.  Targets are the
attention and DeltaNet projections -- adapting only ``q/k/v/o_proj`` would leave
48 of the 64 layers untouched, since three quarters of the stack is Gated
DeltaNet (see :mod:`ttml.models.qwen38.lora`).

Ordering that matters
---------------------
The checkpoint is loaded *before* LoRA is applied.  Wrapping in ``LoraModel``
renames every parameter under a new root, which the loader's name mapping does
not expect, so the reverse order fails to find anything.

MFU
---
Reported MFU is ``achieved / peak`` where peak is per-chip BF16 TFLOPS times all
8 chips, and achieved comes from :func:`ttml.models.qwen38.flops.flops_per_token`
-- which *measures* the reference implementation under ``FlopCounterMode``
rather than assuming the usual ``6N`` formula, because 48 of the 64 layers are
linear-attention and do not fit it.  Token counts exclude the TP axis (TP
replicates data across chips) but include the DP axis.

The figure is deliberately conservative: the FLOPs model counts matmul, conv and
SDPA only, so the DeltaNet's substantial elementwise gating work -- and the
delta rule's ``2 log2(chunk)`` matmuls -- are executed but not counted.  Real
hardware utilization is higher than the number printed.

Examples
--------
Measure MFU without the 54 GB checkpoint (random init, real shapes)::

    ./ttenv.sh sources/examples/qwen38_lora/train.py --random-init --steps 10

Actually fine-tune::

    ./ttenv.sh sources/examples/qwen38_lora/train.py \\
        --model-path /localdev/umales/qwen38-27b --steps 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import ttnn
import ttml
from ttml.common.performance import get_device_peak_tflops_bf16
from ttml.models.qwen38 import Qwen38Config, Qwen38Transformer
from ttml.models.qwen38.flops import flops_per_token
from ttml.models.qwen38.loading import load_from_safetensors
from ttml.models.qwen38.lora import apply_lora, build_lora_config, trainable_summary

DEFAULT_MODEL = "/localdev/umales/qwen38-27b"
DEFAULT_DATA = "/localdev/umales/tt-metal/data/shakespeare.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL, help="dir with config.json + safetensors shards")
    p.add_argument("--data", default=DEFAULT_DATA, help="plain-text corpus")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1, help="micro-batch per DP group")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=2, help="steps excluded from the MFU average")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--include-mlp", action="store_true", help="also adapt the MLPs (~64%% of params)")
    p.add_argument("--dp", type=int, default=2)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--layers", type=int, default=0, help="truncate the stack (debugging; 0 = all 64)")
    p.add_argument(
        "--recompute-deltanet",
        action="store_true",
        help="recompute the DeltaNet mixer in backward instead of keeping its activations; "
        "frees ~80%% of activation memory for ~10%% more FLOPs, and is what makes seq_len=1024 fit",
    )
    p.add_argument("--random-init", action="store_true", help="skip the checkpoint; for MFU measurement")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_tokens(data_path: Path, model_path: Path, vocab_size: int) -> np.ndarray:
    """Tokenize the corpus with the model's own tokenizer, caching the result.

    Tokenizing ~1 MB of text takes long enough to be annoying to repeat, and the
    result is a few MB, so it is cached next to the corpus.
    """
    cache = data_path.with_suffix(".qwen38.npy")
    if cache.exists():
        tokens = np.load(cache)
        print(f"tokens: {len(tokens):,} (cached at {cache})")
        return tokens

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    text = data_path.read_text()
    tokens = np.asarray(tokenizer(text)["input_ids"], dtype=np.uint32)
    if tokens.max() >= vocab_size:
        raise ValueError(f"token id {tokens.max()} exceeds vocab {vocab_size}")
    np.save(cache, tokens)
    print(f"tokens: {len(tokens):,} (tokenized {data_path.name}, cached)")
    return tokens


class Batcher:
    """Samples random ``seq_len + 1`` windows; the extra token is the shifted label."""

    def __init__(self, tokens: np.ndarray, seq_len: int, seed: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        if len(tokens) <= seq_len + 1:
            raise ValueError(f"corpus of {len(tokens)} tokens is too short for seq_len={seq_len}")

    def next(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len - 1, batch)
        window = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts])
        return window[:, :-1], window[:, 1:]


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    num_devices = args.dp * args.tp
    ttml.open_device_mesh(ttml.Mesh((args.dp, args.tp), ("dp", "tp")))
    mesh = ttml.mesh()
    # A 2D mesh requires every axis to be an enabled parallelism axis.
    ttml.autograd.AutoContext.get_instance().initialize_parallelism_context(
        ttml.autograd.DistributedConfig(enable_ddp=args.dp > 1, enable_tp=args.tp > 1)
    )
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    print(f"mesh {mesh.shape} {mesh.axis_names}: dp={args.dp} tp={args.tp} ({num_devices} chips)")

    model_path = Path(args.model_path)
    config = Qwen38Config.from_hf_json(model_path / "config.json")
    if args.layers:
        config.num_hidden_layers = args.layers
        config.layer_types = config.layer_types[: args.layers]
    config.use_tp = args.tp > 1
    config.recompute_deltanet = args.recompute_deltanet

    num_full = sum(1 for i in range(config.num_hidden_layers) if config.is_full_attention(i))
    print(
        f"config: {config.num_hidden_layers} layers "
        f"({config.num_hidden_layers - num_full} DeltaNet + {num_full} attention), "
        f"hidden={config.hidden_size} vocab={config.vocab_size}"
    )
    if args.recompute_deltanet:
        print("DeltaNet mixer activations are recomputed in backward")

    tokens = load_tokens(Path(args.data), model_path, config.vocab_size)
    batcher = Batcher(tokens, args.seq_len, args.seed)

    print("building model ...")
    t0 = time.perf_counter()
    model = Qwen38Transformer(config)
    print(f"built in {time.perf_counter() - t0:.1f}s ({len(model.parameters())} tensors)")

    if not args.random_init:
        print("loading checkpoint (streamed shard by shard) ...")
        t0 = time.perf_counter()
        stats = load_from_safetensors(model, model_path, config)
        print(f"loaded {stats['loaded']} tensors in {time.perf_counter() - t0:.1f}s")
    else:
        print("random init (--random-init): weights are not the pretrained ones")

    # LoRA must come after loading: it renames parameters under a new root.
    lora_config = build_lora_config(rank=args.rank, alpha=args.alpha, include_mlp=args.include_mlp)
    model = apply_lora(model, lora_config)
    summary = trainable_summary(model)
    print(
        f"LoRA rank={args.rank} alpha={args.alpha}: "
        f"{summary['trainable']:,} trainable / {summary['total']:,} total "
        f"({100 * summary['fraction']:.3f}%)"
    )

    trainable = {name: p for name, p in model.parameters().items() if p.get_requires_grad()}
    optimizer = ttml.optimizers.AdamW(
        trainable,
        ttml.optimizers.AdamWConfig.make(args.lr, 0.9, 0.95, 1e-8, args.weight_decay),
    )

    # LoRA freezes the base, so the frozen linears need only the input gradient
    # (dx = dy @ W^T) and not the weight gradient -- roughly 2x forward per layer
    # instead of 3x. flops_per_token reflects that via train_base=False.
    fpt = flops_per_token(config, args.batch_size, args.seq_len, train_base=False)
    peak_tflops = get_device_peak_tflops_bf16() * num_devices
    tokens_per_step = args.batch_size * args.dp * args.seq_len
    print(
        f"flops/token {fpt / 1e9:.2f} GFLOP | peak {peak_tflops:.1f} TFLOP/s "
        f"over {num_devices} chips | {tokens_per_step} tokens/step"
    )

    dp_mapper = mesh.axis_mapper("dp", tdim=0) if args.dp > 1 else None
    tp_axis = mesh.axis_index("tp") if args.tp > 1 else None
    # The loss is replicated over TP and sharded over DP, so it needs a composer
    # rather than a single-buffer read-back.
    loss_composer = ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig(list(range(len(mesh.shape)))))

    print(f"\ntraining {args.steps} steps at seq_len={args.seq_len}\n")
    step_times: list[float] = []
    for step in range(1, args.steps + 1):
        # The global batch spans the DP groups, so sample dp x batch_size and let
        # the mapper hand each group its slice.
        x_np, y_np = batcher.next(args.batch_size * args.dp)
        global_batch = x_np.shape[0]
        inputs = ttml.autograd.Tensor.from_numpy(
            x_np.reshape(global_batch, 1, 1, args.seq_len), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, dp_mapper
        )
        targets = ttml.autograd.Tensor.from_numpy(
            y_np.reshape(global_batch, args.seq_len), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, dp_mapper
        )

        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits = model(inputs, None)
        if args.tp > 1:
            # Logits stay sharded over the vocab dim; materializing the full
            # 248320-wide tensor would cost ~0.5 GB per 1024 tokens.
            loss = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(logits, targets, cluster_axis=tp_axis)
        else:
            loss = ttml.ops.loss.cross_entropy_loss(logits, targets)
        loss.backward(False)
        ctx.reset_graph()
        if args.dp > 1:
            ttml.sync_gradients(model.parameters(), ("dp",))
        optimizer.step()
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - t0

        loss_value = float(np.mean(loss.to_numpy(composer=loss_composer)))
        tps = tokens_per_step / elapsed
        mfu = (tps * fpt / 1e12) / peak_tflops * 100.0
        tag = " (warmup)" if step <= args.warmup_steps else ""
        print(
            f"step {step:4d} | loss {loss_value:7.4f} | {elapsed * 1e3:8.1f} ms | "
            f"{tps:8.1f} tok/s | MFU {mfu:5.2f}%{tag}"
        )
        if step > args.warmup_steps:
            step_times.append(elapsed)

    if step_times:
        # Median rather than mean: it is the robust summary when an occasional
        # step is stretched by host-side work.
        median = float(np.median(step_times))
        tps = tokens_per_step / median
        print(
            f"\nsteady state over {len(step_times)} steps: {median * 1e3:.1f} ms/step | "
            f"{tps:.1f} tok/s | {tps * fpt / 1e12:.2f} TFLOP/s | MFU {(tps * fpt / 1e12) / peak_tflops * 100:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
