# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LoRA fine-tuning of the MiniMax-H3 transformer on a prepared cache, on a Tenstorrent mesh.

    python train.py --cache cache/pixelart-150 --out lora/pixelart --mesh 1,8

Mirrors the CUDA `train_lora.py train`: one sequence per micro-step, `--grad-accum` sequences per optimizer step,
AdamW (0.9, 0.99) with fp32 master weights, linear warmup then constant lr, EMA of the adapter saved as the main
weights with the raw ones in `raw/`, metrics to `<out>/metrics.jsonl`. The cache holds `samples/N.npy` (video rows)
and `embeds/N.npy` (text) from encode_vae.py / encode_text.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np
import ttml

from ttml.common.performance import get_device_peak_tflops_bf16
from ttml.models.minimax_h3 import (
    AUDIO_CHANNELS,
    AUDIO_LATENTS,
    H3Config,
    H3Inputs,
    H3Transformer,
    build_packed_sequence,
    build_row_timesteps,
    calculate_sequence_flops,
    load_transformer,
    read_time_embedder,
    sample_timestep,
    shared_timesteps,
    upload,
)
from lora_utils import (
    Ema,
    clip_grad_norm,
    gather,
    init_lora_A_gaussian,
    inject,
    load_adapter,
    lora_params,
    masked_mse,
    save_adapter,
)

log = logging.getLogger("minimax_h3.train")

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DEFAULT_WEIGHTS = (
    Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / MODEL_REVISION / "transformer"
)
KEYFRAME_NOISE_AUG = 0.999


def open_mesh(dp: int, tp: int, seed: int):
    if dp * tp > 1 and "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
        log.warning("TT_MESH_GRAPH_DESC_PATH is not set")
    ttml.open_device_mesh(ttml.Mesh((dp, tp), ("dp", "tp")))
    ctx = ttml.autograd.AutoContext.get_instance()
    if dp * tp > 1 and not ctx.is_parallelism_context_initialized():
        ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=dp > 1, enable_tp=tp > 1))
    ctx.set_seed(seed)
    return ctx


class Cache:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.manifest = json.loads((folder / "manifest.json").read_text())
        self.meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
        self.keys = sorted(self.manifest)
        missing = [
            k
            for k in self.keys
            if not (folder / "samples" / f"{k}.npy").exists() or not (folder / "embeds" / f"{k}.npy").exists()
        ]
        if missing:
            raise SystemExit(
                f"{len(missing)} samples lack samples/ or embeds/ (e.g. {missing[:3]}); run encode_vae.py and encode_text.py"
            )

    def load(self, key: str) -> tuple[np.ndarray, np.ndarray, int, int]:
        rows = np.load(self.folder / "samples" / f"{key}.npy").astype(np.float32)
        embeds = np.load(self.folder / "embeds" / f"{key}.npy").view(ml_dtypes.bfloat16).astype(np.float32)
        entry = self.manifest[key]
        return rows, embeds, entry["latent_height"], entry["latent_width"]


def scalar(t: ttml.autograd.Tensor) -> float:
    return float(gather(t).reshape(-1)[0])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--mesh", default="1,8", help="dp,tp")
    p.add_argument("--resume", default=None, help="adapter folder to warm-start from")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="global L2 clip over the LoRA grads; 0 disables")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=0, help="default: = rank")
    p.add_argument("--timestep-shift", type=float, default=3.0)
    p.add_argument("--caption-dropout", type=float, default=0.05)
    p.add_argument("--ema-decay", type=float, default=0.999, help="0 disables")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--loss-ema-decay", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sdpa",
        choices=["fused", "composite"],
        default="composite",
        help="the fused kernel's backward is inaccurate on attention-sink rows (grad spikes)",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cache = Cache(Path(args.cache))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trigger = cache.meta.get("trigger")
    log.info("%d cached samples, trigger %r", len(cache.keys), trigger)

    dp, tp = (int(v) for v in args.mesh.split(","))
    if dp != 1:
        raise SystemExit("data parallel is not wired yet; use --mesh 1,N")
    open_mesh(dp, tp, args.seed)
    config = H3Config.from_hf(Path(args.weights) / "config.json", use_tp=tp > 1, sdpa=args.sdpa)
    t0 = time.perf_counter()
    model = H3Transformer(config)
    load_transformer(model, Path(args.weights))
    lora = inject(model, args.rank, args.alpha or args.rank)
    if args.resume:
        log.info("resumed %d adapter tensors from %s", load_adapter(lora, config, Path(args.resume)), args.resume)
    else:
        init_lora_A_gaussian(lora, args.rank, args.seed)
    trainable = lora_params(lora)
    time_weights = read_time_embedder(Path(args.weights))
    log.info(
        "model ready in %.0f s: rank %d, %.1f M trainable params per device",
        time.perf_counter() - t0,
        args.rank,
        sum(np.prod(t.shape()) for t in trainable.values()) / 1e6,
    )

    opt_config = ttml.optimizers.AdamWFullPrecisionConfig.make(args.lr, 0.9, 0.99, 1e-8, args.weight_decay, False)
    optimizer = ttml.optimizers.AdamWFullPrecision(trainable, opt_config)
    ema = Ema(lora, args.ema_decay) if args.ema_decay > 0 else None

    def save(folder: Path, step: int) -> None:
        record = json.dumps(
            {**vars(args), "trigger": trigger, "ema": ema is not None, "step": step}, indent=1, default=str
        )
        if ema is None:
            save_adapter(lora, config, folder)
        else:
            save_adapter(lora, config, folder, ema.shadows)
            save_adapter(lora, config, folder / "raw")
            (folder / "raw" / "training_args.json").write_text(record)
        (folder / "training_args.json").write_text(record)

    rng = random.Random(args.seed)
    noise_rng = np.random.default_rng(args.seed)
    metrics = (out / "metrics.jsonl").open("w", encoding="utf-8")
    samples_log = (out / "samples.jsonl").open("w", encoding="utf-8")  # every sequence: key, t, loss; every step: grad
    order: list[str] = []
    running = running_t = 0.0
    loss_ema = 0.0
    peak_flops = get_device_peak_tflops_bf16() * 1e12 * dp * tp
    window_flops, window_t0 = 0, 0.0
    log.info("peak %.0f TFLOPS bf16 on %d chips (ttml convention)", peak_flops / 1e12, dp * tp)
    lora.train()
    ctx = ttml.autograd.AutoContext.get_instance()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0
        # The step's micro-batches up front, so all their timesteps share one set of AdaLN tables.
        micro = []
        for _ in range(args.grad_accum):
            if not order:
                order = cache.keys[:]
                rng.shuffle(order)
            key = order.pop()
            x0, embeds, latent_h, latent_w = cache.load(key)
            if args.caption_dropout and rng.random() < args.caption_dropout:
                embeds = np.zeros_like(embeds)
            t = sample_timestep(args.timestep_shift, rng)
            layout = build_packed_sequence(embeds.shape[0], 1, latent_h, latent_w)
            micro.append((key, x0, embeds, t, layout))
        timesteps, all_indices = shared_timesteps([build_row_timesteps(m[4], m[3]) for m in micro])
        tables = None
        for (key, x0, embeds, t, layout), timestep_indices in zip(micro, all_indices):
            running_t += t / args.grad_accum
            noise = noise_rng.standard_normal(x0.shape, dtype=np.float32)
            audio = noise_rng.standard_normal(
                (AUDIO_LATENTS * AUDIO_CHANNELS, config.audio_in_channels), dtype=np.float32
            )
            x_t = t * x0 + (1.0 - t) * noise
            inputs = H3Inputs.build(config, time_weights, x_t, audio, embeds, layout, timesteps, timestep_indices)
            if tables is None:
                tables = model.time_tables(inputs.silu_temb)
            target = np.zeros((1, 1, inputs.S_pad, config.video_in_channels), dtype=np.float32)
            target[0, 0, layout.video_indices] = x0 - noise
            row_mask = np.zeros_like(target)
            row_mask[0, 0, layout.video_indices] = 1.0

            pred, _ = lora(inputs, with_audio=False, tables=tables)
            loss = masked_mse(pred, upload(target), upload(row_mask), inputs) * (1.0 / args.grad_accum)
            loss_value = scalar(loss)
            step_loss += loss_value
            loss.backward(False)
            ctx.reset_graph()
            samples_log.write(
                json.dumps(
                    {
                        "step": step,
                        "key": key,
                        "t": round(t, 4),
                        "L": int(embeds.shape[0]),
                        "V": int(x0.shape[0]),
                        "dropout": bool(not embeds.any()),
                        "loss": loss_value * args.grad_accum,
                    }
                )
                + "\n"
            )
            window_flops += calculate_sequence_flops(config, inputs.S_pad, inputs.L_pad, args.rank)
        running += step_loss
        loss_ema = args.loss_ema_decay * loss_ema + (1 - args.loss_ema_decay) * step_loss

        logging_step = step % args.log_every == 0
        grad_norm = clip_grad_norm(lora, args.max_grad_norm)
        samples_log.write(json.dumps({"step": step, "grad_norm": grad_norm}) + "\n")
        samples_log.flush()
        lr_now = args.lr * min(1.0, step / args.warmup) if args.warmup else args.lr
        optimizer.set_lr(lr_now)
        optimizer.step()
        if ema is not None:
            ema.update(step)

        if logging_step:
            elapsed = time.perf_counter() - t0
            window_dt = elapsed - window_t0
            tflops = window_flops / window_dt / 1e12
            mfu = window_flops / window_dt / peak_flops
            loss_avg = running / args.log_every
            loss_ema_hat = loss_ema / (1 - args.loss_ema_decay**step)
            log.info(
                "step %d/%d loss %.4f ema %.4f grad %.3f lr %.2e %.1f s/step | %.0f TFLOPS %.2f%% MFU",
                step,
                args.steps,
                loss_avg,
                loss_ema_hat,
                grad_norm,
                lr_now,
                elapsed / step,
                tflops,
                100 * mfu,
            )
            metrics.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_avg,
                        "loss_ema": loss_ema_hat,
                        "grad_norm": grad_norm,
                        "lr": lr_now,
                        "timestep_mean": running_t / args.log_every,
                        "sec_per_step": elapsed / step,
                        "tflops": tflops,
                        "mfu": mfu,
                    }
                )
                + "\n"
            )
            metrics.flush()
            running = running_t = 0.0
            window_flops, window_t0 = 0, elapsed
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            save(out / f"step-{step:05d}", step)
            log.info("saved %s", out / f"step-{step:05d}")

    save(out, args.steps)
    metrics.close()
    samples_log.close()
    ttml.close_device_mesh()
    log.info("done: adapter in %s (metrics in %s)", out, out / "metrics.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
