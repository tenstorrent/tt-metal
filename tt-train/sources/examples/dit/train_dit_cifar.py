# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Train a class-conditional pixel-space DiT on CIFAR-10 with ttml.

Config-driven like the rest of tt-train:

    python train_dit_cifar.py -c ${TT_METAL_RUNTIME_ROOT}/tt-train/configs/training_configs/training_cifar10_dit_tiny.yaml

Everything host-side is numpy; the model runs on the Tenstorrent device via
ttml autograd. CIFAR-10 is read from the raw python-pickle tarball (no
torchvision dependency).
"""

from __future__ import annotations

import argparse
import os
import pickle
import tarfile
import time
import urllib.request

import numpy as np

import ttml
from ttml.common.config import load_config

from diffusion import DiffusionSchedule, make_training_batch, one_hot, timestep_features, patchify, unpatchify
from dit_ttml import DiT

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def load_cifar10(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns images [50000, 3, 32, 32] fp32 in [-1, 1] and labels [50000]."""
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.isdir(extract_dir):
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.isfile(tar_path):
            print(f"downloading CIFAR-10 to {tar_path}")
            urllib.request.urlretrieve(CIFAR_URL, tar_path)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
    xs, ys = [], []
    for i in range(1, 6):
        # pickle is the official CIFAR-10 distribution format; the tarball is
        # fetched over HTTPS from the canonical cs.toronto.edu source above.
        with open(os.path.join(extract_dir, f"data_batch_{i}"), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"])
        ys.append(np.array(d[b"labels"]))
    x = np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32)
    x = x / 127.5 - 1.0
    y = np.concatenate(ys).astype(np.int64)
    return x, y


def build_model(model_cfg: dict) -> tuple[DiT, int, int]:
    patch = model_cfg["patch_size"]
    image_size = model_cfg["image_size"]
    channels = model_cfg["image_channels"]
    in_dim = patch * patch * channels
    num_tokens = (image_size // patch) ** 2
    model = DiT(
        in_dim=in_dim,
        dim=model_cfg["embedding_dim"],
        depth=model_cfg["num_blocks"],
        num_heads=model_cfg["num_heads"],
        num_tokens=num_tokens,
        num_classes=model_cfg["num_classes"],
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
    )
    return model, patch, num_tokens


def _param_to_numpy(t, composer=None, num_devices: int = 1):
    # precision=NATIVE reads the live bf16 buffer directly. The default
    # precision=FULL runs a device-side typecast whose fp32 view is CACHED and
    # never refreshed (tt-train issue #41657) — with it, every checkpoint/EMA
    # read after the first silently returns stale (init-time) weights.
    arr = t.to_numpy(None, composer, ttml.autograd.PreferredPrecision.NATIVE).astype(np.float32)
    if composer is not None and num_devices > 1:
        # Replicated param read via concat composer stacks the per-device
        # copies along dim 0; keep the first replica.
        arr = arr[: arr.shape[0] // num_devices]
    return arr


def save_checkpoint(model, path: str, state: dict | None = None, composer=None, num_devices: int = 1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if state is None:
        state = {name: _param_to_numpy(t, composer, num_devices) for name, t in model.parameters().items()}
    np.savez(path, **state)


class WeightEma:
    """Host-side EMA of model weights, updated every `interval` steps.

    Per-step decay `decay` is compounded to `decay**interval` so the sparse
    update matches a per-step EMA. ~130MB host pull per update for DiT-S,
    amortized to ~1% overhead at interval=100.
    """

    def __init__(self, model, decay: float = 0.9999, interval: int = 100, composer=None, num_devices: int = 1):
        self.model, self.interval = model, interval
        self.composer, self.num_devices = composer, num_devices
        self.decay_k = decay**interval
        self.state = {
            name: _param_to_numpy(t, composer, num_devices) for name, t in model.parameters().items()
        }

    def maybe_update(self, step: int):
        if step % self.interval:
            return
        for name, t in self.model.parameters().items():
            cur = _param_to_numpy(t, self.composer, self.num_devices)
            self.state[name] = self.decay_k * self.state[name] + (1.0 - self.decay_k) * cur


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup: int = 500, min_lr_frac: float = 0.05) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    import math

    t = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def sample_grid(model, schedule, patch, dim, num_classes, image_size, channels, steps, cfg_scale, seed=0):
    """DDIM sampling, one image per class; returns [num_classes, C, H, W]."""
    rng = np.random.default_rng(seed)
    b = num_classes
    labels = np.arange(num_classes, dtype=np.uint32)
    x = rng.standard_normal((b, channels, image_size, image_size)).astype(np.float32)
    ts = np.linspace(schedule.timesteps - 1, 0, steps).astype(np.int64)

    ctx = ttml.autograd.AutoContext.get_instance()

    def predict(tokens, t_feats, lbl):
        import ttnn

        oh = one_hot(lbl, num_classes + 1)
        pred = model(
            ttml.autograd.Tensor.from_numpy(tokens),
            ttml.autograd.Tensor.from_numpy(t_feats),
            ttml.autograd.Tensor.from_numpy(oh),
        )
        out = pred.to_numpy(ttnn.DataType.FLOAT32)
        ctx.reset_graph()
        return out

    for i, t in enumerate(ts):
        t_feats = timestep_features(np.full((b,), t, dtype=np.int64), dim)
        tokens = patchify(x, patch)
        eps_tok = predict(tokens, t_feats, labels)
        if cfg_scale != 1.0:
            eps_null = predict(tokens, t_feats, np.full((b,), num_classes))
            eps_tok = eps_null + cfg_scale * (eps_tok - eps_null)
        eps = unpatchify(eps_tok, patch, channels, image_size, image_size)

        ab_t = schedule.alphas_bar[t]
        x0 = np.clip((x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t), -1, 1)
        if i + 1 < len(ts):
            ab_prev = schedule.alphas_bar[ts[i + 1]]
            x = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev) * eps
        else:
            x = x0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="training config yaml")
    ap.add_argument("--overfit-batch", action="store_true", help="repeat one fixed batch (bringup sanity)")
    ap.add_argument("--max-steps", type=int, default=None, help="override training_config.max_steps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc = cfg["training_config"]
    mesh_shape = tuple(cfg.get("device_config", {}).get("mesh_shape", [1, 1]))
    ttml.open_device_mesh(ttml.Mesh(mesh_shape, ("dp", "tp")))

    # DDP: shard the (global) batch across the dp axis; grads are averaged
    # with sync_gradients after backward. Single device -> everything no-ops.
    dp_size = mesh_shape[0]
    batch_mapper = ttml.mesh().axis_mapper("dp", tdim=0) if dp_size > 1 else None
    loss_composer = None
    if dp_size > 1:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        loss_composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    dc = cfg.get("diffusion_config", {})
    ec = cfg.get("eval_config", {})
    lc = cfg.get("logging_config", {})
    model_cfg = load_config(tc["model_config"])["dit_config"]

    seed = tc.get("seed", 0)
    batch_size = tc["batch_size"]
    max_steps = args.max_steps or tc["max_steps"]

    run_name = f"dit_d{model_cfg['embedding_dim']}x{model_cfg['num_blocks']}_p{model_cfg['patch_size']}_b{batch_size}_{int(time.time())}"
    run_dir = os.path.join(tc["output_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    wandb_run = None
    if lc.get("wandb", False):
        import wandb

        wandb_run = wandb.init(project=tc.get("project_name", "tt_train_dit"), name=run_name, config=cfg)

    images, labels = load_cifar10(tc["data_path"])
    print(f"CIFAR-10 loaded: {images.shape}, labels {labels.shape}")

    model, patch, num_tokens = build_model(model_cfg)
    num_classes = model_cfg["num_classes"]
    n_params = sum(int(np.prod(t.shape())) for t in model.parameters().values())  # .shape() is a method on ttml tensors
    print(f"model: dim={model_cfg['embedding_dim']} depth={model_cfg['num_blocks']} tokens={num_tokens} params={n_params/1e6:.2f}M")

    schedule = DiffusionSchedule(
        timesteps=dc.get("timesteps", 1000),
        beta_start=dc.get("beta_start", 1e-4),
        beta_end=dc.get("beta_end", 2e-2),
    )
    cfg_drop_prob = dc.get("cfg_drop_prob", 0.1)

    opt = ttml.optimizers.create_optimizer(tc["optimizer"], model.parameters())
    ctx = ttml.autograd.AutoContext.get_instance()
    rng = np.random.default_rng(seed)

    base_lr = tc["optimizer"]["lr"]
    use_cosine = tc.get("lr_schedule", "constant") == "cosine"
    warmup = tc.get("warmup_steps", 500)
    ema = None
    if tc.get("ema_decay", 0):
        ema = WeightEma(
            model, decay=tc["ema_decay"], interval=tc.get("ema_interval", 100),
            composer=loss_composer, num_devices=dp_size,
        )

    log_every = lc.get("log_interval", 50)
    sample_every = ec.get("sample_interval", 0)
    ckpt_every = tc.get("model_save_interval", 0)

    model.train()
    t0 = time.time()
    ema_loss = None
    for step in range(1, max_steps + 1):
        if args.overfit_batch:
            batch_rng = np.random.default_rng(123)
            idx = np.arange(batch_size)
        else:
            batch_rng = rng
            idx = rng.integers(0, images.shape[0], size=batch_size)

        tokens, t_feats, onehot, target = make_training_batch(
            images[idx], labels[idx], schedule, patch, model.dim, batch_rng,
            cfg_drop_prob=cfg_drop_prob, null_class=num_classes,
        )

        import ttnn

        def dev(arr):
            return ttml.autograd.Tensor.from_numpy(
                arr, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, batch_mapper
            )

        if use_cosine:
            opt.set_lr(cosine_lr(step, max_steps, base_lr, warmup))

        opt.zero_grad()
        pred = model(dev(tokens), dev(t_feats), dev(onehot))
        loss = ttml.ops.loss.mse_loss(pred, dev(target))
        loss.backward(False)
        if dp_size > 1:
            ttml.sync_gradients(model.parameters(), ("dp",))
        opt.step()

        loss_val = float(loss.to_numpy(ttnn.DataType.FLOAT32, composer=loss_composer).mean()) if loss_composer \
            else float(loss.to_numpy(ttnn.DataType.FLOAT32).reshape(-1)[0])
        if ema is not None:
            ema.maybe_update(step)
        ctx.reset_graph()
        ema_loss = loss_val if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_val

        if step % log_every == 0:
            ips = batch_size * log_every / (time.time() - t0)
            t0 = time.time()
            print(f"step {step:6d}  loss {loss_val:.4f}  ema {ema_loss:.4f}  {ips:.1f} img/s", flush=True)
            if wandb_run:
                wandb_run.log({"loss": loss_val, "ema_loss": ema_loss, "images_per_sec": ips}, step=step)

        if sample_every and step % sample_every == 0 and dp_size == 1:
            # Sampling is single-device-only for now; on DDP runs, sample
            # offline from a checkpoint instead.
            model.eval()
            grid = sample_grid(
                model, schedule, patch, model.dim, num_classes,
                model_cfg["image_size"], model_cfg["image_channels"],
                steps=ec.get("sample_steps", 50), cfg_scale=ec.get("cfg_scale", 2.0),
            )
            np.save(os.path.join(run_dir, f"samples_{step:06d}.npy"), grid)
            if wandb_run:
                import wandb

                imgs = ((grid.transpose(0, 2, 3, 1) + 1) * 127.5).clip(0, 255).astype(np.uint8)
                wandb_run.log({"samples": [wandb.Image(im, caption=f"class {i}") for i, im in enumerate(imgs)]}, step=step)
            model.train()

        if ckpt_every and step % ckpt_every == 0:
            save_checkpoint(model, os.path.join(run_dir, f"ckpt_{step:06d}.npz"),
                            composer=loss_composer, num_devices=dp_size)
            if ema is not None:
                save_checkpoint(model, os.path.join(run_dir, f"ckpt_ema_{step:06d}.npz"), state=ema.state)

    save_checkpoint(model, os.path.join(run_dir, "ckpt_final.npz"), composer=loss_composer, num_devices=dp_size)
    if ema is not None:
        save_checkpoint(model, os.path.join(run_dir, "ckpt_ema_final.npz"), state=ema.state)
    print(f"done; artifacts in {run_dir}")


if __name__ == "__main__":
    main()
