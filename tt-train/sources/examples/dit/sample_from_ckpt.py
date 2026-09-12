# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Sample a grid from a saved DiT checkpoint (raw or EMA .npz).

    python sample_from_ckpt.py -c <training_config.yaml> --ckpt <ckpt.npz> \
        [--steps 50] [--cfg-scale 2.0] [--out grid.npy]

Loads weights by parameter name into a freshly built model, then runs the
DDIM sampler. Works for checkpoints from single-device or DDP runs.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import ttnn

import ttml
from ttml.common.config import load_config

from diffusion import DiffusionSchedule
from train_dit_cifar import build_model, sample_grid


def load_weights(model, npz_path: str):
    state = np.load(npz_path)
    params = model.parameters()
    missing = [k for k in params if k not in state.files]
    extra = [k for k in state.files if k not in params]
    if missing or extra:
        raise KeyError(f"checkpoint/model mismatch; missing={missing[:3]}... extra={extra[:3]}...")
    for name, tensor in params.items():
        loaded = ttml.autograd.Tensor.from_numpy(
            np.ascontiguousarray(state[name], dtype=np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
        )
        tensor.set_value(loaded.get_value())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ttml.open_device_mesh((1, 1))
    cfg = load_config(args.config)
    model_cfg = load_config(cfg["training_config"]["model_config"])["dit_config"]
    dc = cfg.get("diffusion_config", {})

    model, patch, _ = build_model(model_cfg)
    load_weights(model, args.ckpt)
    model.eval()

    schedule = DiffusionSchedule(
        timesteps=dc.get("timesteps", 1000),
        beta_start=dc.get("beta_start", 1e-4),
        beta_end=dc.get("beta_end", 2e-2),
    )
    grid = sample_grid(
        model, schedule, patch, model_cfg["embedding_dim"], model_cfg["num_classes"],
        model_cfg["image_size"], model_cfg["image_channels"],
        steps=args.steps, cfg_scale=args.cfg_scale, seed=args.seed,
    )
    out = args.out or os.path.splitext(args.ckpt)[0] + f"_samples_s{args.steps}_cfg{args.cfg_scale}.npy"
    np.save(out, grid)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
