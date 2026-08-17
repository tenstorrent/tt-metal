# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""UNet training smoke + throughput probe on real CIFAR-10.

    python smoke_unet_train.py --data-dir <dir> [--model tiny|cifar]
        [--steps 300] [--batch 32] [--probe]

--probe: 30 timed steps after warmup, prints img/s (use with --model cifar
to size the full EDM run).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import ttnn

import ttml

from edm import make_edm_batch_image, nchw_to_nhwc_tokens
from edm_unet import song_unet_cifar, song_unet_tiny
from train_dit_cifar import load_cifar10


def from_np(a):
    return ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", choices=["tiny", "cifar"], default="tiny")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    ttml.open_device_mesh(ttml.Mesh((1, 1), ("dp", "tp")))
    images, labels = load_cifar10(args.data_dir)
    print(f"CIFAR loaded {images.shape}", flush=True)

    model = song_unet_tiny() if args.model == "tiny" else song_unet_cifar()
    n_params = sum(int(np.prod(t.shape())) for t in model.parameters().values())
    print(f"model={args.model} params={n_params/1e6:.2f}M batch={args.batch}", flush=True)

    opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": args.lr}, model.parameters())
    ctx = ttml.autograd.AutoContext.get_instance()
    rng = np.random.default_rng(0)
    emb_feat_dim = model.emb_dim  # host timestep_features dim == emb_dim

    model.train()
    t0 = time.time()
    warmup = 5 if args.probe else 0
    n_steps = (warmup + 30) if args.probe else args.steps
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, images.shape[0], size=args.batch)
        net_in, feats, onehot, target = make_edm_batch_image(
            images[idx], labels[idx], emb_feat_dim, rng, cfg_drop_prob=0.0, null_class=10, hflip=True
        )
        opt.zero_grad()
        pred = model(from_np(nchw_to_nhwc_tokens(net_in)), from_np(feats), from_np(onehot))
        loss = ttml.ops.loss.mse_loss(pred, from_np(nchw_to_nhwc_tokens(target)))
        loss.backward(False)
        opt.step()
        loss_val = float(loss.to_numpy(None, None, ttml.autograd.PreferredPrecision.NATIVE).astype(np.float32).reshape(-1)[0])
        ctx.reset_graph()
        if args.probe and step == warmup:
            t0 = time.time()
        if step % 10 == 0 or step <= 3:
            print(f"step {step:5d} loss {loss_val:.4f}", flush=True)
    if args.probe:
        dt = time.time() - t0
        print(f"PROBE: {30*args.batch/dt:.1f} img/s ({dt/30*1000:.0f} ms/step)", flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
