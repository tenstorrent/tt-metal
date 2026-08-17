# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""UNet training smoke + throughput probe on real CIFAR-10.

    python smoke_unet_train.py --data-dir <dir> [--model tiny|cifar]
        [--steps 300] [--batch 32] [--probe] [--native-conv]

--probe: 30 timed steps after warmup, prints img/s (use with --model cifar
to size the full EDM run), plus a phase breakdown on a fixed batch:
fwd-only / fwd+bwd / full-step ms (each synced by a device read), with the
bwd-only and optimizer-only components derived.
--native-conv (or env EDM_NATIVE_CONV=1): conv FORWARD via native
ttnn.conv2d consuming the flat weight in place (backward stays im2col).
Run the probe once with and once without to compare paths.

Env knobs (see edm_ops.py): EDM_PROFILE_CONV=1 prints cumulative per-bucket
conv timings (fwd_native/fwd_im2col/bwd_im2col_recompute/bwd_col2im/
bwd_matmuls) per probe phase; EDM_SAVE_PATCHES=auto|1|0 sets the conv
backward memory/speed policy.
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


def from_np(a, mapper=None):
    import ttnn as _t

    if mapper is not None:
        return ttml.autograd.Tensor.from_numpy(
            np.ascontiguousarray(a, dtype=np.float32), _t.Layout.TILE, _t.DataType.BFLOAT16, mapper
        )
    return ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def from_np_input(a):
    """Model inputs: no grad needed — lets conv_in skip its dX col2im."""
    t = from_np(a)
    t.set_requires_grad(False)
    return t


def read_scalar(t):
    """Device-sync by pulling one value to host (NATIVE precision: no caching)."""
    return float(t.to_numpy(None, None, ttml.autograd.PreferredPrecision.NATIVE).astype(np.float32).reshape(-1)[0])


def phase_breakdown(model, opt, ctx, batch_np):
    """Time fwd / fwd+bwd / full-step on one fixed batch; print components."""
    import edm_ops

    net_in, feats, onehot, target = batch_np
    x_np, t_np = nchw_to_nhwc_tokens(net_in), nchw_to_nhwc_tokens(target)

    def fwd():
        pred = model(from_np_input(x_np), from_np_input(feats), from_np_input(onehot))
        return ttml.ops.loss.mse_loss(pred, from_np_input(t_np))

    def phase_fwd():
        loss = fwd()
        read_scalar(loss)  # sync
        ctx.reset_graph()

    def phase_fwd_bwd():
        opt.zero_grad()
        loss = fwd()
        loss.backward(False)
        read_scalar(model.out_conv.weight.tensor.get_grad_tensor())  # sync on a grad
        ctx.reset_graph()

    def phase_full():
        opt.zero_grad()
        loss = fwd()
        loss.backward(False)
        opt.step()
        read_scalar(loss)  # sync
        ctx.reset_graph()

    results = {}
    for name, fn in [("fwd", phase_fwd), ("fwd+bwd", phase_fwd_bwd), ("full-step", phase_full)]:
        fn()
        fn()  # warmup x2 (program caches)
        edm_ops.prof_reset()
        n = 10
        t0 = time.time()
        for _ in range(n):
            fn()
        results[name] = (time.time() - t0) / n * 1000
        print(f"PHASE {name:<9s} {results[name]:8.1f} ms", flush=True)
        if edm_ops.PROFILE_CONV:
            edm_ops.prof_report(divisor=n)
    print(
        f"PHASE derived: bwd-only {results['fwd+bwd'] - results['fwd']:.1f} ms, "
        f"opt-only {results['full-step'] - results['fwd+bwd']:.1f} ms",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", choices=["tiny", "cifar"], default="tiny")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--mesh", type=int, default=1, help="dp mesh size (DDP; batch is global)")
    ap.add_argument("--native-conv", action="store_true", help="conv fwd via native ttnn.conv2d")
    args = ap.parse_args()

    import edm_ops

    if args.native_conv:
        edm_ops.NATIVE_CONV = True  # same switch as env EDM_NATIVE_CONV=1
    conv_path = "native" if edm_ops.NATIVE_CONV else "im2col"

    ttml.open_device_mesh(ttml.Mesh((args.mesh, 1), ("dp", "tp")))
    batch_mapper = ttml.mesh().axis_mapper("dp", tdim=0) if args.mesh > 1 else None
    loss_composer = None
    if args.mesh > 1:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        loss_composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    images, labels = load_cifar10(args.data_dir)
    print(f"CIFAR loaded {images.shape}", flush=True)

    model = song_unet_tiny() if args.model == "tiny" else song_unet_cifar()
    n_params = sum(int(np.prod(t.shape())) for t in model.parameters().values())
    print(f"model={args.model} params={n_params/1e6:.2f}M batch={args.batch} conv={conv_path}", flush=True)

    opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": args.lr}, model.parameters())
    ctx = ttml.autograd.AutoContext.get_instance()
    rng = np.random.default_rng(0)
    emb_feat_dim = model.emb_dim  # host timestep_features dim == emb_dim

    model.train()

    if args.probe:  # phase breakdown on one fixed batch, before the img/s loop
        idx = rng.integers(0, images.shape[0], size=args.batch)
        batch_np = make_edm_batch_image(
            images[idx], labels[idx], emb_feat_dim, rng, cfg_drop_prob=0.0, null_class=10
        )
        phase_breakdown(model, opt, ctx, batch_np)

    t0 = time.time()
    warmup = 5 if args.probe else 0
    n_steps = (warmup + 30) if args.probe else args.steps
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, images.shape[0], size=args.batch)
        net_in, feats, onehot, target = make_edm_batch_image(
            images[idx], labels[idx], emb_feat_dim, rng, cfg_drop_prob=0.0, null_class=10, hflip=True
        )
        opt.zero_grad()
        pred = model(from_np_input(nchw_to_nhwc_tokens(net_in)), from_np_input(feats), from_np_input(onehot))
        loss = ttml.ops.loss.mse_loss(pred, from_np_input(nchw_to_nhwc_tokens(target)))
        loss.backward(False)
        opt.step()
        loss_np = loss.to_numpy(None, loss_composer, ttml.autograd.PreferredPrecision.NATIVE)
        loss_val = float(np.asarray(loss_np, dtype=np.float32).mean())
        ctx.reset_graph()
        if args.probe and step == warmup:
            t0 = time.time()
        if step % 10 == 0 or step <= 3:
            print(f"step {step:5d} loss {loss_val:.4f}", flush=True)
    if args.probe:
        dt = time.time() - t0
        print(f"PROBE[conv={conv_path}]: {30*args.batch/dt:.1f} img/s ({dt/30*1000:.0f} ms/step)", flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
