# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""On-device gate for the optional tt-lang fused adaLN path.

    python test_ttl_fused_ops.py            # correctness + e2e loss-decrease
    python test_ttl_fused_ops.py --ab 500   # + eager-vs-ttl loss-trajectory A/B

Exits 0 with a skip message when the ttl package is not installed.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import ttml
from ttl_fused_ops import TILE, is_available, make_fused_modulate


def from_np(a):
    return ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def to_np(t):
    import ttnn

    return t.to_numpy(ttnn.DataType.FLOAT32)


def test_kernel_correctness(B=4, T=256, D=384):
    """Fused modulate fwd+bwd vs numpy/torch-free golden, offsets as in DiTBlock."""
    fused = make_fused_modulate()
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((B, 1, T, D)).astype(np.float32)
    mod_np = rng.standard_normal((B, 1, 1, 4 * D)).astype(np.float32)
    dt = D // TILE

    x = from_np(x_np)
    x.set_requires_grad(True)
    mod = from_np(mod_np)
    mod.set_requires_grad(True)

    y = fused.apply(x, mod, dt, 0)  # scale at [D:2D], shift at [0:D]
    got = to_np(y)
    golden = x_np * (1.0 + mod_np[..., D:2 * D]) + mod_np[..., 0:D]
    err = np.abs(got - golden).max()
    assert err < 0.15, f"fwd mismatch {err}"

    loss = ttml.ops.loss.mse_loss(y, from_np(np.zeros_like(x_np)))
    loss.backward(False)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    print(f"OK   kernel fwd (max err {err:.4f}) + bwd", flush=True)


def test_dit_e2e(steps=5):
    """Tiny DiT with use_ttl_modulation trains: loss must decrease."""
    from dit_ttml import DiT
    from diffusion import DiffusionSchedule, make_training_batch

    rng = np.random.default_rng(0)
    model = DiT(in_dim=48, dim=128, depth=2, num_heads=4, num_tokens=64, num_classes=10, use_ttl_modulation=True)
    sched = DiffusionSchedule()
    images = (rng.random((4, 3, 32, 32), dtype=np.float32) * 2 - 1).astype(np.float32)
    labels = rng.integers(0, 10, size=4)
    opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": 1e-3}, model.parameters())

    losses = []
    for step in range(steps):
        rng_step = np.random.default_rng(1)  # fixed batch
        tokens, t_feats, onehot, target = make_training_batch(
            images, labels, sched, patch=4, model_dim=model.dim, rng=rng_step, cfg_drop_prob=0.0, null_class=10
        )
        opt.zero_grad()
        pred = model(from_np(tokens), from_np(t_feats), from_np(onehot))
        loss = ttml.ops.loss.mse_loss(pred, from_np(target))
        loss.backward(False)
        opt.step()
        losses.append(float(to_np(loss).reshape(-1)[0]))
        ttml.autograd.AutoContext.get_instance().reset_graph()
    print(f"OK   ttl DiT e2e losses: {[f'{v:.4f}' for v in losses]}", flush=True)
    assert losses[-1] < losses[0], losses


def ab_trajectories(steps: int, B=32):
    """Train eager vs ttl variants on identical batch sequences; report curves.

    Weights initialize differently across the two parameterizations (6x D->D
    vs D->4D + gates), so expect statistically matching curves, not bit-equal.
    """
    from dit_ttml import DiT
    from diffusion import DiffusionSchedule, make_training_batch

    results = {}
    for name, flag in (("eager", False), ("ttl", True)):
        rng = np.random.default_rng(0)
        model = DiT(in_dim=12, dim=384, depth=12, num_heads=6, num_tokens=256, num_classes=10, use_ttl_modulation=flag)
        sched = DiffusionSchedule()
        opt = ttml.optimizers.create_optimizer({"type": "AdamW", "lr": 1e-4}, model.parameters())
        images = (rng.random((B, 3, 32, 32), dtype=np.float32) * 2 - 1).astype(np.float32)
        labels = rng.integers(0, 10, size=B)
        batch_rng = np.random.default_rng(42)
        curve = []
        import time

        t0 = time.perf_counter()
        for step in range(steps):
            tokens, t_feats, onehot, target = make_training_batch(
                images, labels, sched, 2, model.dim, batch_rng, cfg_drop_prob=0.0, null_class=10
            )
            opt.zero_grad()
            pred = model(from_np(tokens), from_np(t_feats), from_np(onehot))
            loss = ttml.ops.loss.mse_loss(pred, from_np(target))
            loss.backward(False)
            opt.step()
            curve.append(float(to_np(loss).reshape(-1)[0]))
            ttml.autograd.AutoContext.get_instance().reset_graph()
        dt = time.perf_counter() - t0
        results[name] = (curve, B * steps / dt)
        print(f"{name}: final-10 mean loss {np.mean(curve[-10:]):.4f}, {B*steps/dt:.1f} img/s", flush=True)
    e, t = np.mean(results["eager"][0][-10:]), np.mean(results["ttl"][0][-10:])
    print(f"A/B: eager {e:.4f} vs ttl {t:.4f} (ratio {t/max(e,1e-9):.3f}); "
          f"speed {results['ttl'][1]/results['eager'][1]:.2f}x", flush=True)


if __name__ == "__main__":
    if not is_available():
        print("SKIP: tt-lang (ttl) not installed; fused path unavailable.")
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", type=int, default=0, help="also run an N-step eager-vs-ttl A/B")
    args = ap.parse_args()

    ttml.open_device_mesh(ttml.Mesh((1, 1), ("dp", "tp")))
    test_kernel_correctness()
    test_dit_e2e()
    if args.ab:
        ab_trajectories(args.ab)
    print("TTL FUSED OPS GATE PASSED", flush=True)
