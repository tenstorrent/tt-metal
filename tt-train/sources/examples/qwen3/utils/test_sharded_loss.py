# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test: do ttml loss functions work on vocab-sharded tensors?

Tests the sharded_cross_entropy_loss implementation against PyTorch
reference, both forward value and backward gradients.

Supports testing with different mesh shapes including DP configurations:
  - Pure TP:    --mesh_shape 1 2   (2 devices, vocab-sharded)
  - Pure TP:    --mesh_shape 1 8   (8 devices, vocab-sharded)
  - DP + TP:    --mesh_shape 2 4   (8 devices, 2 DP groups × 4 TP)
  - DP + TP:    --mesh_shape 4 2   (8 devices, 4 DP groups × 2 TP)

Usage:
    python test_sharded_loss.py --mesh_shape 1 2
    python test_sharded_loss.py --mesh_shape 1 8
    python test_sharded_loss.py --mesh_shape 2 4
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QWEN3_ROOT = os.path.dirname(_SCRIPT_DIR)
if _QWEN3_ROOT not in sys.path:
    sys.path.insert(0, _QWEN3_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

import ttnn
import ttml

from utils.sharded_loss import sharded_cross_entropy_loss

RTOL = 0.02
ATOL = 0.15


def check(name, got, ref, rtol=RTOL, atol=ATOL):
    """Return (passed, message)."""
    if got is None:
        return False, "CRASHED"
    diff = abs(got - ref)
    thr = atol + rtol * abs(ref)
    ok = diff <= thr
    tag = "PASS" if ok else "FAIL"
    return ok, f"{tag}  got={got:.6f}  ref={ref:.6f}  diff={diff:.6f}  thr={thr:.6f}"


def try_op(fn):
    """Run fn(), return (value, None) on success or (None, error_str) on failure."""
    try:
        return fn(), None
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, f"{type(e).__name__}: {e}"


def extract_loss(loss_tensor, device, distributed):
    """Extract scalar loss from ttml tensor, averaged across all devices."""
    if distributed:
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        t = ttnn.to_torch(loss_tensor.get_value(), mesh_composer=composer).float()
        return t.mean().item()
    return ttnn.to_torch(loss_tensor.get_value()).float().item()


def pytorch_reference(logits_np, targets_np):
    """Compute PyTorch reference cross-entropy loss + gradients."""
    logits_t = torch.from_numpy(logits_np).float().requires_grad_(True)
    targets_t = torch.from_numpy(targets_np.astype(np.int64))

    B, _, S, V = logits_t.shape
    loss = F.cross_entropy(logits_t.reshape(B * S, V), targets_t.reshape(-1), reduction="mean")
    loss.backward()
    return loss.item(), logits_t.grad.numpy()


def run_test(mesh_shape, batch_size=2, seq_len=32, vocab_size=64):
    dp_size, tp_size = mesh_shape
    total_devices = dp_size * tp_size
    distributed = total_devices > 1
    global_batch = dp_size * batch_size
    raw_local_V = vocab_size // tp_size

    print(f"\n{'=' * 70}")
    print(
        f"Test: sharded_cross_entropy_loss  (mesh=[{dp_size},{tp_size}], "
        f"B={batch_size}, S={seq_len}, V={vocab_size}, local_V={raw_local_V})"
    )
    print(f"{'=' * 70}")

    # ---- device setup ----
    from utils.device_setup import setup_device

    ctx, device = setup_device(dp_size, tp_size)

    # ---- random test data ----
    np.random.seed(42)
    logits_np = np.random.randn(global_batch, 1, seq_len, vocab_size).astype(np.float32)
    targets_np = np.random.randint(0, vocab_size, size=(global_batch, seq_len)).astype(np.uint32)

    ref_loss, ref_grad = pytorch_reference(logits_np, targets_np)
    print(f"\nPyTorch reference  CE={ref_loss:.6f}")

    passed = 0
    failed = 0
    total = 0

    def report(name, val, err, ref_val):
        nonlocal passed, failed, total
        total += 1
        if err is not None:
            failed += 1
            print(f"  {name}: FAIL (crashed: {err})")
        else:
            ok, msg = check(name, val, ref_val)
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"  {name}: {msg}")

    # Build per-device logits: [total_devices * B, 1, S, local_V]
    # Device ordering follows row-major mesh: (dp0,tp0), (dp0,tp1), …, (dp1,tp0), …
    if distributed:
        per_device_chunks = []
        for d in range(dp_size):
            for k in range(tp_size):
                chunk = logits_np[
                    d * batch_size : (d + 1) * batch_size,
                    :,
                    :,
                    k * raw_local_V : (k + 1) * raw_local_V,
                ]
                per_device_chunks.append(chunk)
        logits_stacked = np.concatenate(per_device_chunks, axis=0)
        shard_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)

    # ==================================================================
    # 1. Baseline: ttml.ops.loss.cross_entropy_loss on FULL tensor
    # ==================================================================
    print("\n--- Baseline: cross_entropy_loss on full tensor ---")

    def test_ce_full():
        ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
        logits = ttml.autograd.Tensor.from_numpy(logits_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)
        tgt = ttml.autograd.Tensor.from_numpy(targets_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
        loss = ttml.ops.loss.cross_entropy_loss(logits, tgt, ttml.ops.ReduceType.MEAN)
        val = extract_loss(loss, device, distributed)
        ctx.reset_graph()
        return val

    val, err = try_op(test_ce_full)
    report("CE full (baseline)", val, err, ref_loss)

    # ==================================================================
    # 2. sharded_cross_entropy_loss — forward value
    # ==================================================================
    if tp_size > 1:
        print("\n--- sharded_cross_entropy_loss: forward ---")

        def test_dist_ce_forward():
            ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
            logits = ttml.autograd.Tensor.from_numpy(
                logits_stacked,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                shard_mapper,
            )
            loss = sharded_cross_entropy_loss(
                logits,
                targets_np,
                vocab_size,
                tp_size,
                tp_axis=1,
                dp_size=dp_size,
            )
            val = extract_loss(loss, device, distributed)
            ctx.reset_graph()
            return val

        val, err = try_op(test_dist_ce_forward)
        report("sharded CE forward", val, err, ref_loss)

    # ==================================================================
    # 3. sharded_cross_entropy_loss — backward gradients
    #    For DP, per-group grads are averaged (1/dp_size) to match the
    #    global-mean reference, mirroring training's gradient sync.
    # ==================================================================
    if tp_size > 1:
        print("\n--- sharded_cross_entropy_loss: backward ---")

        def test_dist_ce_backward():
            ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            logits = ttml.autograd.Tensor.from_numpy(
                logits_stacked,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                shard_mapper,
            )
            logits.set_requires_grad(True)
            loss = sharded_cross_entropy_loss(
                logits,
                targets_np,
                vocab_size,
                tp_size,
                tp_axis=1,
                dp_size=dp_size,
            )
            loss.backward(False)

            composer = ttml.core.distributed.concat_mesh_to_tensor_composer(
                device,
                0,
            )
            grad_raw = ttnn.to_torch(logits.get_grad(), mesh_composer=composer).float().numpy()

            # Reconstruct full gradient from per-device shards
            # grad slabs may be tile-padded wider than raw_local_V
            grad_full = np.zeros_like(logits_np)
            for d in range(dp_size):
                for k in range(tp_size):
                    dev_idx = d * tp_size + k
                    slab = grad_raw[dev_idx * batch_size : (dev_idx + 1) * batch_size]
                    grad_full[
                        d * batch_size : (d + 1) * batch_size,
                        :,
                        :,
                        k * raw_local_V : (k + 1) * raw_local_V,
                    ] = slab[:, :, :, :raw_local_V]

            # Average gradients across DP groups to match global mean
            # (mirrors synchronize_gradients in training)
            if dp_size > 1:
                grad_full /= dp_size

            max_diff = np.abs(grad_full - ref_grad).max()
            mean_diff = np.abs(grad_full - ref_grad).mean()
            ctx.reset_graph()
            return max_diff, mean_diff

        val, err = try_op(test_dist_ce_backward)
        total += 1
        if err is not None:
            failed += 1
            print(f"  dist CE backward: FAIL (crashed: {err})")
        else:
            max_diff, mean_diff = val
            ok = max_diff < 0.05
            tag = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"  dist CE backward: {tag}  max_diff={max_diff:.6f}  " f"mean_diff={mean_diff:.6f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print(f"{'=' * 70}")

    ctx.close_device()
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test distributed cross-entropy loss with different DP and TP sizes")
    parser.add_argument(
        "--mesh_shape",
        type=int,
        nargs=2,
        default=[1, 2],
        metavar=("DP", "TP"),
        help="Device mesh shape [dp_size, tp_size]. Default: 1 2",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument(
        "--vocab_size",
        type=int,
        nargs="+",
        default=[64, 240, 1024],
        help="Vocab sizes to test (supports non-tile-aligned). Default: 64 240 1024",
    )
    args = parser.parse_args()

    all_ok = True
    for vs in args.vocab_size:
        ok = run_test(args.mesh_shape, args.batch_size, args.seq_len, vs)
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
