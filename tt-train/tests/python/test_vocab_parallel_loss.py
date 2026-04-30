# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Test: do ttml loss functions work on vocab-sharded tensors?

Tests the vocab_parallel_cross_entropy_loss implementation against PyTorch
reference, both forward value and backward gradients.

Supports testing with different mesh shapes including DP configurations:
  - Pure TP:    --mesh_shape 1 2   (2 devices, vocab-sharded)
  - Pure TP:    --mesh_shape 1 8   (8 devices, vocab-sharded)
  - DP + TP:    --mesh_shape 2 4   (8 devices, 2 DP groups × 4 TP)
  - DP + TP:    --mesh_shape 4 2   (8 devices, 4 DP groups × 2 TP)

Usage:
    python test_vocab_parallel_loss.py --mesh_shape 1 2
    python test_vocab_parallel_loss.py --mesh_shape 1 8
    python test_vocab_parallel_loss.py --mesh_shape 2 4
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

import ttnn
import ttml


RTOL = 0.02
ATOL = 0.15


# ---------------------------------------------------------------------------
# Device setup (inlined from qwen3/utils/device_setup.py)
# ---------------------------------------------------------------------------
def setup_device(dp_size: int, tp_size: int, seed: int = 42):
    """Open a Tenstorrent device (single or mesh) and return ``(ctx, device)``."""
    distributed = dp_size > 1 or tp_size > 1
    total_devices = dp_size * tp_size

    if distributed:
        print(
            f"\nEnabling distributed mode: DP={dp_size}, TP={tp_size} "
            f"({total_devices} devices, mesh [{dp_size}, {tp_size}])"
        )
        ttml.core.distributed.enable_fabric(total_devices)

    ctx = ttml.autograd.AutoContext.get_instance()
    if distributed:
        ctx.open_device([dp_size, tp_size])
        ctx.initialize_parallelism_context(
            ttml.autograd.DistributedConfig(enable_ddp=dp_size > 1, enable_tp=tp_size > 1)
        )
    else:
        ctx.open_device()
    ctx.set_seed(seed)
    return ctx, ctx.get_device()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------
def run_test(ctx, device, mesh_shape, batch_size=2, seq_len=32, vocab_size=64):
    dp_size, tp_size = mesh_shape
    total_devices = dp_size * tp_size
    distributed = total_devices > 1
    global_batch = dp_size * batch_size
    raw_local_V = vocab_size // tp_size

    print(f"\n{'=' * 70}")
    print(
        f"Test: vocab_parallel_cross_entropy_loss  (mesh=[{dp_size},{tp_size}], "
        f"B={batch_size}, S={seq_len}, V={vocab_size}, local_V={raw_local_V})"
    )
    print(f"{'=' * 70}")

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

    if distributed:
        # Natural placements for the C++ vocab_parallel_cross_entropy_loss op:
        #   logits  [dp*B, 1, S, V]  -> Shard(0) on mesh-axis 0 (DP), Shard(3) on mesh-axis 1 (TP)
        #   targets [dp*B, S]        -> Shard(0) on mesh-axis 0 (DP), Replicate on mesh-axis 1 (TP)
        # When dp_size == 1 the row placement degrades to Replicate automatically.
        logits_mapper = ttnn.create_mesh_mapper(
            device,
            ttnn.MeshMapperConfig(
                row_dim=0 if dp_size > 1 else None,
                col_dim=3,
            ),
        )
        targets_mapper = ttnn.create_mesh_mapper(
            device,
            ttnn.MeshMapperConfig(
                row_dim=0 if dp_size > 1 else None,
                col_dim=None,
            ),
        )
        # Gradient composer: reassemble the natural [dp*B, 1, S, V] tensor by concatenating
        # along tensor dim 0 across mesh-axis 0 and along tensor dim 3 across mesh-axis 1.
        grad_composer = ttnn.create_mesh_composer(
            device,
            ttnn.MeshComposerConfig([0, 3]),
        )

        def make_cpp_targets_tensor():
            return ttml.autograd.Tensor.from_numpy(
                targets_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, targets_mapper
            )

        def reconstruct_full_grad_from_mesh(grad_tensor):
            # grad_tensor already has the natural [dp*B, 1, S, V] shape (modulo vocab padding).
            grad_full = ttnn.to_torch(grad_tensor, mesh_composer=grad_composer).float().numpy()
            # Trim vocab padding introduced by tile alignment.
            grad_full = grad_full[:, :, :, :vocab_size]
            # The C++ op averages loss over B*S within each DP group; the PyTorch reference
            # averages over the global dp*B*S, so the gradients differ by a factor of dp_size.
            if dp_size > 1:
                grad_full = grad_full / dp_size
            return grad_full

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
    # 2. C++ vocab_parallel_cross_entropy_loss — forward value
    # ==================================================================
    if tp_size > 1:
        print("\n--- C++ vocab_parallel_cross_entropy_loss: forward ---")

        def test_cpp_ce_forward():
            ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
            logits = ttml.autograd.Tensor.from_numpy(
                logits_np,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                logits_mapper,
            )
            tgt = make_cpp_targets_tensor()
            loss = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(logits, tgt, cluster_axis=1)
            val = extract_loss(loss, device, distributed)
            ctx.reset_graph()
            return val

        val, err = try_op(test_cpp_ce_forward)
        report("C++ sharded CE forward", val, err, ref_loss)

    # ==================================================================
    # 3. C++ vocab_parallel_cross_entropy_loss — backward gradients
    # ==================================================================
    if tp_size > 1:
        print("\n--- C++ vocab_parallel_cross_entropy_loss: backward ---")

        def test_cpp_ce_backward():
            ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            logits = ttml.autograd.Tensor.from_numpy(
                logits_np,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                logits_mapper,
            )
            logits.set_requires_grad(True)
            tgt = make_cpp_targets_tensor()
            loss = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(logits, tgt, cluster_axis=1)
            loss.backward(False)

            grad_full = reconstruct_full_grad_from_mesh(logits.get_grad())

            max_diff = np.abs(grad_full - ref_grad).max()
            mean_diff = np.abs(grad_full - ref_grad).mean()
            ctx.reset_graph()
            return max_diff, mean_diff

        val, err = try_op(test_cpp_ce_backward)
        total += 1
        if err is not None:
            failed += 1
            print(f"  C++ dist CE backward: FAIL (crashed: {err})")
        else:
            max_diff, mean_diff = val
            ok = max_diff < 0.05
            tag = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"  C++ dist CE backward: {tag}  max_diff={max_diff:.6f}  " f"mean_diff={mean_diff:.6f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed, {total} total")
    print(f"{'=' * 70}")

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

    ctx, device = setup_device(*args.mesh_shape)
    try:
        all_ok = True
        for vs in args.vocab_size:
            ok = run_test(
                ctx,
                device,
                args.mesh_shape,
                args.batch_size,
                args.seq_len,
                vs,
            )
            all_ok = all_ok and ok
    finally:
        ctx.close_device()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
