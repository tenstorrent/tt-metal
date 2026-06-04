# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PoC parity + timing test for LoRALinear.

Compares three execution paths at Wan-2.2 attention/FFN shapes:

  baseline   — plain Linear forward, no LoRA
  runtime    — LoRA delta added at forward: base(x) + scale * (x @ A) @ B
  fused      — base weight pre-fused on device: y = base(x; W + scale*A@B)

Reports per-forward latency for each path, plus the one-time swap cost
for both runtime (host pointer flip) and fused (on-device re-fusion).
The break-even line tells you after how many forwards between swaps the
fused-on-swap path overtakes the runtime-delta path.

Run with:
    pytest -xvs models/tt_dit/experimental/tests/test_lora_linear_poc.py
"""
import time

import pytest
import torch

import ttnn

from ...utils.check import assert_quality
from ...utils.tensor import bf16_tensor
from ..lora.lora_linear import LoRALinear


WARMUP_ITERS = 5
TIMED_ITERS = 50
SWAP_ITERS = 200


def _make_lora_pair(in_features: int, out_features: int, rank: int, dtype: torch.dtype):
    """Random LoRA pair in PyTorch convention (small init like real LoRA)."""
    A = torch.randn(rank, in_features, dtype=dtype) * 0.01
    B = torch.randn(out_features, rank, dtype=dtype) * 0.01
    return A, B


def _torch_lora_reference(x, W_pytorch, b_pytorch, A_torch, B_torch, scale):
    """Reference: y = x @ (W + scale * B @ A).T + b"""
    W_eff = W_pytorch + scale * (B_torch @ A_torch)
    return torch.nn.functional.linear(x, W_eff, b_pytorch)


def _bench(fn, mesh_device, label: str, iters: int = TIMED_ITERS) -> float:
    """Returns ms/call. Synchronizes before and after."""
    for _ in range(WARMUP_ITERS):
        fn()
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh_device)
    t1 = time.perf_counter()
    per_call_ms = (t1 - t0) * 1000.0 / iters
    print(f"  {label:<28s} {per_call_ms:9.4f} ms/call   ({iters} iters)")
    return per_call_ms


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    ("seq_len", "in_features", "out_features", "rank"),
    [
        (4096, 5120, 5120, 64),     # Wan 2.2 attention projection
        (4096, 5120, 13824, 64),    # Wan 2.2 ff1
    ],
)
def test_lora_linear_poc(
    mesh_device: ttnn.MeshDevice,
    seq_len: int,
    in_features: int,
    out_features: int,
    rank: int,
) -> None:
    torch.manual_seed(0)
    dtype = torch.bfloat16
    scale = 0.7

    # ---- Reference setup ----
    torch_lin = torch.nn.Linear(in_features, out_features, bias=True).to(dtype=dtype)
    torch_lin.eval()
    A_torch, B_torch = _make_lora_pair(in_features, out_features, rank, dtype)

    x_torch = torch.randn(1, 1, seq_len, in_features, dtype=dtype)
    with torch.no_grad():
        y_base_ref = torch_lin(x_torch)
        y_lora_ref = _torch_lora_reference(
            x_torch, torch_lin.weight, torch_lin.bias, A_torch, B_torch, scale
        )

    # ---- TT setup ----
    tt_lin = LoRALinear(in_features, out_features, bias=True, mesh_device=mesh_device)
    tt_lin.load_torch_state_dict(torch_lin.state_dict())
    idx = tt_lin.register_lora(A_torch, B_torch, scale=scale, name="bench")
    x_tt = bf16_tensor(x_torch, device=mesh_device)

    # ---- Parity: baseline ----
    tt_lin.set_active(None)
    y = tt_lin(x_tt)
    for t in ttnn.get_device_tensors(y):
        assert_quality(y_base_ref, ttnn.to_torch(t), pcc=0.999_500)

    # ---- Parity: runtime delta ----
    tt_lin.set_active(idx)
    y = tt_lin(x_tt)
    for t in ttnn.get_device_tensors(y):
        assert_quality(y_lora_ref, ttnn.to_torch(t), pcc=0.999_000)

    # ---- Parity: fused base ----
    tt_lin.set_active(None)
    tt_lin.fuse_into_base(idx)
    y = tt_lin(x_tt)
    for t in ttnn.get_device_tensors(y):
        assert_quality(y_lora_ref, ttnn.to_torch(t), pcc=0.999_000)
    tt_lin.restore_base()

    # =================================================================
    # Benchmarks
    # =================================================================
    print(
        f"\n=== LoRALinear PoC  "
        f"seq={seq_len}  in={in_features}  out={out_features}  rank={rank} ==="
    )

    # Forward latency
    tt_lin.set_active(None)
    t_base = _bench(lambda: tt_lin(x_tt), mesh_device, "forward: baseline")

    tt_lin.set_active(idx)
    t_runtime = _bench(lambda: tt_lin(x_tt), mesh_device, "forward: runtime delta")

    tt_lin.set_active(None)
    tt_lin.fuse_into_base(idx)
    t_fused = _bench(lambda: tt_lin(x_tt), mesh_device, "forward: fused base")
    tt_lin.restore_base()

    # Swap latency
    def _swap_runtime():
        tt_lin.set_active(idx)
        tt_lin.set_active(None)

    def _swap_fused():
        tt_lin.fuse_into_base(idx)
        tt_lin.restore_base()

    # per pair, divide by 2 to get per-direction
    t_swap_runtime = _bench(_swap_runtime, mesh_device, "swap: runtime (pair)", iters=SWAP_ITERS) / 2.0
    t_swap_fused = _bench(_swap_fused, mesh_device, "swap: fuse+restore (pair)", iters=20) / 2.0

    # =================================================================
    # Summary
    # =================================================================
    print("\nPer-forward overhead vs baseline:")
    delta_runtime = t_runtime - t_base
    delta_fused = t_fused - t_base
    print(
        f"  runtime delta : +{delta_runtime / t_base * 100:6.2f} %   "
        f"(+{delta_runtime:7.4f} ms)"
    )
    print(
        f"  fused base    : +{delta_fused / t_base * 100:6.2f} %   "
        f"(+{delta_fused:7.4f} ms)"
    )

    print("\nOne-time swap cost (per direction):")
    print(f"  runtime (host-side flip)        : {t_swap_runtime * 1000:9.3f} us")
    print(f"  fused (on-device re-fusion)     : {t_swap_fused * 1000:9.3f} us")

    print("\nBreak-even analysis:")
    forward_diff = t_runtime - t_fused
    if forward_diff > 0:
        n_break = (t_swap_fused - t_swap_runtime) / forward_diff
        print(
            f"  fuse-on-swap overtakes runtime-delta after "
            f"~{n_break:.1f} forwards between swaps"
        )
    else:
        print("  runtime-delta is at-or-faster than fused (unexpected; check setup)")

    # Cleanup
    tt_lin.unregister_lora(idx)
