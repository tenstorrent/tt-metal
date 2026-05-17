"""SigLIP softmax kernel parity vs torch.softmax fp32.

Sandbox: (M=256, K=1152) bf16 input, softmax along K. Validates the
row-wise softmax primitive before integrating into SDPA.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import pcc  # noqa: E402

M, K = 256, 1152


def make_softmax_input(seed: int = 42) -> torch.Tensor:
    """Synthetic input mimicking attention scores: small values around 0."""
    g = torch.Generator().manual_seed(seed)
    # Mimic attention scores: Q @ K^T / sqrt(head_dim) typically bounded
    # in [-10, 10] range. Use Gaussian with std=2.
    return torch.randn(M, K, generator=g, dtype=torch.bfloat16) * 2.0


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_device_kernel_pcc(device):
    from softmax_op import SigLIPSoftmaxOp, build_tensors_for_softmax_test

    x_torch = make_softmax_input(seed=42)
    y_golden = F.softmax(x_torch.float(), dim=-1).to(torch.bfloat16)

    (activation_tt, scaler_tt, max_tt, exp_tt, sum_tt, isum_tt, output_tt) = build_tensors_for_softmax_test(
        device, x_torch
    )

    SigLIPSoftmaxOp.op(activation_tt, scaler_tt, output_tt, max_tt, exp_tt, sum_tt, isum_tt)

    import ttnn as _ttnn

    y_device = _ttnn.to_torch(output_tt)
    p = pcc(y_golden, y_device)
    print(f"\nPCC (softmax, kernel vs torch) = {p:.6f}")

    # Sanity: each row should sum to ~1 (softmax property).
    row_sums = y_device.float().sum(dim=-1)
    print(
        f"  Row sum mean={row_sums.mean().item():.6f}, min={row_sums.min().item():.6f}, max={row_sums.max().item():.6f}"
    )

    assert p >= 0.99, f"PCC {p} below 0.99 gate"
