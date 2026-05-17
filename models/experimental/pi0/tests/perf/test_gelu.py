"""SigLIP GELU primitive: parity vs torch GELU (tanh approximation).

(M=256, D=4320) bf16 elementwise. Matches SigLIP-So400m's gelu_pytorch_tanh.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import pcc  # noqa: E402

M, D = 256, 4320


def make_input(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    # Mimic post-FC1 activations: small std, slightly larger range than gaussian.
    return torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 1.0


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_device_kernel_pcc(device):
    from gelu_op import SigLIPGeluOp, build_tensors_for_gelu_test

    x = make_input(seed=42)
    y_golden = F.gelu(x.float(), approximate="tanh").to(torch.bfloat16)

    in_tt, out_tt = build_tensors_for_gelu_test(device, x, num_cores=8)
    SigLIPGeluOp.op(in_tt, out_tt)

    import ttnn as _ttnn

    y_device = _ttnn.to_torch(out_tt)
    p = pcc(y_golden, y_device)
    print(f"\nPCC (GELU, kernel vs torch tanh-approx) = {p:.6f}")
    assert p >= 0.99, f"PCC {p} below 0.99 gate"
