"""SigLIP residual-add primitive: parity vs torch a + b.

(M=256, D=1152) bf16 elementwise add. 8 cores × 1 M-tile each.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import pcc  # noqa: E402

M, D = 256, 1152


def make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(M, D, generator=g, dtype=torch.bfloat16)
    b = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.3  # residual is typically smaller
    return a, b


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_device_kernel_pcc(device):
    from residual_op import SigLIPResidualAddOp, build_tensors_for_residual_test

    a_t, b_t = make_inputs(seed=42)
    y_golden = (a_t.float() + b_t.float()).to(torch.bfloat16)

    a_tt, b_tt, out_tt = build_tensors_for_residual_test(device, a_t, b_t, num_cores=8)
    SigLIPResidualAddOp.op(a_tt, b_tt, out_tt)

    import ttnn as _ttnn

    y_device = _ttnn.to_torch(out_tt)
    p = pcc(y_golden, y_device)
    print(f"\nPCC (residual add, kernel vs torch) = {p:.6f}")
    assert p >= 0.999, f"PCC {p} below 0.999 gate"
