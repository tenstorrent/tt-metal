"""POC test: pi05_siglip_ops::Softmax Op-struct.

Validates the Op-struct port preserves the 5-phase numerically-stable
softmax (max, exp, sum, isum, scale). Compares against torch fp32 softmax
(PCC) and against the monolithic siglip_softmax_kernel.cpp (bit-identical).
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
from golden_fc1 import pcc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attention_block"))
from op_struct_softmax_poc import (  # noqa: E402
    SigLIPSoftmaxOpStruct,
    build_tensors_for_softmax_test,
)

M, K = 256, 1152


def make_softmax_input(seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(M, K, generator=g, dtype=torch.bfloat16) * 2.0


def _run(device, op_cls, x_torch):
    tensors = build_tensors_for_softmax_test(device, x_torch)
    activation_tt, scaler_tt, max_tt, exp_tt, sum_tt, isum_tt, output_tt = tensors
    op_cls.op(activation_tt, scaler_tt, output_tt, max_tt, exp_tt, sum_tt, isum_tt)
    import ttnn as _ttnn

    return _ttnn.to_torch(output_tt)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_softmax_op_struct_pcc(device):
    x_torch = make_softmax_input(seed=42)
    y_golden = F.softmax(x_torch.float(), dim=-1).to(torch.bfloat16)

    y_device = _run(device, SigLIPSoftmaxOpStruct, x_torch)
    p = pcc(y_golden, y_device)
    print(f"\nPCC (Softmax Op-struct vs torch fp32) = {p:.6f}")
    assert p >= 0.99, f"PCC {p} below 0.99 gate"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_softmax_op_struct_matches_monolithic(device):
    from softmax_op import SigLIPSoftmaxOp  # noqa: E402

    x_torch = make_softmax_input(seed=42)
    y_op_struct = _run(device, SigLIPSoftmaxOpStruct, x_torch)
    y_mono = _run(device, SigLIPSoftmaxOp, x_torch)

    max_diff = float((y_op_struct.float() - y_mono.float()).abs().max())
    print(f"\nmax abs diff (Softmax Op-struct vs monolithic) = {max_diff:.6e}")
    assert max_diff == 0.0, f"Op-struct path diverged from monolithic by {max_diff}"
