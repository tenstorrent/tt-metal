"""POC test: pi05_siglip_ops::ResidualAdd Op-struct vs torch a + b.

Validates the Op-struct refactor pattern. If this matches the monolithic
siglip_residual_kernel.cpp PCC bar (≥0.999), the same pattern can be applied
to LN, matmul, and SDPA.

Shape: (M=256, D=1152) bf16 elementwise add. 8 cores × 1 M-tile each.
"""
import sys
from pathlib import Path

import pytest
import torch

# Reuse the existing perf-test golden helper.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
from golden_fc1 import pcc  # noqa: E402

# Make the Op-struct POC importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attention_block"))
from op_struct_residual_poc import (  # noqa: E402
    SigLIPResidualAddOpStruct,
    build_tensors_for_residual_test,
)

M, D = 256, 1152


def make_inputs(seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(M, D, generator=g, dtype=torch.bfloat16)
    b = torch.randn(M, D, generator=g, dtype=torch.bfloat16) * 0.3
    return a, b


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_residual_op_struct_pcc(device):
    a_t, b_t = make_inputs(seed=42)
    y_golden = (a_t.float() + b_t.float()).to(torch.bfloat16)

    a_tt, b_tt, out_tt = build_tensors_for_residual_test(device, a_t, b_t, num_cores=8)
    SigLIPResidualAddOpStruct.op(a_tt, b_tt, out_tt)

    import ttnn as _ttnn

    y_device = _ttnn.to_torch(out_tt)
    p = pcc(y_golden, y_device)
    print(f"\nPCC (ResidualAdd Op-struct vs torch) = {p:.6f}")
    assert p >= 0.999, f"PCC {p} below 0.999 gate"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_residual_op_struct_matches_monolithic(device):
    """Bit-by-bit: Op-struct output should equal monolithic kernel output."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
    from residual_op import SigLIPResidualAddOp  # noqa: E402

    a_t, b_t = make_inputs(seed=42)

    # Op-struct path
    a_tt, b_tt, out_tt = build_tensors_for_residual_test(device, a_t, b_t, num_cores=8)
    SigLIPResidualAddOpStruct.op(a_tt, b_tt, out_tt)

    import ttnn as _ttnn

    y_op_struct = _ttnn.to_torch(out_tt)

    # Monolithic path (reuses the same tensor builder shape).
    a_tt2, b_tt2, out_tt2 = build_tensors_for_residual_test(device, a_t, b_t, num_cores=8)
    SigLIPResidualAddOp.op(a_tt2, b_tt2, out_tt2)
    y_mono = _ttnn.to_torch(out_tt2)

    diff = (y_op_struct.float() - y_mono.float()).abs()
    max_diff = float(diff.max())
    print(f"\nmax abs diff (Op-struct vs monolithic) = {max_diff:.6e}")
    assert max_diff == 0.0, f"Op-struct path diverged from monolithic by {max_diff}"
