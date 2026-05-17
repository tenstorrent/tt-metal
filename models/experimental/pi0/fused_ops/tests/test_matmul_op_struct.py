"""POC test: pi05_siglip_ops::EncoderMatmul Op-struct.

Validates the Op-struct port handles both N_TILES_PER_CORE settings used by
the SigLIP attention sub-block:
  * QKV   (N_per_core=3, N=3456, 36 cores)
  * O-proj (N_per_core=1, N=1152, 36 cores)

Compares against torch fp32 (PCC) and against the monolithic
qkv_matmul_kernel.cpp output (bit-identical).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
from golden import K, M, N_FUSED, golden, load_layer0_qkv, make_input, pcc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attention_block"))
from op_struct_matmul_poc import (  # noqa: E402
    SigLIPQKVMatmulOpStruct,
    SigLIPOprojMatmulOpStruct,
    build_qkv_tensors,
    build_tensors_for_oproj_test,
)


# ----------------------------------------------------------------------
# QKV shape: N_per_core=3
# ----------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qkv_matmul_op_struct_pcc(device):
    w_torch, b_torch = load_layer0_qkv()
    x_torch = make_input(seed=42)

    y_golden_no_bias = (x_torch.float() @ w_torch.float()).to(x_torch.dtype)

    a_tt, w_tt, out_tt = build_qkv_tensors(device, w_torch, b_torch, x_torch, num_cores=36)
    SigLIPQKVMatmulOpStruct.op(a_tt, w_tt, out_tt, num_cores=36)

    import ttnn as _ttnn

    y_device = _ttnn.to_torch(out_tt)
    p = pcc(y_golden_no_bias, y_device)
    print(f"\nPCC (QKV Op-struct, no bias, vs torch) = {p:.6f}")
    assert p >= 0.99, f"PCC {p} below 0.99 gate"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qkv_matmul_op_struct_matches_monolithic(device):
    """Each tensor set ~1.3 MB/core; two sets don't co-fit on 36 cores.
    Run Op-struct, copy out + deallocate, then run monolithic on a fresh set."""
    import ttnn as _ttnn

    from qkv_op import SigLIPQKVMatmulOp  # noqa: E402

    w_torch, b_torch = load_layer0_qkv()
    x_torch = make_input(seed=42)

    a1, w1, o1 = build_qkv_tensors(device, w_torch, b_torch, x_torch, num_cores=36)
    SigLIPQKVMatmulOpStruct.op(a1, w1, o1, num_cores=36)
    y_op_struct = _ttnn.to_torch(o1)
    _ttnn.deallocate(a1)
    _ttnn.deallocate(w1)
    _ttnn.deallocate(o1)

    a2, w2, o2 = build_qkv_tensors(device, w_torch, b_torch, x_torch, num_cores=36)
    SigLIPQKVMatmulOp.op(a2, w2, o2, num_cores=36)
    y_mono = _ttnn.to_torch(o2)

    max_diff = float((y_op_struct.float() - y_mono.float()).abs().max())
    print(f"\nmax abs diff (QKV Op-struct vs monolithic) = {max_diff:.6e}")
    assert max_diff == 0.0, f"Op-struct path diverged from monolithic by {max_diff}"


# ----------------------------------------------------------------------
# O-proj shape: N_per_core=1
# ----------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_oproj_matmul_op_struct_pcc(device):
    # Synthetic weights — same shape as O-proj. We only care PCC against
    # the matmul reference, not real layer-0 weights for this POC.
    torch.manual_seed(7)
    M_, K_, N_ = 256, 1152, 1152
    w_torch = torch.randn(K_, N_, dtype=torch.bfloat16)
    x_torch = torch.randn(M_, K_, dtype=torch.bfloat16)
    y_golden = (x_torch.float() @ w_torch.float()).to(x_torch.dtype)

    a_tt, w_tt, out_tt = build_tensors_for_oproj_test(device, w_torch, x_torch, num_cores=36)
    SigLIPOprojMatmulOpStruct.op(a_tt, w_tt, out_tt, num_cores=36)

    import ttnn as _ttnn

    y_device = _ttnn.to_torch(out_tt)
    p = pcc(y_golden, y_device)
    print(f"\nPCC (O-proj Op-struct, vs torch) = {p:.6f}")
    assert p >= 0.99, f"PCC {p} below 0.99 gate"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_oproj_matmul_op_struct_matches_monolithic(device):
    import ttnn as _ttnn

    from oproj_op import SigLIPOprojMatmulOp  # noqa: E402

    torch.manual_seed(7)
    M_, K_, N_ = 256, 1152, 1152
    w_torch = torch.randn(K_, N_, dtype=torch.bfloat16)
    x_torch = torch.randn(M_, K_, dtype=torch.bfloat16)

    a1, w1, o1 = build_tensors_for_oproj_test(device, w_torch, x_torch, num_cores=36)
    SigLIPOprojMatmulOpStruct.op(a1, w1, o1, num_cores=36)
    y_op_struct = _ttnn.to_torch(o1)
    _ttnn.deallocate(a1)
    _ttnn.deallocate(w1)
    _ttnn.deallocate(o1)

    a2, w2, o2 = build_tensors_for_oproj_test(device, w_torch, x_torch, num_cores=36)
    SigLIPOprojMatmulOp.op(a2, w2, o2, num_cores=36)
    y_mono = _ttnn.to_torch(o2)

    max_diff = float((y_op_struct.float() - y_mono.float()).abs().max())
    print(f"\nmax abs diff (O-proj Op-struct vs monolithic) = {max_diff:.6e}")
    assert max_diff == 0.0, f"Op-struct path diverged from monolithic by {max_diff}"
