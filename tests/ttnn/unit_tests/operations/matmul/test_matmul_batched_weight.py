# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched-weight (true batched matmul) matrix for the 2D dual-multicast matmul
(Refinement 3).

A batched weight B (..., K, N) carries leading dims matching the activation's —
one independent K×N matrix per batch. The reader's in1 (weight) tile-id gains a
per-batch b*Kt*Nt offset (weight_batch_stride): batch b reads weight matrix b
instead of re-reading a shared 2D block. The activation read, writer, batch
loop, and dual-multicast topology are unchanged.

This matrix asserts BOTH PCC and relative-RMS at the SAME per-(effective-dtype,
acc) bands the golden suite grades by (helpers.TOLERANCES). It covers:
  * the 3 golden batched INPUTS shapes (batch=4, batch=8, rank-4 2×4 batch grid),
  * each across fp32/bf16/bf8b + acc T/F + one mixed bf16/fp32 path,
  * a non-regression check that the SHARED-weight path (2D weight against a
    batched activation) still produces the shared result,
  * a numerical cross-check that batched-weight output is NOT just the
    shared-weight result (i.e. the per-batch offset is actually applied),
  * structural validate() checks (mismatched batched leading dims raise).

Device comes from the dir conftest's module-scoped fixture.
"""

import pytest
import torch
import ttnn

from ttnn.operations.matmul import matmul
from models.common.utility_functions import comp_pcc


_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}

_COARSENESS = {ttnn.float32: 0, ttnn.bfloat16: 1, ttnn.bfloat8_b: 2}

# Mirror eval/golden_tests/matmul/helpers.py::TOLERANCES (PCC, relRMS), keyed on
# the COARSER of activation/weight dtype × fp32_dest_acc_en.
_TOLERANCES = {
    (ttnn.float32, True): (0.999, 0.02),
    (ttnn.bfloat16, True): (0.997, 0.04),
    (ttnn.bfloat16, False): (0.99, 0.10),
    (ttnn.bfloat8_b, True): (0.98, 0.12),
    (ttnn.bfloat8_b, False): (0.98, 0.15),
}


def _effective_dtype(dtype, weight_dtype):
    return dtype if _COARSENESS[dtype] >= _COARSENESS[weight_dtype] else weight_dtype


# (A_shape, B_shape) — B carries a real batch dim matching A's leading dims.
SHAPES = [
    pytest.param((4, 128, 512), (4, 512, 512), id="batch4_128x512x512"),
    pytest.param((8, 64, 128), (8, 128, 64), id="batch8_small_64x128x64"),
    pytest.param((2, 4, 128, 256), (2, 4, 256, 128), id="rank4_2x4_128x256x128"),
]


# (act_dtype, weight_dtype, fp32_acc) — fp32 only with acc=True (the {fp32,
# acc=False} op EXCLUSION); bf16/bf8b at both accumulators; one mixed path.
CONFIGS = [
    pytest.param(ttnn.float32, ttnn.float32, True, id="fp32_fp32_accT"),
    pytest.param(ttnn.bfloat16, ttnn.bfloat16, True, id="bf16_bf16_accT"),
    pytest.param(ttnn.bfloat16, ttnn.bfloat16, False, id="bf16_bf16_accF"),
    pytest.param(ttnn.bfloat8_b, ttnn.bfloat8_b, True, id="bf8b_bf8b_accT"),
    pytest.param(ttnn.bfloat8_b, ttnn.bfloat8_b, False, id="bf8b_bf8b_accF"),
    pytest.param(ttnn.bfloat16, ttnn.float32, True, id="bf16_fp32_mixed_accT"),
]


@pytest.mark.parametrize("a_shape, b_shape", SHAPES)
@pytest.mark.parametrize("dtype, weight_dtype, fp32_acc", CONFIGS)
def test_matmul_batched_weight_matrix(device, a_shape, b_shape, dtype, weight_dtype, fp32_acc):
    # Guard: B must be a real batched weight with matching leading dims.
    assert list(b_shape[:-2]) == list(a_shape[:-2]), "test expects matching batched leading dims"
    assert any(d > 1 for d in b_shape[:-2]), "test expects a real batch dim in B"

    torch.manual_seed(0)
    A = torch.randn(a_shape, dtype=_TORCH_DTYPE[dtype])
    B = torch.randn(b_shape, dtype=_TORCH_DTYPE[weight_dtype])
    # torch.matmul over matching leading dims = independent per-batch matmul.
    expected = torch.matmul(A.to(torch.float32), B.to(torch.float32))

    ttnn_a = ttnn.from_torch(A, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(B, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
    )
    out = ttnn.to_torch(matmul(ttnn_a, ttnn_b, compute_kernel_config=config)).to(torch.float32)

    expected_shape = list(a_shape[:-1]) + [b_shape[-1]]
    assert list(out.shape) == expected_shape, f"shape {list(out.shape)} != {expected_shape}"

    eff = _effective_dtype(dtype, weight_dtype)
    pcc_band, rms_band = _TOLERANCES[(eff, fp32_acc)]

    e = expected.flatten().to(torch.float64)
    a = out.flatten().to(torch.float64)
    abs_err = (e - a).abs()
    denom = float(e.pow(2).mean().sqrt()) or 1.0
    rel_rms = float(abs_err.pow(2).mean().sqrt()) / denom
    _, pcc_val = comp_pcc(expected, out, pcc=pcc_band)

    print(
        f"\n[batched-weight] {a_shape}@{b_shape} act={dtype.name} wt={weight_dtype.name} "
        f"eff={eff.name} fp32_acc={fp32_acc} | PCC={pcc_val} relRMS={rel_rms:.5g} "
        f"| bands PCC>={pcc_band} RMS<={rms_band}"
    )

    assert pcc_val >= pcc_band, f"PCC {pcc_val} < {pcc_band} (eff {eff.name}, acc={fp32_acc})"
    assert rel_rms <= rms_band, f"relRMS {rel_rms} > {rms_band} (eff {eff.name}, acc={fp32_acc})"


def test_batched_weight_differs_from_shared(device):
    """The per-batch offset must actually be applied: a batched weight whose
    matrices differ per batch must produce DIFFERENT output than feeding only
    batch-0's weight as a shared 2D weight against the same activation.

    Without the b*Kt*Nt offset the reader would read batch-0's weight for every
    batch, so this test would see batch>0 outputs match the shared-weight result
    — i.e. it pins the regression the offset prevents.
    """
    torch.manual_seed(1)
    a_shape = (3, 64, 96)
    b_shape = (3, 96, 128)
    A = torch.randn(a_shape, dtype=torch.float32)
    B = torch.randn(b_shape, dtype=torch.float32)

    ttnn_a = ttnn.from_torch(A, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(B, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b0 = ttnn.from_torch(B[0], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    out_batched = ttnn.to_torch(matmul(ttnn_a, ttnn_b, compute_kernel_config=cfg)).to(torch.float32)
    out_shared = ttnn.to_torch(matmul(ttnn_a, ttnn_b0, compute_kernel_config=cfg)).to(torch.float32)

    expected_batched = torch.matmul(A, B)
    # Batched matches the true per-batch reference.
    _, pcc_b = comp_pcc(expected_batched, out_batched, pcc=0.999)
    assert pcc_b >= 0.999, f"batched PCC {pcc_b} < 0.999"

    # batch 0 of the two paths agrees (same weight matrix B[0]); a later batch
    # must DIFFER (different weight matrix) — proving the offset is live.
    assert torch.allclose(out_batched[0], out_shared[0], rtol=1e-3, atol=1e-3), "batch 0 should match shared B[0]"
    diff = (out_batched[2] - out_shared[2]).abs().max().item()
    assert diff > 1.0, f"batch 2 should differ from shared B[0] (max abs diff {diff})"


def test_validate_batched_weight_now_supported(device):
    """A batched weight with matching leading dims is now SUPPORTED (no raise)."""
    A = torch.randn((4, 64, 64), dtype=torch.float32)
    B = torch.randn((4, 64, 64), dtype=torch.float32)
    ttnn_a = ttnn.from_torch(A, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(B, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(matmul(ttnn_a, ttnn_b)).to(torch.float32)
    _, pcc = comp_pcc(torch.matmul(A, B), out, pcc=0.999)
    assert pcc >= 0.999


def test_batched_weight_broadcast_over_size1_A_dim(device):
    """torch.matmul broadcast over a SIZE-1 activation leading dim.

    A=(1, 2, M, K) against B=(2, K, N): B_lead=[2] right-aligns with A_lead=[1,2]
    and broadcasts over A's leading size-1 dim. prod(A_lead)==prod(B_lead)==2, so
    the flattened batch correspondence is the identity map and the reader's
    b*Kt*Nt offset is correct. This is the test_translated
    test_matmul_with_transpose_and_configs[1-2-4096-32-256] scenario (which feeds
    a squeezed (2,K,N) weight against a (1,2,M,K) activation).
    """
    torch.manual_seed(0)
    A = torch.rand((1, 2, 4096, 32), dtype=torch.bfloat16)
    B = torch.rand((2, 32, 256), dtype=torch.bfloat16)  # B_lead=[2] vs A_lead=[1,2]
    expected = torch.matmul(A.to(torch.float32), B.to(torch.float32))  # (1,2,4096,256)

    ttnn_a = ttnn.from_torch(A, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(B, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(matmul(ttnn_a, ttnn_b)).to(torch.float32)

    assert list(out.shape) == [1, 2, 4096, 256], f"shape {list(out.shape)}"
    _, pcc = comp_pcc(expected, out, pcc=0.999)
    assert pcc >= 0.999, f"broadcast-over-size1 PCC {pcc} < 0.999"


def test_genuine_broadcast_replication_raises(device):
    """A GENUINE broadcast that replicates one weight across many distinct
    A-batches (A_lead=[3,2], B_lead=[2]) changes the per-batch mapping and is out
    of this refinement's scope — must raise ValueError, not silently miscompute."""
    A = torch.randn((3, 2, 64, 64), dtype=torch.float32)
    B = torch.randn((2, 64, 64), dtype=torch.float32)  # B_lead=[2] vs A_lead=[3,2]
    ttnn_a = ttnn.from_torch(A, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(B, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(ValueError):
        matmul(ttnn_a, ttnn_b)


def test_validate_batched_weight_leading_dim_mismatch_raises(device):
    """A batched weight whose leading dims don't match the activation's still
    raises a structural ValueError (unchanged contract)."""
    A = torch.randn((4, 64, 64), dtype=torch.float32)
    B = torch.randn((8, 64, 64), dtype=torch.float32)  # batch 8 != activation batch 4
    ttnn_a = ttnn.from_torch(A, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(B, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(ValueError):
        matmul(ttnn_a, ttnn_b)


def test_shared_weight_against_batched_activation_nonregression(device):
    """Phase-0 path: a shared 2D weight against a batched activation still works
    (weight_batch='single', weight_batch_stride=0)."""
    A = torch.randn((4, 128, 512), dtype=torch.float32)
    B = torch.randn((512, 512), dtype=torch.float32)  # shared 2D weight
    ttnn_a = ttnn.from_torch(A, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(B, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(matmul(ttnn_a, ttnn_b)).to(torch.float32)
    expected = torch.matmul(A, B)
    _, pcc = comp_pcc(expected, out, pcc=0.999)
    assert pcc >= 0.999, f"shared-weight non-regression PCC {pcc} < 0.999"
