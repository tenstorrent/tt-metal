# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for scaled_dot_product_attention (Refinement 1).

Authoritative precision characterization across the numeric surface opened by
R1:
- dtype:            bfloat16, float32, bfloat8_b
- fp32_dest_acc_en: True, False  (driven via ttnn.ComputeKernelConfig)
- math_fidelity:    HiFi4 / HiFi2 / LoFi

The R1 load-bearing change is storing the online-softmax running statistics
(cb_o / cb_l / cb_m and the per-iteration cb_pv / cb_o_resc) in fp32 when the
DEST accumulator is fp32 — so the recurrence stops re-rounding to bf16 every
KV-chunk. The long-context shapes (S in {2048, 4096}) are the cells that were
`supported_fail` at Phase 0; they must pass here for bf16 + fp32 DEST acc.

The (float32, fp32_dest_acc_en=False) cell is an op-side EXCLUSION (maxed
input + non-maxed DEST acc is refused) — see test_exclusion_fp32_no_acc.

PCC is the gate; abs/RMS are printed for observability.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}


def _reference_sdpa(Q, K, V, *, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf)


# PCC thresholds per (dtype, fp32_dest_acc_en). Mirrors the golden TOLERANCES;
# bf8b/16-bit-DEST are inherently looser.
_PCC = {
    (ttnn.float32, True): 0.999,
    (ttnn.bfloat16, True): 0.995,
    (ttnn.bfloat16, False): 0.99,
    (ttnn.bfloat8_b, True): 0.99,
    (ttnn.bfloat8_b, False): 0.99,
}


SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 128, 64), id="multi_tile"),
    pytest.param((1, 4, 128, 64), id="multi_head"),
    pytest.param((2, 8, 256, 64), id="batched_multi_head"),
    pytest.param((1, 8, 512, 128), id="larger"),
    pytest.param((1, 1, 2048, 64), id="long_context_2048"),  # R1 supported_fail cell
    pytest.param((1, 1, 4096, 64), id="long_context_4096"),  # R1 supported_fail cell
]


@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
    ],
)
@pytest.mark.parametrize(
    "fp32_acc",
    [
        pytest.param(True, id="fp32_acc"),
        pytest.param(False, id="bf16_acc"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bfp8"),
    ],
)
@pytest.mark.parametrize("shape", SHAPES)
def test_scaled_dot_product_attention_precision_matrix(device, shape, dtype, fp32_acc, math_fidelity):
    # (float32, acc=False) is an op-side EXCLUSION — skip (covered separately).
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSION: float32 + fp32_dest_acc_en=False is refused (see test_exclusion_fp32_no_acc)")

    B, H, S, D = shape
    torch.manual_seed(0)
    torch_dtype = _TORCH_DTYPE[dtype]
    Q = torch.randn((B, H, S, D), dtype=torch_dtype)
    K = torch.randn((B, H, S, D), dtype=torch_dtype)
    V = torch.randn((B, H, S, D), dtype=torch_dtype)

    ref = _reference_sdpa(Q, K, V).float()

    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )

    tq = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=config)
    got = ttnn.to_torch(out).float()

    abs_err = (got - ref).abs()
    max_abs = abs_err.max().item()
    rms = torch.sqrt((abs_err**2).mean()).item()
    rel_rms = rms / (ref.std().item() + 1e-12)
    pcc_pass, pcc_str = comp_pcc(ref, got, pcc=_PCC[(dtype, fp32_acc)])
    _, allclose_str = comp_allclose(ref, got)
    print(
        f"\n[precmatrix] shape={shape} dtype={dtype} fp32_acc={fp32_acc} "
        f"fid={math_fidelity} max_abs={max_abs:.5f} rel_rms={rel_rms:.5f} "
        f"| {pcc_str} | {allclose_str}"
    )

    assert pcc_pass, f"PCC below threshold {_PCC[(dtype, fp32_acc)]}: {pcc_str}"


def test_exclusion_fp32_no_acc(device):
    """float32 + fp32_dest_acc_en=False must be refused (ExcludedCell)."""
    Q = torch.randn((1, 1, 32, 32), dtype=torch.float32)
    tq = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    with pytest.raises(NotImplementedError):
        scaled_dot_product_attention(tq, tq, tq, compute_kernel_config=config)


def test_bfloat8_b_supported(device):
    """bfloat8_b is accepted (smoke — block-float pack/unpack is transparent)."""
    torch.manual_seed(0)
    Q = torch.randn((1, 2, 128, 64), dtype=torch.bfloat16)
    K = torch.randn((1, 2, 128, 64), dtype=torch.bfloat16)
    V = torch.randn((1, 2, 128, 64), dtype=torch.bfloat16)
    ref = _reference_sdpa(Q, K, V).float()
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tq, tk, tv)
    got = ttnn.to_torch(out).float()
    pcc_pass, pcc_str = comp_pcc(ref, got, pcc=0.99)
    print(f"\n[bf8b] {pcc_str}")
    assert pcc_pass, pcc_str
