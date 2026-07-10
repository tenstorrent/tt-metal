# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for Flash-Attention scaled_dot_product_attention.

Single authoritative precision characterization test (numeric-formats-metal §10).
Cross-product of dtype × math_fidelity × fp32_dest_acc_en × input distribution
over a set of tile-aligned SDPA shapes (Refinement 1 supports tile_aligned only).

Asserts on PCC only; all other metrics are printed for observability. Cells that
match the op EXCLUSIONS ({float32, fp32_dest_acc_en=False}) are skipped.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# (Q_shape, K_shape, V_shape) — tile-aligned self/cross/GQA, small → large.
SHAPES = [
    pytest.param(((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)), id="1x1x32x32_single_tile"),
    pytest.param(((1, 1, 32, 64), (1, 1, 32, 64), (1, 1, 32, 64)), id="1x1x32x64"),
    pytest.param(((1, 1, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64)), id="1x1x128x64"),
    pytest.param(((1, 2, 256, 64), (1, 2, 256, 64), (1, 2, 256, 64)), id="1x2x256x64_2kv_blocks"),
    pytest.param(((2, 4, 512, 64), (2, 4, 512, 64), (2, 4, 512, 64)), id="2x4x512x64_multi"),
    # D=128 (Dt=4) at small S — exercises the wider head_dim tiling within fp32's
    # doubled CB footprint. Larger-S D=128 fp32 OOMs (L1 budget, Refinement 4).
    pytest.param(((1, 2, 32, 128), (1, 2, 32, 128), (1, 2, 32, 128)), id="1x2x32x128_wide_D"),
    pytest.param(((1, 4, 128, 64), (1, 4, 64, 64), (1, 4, 64, 64)), id="cross_Sq_gt_Skv"),
    pytest.param(((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64)), id="gqa_4to1"),
]

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,  # no native torch bf8b; reference in bf16
}


def _pcc_gate(math_fidelity, distribution):
    """PCC assertion gate (numeric-formats-metal §11).

    The realistic slice — `normal` (randn) inputs, matching the golden suite and
    real models — gates at 0.99 across ALL dtypes and fidelities (empirically all
    pass). `uniform` [0,1) all-positive inputs are a documented pathological case
    for attention (near-uniform softmax + reduced-fidelity matmul over positive-
    only operands), independent of dtype; LoFi/HiFi2 there legitimately drop to
    ~0.78/0.96 — expected hardware behavior, gated loosely as characterization,
    not a correctness bug. All other metrics are printed regardless of gate.
    """
    if distribution == "randn":
        return 0.99
    # uniform stress case, by fidelity
    if math_fidelity == ttnn.MathFidelity.HiFi4:
        return 0.96
    if math_fidelity == ttnn.MathFidelity.HiFi2:
        return 0.95
    return 0.75  # LoFi


@pytest.mark.parametrize(
    "distribution",
    [pytest.param("randn", id="normal"), pytest.param("rand", id="uniform")],
)
@pytest.mark.parametrize(
    "fp32_acc",
    [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")],
)
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
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
@pytest.mark.parametrize("shapes", SHAPES)
def test_precision_matrix(device, shapes, dtype, math_fidelity, fp32_acc, distribution):
    # Op EXCLUSION: float32 + fp32_dest_acc_en=False (maxed input, non-maxed acc).
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSION: {float32, fp32_dest_acc_en=False}")

    q_shape, k_shape, v_shape = shapes
    tdt = _TORCH_DTYPE[dtype]
    torch.manual_seed(2026)
    gen = torch.randn if distribution == "randn" else torch.rand
    Q = gen(q_shape, dtype=tdt)
    K = gen(k_shape, dtype=tdt)
    V = gen(v_shape, dtype=tdt)

    # fp32 reference with GQA/MQA head broadcast.
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        r = H_q // H_kv
        Kf, Vf = Kf.repeat_interleave(r, 1), Vf.repeat_interleave(r, 1)
    expected = torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf)

    to_dev = lambda t: ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_acc, math_approx_mode=False)
    ttnn_out = scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V), compute_kernel_config=cfg)
    out = ttnn.to_torch(ttnn_out).float()

    diff = (expected - out).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    median_abs = diff.median().item()
    p99_abs = torch.quantile(diff.flatten(), 0.99).item()
    rms = torch.sqrt((diff**2).mean()).item()
    denom = torch.sqrt((expected.float() ** 2).mean()).clamp(min=1e-10).item()
    rel_rms = rms / denom
    gate = _pcc_gate(math_fidelity, distribution)
    _, pcc_msg = comp_pcc(expected, out, pcc=gate)
    _, allclose_msg = comp_allclose(expected, out, rtol=0.1, atol=0.1)

    print(
        f"\n[precision-matrix] shape={q_shape} dtype={dtype} fidelity={math_fidelity} "
        f"fp32_acc={fp32_acc} dist={distribution} | {pcc_msg} max_abs={max_abs:.5f} "
        f"mean_abs={mean_abs:.5f} median_abs={median_abs:.5f} p99_abs={p99_abs:.5f} "
        f"rel_rms={rel_rms:.5f} | {allclose_msg}"
    )

    passed, msg = comp_pcc(expected, out, pcc=gate)
    assert passed, msg
