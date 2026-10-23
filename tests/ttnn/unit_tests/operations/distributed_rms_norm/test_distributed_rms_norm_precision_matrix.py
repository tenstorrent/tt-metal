# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision matrix for distributed_rms_norm (Refinement 4): shape x dtype x DEST width x math fidelity x distribution.

System 2x2 mesh, group along cluster_axis 1 (G = 2, Linear), FABRIC_2D_TORUS_XY, bf16 TILE gamma. Metrics are taken on
the WORST device of the group and printed as one `PRECISION_MATRIX` line per case (precision_matrix_results.md is built
from them). Asserts PCC only (0.99 matrix floor). {float32, fp32_dest_acc_en=False} is an EXCLUSION -> skipped.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_precision_matrix.py
"""

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import (
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    quantize_like_device,
)
from models.common.utility_functions import calculate_detailed_ulp_stats, comp_allclose, comp_pcc
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

CLUSTER_AXIS = 1
EPS = 1e-6
PCC_FLOOR = 0.99
_TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]

SHAPES = [
    pytest.param((1, 1, 32, 64), id="32x64_small"),
    pytest.param((1, 1, 32, 1792), id="32x1792_decode_R3"),
    pytest.param((1, 1, 64, 1024), id="64x1024"),
    pytest.param((1, 1, 640, 1792), id="640x1792_R3_waves"),
    pytest.param((1, 1, 2048, 1280), id="2048x1280_large_R1"),
    pytest.param((1, 1, 32, 50), id="32x50_W_non_aligned"),
    pytest.param((1, 1, 17, 1280), id="17x1280_H_non_aligned"),
    pytest.param((2, 1, 100, 100), id="100x100_both_non_aligned"),
]


def _stacked(mesh, shape, dist, seed):
    torch.manual_seed(seed)
    rows, cols = tuple(mesh.shape)
    full = (rows, cols, *shape)
    return torch.rand(full) if dist == "rand" else torch.randn(full)


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("distribution", [pytest.param("rand", id="uniform"), pytest.param("randn", id="normal")])
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi3, id="HiFi3"),
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
@pytest.mark.parametrize("shape", SHAPES)
def test_distributed_rms_norm_precision_matrix(mesh_device, shape, dtype, math_fidelity, fp32_acc, distribution):
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSION {dtype: float32, fp32_dest_acc_en: False}: fp32 input needs fp32 accumulation")
    x = quantize_like_device(_stacked(mesh_device, shape, distribution, seed=11).to(_TORCH[dtype]), dtype)
    g = _stacked(mesh_device, (1, 1, 1, shape[-1]), "randn", seed=12).to(torch.bfloat16)
    expected = pytorch_distributed_rms_norm(x, g, CLUSTER_AXIS, EPS)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(g, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_acc, math_approx_mode=False)
    out = distributed_rms_norm(tx, cluster_axis=CLUSTER_AXIS, gamma=tg, epsilon=EPS, compute_kernel_config=cfg)

    cols = tuple(mesh_device.shape)[1]
    worst = None
    for idx, dev in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        got = ttnn.to_torch(dev).float()
        ref = expected[r, c].float()
        _, pcc = comp_pcc(ref, got, PCC_FLOOR)
        if worst is None or pcc < worst[0]:
            worst = (pcc, got, ref)
    pcc, got, ref = worst
    _, allclose_msg = comp_allclose(ref, got)
    ulp = calculate_detailed_ulp_stats(ref.to(torch.bfloat16), got.to(torch.bfloat16))
    abs_err = (got - ref).abs()
    rel_rms = (abs_err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp(min=1e-10)).item()
    print(
        f"\nPRECISION_MATRIX shape={'x'.join(map(str, shape))} dtype={dtype} fidelity={math_fidelity} "
        f"fp32_acc={fp32_acc} dist={distribution} pcc={pcc:.6f} rel_rms={rel_rms:.4g} "
        f"median_abs={abs_err.median().item():.4g} p99_abs={torch.quantile(abs_err.flatten(), 0.99).item():.4g} "
        f"ulp={ulp} | {allclose_msg}"
    )
    assert pcc >= PCC_FLOOR, f"PCC {pcc} < {PCC_FLOOR}"
