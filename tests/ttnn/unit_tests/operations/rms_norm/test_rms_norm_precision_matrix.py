# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for ttnn.operations.rms_norm — the single authoritative precision characterization.

Axes (full cross-product): shape x dtype (every SUPPORTED dtype) x fp32_dest_acc_en x math_fidelity x
input distribution. Every metric is printed for every cell; only PCC is asserted (>= 0.99). The
{float32, fp32_dest_acc_en=False} cell is a permanent op-side EXCLUSION and is skipped here for that
reason. The reference is computed in fp32 from the dtype-ROUNDED device inputs (what the kernel
actually saw), so the numbers measure kernel error, not host quantization of the inputs.

At module teardown the collected rows are written to `precision_matrix_results.md` beside this file.
"""

from datetime import date
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import calculate_detailed_ulp_stats, comp_allclose
from ttnn.operations.rms_norm import EXCLUSIONS, SUPPORTED, rms_norm

PCC_THRESHOLD = 0.99
RESULTS_PATH = Path(__file__).with_name("precision_matrix_results.md")

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
DTYPE_ID = {ttnn.float32: "fp32", ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}
FIDELITY_ID = {
    ttnn.MathFidelity.HiFi4: "HiFi4",
    ttnn.MathFidelity.HiFi2: "HiFi2",
    ttnn.MathFidelity.LoFi: "LoFi",
}

SHAPES = [
    pytest.param((32, 32), id="32x32_single_tile"),
    pytest.param((1, 1, 64, 128), id="64x128"),
    pytest.param((4, 128, 512), id="3d_4x128x512"),
    pytest.param((2, 4, 128, 512), id="2x4x128x512_R1"),
    pytest.param((1, 1, 2048, 256), id="2048x256_tall"),
    pytest.param((1, 1, 32, 4096), id="32x4096_R2_occupancy"),
    pytest.param((128, 8192), id="128x8192_wide"),
    pytest.param((1, 1, 64, 12288), id="64x12288_R2_residency"),
]


def torch_rms_norm(x, gamma, epsilon=1e-6):
    xf = x.to(torch.float32)
    return xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon) * gamma.to(torch.float32).reshape(-1)


def _is_excluded(axes):
    return any(all(axes.get(k) == v for k, v in exc.items()) for exc in EXCLUSIONS)


def precision_metrics(expected, actual):
    e, a = expected.flatten(), actual.flatten()
    pcc = torch.corrcoef(torch.stack([e, a]))[0, 1].item()
    abs_err = (a - e).abs()
    return {
        "pcc": pcc,
        "max_abs_err": abs_err.max().item(),
        "median_abs_err": abs_err.median().item(),
        "p99_abs_err": torch.quantile(abs_err[: 2**24], 0.99).item(),
        "rel_rms_err": (abs_err.pow(2).mean().sqrt() / e.pow(2).mean().sqrt().clamp(min=1e-10)).item(),
    }


@pytest.fixture(scope="module", autouse=True)
def results_table():
    rows = []
    yield rows
    if not rows:
        return
    header = (
        "| shape | dtype | fidelity | fp32_dest_acc | dist | PCC | max_abs | median_abs | p99_abs | rel_RMS | "
        "max_ULP | mean_ULP | p99_ULP | allclose(1e-2) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['shape']} | {r['dtype']} | {r['fidelity']} | {r['fp32_acc']} | {r['dist']} | {r['pcc']:.6f} | "
        f"{r['max_abs_err']:.4g} | {r['median_abs_err']:.4g} | {r['p99_abs_err']:.4g} | {r['rel_rms_err']:.4g} | "
        f"{r['max_ulp']:.3g} | {r['mean_ulp']:.3g} | {r['p99_ulp']:.3g} | {r['allclose']} |\n"
        for r in rows
    )
    RESULTS_PATH.write_text(
        f"# rms_norm precision matrix\n\n"
        f"Last run: {date.today().isoformat()} — {len(rows)} cells, gate PCC >= {PCC_THRESHOLD} "
        f"(all other metrics observational). Reference: fp32 RMSNorm of the dtype-rounded device inputs.\n\n"
        f"Skipped: `{{float32, fp32_dest_acc_en=False}}` (permanent op-side EXCLUSION: fp32 input with a 16-bit "
        f"DEST accumulation is lossy by construction).\n\n" + header + body
    )


@pytest.mark.parametrize("distribution", [pytest.param("rand", id="uniform"), pytest.param("randn", id="normal")])
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize("math_fidelity", list(FIDELITY_ID), ids=list(FIDELITY_ID.values()))
@pytest.mark.parametrize("dtype", list(DTYPE_ID), ids=list(DTYPE_ID.values()))
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_precision_matrix(device, shape, dtype, math_fidelity, fp32_acc, distribution, results_table):
    assert dtype in SUPPORTED["dtype"] and fp32_acc in SUPPORTED["fp32_dest_acc_en"]
    if _is_excluded({"dtype": dtype, "fp32_dest_acc_en": fp32_acc}):
        pytest.skip("permanent EXCLUSION: float32 input with 16-bit DEST accumulation is lossy by construction")

    torch.manual_seed(7)
    gen = torch.rand if distribution == "rand" else torch.randn
    x = gen(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    g = torch.randn(shape[-1], dtype=torch.float32).to(TORCH_DTYPE[dtype])

    ttnn_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_g = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_acc, math_approx_mode=False
    )
    actual = ttnn.to_torch(rms_norm(ttnn_x, gamma=ttnn_g, compute_kernel_config=config)).to(torch.float32)

    # Reference from what the device actually saw (bf8b block-quantizes the inputs on upload).
    expected = torch_rms_norm(ttnn.to_torch(ttnn_x), ttnn.to_torch(ttnn_g))

    m = precision_metrics(expected, actual)
    ulp = calculate_detailed_ulp_stats(expected, actual)
    allclose_ok, allclose_msg = comp_allclose(expected, actual, rtol=1e-2, atol=1e-2)
    row = {
        "shape": "x".join(map(str, shape)),
        "dtype": DTYPE_ID[dtype],
        "fidelity": FIDELITY_ID[math_fidelity],
        "fp32_acc": fp32_acc,
        "dist": distribution,
        **m,
        "max_ulp": ulp["max_ulp"],
        "mean_ulp": ulp["mean_ulp"],
        "p99_ulp": ulp["p99_ulp"],
        "allclose": bool(allclose_ok),
    }
    results_table.append(row)
    print(
        f"\nPRECISION shape={row['shape']} dtype={row['dtype']} fid={row['fidelity']} fp32_acc={fp32_acc} "
        f"dist={distribution} pcc={m['pcc']:.6f} max_abs={m['max_abs_err']:.4g} median_abs={m['median_abs_err']:.4g} "
        f"p99_abs={m['p99_abs_err']:.4g} rel_rms={m['rel_rms_err']:.4g} ulp(max/mean/p99)={ulp['max_ulp']:.3g}/"
        f"{ulp['mean_ulp']:.3g}/{ulp['p99_ulp']:.3g} | {allclose_msg}"
    )
    assert torch.isfinite(actual).all(), "non-finite output"
    assert m["pcc"] >= PCC_THRESHOLD, f"PCC {m['pcc']:.6f} < {PCC_THRESHOLD}"
