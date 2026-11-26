# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for a2a_wkv_b2.

The op has exactly one legal shape (per chip attn [1, 96, 160, 512], W [1, 96, 512, 128] on a
4-chip ring), so instead of shapes the baseline sweeps the input distribution: unit normal at
three seeds, small-magnitude (x0.01) and large-magnitude (x8) attn. All 8 chips' outputs are
gated together (PCC + relative RMS, as eval.metrics.check_output does) and the got/true ratio
spread is reported as the scale-bug detector.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.a2a_wkv_b2 import a2a_wkv_b2

from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import (
    MESH_SHAPE,
    _expected,
    _make_inputs,
)

PCC_GATE = 0.9995  # feature_spec LOOSE_CASES pcc_threshold
REL_RMS_GATE = 0.05

_mesh = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
_fabric = pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}], indirect=True
)


@_fabric
@_mesh
@pytest.mark.parametrize(
    "seed, scale",
    [(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.01), (3, 8.0)],
    ids=["seed0", "seed1", "seed2", "small_x0.01", "large_x8"],
)
def test_a2a_wkv_b2_precision_baseline(mesh_device, seed, scale):
    chips, attn, w = _make_inputs(mesh_device, seed, scale=scale)
    out = a2a_wkv_b2(attn, w)
    expected = torch.stack(_expected(mesh_device, chips, w)).to(torch.float32)
    actual = torch.stack([ttnn.to_torch(t).to(torch.float32) for t in ttnn.get_device_tensors(out)])

    _, pcc = comp_pcc(expected, actual)
    _, allclose_msg = comp_allclose(expected, actual)
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_rms = (torch.sqrt(((actual - expected) ** 2).mean()) / expected.std()).item()

    nz = (expected.abs() > 1e-3 * expected.abs().max()) & torch.isfinite(actual)
    ratio = actual[nz] / expected[nz]
    q = torch.quantile(ratio[: 1 << 24].double(), torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    print(
        f"\nPRECISION seed={seed} scale={scale}: pcc={pcc:.6f} max_abs={max_abs:.5g} mean_abs={mean_abs:.5g} "
        f"rel_rms={rel_rms:.5f} ratio_median={q[1].item():.5f} ratio_p5={q[0].item():.5f} "
        f"ratio_p95={q[2].item():.5f} | {allclose_msg}"
    )
    assert torch.isfinite(actual).all()
    assert pcc >= PCC_GATE, f"pcc {pcc} < {PCC_GATE}"
    assert rel_rms <= REL_RMS_GATE, f"rel_rms {rel_rms} > {REL_RMS_GATE}"
    # Scale-bug detector. Measured: the median got/true ratio sits at ~1.013 with a broad p5..p95 spread
    # (~0.95..1.09) — the 16-bit DEST accumulate / bf8_b pack rounding bias on Blackhole (it grows with
    # the number of fidelity phases: LoFi 0.998, HiFi2 1.013, HiFi4 1.015; stock ttnn.matmul with the same
    # dtypes/config shows 1.06). A structural scale bug would show a TIGHT cluster at a non-1 constant.
    assert abs(q[1].item() - 1.0) < 0.03, f"median got/true ratio {q[1].item()} far from 1 (scale bug?)"
    assert (q[2] - q[0]).item() > 0.02, "got/true ratio tightly clustered: looks like a uniform scale bug"
