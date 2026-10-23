# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for distributed_rms_norm (HiFi4; every supported dtype x fp32_dest_acc_en corner).

Measures, per shape x dtype (gamma bf16, TILE, Linear on the system mesh), against the fp32 torch reference
(torch.nn.functional.rms_norm over the concatenated group row):
  PCC, max / mean abs error, relative RMS error, ULP distance (in the output dtype) and the got/true ratio spread
  (median and p5..p95 over finite, |ref| > 1e-3 elements — the scale-bug detector: a tight cluster around a non-1.0
  constant is a structural bug, a spread centred on 1.0 is rounding noise).
Results are printed and appended to $DRMS_PRECISION_JSON (if set) for the verification report.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _mesh_shape_of
from eval.golden_tests.distributed_rms_norm.helpers import (
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
)
from eval.golden_tests.distributed_rms_norm.topology import topology_feasible
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc, ulp_distance
from ttnn.operations.distributed_rms_norm import distributed_rms_norm


@pytest.fixture
def mesh_device(request, _fabric_mesh_cache):
    """The golden fabric mesh, shared with any other module of this session that already opened it.

    Every module that imports the golden `_fabric_mesh_cache` gets its own session FixtureDef, hence its own cache;
    a second cache would try to re-set the fabric config while the first module's mesh is still open (TT_FATAL).
    Reuse whichever cache in this session already holds an open mesh (e.g. the acceptance module's).
    """
    for fdef in request._fixturemanager._arg2fixturedefs.get("_fabric_mesh_cache", []):
        cached = getattr(fdef, "cached_result", None)
        if cached is not None and cached[0] is not None and getattr(cached[0], "mesh", None) is not None:
            return cached[0].get(_mesh_shape_of(request))
    return _fabric_mesh_cache.get(_mesh_shape_of(request))


LINEAR_AXES = [a for a in (0, 1) if topology_feasible(ttnn.Topology.Linear, a) is None]

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
PCC_FLOOR = {ttnn.float32: 0.9999, ttnn.bfloat16: 0.999, ttnn.bfloat8_b: 0.99}

# (dtype, fp32_dest_acc_en) corners; {float32, False} is EXCLUDED (precision convention)
CORNERS = [
    pytest.param(ttnn.bfloat16, True, id="bf16-fp32acc"),
    pytest.param(ttnn.float32, True, id="fp32-fp32acc"),
    pytest.param(ttnn.bfloat8_b, True, id="bfp8-fp32acc"),
    pytest.param(ttnn.bfloat16, False, id="bf16-bf16acc"),
    pytest.param(ttnn.bfloat8_b, False, id="bfp8-bf16acc"),
]

SHAPES = [
    (1, 1, 32, 32),  # single tile (R5 point: one chunk)
    (1, 1, 32, 1792),  # decode, one row-group x 14 chunks
    (1, 1, 640, 1792),  # PERF FOCUS: 20 row-groups x 5 chunks
    (1, 1, 2048, 1280),  # box prefill chunk
]


@pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")
@pytest.mark.parametrize("dtype, fp32_acc", CORNERS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(shape, dtype, fp32_acc, mesh_device):
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    cluster_axis = LINEAR_AXES[-1]
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    gamma = torch.randn((rows, cols, 1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(gamma, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_acc
    cfg.math_approx_mode = False
    out = distributed_rms_norm(tx, cluster_axis=cluster_axis, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg)

    expected = pytorch_distributed_rms_norm(x.to(torch.float32), gamma, cluster_axis, 1e-6)  # fp32 reference
    actual = torch.stack(
        [ttnn.to_torch(t).to(torch.float32).reshape(shape) for t in ttnn.get_device_tensors(out)]
    ).reshape(expected.shape)

    _, pcc_msg = assert_with_pcc(expected, actual, PCC_FLOOR[dtype])
    _, allclose_msg = comp_allclose(expected, actual)

    err = (actual - expected).abs()
    rel_rms = float(torch.sqrt(((actual - expected) ** 2).mean()) / torch.sqrt((expected**2).mean()))
    # ULP in the output dtype: round the fp32 reference to it, compare bit patterns
    td = TORCH_DTYPE[dtype]
    ulp = ulp_distance(actual.to(td), expected.to(td)).to(torch.float64)
    mask = torch.isfinite(expected) & (expected.abs() > 1e-3)
    ratio = (actual[mask] / expected[mask]).to(torch.float64)
    q = torch.quantile(ratio[: 1 << 24], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)).tolist()

    rec = dict(
        shape="x".join(map(str, shape)),
        dtype=str(dtype).split(".")[-1],
        fp32_dest_acc_en=fp32_acc,
        mesh=f"{rows}x{cols}",
        cluster_axis=cluster_axis,
        pcc=pcc_msg,
        max_abs=float(err.max()),
        mean_abs=float(err.mean()),
        rel_rms=rel_rms,
        ulp_max=float(ulp.max()),
        ulp_mean=float(ulp.mean()),
        ulp_p99=float(torch.quantile(ulp.flatten()[: 1 << 24], 0.99)),
        ratio_p5=q[0],
        ratio_median=q[1],
        ratio_p95=q[2],
        allclose=allclose_msg,
    )
    print("PRECISION_BASELINE", json.dumps(rec))
    path = os.environ.get("DRMS_PRECISION_JSON")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    # scale-bug guard: the ratio must centre on 1.0
    assert abs(q[1] - 1.0) < 1e-2, f"got/true ratio median {q[1]} is not 1.0 (structural scale error)"
