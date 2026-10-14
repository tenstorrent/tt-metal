# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for distributed_rms_norm (Phase 0 SUPPORTED cells).

Measures, per shape x dtype, over every device of the mesh (worst device reported):
PCC, max / mean abs error, relative RMS error, max / mean ULP error (in the output dtype, against the fp32
reference rounded to that dtype), and the got/true ratio spread (median, p5, p95) over finite, non-tiny
reference elements — the scale-bug detector: a tight cluster around a non-1.0 constant is a structural bug,
a spread centred on 1.0 is rounding noise.

Prints one `PRECISION_BASELINE` line per case (collected into verification_report.md).
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.distributed_rms_norm import distributed_rms_norm


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


SYSTEM_MESH = _system_mesh_shape()
FABRIC_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "fabric_router_config": _router_config(),
    "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
}

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
MANTISSA_BITS = {ttnn.float32: 23, ttnn.bfloat16: 7}
PCC_GATE = {ttnn.float32: 0.9999, ttnn.bfloat16: 0.999}  # golden TOLERANCES (fp32_dest_acc_en=True)
EPS = 1e-6
CLUSTER_AXIS = 1

SHAPES = [
    (1, 1, 32, 1792),  # decode: S_w > 1, one row group
    (1, 1, 640, 1792),  # PERF FOCUS
    (1, 1, 2048, 1280),  # box prefill, multi-row groups
    (32, 4096),  # wide: several combine tiles
]


def _reference(x, g, cluster_axis):
    x = x.to(torch.float32)
    g = None if g is None else g.to(torch.float32)
    rows, cols = x.shape[0], x.shape[1]
    w_local = x.shape[-1]
    out = torch.empty_like(x)
    for other in range(cols if cluster_axis == 0 else rows):
        members = [(r, other) for r in range(rows)] if cluster_axis == 0 else [(other, c) for c in range(cols)]
        full = torch.cat([x[r, c] for r, c in members], dim=-1)
        weight = None if g is None else torch.cat([g[r, c].reshape(-1) for r, c in members])
        normed = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=weight, eps=EPS)
        for p, (r, c) in enumerate(members):
            out[r, c] = normed[..., p * w_local : (p + 1) * w_local]
    return out


def _to_mesh(stacked, mesh_device, *, dtype, layout):
    rows = stacked.shape[0]
    global_tensor = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        global_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


def _ulp_error(actual, expected, dtype):
    """|actual - round(expected)| in units of the output dtype's ulp at |expected|."""
    exp_q = expected.to(TORCH_DTYPE[dtype]).to(torch.float64)
    _, e = torch.frexp(exp_q.abs().clamp_min(torch.finfo(TORCH_DTYPE[dtype]).tiny))
    ulp = torch.ldexp(torch.ones_like(exp_q), e - 1 - MANTISSA_BITS[dtype])
    return (actual.to(torch.float64) - exp_q).abs() / ulp


def _metrics(actual, expected, dtype):
    a, e = actual.to(torch.float64).flatten(), expected.to(torch.float64).flatten()
    ac, ec = a - a.mean(), e - e.mean()
    pcc = float((ac * ec).sum() / (ac.norm() * ec.norm()))
    diff = (a - e).abs()
    rel_rms = float(((a - e) ** 2).mean().sqrt() / (e**2).mean().sqrt())
    ulp = _ulp_error(a, e, dtype)
    mask = torch.isfinite(e) & (e.abs() > 1e-3 * e.abs().max())
    r = a[mask] / e[mask]
    q = torch.quantile(r[: 1 << 24].float(), torch.tensor([0.05, 0.5, 0.95]))
    return dict(
        pcc=pcc,
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        rel_rms=rel_rms,
        max_ulp=float(ulp.max()),
        mean_ulp=float(ulp.mean()),
        ratio_p5=float(q[0]),
        ratio_median=float(q[1]),
        ratio_p95=float(q[2]),
    )


@pytest.mark.parametrize("mesh_device", [SYSTEM_MESH], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC_PARAMS], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(mesh_device, shape, dtype):
    if tuple(mesh_device.shape)[CLUSTER_AXIS] < 2:
        pytest.skip("cluster axis has size 1")
    torch.manual_seed(1234)
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    g = torch.randn((rows, cols, 1, 1, 1, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    expected = _reference(x, g, CLUSTER_AXIS)

    tx = _to_mesh(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tg = _to_mesh(g, mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    out = distributed_rms_norm(tx, cluster_axis=CLUSTER_AXIS, gamma=tg, epsilon=EPS)

    worst = None
    for idx, dev in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        actual = ttnn.to_torch(dev).to(torch.float32)
        m = _metrics(actual, expected[r, c], dtype)
        if worst is None or m["pcc"] < worst[1]["pcc"]:
            worst = ((r, c), m, actual, expected[r, c])
    (r, c), m, actual, exp_rc = worst
    _, allclose_msg = comp_allclose(exp_rc, actual)
    print(
        f"\nPRECISION_BASELINE shape={shape} dtype={dtype} worst_device=({r},{c}) "
        + " ".join(f"{k}={v:.6g}" for k, v in m.items())
        + f" | comp_allclose: {allclose_msg}"
    )
    assert_with_pcc(exp_rc, actual, PCC_GATE[dtype])
