# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for matmul_reduce_scatter (verifier, Phase 0).

Measures, per device and worst over devices: PCC, max / mean abs error, relative RMS error, bf16 ULP error
(|actual - expected| in units of the bf16 spacing at |expected|), and the got/true ratio spread (the scale-bug
detector: a tight cluster of actual/expected around a non-1.0 constant = structural scale bug).

    scripts/run_safe_pytest.sh --dev --run-all \
        tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter_precision_baseline.py -s
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter


def _system_mesh_shape():
    shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    return tuple(shape[i] for i in range(shape.dims()))


@pytest.fixture(scope="module")
def mesh_device():
    from tests.scripts.common import get_updated_device_params

    shape = _system_mesh_shape()
    if len(shape) != 2 or shape[0] * shape[1] < 2:
        pytest.skip(f"needs a 2-D mesh with >= 2 devices, system mesh is {shape}")
    fabric = ttnn.FabricConfig.FABRIC_2D
    ttnn.set_fabric_config(fabric, ttnn.FabricReliabilityMode.STRICT_INIT)
    params = get_updated_device_params({"fabric_config": fabric})
    params.pop("fabric_config")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)
    yield mesh
    for submesh in mesh.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _stacked_randn(mesh_device, shape, seed, scale=1.0):
    torch.manual_seed(seed)
    rows, cols = tuple(mesh_device.shape)
    return (torch.randn((rows, cols, *shape), dtype=torch.float32) * scale).to(torch.bfloat16)


def _as_device_holds(stacked, dtype):
    if dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        t = ttnn.from_torch(stacked.reshape(-1, stacked.shape[-1]).float(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
        return ttnn.to_torch(t).reshape(stacked.shape).to(torch.bfloat16)
    return stacked


def _to_mesh(stacked, mesh_device, dtype):
    rows, cols = stacked.shape[0], stacked.shape[1]
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )


def _reference(a_stacked, w_stacked, cluster_axis, scatter_dim):
    a = a_stacked.float()
    w = w_stacked.float().reshape(*w_stacked.shape[:2], *([1] * (a.dim() - 4)), *w_stacked.shape[2:])
    total = torch.matmul(a, w).sum(dim=cluster_axis, keepdim=True).expand(*a.shape[:-1], w.shape[-1])
    rows, cols = a.shape[0], a.shape[1]
    g = (rows, cols)[cluster_axis]
    return torch.stack(
        [
            torch.stack([torch.chunk(total[r, c], g, dim=scatter_dim)[(r, c)[cluster_axis]] for c in range(cols)])
            for r in range(rows)
        ]
    )


def _bf16_ulp(x):
    """Spacing of bf16 at |x| (8 mantissa bits incl. implicit): 2^(floor(log2|x|) - 7)."""
    mag = x.abs().clamp_min(2.0**-126)
    return torch.pow(2.0, torch.floor(torch.log2(mag)) - 7)


def _metrics(actual, expected):
    a, e = actual.double().flatten(), expected.double().flatten()
    err = (a - e).abs()
    rel_rms = float(((a - e) ** 2).mean().sqrt() / (e**2).mean().sqrt().clamp_min(1e-30))
    ulp = err / _bf16_ulp(e)
    nz = e.abs() > 1e-3 * e.abs().max()
    r = a[nz] / e[nz]
    return {
        "max_abs": float(err.max()),
        "mean_abs": float(err.mean()),
        "rel_rms": rel_rms,
        "ulp_mean": float(ulp.mean()),
        "ulp_p99": float(torch.quantile(ulp.float(), 0.99)),
        "ratio_median": float(r.median()),
        "ratio_p5": float(torch.quantile(r.float(), 0.05)),
        "ratio_p95": float(torch.quantile(r.float(), 0.95)),
    }


HIFI2_BF16_ACC = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
HIFI2_FP32_ACC = dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
HIFI4_FP32_ACC = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)

# (A, W, cluster_axis, scatter_dim, weight dtype, compute config, PCC floor[, activation dtype])
CASES = [
    pytest.param((1, 1, 256, 512), (512, 1024), 1, -2, ttnn.bfloat16, HIFI4_FP32_ACC, 0.999, id="small_bf16_hifi4"),
    pytest.param((1, 1, 640, 512), (512, 7168), 1, -1, ttnn.bfloat8_b, HIFI2_FP32_ACC, 0.998, id="shared_down_bf8b"),
    pytest.param((1, 1, 640, 2048), (2048, 7168), 1, -1, ttnn.bfloat8_b, HIFI2_BF16_ACC, 0.998, id="focus_bf16acc"),
    pytest.param((1, 1, 640, 2048), (2048, 7168), 1, -1, ttnn.bfloat8_b, HIFI2_FP32_ACC, 0.998, id="focus_fp32acc"),
    pytest.param((1, 1, 2048, 2048), (2048, 4096), 1, -2, ttnn.bfloat8_b, HIFI2_FP32_ACC, 0.998, id="mimo_rows"),
    pytest.param((1, 1, 640, 8448), (8448, 7168), 0, -1, ttnn.bfloat16, HIFI2_FP32_ACC, 0.999, id="large_k_bf16"),
    # Refinement 2: bfloat8_b activations and/or bfloat4_b weights (golden floor for bfloat4_b: PCC 0.99)
    pytest.param(
        (1, 1, 640, 2048), (2048, 7168), 1, -1, ttnn.bfloat8_b, HIFI2_FP32_ACC, 0.998, ttnn.bfloat8_b, id="focus_a_bf8b"
    ),
    pytest.param((1, 1, 640, 2048), (2048, 7168), 1, -1, ttnn.bfloat4_b, HIFI2_FP32_ACC, 0.99, id="focus_w_bf4b"),
    pytest.param(
        (1, 1, 640, 2048),
        (2048, 7168),
        1,
        -1,
        ttnn.bfloat4_b,
        HIFI2_BF16_ACC,
        0.99,
        ttnn.bfloat8_b,
        id="focus_bf8b_bf4b_bf16acc",
    ),
    pytest.param(
        (1, 1, 2048, 2048),
        (2048, 4096),
        1,
        -2,
        ttnn.bfloat4_b,
        HIFI2_FP32_ACC,
        0.99,
        ttnn.bfloat8_b,
        id="mimo_rows_bf8b_bf4b",
    ),
    pytest.param(
        (1, 1, 640, 8448),
        (8448, 7168),
        0,
        -1,
        ttnn.bfloat4_b,
        HIFI2_FP32_ACC,
        0.99,
        ttnn.bfloat8_b,
        id="large_k_bf8b_bf4b",
    ),
]


CASES = [c if len(c.values) == 8 else pytest.param(*c.values, ttnn.bfloat16, id=c.id) for c in CASES]


@pytest.mark.parametrize("a_shape,w_shape,cluster_axis,scatter_dim,weight_dtype,cfg,pcc_floor,a_dtype", CASES)
def test_precision_baseline(
    mesh_device, a_shape, w_shape, cluster_axis, scatter_dim, weight_dtype, cfg, pcc_floor, a_dtype
):
    g = tuple(mesh_device.shape)[cluster_axis]
    extent = a_shape[-2] if scatter_dim == -2 else w_shape[-1]
    if g < 2 or extent % (32 * g):
        pytest.skip(f"G={g} cannot split extent {extent}")
    a = _as_device_holds(_stacked_randn(mesh_device, a_shape, 0), a_dtype)
    w = _as_device_holds(_stacked_randn(mesh_device, w_shape, 1, scale=w_shape[0] ** -0.5), weight_dtype)
    expected = _reference(a, w, cluster_axis, scatter_dim)
    out = matmul_reduce_scatter(
        _to_mesh(a, mesh_device, a_dtype),
        _to_mesh(w, mesh_device, weight_dtype),
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(**cfg),
    )
    cols = tuple(mesh_device.shape)[1]
    worst = None
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        actual = ttnn.to_torch(t).float()
        ref = expected[r, c]
        _, pcc = assert_with_pcc(ref, actual, pcc_floor)
        _, allclose_msg = comp_allclose(ref, actual)
        m = _metrics(actual, ref)
        m["pcc"] = float(pcc)
        if worst is None or m["rel_rms"] > worst[1]["rel_rms"]:
            worst = ((r, c), m, allclose_msg)
    (r, c), m, allclose_msg = worst
    logger.info(
        f"PRECISION {a_shape}x{w_shape} axis={cluster_axis} dim={scatter_dim} a={a_dtype} w={weight_dtype} cfg={cfg} "
        f"worst_dev=({r},{c}) pcc={m['pcc']:.6f} max_abs={m['max_abs']:.5f} mean_abs={m['mean_abs']:.6f} "
        f"rel_rms={m['rel_rms']:.5f} ulp_mean={m['ulp_mean']:.3f} ulp_p99={m['ulp_p99']:.2f} "
        f"ratio_median={m['ratio_median']:.5f} ratio_p5={m['ratio_p5']:.4f} ratio_p95={m['ratio_p95']:.4f} "
        f"| {allclose_msg}"
    )
    # scale-bug detector: the median got/true ratio of a correct op sits at 1.0. With fp32 DEST accumulation it is
    # within ~0.3% (the HiFi2 operand truncation pulls it slightly below 1). With bf16 DEST accumulation
    # (fp32_dest_acc_en=False, the FOCUS production floor) the matmul's bf16 accumulation carries a K-dependent
    # rounding bias (verifier triage, Phase 0: slope 1.007 at K=512, 1.016 at K=2048; fp32 DEST on the same data and
    # transport: 0.999), so the bound is looser there — a structural bug (dropped / doubled K-block or partial) moves
    # it by >= 1/num_k_blocks or 1/G, far outside either bound.
    ratio_tol = 0.01 if cfg["fp32_dest_acc_en"] else 0.03
    assert abs(m["ratio_median"] - 1.0) < ratio_tol, f"median got/true ratio {m['ratio_median']} (scale bug?)"
