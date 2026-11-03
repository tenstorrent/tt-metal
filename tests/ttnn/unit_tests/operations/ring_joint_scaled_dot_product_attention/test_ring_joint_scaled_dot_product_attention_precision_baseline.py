# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for ring_joint_scaled_dot_product_attention (Phase 0 SUPPORTED: bf16, fp32 DEST, HiFi2 default).

Measures, per shape, over every device's valid rows (worst device reported): PCC, max / mean abs error, relative
RMS error, bf16 ULP error (p50 / p99 / max, in units of the bf16 spacing at the reference value), and the got/true
ratio spread (median, p5, p95 over elements whose |reference| is not tiny) — the scale-bug detector: a tight cluster
around a non-1.0 constant is a structural scale bug, a broad spread around 1.0 is ordinary rounding noise.

Run with -s to see the table. Assertions are the golden gates for this config (PCC >= 0.999, rel RMS <= 0.02).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.ring_joint_scaled_dot_product_attention.conftest import (  # noqa: F401 — fixtures
    _fabric_mesh_cache,
    mesh_device,
)
from eval.golden_tests.ring_joint_scaled_dot_product_attention.helpers import (
    SEQ_PAD,
    build_inputs,
    call_op,
    pytorch_ring_joint_scaled_dot_product_attention,
    valid_joint_rows,
    valid_query_rows,
)
from eval.golden_tests.ring_joint_scaled_dot_product_attention.topology import ring_mock_meshes, system_mesh_shape
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC_TARGET, RMS_TARGET = 0.999, 0.02

# (id, per-device Q/K/V(/joint Q logical) shapes, padded)
SHAPES = [
    ("small_1x1x64x64", ((1, 1, 64, 64),) * 3, False),
    ("medium_1x4x256x128_padded", ((1, 4, 256, 128),) * 3, True),
    ("wan_like_1x10x512x128_padded", ((1, 10, 512, 128),) * 3, True),
    ("flux_joint_1x6x512x128_L512_padded", ((1, 6, 512, 128),) * 4, True),
]


def _placements():
    out = [("linear_axis0_G2", system_mesh_shape(), ttnn.FabricConfig.FABRIC_2D, 0, ttnn.Topology.Linear)]
    for mesh_shape, axis, configs in ring_mock_meshes():
        if axis == 1 and ttnn.FabricConfig.FABRIC_2D_TORUS_XY in configs and mesh_shape[1] >= 3:
            out.append(("ring_1xN", mesh_shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY, 1, ttnn.Topology.Ring))
            break
    return out


def _bf16_ulp(x):
    """Spacing of bf16 at |x| (2^(e-7)), floored at the smallest normal spacing."""
    ax = x.abs().clamp_min(2.0**-126)
    return torch.pow(2.0, torch.floor(torch.log2(ax)) - 7)


def _metrics(actual, expected):
    a, e = actual.double().flatten(), expected.double().flatten()
    diff = (a - e).abs()
    rel_rms = float(torch.sqrt((diff**2).mean()) / torch.sqrt((e**2).mean()).clamp_min(1e-30))
    ulp = diff / _bf16_ulp(e)
    big = e.abs() > 0.05 * float(e.abs().max())
    r = a[big] / e[big]
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=r.dtype)) if r.numel() else torch.zeros(3)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_rms": rel_rms,
        "ulp_p50": float(ulp.median()),
        "ulp_p99": float(torch.quantile(ulp.float(), 0.99)),
        "ulp_max": float(ulp.max()),
        "ratio_p5": float(q[0]),
        "ratio_med": float(q[1]),
        "ratio_p95": float(q[2]),
    }


@pytest.mark.parametrize("shape_id,inputs,padded", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize(
    "placement_id,mesh_shape,fabric_config,cluster_axis,topology", _placements(), ids=[p[0] for p in _placements()]
)
def test_precision_baseline(
    placement_id, mesh_shape, fabric_config, cluster_axis, topology, shape_id, inputs, padded, mesh_device
):
    G = mesh_shape[cluster_axis]
    n = inputs[0][-2]
    logical_n = G * n - (SEQ_PAD if padded else 0)
    joint_mode = "replicated" if len(inputs) == 4 else "none"
    q, k, v, jq, jk, jv, _ = build_inputs(
        inputs,
        mesh_shape,
        cluster_axis=cluster_axis,
        joint_mode=joint_mode,
        dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat16,
        seed=1234,
    )
    out, jout = call_op(
        mesh_device,
        q,
        k,
        v,
        jq,
        jk,
        jv,
        dtype=ttnn.bfloat16,
        kv_dtype=ttnn.bfloat16,
        logical_n=logical_n,
        cluster_axis=cluster_axis,
        topology=topology,
    )
    exp_o, exp_j = pytorch_ring_joint_scaled_dot_product_attention(
        q, k, v, jq, jk, jv, cluster_axis=cluster_axis, logical_n=logical_n
    )
    rows, cols = mesh_shape
    checks = [("out", out, exp_o)] + ([("joint_out", jout, exp_j)] if jout is not None else [])
    for name, t, exp in checks:
        devs = ttnn.get_device_tensors(t)
        worst = None
        for idx, dt in enumerate(devs):
            r, c = divmod(idx, cols)
            p = c if cluster_axis == 1 else r
            if name == "out":
                mask = valid_query_rows(p, n, n, G, logical_n=logical_n, is_balanced=False, cross=False)
            else:
                mask = valid_joint_rows(p, exp.shape[-2], logical_l=0, sharded=False)
            got = ttnn.to_torch(dt).float()[:, :, mask]
            want = exp[r, c][:, :, mask].float()
            assert_with_pcc(want, got, PCC_TARGET)
            _, pcc_msg = comp_allclose(want, got)
            m = _metrics(got, want)
            pcc = float(torch.corrcoef(torch.stack([got.flatten().double(), want.flatten().double()]))[0, 1])
            if worst is None or pcc < worst[0]:
                worst = (pcc, (r, c), m, pcc_msg)
        pcc, dev, m, pcc_msg = worst
        print(
            f"\nPRECISION {placement_id} {shape_id} {name} worst_dev={dev} pcc={pcc:.6f} "
            f"max_abs={m['max_abs']:.4g} mean_abs={m['mean_abs']:.4g} rel_rms={m['rel_rms']:.4g} "
            f"ulp_p50={m['ulp_p50']:.2f} ulp_p99={m['ulp_p99']:.2f} ulp_max={m['ulp_max']:.1f} "
            f"ratio[p5/med/p95]={m['ratio_p5']:.4f}/{m['ratio_med']:.4f}/{m['ratio_p95']:.4f} | {pcc_msg}"
        )
        assert m["rel_rms"] <= RMS_TARGET, f"{name}: rel RMS {m['rel_rms']:.4f} > {RMS_TARGET}"
