# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for distributed_rms_norm (verifier, Phase 0).

Measures, per shape / dtype, on the WORST device of the group: PCC, max / mean abs error, relative RMS error, ULP
distance (in the output dtype) and the got/true ratio spread (the scale-bug detector: a tight cluster of
actual / expected around a constant != 1.0 means a structural scale error, a broad spread centred on 1.0 is
ordinary rounding noise). Each case prints one `PRECISION_BASELINE` line; the table in verification_report.md is
built from them.

Runs on the system 2x2 mesh, group along cluster_axis 1 (G = 2, Linear), FABRIC_2D_TORUS_XY, bf16 gamma,
HiFi4 + fp32_dest_acc_en (the Phase 0 corner). The shapes pin both local regimes: R3-X (decode 32x1792 and the
perf-focus 640x1792) and R1-X (2048x1280, the batched rank-4 shape).

    scripts/run_safe_pytest.sh --dev tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_precision_baseline.py
"""

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import (
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    random_stacked,
)
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc, ulp_distance
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

CLUSTER_AXIS = 1
EPS = 1e-6
_TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
# Gate = the golden suite's PCC tolerance (helpers.TOLERANCES); the baseline measures, it does not tighten.
_PCC_GATE = {ttnn.bfloat16: 0.999, ttnn.float32: 0.9999}


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]

SHAPES = [
    (1, 1, 32, 1792),  # decode, R3-X W-split (one tile-row)
    (1, 1, 640, 1792),  # perf focus, R3-X row-groups
    (1, 1, 2048, 1280),  # prefill, R1-X
    (2, 4, 128, 512),  # batched rank 4, R1-X
]


def _metrics(actual, expected, dtype):
    a = actual.to(torch.float64).flatten()
    e = expected.to(torch.float64).flatten()
    diff = (a - e).abs()
    rel_rms = (((a - e) ** 2).mean().sqrt() / (e**2).mean().sqrt().clamp_min(1e-30)).item()
    ac, ec = a - a.mean(), e - e.mean()
    pcc = ((ac * ec).sum() / (ac.norm() * ec.norm()).clamp_min(1e-30)).item()
    tdt = _TORCH[dtype]
    ulp = ulp_distance(actual.to(tdt).flatten(), expected.to(tdt).flatten()).to(torch.float64)
    # got/true ratio over elements whose reference is clearly non-zero (a near-zero reference makes the ratio noise)
    mask = torch.isfinite(a) & (e.abs() > 0.05 * e.abs().max())
    r = a[mask] / e[mask]
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return dict(
        pcc=pcc,
        max_abs=diff.max().item(),
        mean_abs=diff.mean().item(),
        rel_rms=rel_rms,
        ulp_max=ulp.max().item(),
        ulp_mean=ulp.mean().item(),
        ratio_p5=q[0].item(),
        ratio_med=q[1].item(),
        ratio_p95=q[2].item(),
    )


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_precision_baseline(mesh_device, dtype):
    # Shapes loop inside one test: one fabric mesh open per dtype (re-opening a fabric mesh in-process under the
    # --dev watcher is an infrastructure flake, see verification_report.md).
    failures = []
    for shape in SHAPES:
        x = random_stacked(mesh_device, shape, _TORCH[dtype], seed=7)
        g = random_stacked(mesh_device, (1, 1, 1, shape[-1]), torch.bfloat16, seed=8)
        expected = pytorch_distributed_rms_norm(x, g, CLUSTER_AXIS, EPS)
        tx = create_ttnn_input_tensor(x, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        tg = create_ttnn_gamma(g, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = distributed_rms_norm(tx, cluster_axis=CLUSTER_AXIS, gamma=tg, epsilon=EPS)

        cols = tuple(mesh_device.shape)[1]
        worst = None
        for idx, dev in enumerate(ttnn.get_device_tensors(out)):
            r, c = divmod(idx, cols)
            actual = ttnn.to_torch(dev)
            ref = expected[r, c]
            m = _metrics(actual, ref, dtype)
            _, allclose_msg = comp_allclose(ref.float(), actual.float())
            if worst is None or m["pcc"] < worst[0]["pcc"]:
                worst = (m, allclose_msg, actual, ref)
        m, allclose_msg, actual, ref = worst
        print(
            f"\nPRECISION_BASELINE shape={'x'.join(map(str, shape))} dtype={dtype} pcc={m['pcc']:.6f} "
            f"max_abs={m['max_abs']:.4g} mean_abs={m['mean_abs']:.4g} rel_rms={m['rel_rms']:.4g} "
            f"ulp_max={m['ulp_max']:.0f} ulp_mean={m['ulp_mean']:.3f} "
            f"ratio[p5,med,p95]=[{m['ratio_p5']:.5f},{m['ratio_med']:.5f},{m['ratio_p95']:.5f}] | {allclose_msg}"
        )
        try:
            assert_with_pcc(ref.float(), actual.float(), pcc=_PCC_GATE[dtype])
            # scale-bug guard: the median got/true ratio must sit on 1.0 (a structural scale error moves it as a whole)
            assert abs(m["ratio_med"] - 1.0) < 1e-2, f"median got/true ratio {m['ratio_med']} is off 1.0: scale bug"
        except AssertionError as e:
            failures.append(f"{shape}: {e}")
    assert not failures, "\n".join(failures)
