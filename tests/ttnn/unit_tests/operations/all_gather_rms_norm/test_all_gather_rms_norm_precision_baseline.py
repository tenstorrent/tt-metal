# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for all_gather_rms_norm (verifier). Records, per supported precision corner and shape:
PCC, max / mean abs error, relative RMS error, ULP error (p99 / max, in the output dtype), and the got/true ratio
spread (median, p5, p95) — the scale-bug detector: a tight cluster of got/true around a constant != 1 is a structural
bug (wrong scaler, stale partial), a broad spread centred on 1.0 is rounding.

Group: the 4 devices of a 1x4 mesh on cluster_axis 1, production transport (Ring where the box closes it, 2 links),
gamma = bf16 ROW_MAJOR sticks (MLA's format). Reference: torch rms_norm over the device-quantized gathered row in fp32.

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/all_gather_rms_norm/test_all_gather_rms_norm_precision_baseline.py -s
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

EPS = 1e-6
TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT
PCC = {ttnn.float32: 0.9999, ttnn.bfloat16: 0.999, ttnn.bfloat8_b: 0.99}
TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
MANTISSA_BITS = {ttnn.float32: 23, ttnn.bfloat16: 7, ttnn.bfloat8_b: 7}


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]


def _topology(mesh_shape):
    try:
        from eval.golden_tests.all_gather_rms_norm.topology import ring_mock_meshes

        if (tuple(mesh_shape), 1) in ring_mock_meshes():
            return ttnn.Topology.Ring
    except ImportError:
        pass
    return ttnn.Topology.Linear


def _to_mesh(stacked, mesh_device, *, dtype, layout):
    rows, cols = tuple(mesh_device.shape)
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def _quantize(stacked, dtype):
    if dtype != ttnn.bfloat8_b:
        return stacked
    flat = stacked.reshape(-1, stacked.shape[-1])
    return ttnn.to_torch(ttnn.from_torch(flat, dtype=dtype, layout=TILE)).reshape(stacked.shape).to(torch.bfloat16)


def _ulp_error(got, ref, dtype):
    """|got - ref| in units of the output dtype's ulp at |ref| (bfloat8_b: its 7-bit mantissa, per element)."""
    mag = ref.abs().clamp_min(torch.finfo(torch.float32).tiny)
    ulp = torch.pow(2.0, torch.floor(torch.log2(mag)) - MANTISSA_BITS[dtype])
    return (got - ref).abs() / ulp


# (shape, id): tiny single tile-row (k = 1), the PERF FOCUS case (k = 8 symmetric combine), the wide dense-FFN case
# (k = 12 owner combine), W_local non-aligned (regime N compose).
SHAPES = [
    pytest.param((1, 1, 32, 64), id="32x64"),
    pytest.param((1, 1, 640, 384), id="focus_640x384"),
    pytest.param((1, 1, 640, 1792), id="640x1792"),
    pytest.param((1, 1, 64, 1000), id="64x1000_wna"),
]
CORNERS = [
    pytest.param(ttnn.bfloat16, True, id="bf16-fp32dest"),
    pytest.param(ttnn.bfloat16, False, id="bf16-fp16dest"),
    pytest.param(ttnn.float32, True, id="fp32-fp32dest"),
    pytest.param(ttnn.bfloat8_b, True, id="bf8b-fp32dest"),
]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
@pytest.mark.parametrize("dtype,fp32_dest", CORNERS)
@pytest.mark.parametrize("shape", SHAPES)
def test_precision_baseline(mesh_device, shape, dtype, fp32_dest):
    if dtype == ttnn.bfloat8_b and shape[-1] % 32:
        pytest.skip("EXCLUSION: bfloat8_b x w_non_aligned")
    torch.manual_seed(1234)
    rows, cols = tuple(mesh_device.shape)
    G = cols
    W = G * shape[-1]
    x = _quantize(torch.randn((rows, cols, *shape)).to(TORCH[dtype]), dtype)
    gamma = torch.randn((W,)).to(torch.bfloat16)
    padded = -(-W // 32) * 32
    sticks = torch.nn.functional.pad(gamma, (0, padded - W)).reshape(1, 1, padded // 32, 32)
    tg = _to_mesh(
        sticks.reshape(1, 1, *sticks.shape).expand(rows, cols, *sticks.shape).contiguous(),
        mesh_device,
        dtype=ttnn.bfloat16,
        layout=RM,
    )
    tx = _to_mesh(x, mesh_device, dtype=dtype, layout=TILE)

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = False
    out = all_gather_rms_norm(
        tx,
        cluster_axis=1,
        gamma=tg,
        epsilon=EPS,
        topology=_topology(mesh_device.shape),
        num_links=2,
        compute_kernel_config=cfg,
    )

    worst = None
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        full = torch.cat([x[r, cc].float() for cc in range(cols)], dim=-1)
        ref = torch.nn.functional.rms_norm(full, (W,), weight=gamma.float(), eps=EPS)
        got = ttnn.to_torch(t).float()
        assert torch.isfinite(got).all(), f"device ({r},{c}): non-finite output"
        err = got - ref
        rel_rms = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        ulp = _ulp_error(got, ref, dtype)
        nz = ref.abs() > 1e-3
        ratio = (got[nz] / ref[nz]).flatten()
        stats = dict(
            dev=(r, c),
            max_abs=err.abs().max().item(),
            mean_abs=err.abs().mean().item(),
            rel_rms=rel_rms,
            ulp_p99=ulp.flatten().kthvalue(max(1, int(0.99 * ulp.numel()))).values.item(),
            ulp_max=ulp.max().item(),
            r_med=ratio.median().item(),
            r_p5=ratio.kthvalue(max(1, int(0.05 * ratio.numel()))).values.item(),
            r_p95=ratio.kthvalue(max(1, int(0.95 * ratio.numel()))).values.item(),
            allclose=comp_allclose(ref, got, rtol=1e-2, atol=1e-2)[1],
        )
        _, pcc_val = assert_with_pcc(ref, got, PCC[dtype])
        stats["pcc"] = float(pcc_val)
        if worst is None or stats["rel_rms"] > worst["rel_rms"]:
            worst = stats
    print(
        f"\nBASELINE {list(shape)} {dtype} fp32_dest={fp32_dest} worst_dev={worst['dev']} "
        f"pcc={worst['pcc']:.6f} max_abs={worst['max_abs']:.4g} mean_abs={worst['mean_abs']:.4g} "
        f"rel_rms={worst['rel_rms']:.4g} ulp_p99={worst['ulp_p99']:.3g} ulp_max={worst['ulp_max']:.3g} "
        f"ratio med={worst['r_med']:.5f} p5={worst['r_p5']:.5f} p95={worst['r_p95']:.5f} | {worst['allclose']}"
    )
