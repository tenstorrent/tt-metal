# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Fused V4.1 mHC split projection (tt/v41/mhc.py ``fused_project``, bead 8y7.9.5) vs fp64 torch and the composite
``tt_mhc._project`` (matmul + x*x + sum, two TP all-reduces).

Both paths feed the matmul unit fp32 operands (tf32 multiplies, fp32 accumulate), so the fused op is held to the
composite path's own error against fp64 rather than to bit identity: the sum of squares moves from an SFPU
square + FPU reduce to the diagonal of the FPU Gram tile x @ x^T. ``test_mhc_split_traced_time`` logs the traced
per-call split time on the 2x4 mesh at the production per-chip shape (chunk 5120: 2560 tokens, hidden slice 1280)
as ``V41_MHC_PERF`` JSON lines.
"""

import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap, _compute_kernel_config, _project
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.mhc import _V41Site, fused_project, fused_rms_project, mhc_config

N = 4
MIX = (2 + N) * N
EPS = 1e-6
DRAM_GBPS = 512.0  # Blackhole p150 DRAM peak (G2 bound)
REPLAYS = 20
REF = SimpleNamespace(norm_eps=C.RMS_NORM_EPS, hc_mult=N, hc_sinkhorn_iters=C.HC_SINKHORN_ITERS, hc_eps=C.HC_EPS)
SHAPES = {"small": (128, 256), "odd": (64, 96), "production": (2560, 1280)}  # (tokens, hidden/tp)


def _inputs(tokens, hidden, seed=0):
    gen = torch.Generator().manual_seed(seed)
    k = N * hidden
    return torch.randn(1, 1, tokens, k, generator=gen), torch.randn(MIX, k, generator=gen) * k**-0.5


def _upload(device, t):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _max_err(a, ref):
    return (a.double() - ref).abs().max().item()


@pytest.mark.parametrize("shape", list(SHAPES))
def test_mhc_fused_project(device, shape):
    tokens, hidden = SHAPES[shape]
    x, fn = _inputs(tokens, hidden)
    k = x.shape[-1]
    fn_pad = torch.cat([fn, torch.zeros(32 - MIX, k)])
    tx, tw, tw24 = (
        _upload(device, x),
        _upload(device, fn_pad.t().reshape(1, 1, k, 32)),
        _upload(device, fn.t()[None, None]),
    )

    xd, fd = x.double()[0, 0], fn.double()
    ref_mixes = xd @ fd.t()
    ref_ms = xd.square().mean(-1, keepdim=True) + EPS
    ref_proj = ref_mixes * ref_ms.rsqrt()

    t0 = time.perf_counter()
    raw = [ttnn.to_torch(fused_project(tx, tw, MIX, k, EPS))[0, 0] for _ in range(2)]
    proj = [ttnn.to_torch(fused_rms_project(tx, tw, MIX, EPS))[0, 0] for _ in range(2)]
    logger.info(f"fused project {shape}: 4 calls {time.perf_counter() - t0:.2f}s (incl. compile)")
    assert torch.equal(raw[0], raw[1]) and torch.equal(proj[0], proj[1]), "fused mHC projection is not deterministic"

    ckc = _compute_kernel_config()
    comp_mixes = ttnn.to_torch(ttnn.matmul(tx, tw24, compute_kernel_config=ckc))[0, 0]
    comp_ss = ttnn.to_torch(ttnn.sum(ttnn.multiply(tx, tx), dim=-1, keepdim=True))[0, 0, :, :1]
    comp_proj = ttnn.to_torch(_project(tx, tw24, EPS, ckc))[0, 0]

    errs = {
        "mixes": (_max_err(raw[0][:, :MIX], ref_mixes), _max_err(comp_mixes, ref_mixes)),
        "mean_sq": (_max_err(raw[0][:, MIX : MIX + 1], ref_ms), _max_err(comp_ss / k + EPS, ref_ms)),
        "projection": (_max_err(proj[0][:, :MIX], ref_proj), _max_err(comp_proj, ref_proj)),
    }
    logger.info(f"{shape}: max |err| vs fp64 (fused, composite): {errs}")
    assert torch.all(raw[0][:, MIX + 1 :] == 0), "mixes columns past the mean of squares must stay zero"
    for key, (fused_err, comp_err) in errs.items():
        assert fused_err <= 2 * comp_err + 1e-6, (key, errs)


def _traced_us(mesh_device, fn):
    fn()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / REPLAYS * 1e6
    ttnn.release_trace(mesh_device, tid)
    for t in out:
        t.deallocate()
    return us


MESH = pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(trace_region_size=64 << 20),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)


def _site_weights(seed=1):
    gen = torch.Generator().manual_seed(seed)
    width = N * C.EMB_SIZE
    return (
        torch.randn(MIX, width, generator=gen) * width**-0.5,
        0.5 * torch.randn(MIX, generator=gen),
        0.5 * torch.randn(3, generator=gen),
    )


def _sites(mesh_device, hc):
    cfg, topology = mhc_config(C), per_axis_topology()[1]
    composite = TtMHCWrap(mesh_device, cfg, *hc, tp_axis=1, topology=topology)
    fused = _V41Site(mesh_device, cfg, *hc, tp_axis=1, topology=topology)
    return composite, fused


def _mesh_streams(mesh_device, tokens, hidden, seed=0):
    """-> (host [1, 1, sp * tokens, tp * N * hidden], device: chip (i, j) holds token block i, hidden slice j)."""
    sp, tp = tuple(mesh_device.shape)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, sp * tokens, tp * N * hidden, generator=gen)
    return x, ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=(2, 3)),
    )


@MESH
def test_mhc_split_matches_composite_mesh(mesh_device, device_params):
    """Production per-chip shape on the mesh: fused and composite split vs the fp32 CPU reference ``hc_mixes``
    (both round the Sinkhorn op's matmul operands to tf32, so they agree to a few tf32 ulps, not bitwise)."""
    tokens, hidden = SHAPES["production"]
    sp, tp = tuple(mesh_device.shape)
    hc = _site_weights()
    x_host, x = _mesh_streams(mesh_device, tokens, hidden)
    t0 = time.perf_counter()
    streams = x_host[0, 0].reshape(sp * tokens, tp, N, hidden).permute(0, 2, 1, 3).reshape(sp * tokens, N * tp * hidden)
    ref = v41.Block.hc_mixes(REF, streams.reshape(1, sp * tokens, N, tp * hidden), hc[0], hc[2], hc[1])
    ref = [r[0].flatten(1) for r in ref]  # [S, N], [S, N], [S, N * N]
    logger.info(f"CPU reference hc_mixes {time.perf_counter() - t0:.1f}s")
    composite, fused = _sites(mesh_device, hc)
    # split outputs are replicated across TP: keep chip column 0
    down = lambda t: ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, tuple(mesh_device.shape), dims=(2, 3))
    )[0, 0]
    a = [down(t) for t in composite.split(x)]
    b = [down(t) for t in fused.split(x)]
    b2 = [down(t) for t in fused.split(x)]
    for name, r, u, v, v2 in zip(("pre", "post", "comb"), ref, a, b, b2):
        k = r.shape[-1]
        assert torch.equal(v, v2), f"fused split {name} is not deterministic"
        comp_err, fused_err = _max_err(u[:, :k], r.double()), _max_err(v[:, :k], r.double())
        logger.info(f"split {name}: max |err| vs CPU fused {fused_err:.3e}, composite {comp_err:.3e}")
        assert fused_err <= 2 * comp_err + 1e-5, (name, fused_err, comp_err)


@pytest.mark.parametrize("impl", ["fused", "composite"])
@MESH
def test_mhc_split_traced_time(mesh_device, device_params, impl):
    tokens, hidden = SHAPES["production"]
    _, x = _mesh_streams(mesh_device, tokens, hidden)
    composite, fused = _sites(mesh_device, _site_weights())
    site = fused if impl == "fused" else composite
    us = _traced_us(mesh_device, lambda: site.split(x))
    bound_us = 4 * tokens * N * hidden / (DRAM_GBPS * 1e3)  # read the streams once
    record = {
        "op": "split",
        "impl": impl,
        "tokens": tokens,
        "hidden_per_chip": hidden,
        "traced_us": round(us, 1),
        "dram_bound_us": round(bound_us, 1),
        "dram_util": round(bound_us / us, 3),
    }
    line = "V41_MHC_PERF " + json.dumps(record)
    logger.info(line)
    with open(os.environ.get("V41_MHC_PERF_OUT", "generated/v41_mhc_perf.log"), "a") as f:
        f.write(line + "\n")
