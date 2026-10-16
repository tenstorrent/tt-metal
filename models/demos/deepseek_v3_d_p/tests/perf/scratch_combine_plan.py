# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: the combine planner (ttnn.bringup.combine_rf.combine_plan) on the LoudBox as a 2 x 4 mesh against its
host model, bit for bit, for the overlap test's routings (K3 per-chip load). Times one planner call."""

import os
import statistics

import pytest
import torch
from loguru import logger
from ttnn.bringup.combine_rf.combine_plan import combine_planner_prepare, plan_reference
from ttnn.bringup.combine_rf.sparse_all_gather_2d import DIRS, mesh_plan

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.perf.scratch_flat_combine_rf import EPC, C, K, L, N, R, S, _chip_ns, _ids

A = int(os.environ.get("FC_NACC", "15"))
ROWS_CAP = int(os.environ.get("FC_ROWS_CAP", "40960"))
_PARAMS = [
    pytest.param(
        (R, C), fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0), id="fabric2d-mesh-2x4-2link"
    )
]


def port_luts(mesh_device):
    """Per chip: lut[direction * 2 + link] = port id (the chip's ports in DIRS / link order), and the port count."""
    chips, _, _ = mesh_plan(mesh_device, L)
    luts, n = {}, {}
    for ch, ports in chips.items():
        lut, j = [0xFF] * 8, 0
        for di, d in enumerate(DIRS):
            if d in ports:
                for li in range(L):
                    lut[di * 2 + li] = j
                    j += 1
        luts[ch], n[ch] = lut, j
    return luts, n


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("routing", os.environ.get("FC_ROUTINGS", "uniform,zipf_spread_1.2,zipf_packed_1.2").split(","))
def test_combine_plan(mesh_device, device_params, routing):
    g = torch.Generator().manual_seed(0)
    ids = _ids(routing, g)  # [N, S, K]
    gates = torch.softmax(torch.randn(N, S, K, generator=g), -1).to(torch.bfloat16)
    wbits = gates.view(torch.int16).to(torch.int64) & 0xFFFF
    luts, nps = port_luts(mesh_device)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    ROWS = os.environ.get("FC_PLAN_ROWS") == "1"  # (the routing as [N S, K] row-major: a page per token)
    shp = (N * S, K) if ROWS else (1, -1)
    ids_t = ttnn.from_torch(
        ids.reshape(*shp).to(torch.int32),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    w_t = ttnn.from_torch(
        gates.reshape(*shp),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    m = int(os.environ.get("FC_PLAN_CORES", "8"))  # (0: the one-core planner)
    cores = [(i % 8, i // 8) for i in range(m)] if m else None
    run = combine_planner_prepare(
        mesh_device, S=S, K=K, EPC=EPC, A=A, L=L, port_luts=luts, n_ports=nps, rows_cap=ROWS_CAP, cores=cores, row_pages=ROWS
    )
    outs = run(ids_t, w_t)
    ttnn.synchronize_device(mesh_device)
    got = {
        k: [ttnn.to_torch(t).to(torch.int64).reshape(-1) & 0xFFFFFFFF for t in ttnn.get_device_tensors(v)]
        for k, v in outs.items()
    }
    chips = [(r, c) for r in range(R) for c in range(C)]
    for ci, ch in enumerate(chips):
        ref = plan_reference(ids, wbits, ci, R, C, S, K, EPC, N * EPC, A, L, nps[ch], luts[ch], ROWS_CAP)
        for k, v in ref.items():
            exp = torch.tensor(v, dtype=torch.int64)
            bad = (got[k][ci][: exp.numel()] != exp).nonzero().flatten()
            assert bad.numel() == 0, f"{routing} chip {ci} {k}: {bad.numel()} words differ, first {bad[:8].tolist()}"
        logger.info(f"{routing} chip {ci}: plan matches the host model (rows {ref['plan'][600]}, ports {nps[ch]})")
    _chip_ns(mesh_device)
    per = []
    for _ in range(5):
        run(ids_t, w_t)
        ttnn.synchronize_device(mesh_device)
        per.append(max(_chip_ns(mesh_device)) / 1e3)
    if run.tdbg is not None:  # (phases of the last call, on the chip whose planner ended last)
        names = "read sort send wait xp+place rows+home final end".split()
        worst = None
        for ci, t in enumerate(ttnn.get_device_tensors(run.tdbg)):
            d = (ttnn.to_torch(t).to(torch.int64).reshape(m, 16) & 0xFFFFFFFF).tolist()
            t0 = min(r[0] for r in d)
            span = max(r[8] for r in d) - t0
            if worst is None or span > worst[0]:
                worst = (span, ci, d, t0)
        span, ci, d, t0 = worst
        for pi, r in enumerate(d):
            logger.info(
                f"PLAN {routing} chip {ci} core {pi}: "
                + " ".join(f"{n} {(r[i + 1] - r[i]) / 1350:.1f}" for i, n in enumerate(names))
                + f" | start +{(r[0] - t0) / 1350:.1f} end {(r[8] - t0) / 1350:.1f} us"
            )
    logger.info(
        f"PLAN {routing}: planner {statistics.median(per):.1f} us ({m or 1} cores per chip, {N * S * K} routing entries)"
    )
