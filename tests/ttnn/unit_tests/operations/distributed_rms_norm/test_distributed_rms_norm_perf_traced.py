# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced per-chip perf breakdown of distributed_rms_norm on the perf cases (measurement driver, not a gate).

The LOOSE_CASES geometry: the box's 4-chip snake ring opened as 1 x 4, FABRIC_2D_TORUS_XY, 14 KiB + 64 B payload,
Ring, 2 links, bf16 x / TILE bf16 gamma, HiFi4 + fp32 acc, eps 1e-5. Metric: DEVICE KERNEL DURATION per chip and call
(in-process device profiler, as the fabric_ring_exchange bench), reported as
  untraced last-launched / untraced slowest / traced slowest-chip-per-call (median)
for the op and for the op with the exchange stubbed (DRMS_ABLATE=exchange: ports idle, landed waits skipped — its
output is garbage, timing only). Exposed transport (OVERLAP) = traced slowest(op) - traced slowest(stubbed).

    scripts/run_safe_pytest.sh --no-precompile --run-all \
        tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_perf_traced.py -s

Env: DRMS_PT_CASES ("640x1792,32x1792,2048x1280"), DRMS_PT_UNTRACED (5), DRMS_PT_TRACED (8),
DRMS_PT_VARIANTS ("op,stub"), DRMS_PT_EXTRA_ABLATE (payload ablations added to both variants: reads, writes, gamma).
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from ttnn.operations.distributed_rms_norm import distributed_rms_norm, plan_summary

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_list = lambda name, default: [x for x in os.environ.get(name, default).split(",") if x]
CASES = [tuple(int(v) for v in c.split("x")) for c in _list("DRMS_PT_CASES", "640x1792,32x1792,2048x1280")]
VARIANTS = _list("DRMS_PT_VARIANTS", "op,stub")
UNTRACED = int(os.environ.get("DRMS_PT_UNTRACED", "5"))
TRACED = int(os.environ.get("DRMS_PT_TRACED", "8"))

pytestmark = pytest.mark.skipif(os.environ.get("UP_FRONT_COLLECT") == "1", reason="run with --no-precompile")


def _per_chip_ns(mesh):
    ttnn.ReadDeviceProfiler(mesh)
    out = {}
    for dev, programs in (ttnn.get_latest_programs_perf_data() or {}).items():
        out[dev] = [
            float(p.program_analyses_results[_DURATION_KEY].duration)
            for p in programs
            if _DURATION_KEY in (getattr(p, "program_analyses_results", None) or {})
        ]
    return out


def _measure(mesh, run):
    run()
    ttnn.synchronize_device(mesh)
    _per_chip_ns(mesh)  # drop compile + first call
    last, slow = [], []
    for _ in range(UNTRACED):
        run()
        ttnn.synchronize_device(mesh)
        per = [v[-1] for v in _per_chip_ns(mesh).values() if v]
        last.append(min(per))
        slow.append(max(per))
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(TRACED):
        run()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.release_trace(mesh, tid)
    ttnn.synchronize_device(mesh)
    per = [v[-TRACED:] for v in _per_chip_ns(mesh).values() if len(v) >= TRACED]
    traced = [max(chip[i] for chip in per) for i in range(TRACED)]
    traced_min = [min(chip[i] for chip in per) for i in range(TRACED)]
    us = lambda v: statistics.median(v) / 1000
    return us(last), us(slow), us(traced), us(traced_min)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "fabric_router_config": production_router_config(),
            "trace_region_size": 1 << 23,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
def test_perf_traced(mesh_device):
    mesh = mesh_device
    G = 4
    for shape in CASES:
        M, W = shape
        x = torch.randn((G, 1, 1, M, W), dtype=torch.float32).to(torch.bfloat16)
        g = torch.randn((G, 1, 1, 1, W), dtype=torch.float32).to(torch.bfloat16)
        put = lambda t: ttnn.from_torch(
            torch.cat(list(t), dim=-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 3), mesh_shape=(1, 4)),
        )
        tx, tg = put(x), put(g)
        print(f"\nPLAN {shape} {plan_summary(tx, cluster_axis=1, gamma=tg, topology=ttnn.Topology.Ring, num_links=2)}")
        run = lambda: distributed_rms_norm(
            tx, cluster_axis=1, gamma=tg, epsilon=1e-5, topology=ttnn.Topology.Ring, num_links=2
        )
        # correctness of the real variant first
        out = run()
        full = torch.cat(list(x.float()), dim=-1)
        ref = torch.nn.functional.rms_norm(full, (full.shape[-1],), torch.cat(list(g.float().reshape(G, -1))), 1e-5)
        got = torch.cat([ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(out)], dim=-1)
        assert torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1] > 0.999
        res = {}
        for variant in VARIANTS:
            # DRMS_PT_EXTRA_ABLATE (measurement only): payload ablations added to both variants, e.g. "reads,writes"
            ablate = [a for a in os.environ.get("DRMS_PT_EXTRA_ABLATE", "").split(",") if a]
            if variant == "stub":
                ablate.append("exchange")
            if ablate:
                os.environ["DRMS_ABLATE"] = ",".join(ablate)
            try:
                res[variant] = _measure(mesh, run)
            finally:
                os.environ.pop("DRMS_ABLATE", None)
            l, s, t, tm = res[variant]
            print(
                f"PERF {M}x{W} {variant:4s}: untraced last {l:6.2f} | untraced slowest {s:6.2f} | "
                f"traced slowest {t:6.2f} | traced fastest {tm:6.2f} us"
            )
        if "op" in res and "stub" in res:
            print(f"PERF {M}x{W} exposed transport (traced slowest op - stub): {res['op'][2] - res['stub'][2]:.2f} us")
