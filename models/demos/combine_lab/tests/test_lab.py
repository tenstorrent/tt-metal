# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""(local) Combine lab on the Galaxy's 4 x 4 torus (both rings, FABRIC_2D_TORUS_XY): the bare all-to-all baseline
and the lab's configurations (lab.py), GLM-5.3 4x4 10240 traffic, uniform. Each configuration: built, one warm call,
then LAB_RUNS calls (a mesh barrier ahead of each), the device time of each (slowest chip) from the real-time profiler.

LAB_CASES: ';'-separated "name:key=val,key=val" over LabCfg's fields (ints / bools); "a2a" runs the bare all-to-all.
Run with TT_VISIBLE_DEVICES = the 4 x 4 subtorus and TT_MESH_GRAPH_DESC_PATH =
models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto."""

import dataclasses
import os
import statistics
import sys
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.combine_lab import a2a, lab

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests/ttnn/unit_tests/operations/experimental/moe_block"))
from op_checks import MeshBarrier, op_device_times  # noqa: E402

PAYLOAD = int(os.environ.get("LAB_PAYLOAD", "12352"))  # (the fabric router's payload: 6208 for half-row packets)
RUNS = int(os.environ.get("LAB_RUNS", "7"))
DEFAULT = "a2a;nocredit:credit=0;d3:;d3_thr1:cred_thr=1;d3_thr8:cred_thr=8;d1:D=1;d2:D=2;d6:D=6;d3_read:read_row=1;a2a_end"


def _cases():
    out = []
    for item in [x for x in os.environ.get("LAB_CASES", DEFAULT).split(";") if x.strip()]:
        name, _, rest = item.strip().partition(":")
        kv = dict(p.split("=", 1) for p in rest.split(",") if p.strip())
        out.append((name, kv))
    return out


def _cfg(kv):
    cfg = lab.LabCfg()
    for k, v in kv.items():
        f = {f.name: f for f in dataclasses.fields(cfg)}[k]
        if f.type in (str, "str"):
            setattr(cfg, k, v)
        else:
            setattr(cfg, k, (v not in ("0", "false", "False")) if f.type in (bool, "bool") else int(v))
    return cfg


def _time(mesh, io, mpd, ops):
    barrier = MeshBarrier(mesh)
    try:
        ttnn.generic_op(io, mpd)  # (warm: program build)
        ttnn.synchronize_device(mesh)

        def run():
            for _ in range(RUNS):
                barrier()
                ttnn.generic_op(io, mpd)

        t = op_device_times(mesh, run, ops)
    finally:
        barrier.close()
    return [x / 1e3 for x in t[next(iter(ops))]]


@pytest.mark.parametrize("mesh_device", [(4, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [dict(torus_xy_device_params(fabric_payload_size=PAYLOAD), l1_small_size=8192)],
    ids=["torus_xy"],
    indirect=True,
)
def test_lab(mesh_device):
    chips, fh = lab.fabric_first_hops(mesh_device)
    res = {}
    for name, kv in _cases():
        if name.startswith("a2a"):
            cfg = a2a.A2ACfg(PB=12320, rows_per_dst=320, senders=[(0, y) for y in range(8)])
            orig = a2a.first_hop
            a2a.first_hop = lambda s, d: fh[(s, d)]
            try:
                mpd, io, _, _ = a2a.build(mesh_device, cfg)
            finally:
                a2a.first_hop = orig
            us = _time(mesh_device, io, mpd, {"a2a": "combine_lab/kernels/a2a"})
        else:
            cfg = _cfg(kv)
            mpd, io, stats = lab.build(mesh_device, cfg)
            if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":  # (profile mode: the LAB_TIMING counters, no timing)
                for _ in range(1 + int(os.environ.get("PROF_CALLS", "2"))):
                    ttnn.generic_op(io, mpd)
                    ttnn.synchronize_device(mesh_device)
                    ttnn.ReadDeviceProfiler(mesh_device)
                logger.info(f"LABPROF {name}: profiled {kv}")
                for t in io:
                    ttnn.deallocate(t)
                continue
            us = _time(mesh_device, io, mpd, {"lab": "combine_lab/kernels/lab_"})
            per_port = sorted(set(stats.values()))
            logger.info(f"LABCFG {name}: {cfg} | rows a port {per_port[0]}..{per_port[-1]}")
        for t in io:
            ttnn.deallocate(t)
        res[name] = us
        logger.info(f"LAB {name}: {statistics.median(us):.1f} us  runs {[round(x) for x in us]}  {kv}")
    for name, us in res.items():
        logger.info(f"LABSUM {name:12} {statistics.median(us):8.1f}  min {min(us):8.1f}")
