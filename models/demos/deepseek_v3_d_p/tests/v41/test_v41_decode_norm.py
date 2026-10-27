# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode sublayer norm on a core grid (``tt/v41/decode_norm.py``, ``V41_DECODE_NORM=sharded``) against
the
block's ``TtDistributedRmsNorm`` on the decode tile.

No weights: a random gamma, ``V41_DN_TRIALS`` random ``[1, 1, 32, hidden/tp]`` bf16 tiles a chip (every chip's
TP
slice of its own rows), the two outputs compared on every chip (the interleaved op's rounding against the
sharded
program's: bf16-level), then each path under trace (``V41_DN_REPLAYS`` replays of 4 calls) for the per-site
device
time, for every core count of ``V41_DN_CORES`` (comma list; default 10,20,40). Results: ``V41_DN_RESULT``,
``V41_DN_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives, fabric_num_links
from models.demos.deepseek_v3_d_p.tt.v41.decode_norm import DecodeRmsNorm
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=64 * 1024 * 1024), indirect=True)
def test_v41_decode_norm(mesh_device, device_params):
    trials = int(os.environ.get("V41_DN_TRIALS", 4))
    replays = int(os.environ.get("V41_DN_REPLAYS", 20))
    core_counts = [int(c) for c in os.environ.get("V41_DN_CORES", "10,20,40").split(",")]
    sp, tp = tuple(mesh_device.shape)
    hidden = C.EMB_SIZE
    width = hidden // tp
    torch.manual_seed(5)
    gamma = torch.rand(hidden) * 1.5 + 0.25
    norm = TtDistributedRmsNorm(
        mesh_device=mesh_device,
        emb_dim=hidden,
        torch_weight=gamma,
        epsilon=C.RMS_NORM_EPS,
        cluster_axis=TP_AXIS,
        num_links=fabric_num_links(),
        topology=per_axis_topology()[TP_AXIS],
    )
    ccl = V41Collectives(mesh_device)
    decode_norms = {cores: DecodeRmsNorm(norm, width, ccl.tp_all_gather, cores) for cores in core_counts}

    def tile(h: torch.Tensor):
        return ttnn.from_torch(
            h.view(1, 1, sp * 32, hidden),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(2, 3)),
        )

    result = {"mesh": [sp, tp], "width": width, "trials": [], "cores": core_counts}
    worst = {str(c): {"pcc": 1.0, "max_abs": 0.0, "rel": 0.0} for c in core_counts}
    for trial in range(trials):
        h = torch.randn(sp, 32, hidden) * (1.0 + trial)
        x = tile(h)
        ref = norm(x)
        ref_t = ttnn.to_torch(ref, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, (sp, tp), dims=(2, 3))).float()
        for cores, dn in decode_norms.items():
            out = dn(x)
            assert tuple(out.shape) == tuple(ref.shape) and out.dtype == ref.dtype, (out.shape, out.dtype)
            assert not out.memory_config().is_sharded()
            out_t = ttnn.to_torch(
                out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, (sp, tp), dims=(2, 3))
            ).float()
            max_abs = float((ref_t - out_t).abs().max())
            rel = max_abs / max(float(ref_t.abs().max()), 1e-6)
            pcc = _pcc(ref_t, out_t)
            entry = {
                "trial": trial,
                "cores": cores,
                "pcc": round(pcc, 7),
                "max_abs": round(max_abs, 5),
                "rel": round(rel, 6),
            }
            result["trials"].append(entry)
            logger.info(f"decode norm trial {json.dumps(entry)}")
            w = worst[str(cores)]
            w["pcc"], w["max_abs"], w["rel"] = min(w["pcc"], pcc), max(w["max_abs"], max_abs), max(w["rel"], rel)
            ttnn.deallocate(out)
        ttnn.deallocate(ref)
        ttnn.deallocate(x)
    result["worst"] = worst
    # the per-site device time of each path, under trace
    x = tile(torch.randn(sp, 32, hidden))
    timing = {}
    paths = [("interleaved", lambda: norm(x))] + [
        (f"sharded-{c}", (lambda d=dn: d(x))) for c, dn in decode_norms.items()
    ]
    for name, fn in paths:
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [fn() for _ in range(4)]
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        best = None
        for _ in range(replays):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            dt = (time.perf_counter() - t0) / 4
            best = dt if best is None else min(best, dt)
        ttnn.release_trace(mesh_device, tid)
        for o in outs:
            ttnn.deallocate(o)
        timing[name] = round(best * 1e6, 1)
        logger.info(f"decode norm traced {name}: {timing[name]} us a call (min of {replays} replays of 4 calls)")
    result["site_us"] = timing
    logger.info(f"V41_DN_RESULT {json.dumps(result)}")
    out_path = os.environ.get("V41_DN_OUT")
    if out_path:
        with open(os.path.expanduser(out_path), "w") as f:
            json.dump(result, f, indent=1)
    for c, w in worst.items():
        assert w["pcc"] > 0.9999 and w["rel"] < 0.02, (c, w)
