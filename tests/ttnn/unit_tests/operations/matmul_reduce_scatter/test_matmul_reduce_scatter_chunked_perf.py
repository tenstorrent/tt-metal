# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunk-split perf sweeps for matmul_reduce_scatter on the QB2 perf placement (1x4 ring, FABRIC_2D_TORUS_XY, 2 links).

Every config is measured traced (WARMUP eager calls, one trace capture, REPLAYS replays) and read in-process from the
device profiler: per replay, the op latency is the MAX over the 4 devices of DEVICE KERNEL DURATION; a config reports
the median replay. Each config's output (last eager call) is also checked against the fp32 torch reference.

Sweeps (MRS_SWEEPS, comma list; default all):
  kimi   Kimi K2.7 o_proj 640x2048x7168 (fits unchunked): m-only, n-only and 2D splits, ZONE_SETS 1 vs 2
  wan    Wan2.2 FF2 720p 9472x3456x5120 bf16 (does not fit unchunked): the auto split and the fitting splits
  synth  8192x4096x8192 bf16 heatmap: n_m in {1,2,4,8,16,32} x n_n in {1,2,4,8,16}

Results are appended as JSON lines to MRS_SWEEP_OUT (default generated/mrs_chunk_sweep.jsonl).

Run (profiler env needed for the in-process durations):
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
  TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 MRS_SWEEPS=kimi \\
  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter_chunked_perf.py -s
"""

from __future__ import annotations

import json
import os
import statistics
import time

import pytest
import torch
import ttnn

import ttnn.operations.matmul_reduce_scatter as pkg
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as desc
from eval.golden_tests.matmul_reduce_scatter.helpers import (
    create_ttnn_input_tensor,
    pytorch_matmul_reduce_scatter,
    quantize_like_device,
    random_stacked,
)

WARMUP = 2
REPLAYS = 5
TRACE_REGION_BYTES = 4 << 20
OUT = os.environ.get("MRS_SWEEP_OUT", "generated/mrs_chunk_sweep.jsonl")
BF8, BF16 = ttnn.bfloat8_b, ttnn.bfloat16


def _kimi_configs():
    cfg = [(1, 1, 2)]
    cfg += [(m, 1, 2) for m in (2, 4, 5, 10, 20)]  # m only (resident A reloads per m chunk)
    cfg += [(1, n, 2) for n in (2, 4, 7, 8, 14, 28, 56)]  # n only (A stays resident)
    cfg += [(m, n, 2) for m, n in ((2, 2), (2, 4), (4, 2), (4, 4), (2, 7), (5, 4), (4, 8), (10, 4), (5, 14), (10, 14))]
    cfg += [(1, n, 1) for n in (2, 4, 8, 14)] + [(2, 2, 1), (4, 4, 1)]  # one zone set
    return cfg


def _wan_configs():
    cfg = [None]  # the planner's own choice
    cfg += [(m, 1, 2) for m in (3, 4, 6, 8, 12, 16, 24, 32)]
    cfg += [(m, n, 2) for m, n in ((2, 2), (2, 4), (4, 2), (4, 4), (2, 8), (8, 2), (8, 4), (4, 8), (16, 2), (6, 6))]
    cfg += [(m, n, 2) for m, n in ((8, 8), (16, 4), (12, 4), (16, 8), (32, 4), (32, 8))]
    return cfg


_KIMI_KB = [(1, 1), (1, 2), (1, 4), (1, 8), (2, 2), (2, 1), (4, 4)]
_WAN_KB = [(3, 1), (4, 1), (8, 4), (16, 2), (16, 4)]
_SYNTH_KB = [(4, 4), (8, 2), (8, 4), (16, 4), (16, 8)]


def _kb_configs(splits):
    return [(m, n, 2, kb) for m, n in splits for kb in (8, 16, 32, 64, 128)]


def _synth_configs():
    return [(m, n, 2) for m in (1, 2, 4, 8, 16, 32) for n in (1, 2, 4, 8, 16)]


SWEEPS = {
    # name: (inputs, weight dtype, fp32 DEST acc, configs)
    "kimi": (((1, 1, 640, 2048), (2048, 7168)), BF8, False, _kimi_configs),
    "wan": (((1, 1, 9472, 3456), (3456, 5120)), BF16, True, _wan_configs),
    "synth": (((1, 1, 8192, 4096), (4096, 8192)), BF16, True, _synth_configs),
    # the K-block cap per chunked config (each chunk re-walks Kt / kb K-steps per entry)
    "kimi_kb": (((1, 1, 640, 2048), (2048, 7168)), BF8, False, lambda: _kb_configs(_KIMI_KB)),
    "wan_kb": (((1, 1, 9472, 3456), (3456, 5120)), BF16, True, lambda: _kb_configs(_WAN_KB)),
    "synth_kb": (((1, 1, 8192, 4096), (4096, 8192)), BF16, True, lambda: _kb_configs(_SYNTH_KB)),
    # the heatmaps again with the K-block cap at 32
    "synth32": (((1, 1, 8192, 4096), (4096, 8192)), BF16, True, lambda: [c[:3] + (32,) for c in _synth_configs()]),
    "wan32": (((1, 1, 9472, 3456), (3456, 5120)), BF16, True, lambda: [c[:3] + (32,) for c in _wan_configs()[1:]]),
    # the planner's own split (cost model), and its neighbours
    "auto": (
        ((1, 1, 9472, 3456), (3456, 5120)),
        BF16,
        True,
        lambda: [None, (9, 1, 2), (8, 1, 2), (10, 1, 2), (12, 1, 2)],
    ),
    "synth_auto": (((1, 1, 8192, 4096), (4096, 8192)), BF16, True, lambda: [None]),
    # stage peel (MRS_ABLATE; outputs are wrong by design): full, no DRAM, + no multicast, + no compute (transport)
    "kimi_ablate": (((1, 1, 640, 2048), (2048, 7168)), BF8, False, lambda: _ablate_configs(_KIMI_ABL)),
}
_KIMI_ABL = [(1, 1), (1, 2), (1, 4), (1, 8), (2, 1)]


def _ablate_configs(splits):
    peel = ("", "dram", "dram,mcast", "dram,mcast,compute")
    return [(m, n, 2, 16, ab) for m, n in splits for ab in peel]


SELECTED = [s for s in os.environ.get("MRS_SWEEPS", "kimi,wan,synth").split(",") if s]


def _open():
    from tests.scripts.common import get_updated_device_params

    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
    fc = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    ttnn.set_fabric_config(
        fc,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        cfg,
    )
    params = get_updated_device_params({"fabric_config": fc})
    params.pop("fabric_config")
    params["trace_region_size"] = TRACE_REGION_BYTES
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), **params)


@pytest.fixture(scope="module")
def mesh():
    m = _open()
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _uid(p):
    u = p.program_execution_uid
    return (u.runtime_id, u.trace_id, u.trace_id_counter)


def _traced_durations(mesh, run):
    """[op latency per replay (us)] = max over devices of DEVICE KERNEL DURATION, plus host us per replay."""
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    seen = {dev: {_uid(p) for p in progs} for dev, progs in ttnn.get_all_programs_perf_data().items()}
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = run()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    host_us = (time.perf_counter() - t0) / REPLAYS * 1e6
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)
    ttnn.ReadDeviceProfiler(mesh)
    per_replay = {}
    for dev, progs in ttnn.get_all_programs_perf_data().items():
        for p in progs:
            uid = _uid(p)
            if uid in seen.get(dev, set()) or uid[2] >= 1 << 62:  # earlier programs / non-traced runs
                continue
            dur = p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration / 1000
            per_replay.setdefault(uid[2], []).append(dur)
    reps = [max(v) for _, v in sorted(per_replay.items()) if len(v) == 4]
    return reps, host_us


@pytest.mark.parametrize("sweep", SELECTED)
def test_chunk_sweep(mesh, sweep):
    (a_shape, w_shape), wdt, fp32_acc, configs = SWEEPS[sweep]
    a = random_stacked(mesh, a_shape, torch.bfloat16, seed=0)
    w = quantize_like_device(random_stacked(mesh, w_shape, torch.bfloat16, seed=1, scale=w_shape[0] ** -0.5), wdt)
    ta = create_ttnn_input_tensor(a, mesh, dtype=BF16, layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(w, mesh, dtype=wdt, layout=ttnn.TILE_LAYOUT)
    expected = pytorch_matmul_reduce_scatter(a, w, 1, -1).double()
    ref_norm = [expected[0, d].pow(2).mean().sqrt().item() for d in range(4)]
    ckc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_acc)
    run = lambda: matmul_reduce_scatter(
        ta, tw, cluster_axis=1, scatter_dim=-1, topology=ttnn.Topology.Ring, num_links=2, compute_kernel_config=ckc
    )
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    old_sets, old_kb, old_kbc = desc.ZONE_SETS, desc.KB_MAX, desc.KB_MAX_CHUNKED
    for cfg in configs():
        rec = dict(sweep=sweep, M=a_shape[-2], K=a_shape[-1], N=w_shape[-1], forced=cfg)
        pkg._FORCE_CHUNKS = None if cfg is None else cfg[:2]
        desc.ZONE_SETS = old_sets if cfg is None else cfg[2]
        desc.KB_MAX = old_kb if cfg is None or len(cfg) < 4 else cfg[3]
        desc.KB_MAX_CHUNKED = old_kbc if cfg is None or len(cfg) < 4 else cfg[3]
        ablate = cfg[4] if cfg is not None and len(cfg) > 4 else ""
        rec["ablate"] = ablate
        os.environ["MRS_ABLATE"] = ablate
        try:
            out = None
            for _ in range(WARMUP):
                if out is not None:
                    ttnn.deallocate(out)
                out = run()
            rec["plan"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in pkg._LAST_PLAN.items()}
            ttnn.synchronize_device(mesh)
            rec["rel_rms"] = max(
                (ttnn.to_torch(t).double() - expected[0, d]).pow(2).mean().sqrt().item() / ref_norm[d]
                for d, t in enumerate(ttnn.get_device_tensors(out))
            )
            ttnn.deallocate(out)
            reps, host_us = _traced_durations(mesh, run)
            rec.update(replays_us=reps, median_us=statistics.median(reps) if reps else None, host_us=host_us)
        except NotImplementedError as e:
            rec["error"] = f"nofit: {e}"
        except ValueError as e:
            rec["error"] = f"invalid: {e}"
        finally:
            pkg._FORCE_CHUNKS = None
            desc.ZONE_SETS = old_sets
            desc.KB_MAX = old_kb
            desc.KB_MAX_CHUNKED = old_kbc
            os.environ.pop("MRS_ABLATE", None)
        line = json.dumps(rec)
        print("SWEEP", line, flush=True)
        with open(OUT, "a") as f:
            f.write(line + "\n")
        if "rel_rms" in rec and not ablate:
            assert rec["rel_rms"] <= (0.035 if wdt == BF8 else 0.01), rec
