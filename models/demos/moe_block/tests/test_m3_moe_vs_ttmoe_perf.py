# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3's MoE layer on its production stage (one [2, 4] tray, SP 2 x TP 4, Fabric2D, no wrap) over the production
packings (test_moe_block_models.M3_PRODUCTION), timed two ways for each implementation:

* moe_block: the shared MoE block (TtK3MoEBlockFfn: moe_head -> sparse_dispatch_2d -> routed_experts_combine ->
  moe_tail), x [1, 1, S/SP, H/TP] a chip;
* ttmoe: M3's own production MLP (models/demos/minimax_m3/tt/mlp.py: TopKRouter, the shared expert on its own
  sub-device concurrently with dispatch, TtMiniMaxMoE - DeepSeek's TtMoe dispatch / unified_routed_expert_moe /
  combine / reduce-scatter), x [1, 1, S/SP, H] a chip (the layer's pre-MLP all-gather's output), bf4 experts
  (EXPERT_DTYPE's default), the padded chunks through its PrefillChunk padding config.

Per case, runs of CALLS calls back to back, one sync each: the wall time a call (what a layer costs the pipeline - it counts overlap,
e.g. TtMoe's shared expert beside its dispatch, and host dispatch) and each program group's device time a call (its
slowest chip, grouped by the ttnn op directory its kernels come from):
* eager, both (M3 runs its prefill untraced: TtPrefillRuntime.use_trace False - its shared-expert overlap switches the
  sub-device manager mid-forward, which a trace capture cannot hold: TT_FATAL, then the device is lost);
* traced replays, the block only.
TtMiniMaxMoE frees its input after dispatch (the layer's tensor in the model), so each TtMoe call takes its own copy,
made before the timed calls. Every number is the median of op_checks.PERF_RUNS (5) runs (wall: runs of CALLS calls;
device: single calls, each on its own profiler window); the wall and device times a call are gated against
m3_moe_perf_baselines.json (per machine; MOE_PERF_RECORD=1 records them).
Random weights, every expert distinct. Run as test_moe_block_models' 2x4 cases (its descriptor, one tray)."""

import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.moe_block.tests.test_moe_block_models import M3_PRODUCTION, MESHES, MODELS, _setup
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests/ttnn/unit_tests/operations/experimental/moe_block"))
from op_checks import PERF_RUNS, PerfBaselines, median_of_runs  # noqa: E402

CALLS = 10  # (calls a timed wall run)
WALL_MARGIN, DEVICE_MARGIN = 0.10, 0.05
EAGER_MARGIN = 0.20  # (eager numbers carry host dispatch: a program that waits on the host shows it as device time)
BASELINES = PerfBaselines(Path(__file__).with_name("m3_moe_perf_baselines.json"))
CASES = [c for c in M3_PRODUCTION]


def _group(srcs) -> str:
    """A program's group: the ttnn op directory of its first kernel source (operations/<...>/device), else the file."""
    for s in srcs:
        s = s.replace("\\", "/")
        if "/operations/" in s:
            rest = s.split("/operations/", 1)[1]
            return rest.split("/device/", 1)[0] if "/device/" in rest else rest.rsplit("/", 1)[0]
    return srcs[0].rsplit("/", 1)[-1] if srcs else "?"


def _traced(mesh_device, fn, label):
    """fn() captured once in a trace: (wall us a replay, {group: median device us a replay}, device us a replay)."""
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(2):  # (warm replays)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)

    def wall_run():
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(CALLS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / CALLS * 1e6

    wall, _ = median_of_runs(wall_run)
    per = defaultdict(list)
    totals = []
    for _ in range(PERF_RUNS):  # (each replay on its own profiler window: a trace's programs keep one runtime id)
        _, progs = profile_realtime_program_merged(
            mesh_device, lambda: ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        )
        sums = defaultdict(float)
        for e in progs.values():
            sums[_group(e["kernel_sources"])] += e["duration_ns"] / 1e3
        for g, v in sums.items():
            per[g].append(v)
        totals.append(sum(sums.values()))
    ttnn.release_trace(mesh_device, tid)
    groups = {g: statistics.median(v) for g, v in per.items()}
    dev = statistics.median(totals)
    logger.info(
        f"E2E {label}: wall {wall:.0f} us a traced replay, device programs {dev:.0f} us | "
        + " | ".join(f"{g} {v:.0f}" for g, v in sorted(groups.items(), key=lambda kv: -kv[1]))
    )
    return wall, groups, dev, out


def _eager(mesh_device, fn, make_arg, label):
    """Eager calls of fn(arg), each on its own arg made beforehand: (wall us a call: the median of PERF_RUNS runs of
    CALLS calls, {group: device us a call}, device us a call: medians over PERF_RUNS single calls)."""

    def wall_run():
        args = [make_arg() for _ in range(CALLS)]
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for a in args:
            ttnn.deallocate(fn(a))
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / CALLS * 1e6

    wall, _ = median_of_runs(wall_run)
    per, totals = defaultdict(list), []
    for _ in range(PERF_RUNS):
        a = make_arg()
        ttnn.synchronize_device(mesh_device)
        _, progs = profile_realtime_program_merged(mesh_device, lambda: ttnn.deallocate(fn(a)))
        one = defaultdict(float)
        for e in progs.values():
            one[_group(e["kernel_sources"])] += e["duration_ns"] / 1e3
        for g, v in one.items():
            per[g].append(v)
        totals.append(sum(one.values()))
    sums = {g: statistics.median(v) for g, v in per.items()}
    dev = statistics.median(totals)
    logger.info(
        f"E2E {label} eager: wall {wall:.0f} us a call, device programs {dev:.0f} us | "
        + " | ".join(f"{g} {v:.0f}" for g, v in sorted(sums.items(), key=lambda kv: -kv[1]))
    )
    return wall, dict(sums), dev


def _params():
    shape, make_params, topology = MESHES["2x4"]
    out = []
    for impl in ("moe_block", "ttmoe"):
        for c in CASES:
            out.append(
                pytest.param(
                    impl,
                    c,
                    shape,
                    make_params(**MODELS["minimax_m3"].mesh_params()),
                    marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topology),
                    id=f"{impl}-{c.id}",
                )
            )
    return out


def _ttmoe(mesh_device, case):
    """M3's MLP (router + shared expert + TtMiniMaxMoE) on the stage, its input, and the forward's PrefillChunks."""
    from models.demos.common.prefill.prefill_chunk import PrefillChunk
    from models.demos.minimax_m3.config import MeshConfig
    from models.demos.minimax_m3.tt.ccl import CCLManager
    from models.demos.minimax_m3.tt.mlp import MLP
    from models.demos.minimax_m3.utils.general_utils import get_default_num_links
    from models.demos.minimax_m3.utils.model_variant import load_config

    cfg = load_config()
    H, E, I, Is = cfg["hidden_size"], cfg["num_local_experts"], cfg["intermediate_size"], cfg["shared_intermediate_size"]
    R, C = tuple(mesh_device.shape)
    g = torch.Generator().manual_seed(13)
    r = lambda *s, sc: torch.randn(*s, generator=g) * sc  # noqa: E731
    state = {
        "gate.weight": r(E, H, sc=H**-0.5),
        "e_score_correction_bias": r(E, sc=0.01),
        "shared_experts.gate_proj.weight": r(Is, H, sc=H**-0.5),
        "shared_experts.up_proj.weight": r(Is, H, sc=H**-0.5),
        "shared_experts.down_proj.weight": r(H, Is, sc=Is**-0.5),
    }
    for e in range(E):
        state[f"experts.{e}.w1.weight"] = r(I, H, sc=H**-0.5)
        state[f"experts.{e}.w3.weight"] = r(I, H, sc=H**-0.5)
        state[f"experts.{e}.w2.weight"] = r(H, I, sc=I**-0.5)
    mlp = MLP(
        mesh_device=mesh_device,
        hf_config=SimpleNamespace(**cfg),
        state_dict=state,
        ccl_manager=CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear),
        mesh_config=MeshConfig((R, C), tp=C),
        expert_weight_dtype=ttnn.bfloat4_b,  # (EXPERT_DTYPE's production default)
        use_ep_moe=True,
        ep_seq_len_per_chip=case.chunk // R,
        overlap_shared_expert=True,  # (PREFILL_OVERLAP_SHARED_EXPERT's production default)
    )
    del state
    x = torch.randn(R, 1, case.chunk // R, H, generator=g).to(torch.bfloat16)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(R, C)),
    )
    if case.pad is not None:
        chunks = [PrefillChunk(0, case.pad[0], case.pad[1])]
    elif case.packing == "4xc1024":
        chunks = [PrefillChunk(0, k * 1024, (k + 1) * 1024) for k in range(4)]
    else:
        chunks = None
    return mlp, x_tt, chunks


@pytest.mark.parametrize("impl, case, mesh_device, device_params", _params(), indirect=["mesh_device", "device_params"])
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
def test_m3_moe_e2e_perf(impl, case, mesh_device, device_params):
    label = f"{impl} {case.id}"
    if impl == "moe_block":
        run = _setup(case, mesh_device)
        fn, make_arg = (lambda _: run()), (lambda: None)
    else:
        mlp, x_tt, chunks = _ttmoe(mesh_device, case)
        fn, make_arg = (lambda x: mlp(x, chunks=chunks)), (lambda: ttnn.clone(x_tt))
    for _ in range(2):  # (weights, programs, the padding config)
        ttnn.deallocate(fn(make_arg()))
    wall, _, dev = _eager(mesh_device, fn, make_arg, label)
    BASELINES.check(f"{label} eager wall", wall, EAGER_MARGIN)
    BASELINES.check(f"{label} eager device", dev, EAGER_MARGIN)
    if impl == "moe_block":
        wall, _, dev, _ = _traced(mesh_device, run, label)
        BASELINES.check(f"{label} traced wall", wall, WALL_MARGIN)
        BASELINES.check(f"{label} traced device", dev, DEVICE_MARGIN)
