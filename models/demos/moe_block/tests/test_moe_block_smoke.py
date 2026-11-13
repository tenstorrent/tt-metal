# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MoEBlock (models/demos/moe_block/tt/moe_block.py) end to end on an 8x4 Galaxy mesh: head -> dispatch -> experts +
combine -> tail with the shared arena, on zero / uninitialized weights. Checks that the four ops plan, build and launch
together (the arena spec accepted by the head and the combine, the output's shape) and gates the block's host time a
forward on program-cache hits: the median of op_checks.PERF_RUNS (5) runs, each the best of HOST_TRIES timings of
HOST_CALLS forwards (few: the command queue never fills, so no forward waits on the device), the timed thread pinned to
a core no busy thread shares (_quiet_cpu), against smoke_perf_baselines.json (per machine; MOE_PERF_RECORD=1 records
it; not on the mock Galaxy). No numerics (the ops' own tests check those bit-exact).

Needs an 8x4 Galaxy, or the mock Galaxy (plan, build and launch only):

    TT_METAL_MOCK_CLUSTER_DESC_PATH=$TT_METAL_HOME/tt_metal/third_party/umd/tests/cluster_descriptor_examples/\
blackhole_galaxy.yaml pytest models/demos/moe_block/tests/test_moe_block_smoke.py
"""

import os
import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.moe_block.tt.flat_gather import flat_weights_module
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests/ttnn/unit_tests/operations/experimental/moe_block"))
from op_checks import PerfBaselines, median_of_runs  # noqa: E402

R, C = 8, 4
CHUNK = 5120
ACCUMULATORS = 11  # (what the combine planner gives GLM-5.3 at 8x4, chunk 5120: the block derives it)
HOST_CALLS = 4  # (forwards a timing)
HOST_TRIES = 3  # (timings a run: its best)
# A busy spin ahead of each timing: the Galaxy host's cores scale their clock with load (schedutil, 1.5 - 3.8 GHz), and
# the timed thread idles at each sync - a timing that starts on a slow clock reads ~2x (700 vs 1500 us a forward).
HOST_SPIN_S = 0.05


def _quiet_cpu():
    """A logical CPU whose core (both SMT siblings) no busy thread of the host runs on - the timed thread is pinned
    there. On the Galaxy's host, other processes (wekanode: two cores at 100%) and the runtime's own spinning thread
    halve the timed thread's speed when it shares a core with one of them, and the scheduler then keeps it there: a run
    drifts from ~700 to ~1500 us a forward. MOE_HOST_CPU overrides; -1: no pinning."""
    want = int(os.environ.get("MOE_HOST_CPU", "-2"))
    if want != -2:
        return None if want < 0 else want

    def ticks():
        out = {}
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                for tid in os.listdir(f"/proc/{pid}/task"):
                    f = open(f"/proc/{pid}/task/{tid}/stat").read().rsplit(")", 1)[1].split()
                    out[(pid, tid)] = (int(f[11]) + int(f[12]), int(f[36]))
            except OSError:
                pass
        return out

    t0 = ticks()
    time.sleep(0.5)
    t1 = ticks()
    hz = os.sysconf("SC_CLK_TCK")
    busy = {t1[k][1] for k in t1 if k in t0 and (t1[k][0] - t0[k][0]) / hz > 0.25}  # (over 50% of the 0.5 s)

    def core(c):
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
        except OSError:
            return {c}
        out = set()
        for part in sib.split(","):
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
        return out

    allowed = sorted(os.sched_getaffinity(0))
    for c in reversed(allowed):  # (from the top: away from the low cores system daemons favour)
        if not core(c) & busy:
            return c
    return None
HOST_MARGIN = 0.15  # (host time: noisier than device time)
BASELINES = PerfBaselines(Path(__file__).with_name("smoke_perf_baselines.json"))
MOCK = "blackhole_galaxy" in os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")


def _glm53():
    mb = ttnn.experimental.moe_block
    silu = mb.ExpertActivation(kind=mb.ExpertActivationKind.Silu)
    return mb.MoEModelDims(
        num_experts=256,
        top_k=8,
        emb_dim=6144,
        routed_expert_hidden_dim=2048,
        routed_expert_activation=silu,
        router=mb.RouterSpec(score_func=mb.RouterScoreFunc.Sigmoid, route_scale=2.5),
        shared_expert_hidden_dim=2048,
        shared_expert_activation=silu,
    )


@pytest.fixture(scope="module")
def mesh():
    if not MOCK and ttnn.get_num_devices() < R * C:
        pytest.skip(f"needs a {R}x{C} Galaxy or the mock Galaxy")
    # (the combine sends one row of H bf16 + its 32 B trailer a fabric packet)
    rc = ttnn._ttnn.fabric.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = (2 * _glm53().emb_dim + 32 + 63) // 64 * 64
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ttnn.FabricReliabilityMode.RELAXED_INIT, router_config=rc
    )
    m = ttnn.open_mesh_device(ttnn.MeshShape(R, C), l1_small_size=5888)  # (the block's words and semaphores)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_moe_block_smoke(mesh):
    mb = ttnn.experimental.moe_block
    fw = flat_weights_module()
    dims = _glm53()
    E, H, I, Is, N = dims.num_experts, dims.emb_dim, dims.routed_expert_hidden_dim, 2048, R * C
    dtypes = mb.MoEDtypes()
    res = mb.MoEBlockResources(
        mesh, dims, mb.MoEBlockConfig(chunk_tokens=CHUNK, num_links=2, persist_l1_small=True), dtypes
    )
    try:
        block = MoEBlock(mesh, res, dtypes=dtypes)
        assert block.accumulators == ACCUMULATORS, f"the combine planner's accumulators {block.accumulators}"
        host = lambda *s: ttnn.from_torch(torch.zeros(*s), dtype=ttnn.float32)
        head_w = mb.prepare_moe_head_weights(
            host(Is, H), host(Is, H), host(H, Is), None, host(E, H), host(E),
            resources=res, dtypes=dtypes, accumulators=block.accumulators,
        )  # fmt: skip
        EPC = E // N
        gids = [list(range(c * EPC, (c + 1) * EPC)) for c in range(N)]
        experts = fw.FlatRoutedExpert(
            mesh, "uninitialized", m=block.expert_kw["max_tokens_per_expert"], H=dims.routed_width, I=I, gids=gids,
            n_global=E, act="silu", pin=1, layout=block.expert_kw["layout"],
        )  # fmt: skip
        weights = MoEBlockWeights(head=head_w, expert_ids=experts.gidx, gate_up=experts.w_gu, down=experts.w_d)
        x = ttnn.from_torch(
            torch.zeros(1, 1, CHUNK, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
        )
        y = block(x, weights)  # (a miss on every op)
        assert tuple(y.shape) == (1, 1, CHUNK // R, H // C), y.shape
        ttnn.deallocate(y)
        ttnn.deallocate(block(x, weights))

        def timing():
            ttnn.synchronize_device(mesh)
            spin = time.perf_counter() + HOST_SPIN_S  # (the core's clock up first: see HOST_SPIN_S)
            while time.perf_counter() < spin:
                pass
            t0 = time.perf_counter()
            for _ in range(HOST_CALLS):
                ttnn.deallocate(block(x, weights))
            us = (time.perf_counter() - t0) / HOST_CALLS * 1e6
            ttnn.synchronize_device(mesh)
            return us

        def host_us():
            return min(timing() for _ in range(HOST_TRIES))

        cpu = _quiet_cpu()
        saved = os.sched_getaffinity(0)
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})  # (this thread only)
        try:
            us, runs = median_of_runs(host_us)
        finally:
            os.sched_setaffinity(0, saved)
        print(
            f"MoEBlock glm53 {R}x{C} chunk {CHUNK}: {us:.1f} us of host a forward (program-cache hits, untraced; "
            f"median of {', '.join(f'{r:.0f}' for r in runs)}; timed thread on cpu {cpu})"
        )
        if not MOCK:
            BASELINES.check(f"glm53 {R}x{C} {CHUNK} host a forward", us, HOST_MARGIN)
        experts.close()
    finally:
        res.release()
