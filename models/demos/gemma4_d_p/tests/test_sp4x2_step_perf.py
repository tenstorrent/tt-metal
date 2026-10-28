# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SP4 x 2 study: batched prefill steps on two CP4 x TP4 halves of one Galaxy, each a full model copy.

The 8x4 Galaxy is split into two 4x4 submeshes with one Gemma4 instance each. A step gives each half B requests x 2048
tokens (B = 1 .. SP4X2_LANES); both halves replay their traces at once and the step time is the wall time until both
finish (trace replay only, as StepRunner.step times it). Compare with test_batched_prefill.py::test_batched_step_perf
on the whole 8x4 mesh, which runs 2 B requests in one step at the same prefixes.

Needs GEMMA4_ALLOW_SUBMESH=1 and a TT_CACHE_PATH holding a tensor_cache_bf16_mesh4x4 cache (GEMMA4_PREFILL_LOAD_FULL_
WEIGHTS=1 builds it on the first run). GEMMA4_SLIDING_LANES=0 runs one sliding ring SDPA per request.
SP4X2_MIXES picks the prefix mixes, G4B_REPS the timed replays.
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4_d_p.demo.batched_steps import TRACE_REGION_SIZE, StepRunner, build_batched_model
from models.demos.gemma4_d_p.demo.text_demo_prefill import _hf_model_id, _text_token_stream
from models.demos.gemma4_d_p.tests.test_batched_prefill import _MIXES
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric


@torch.no_grad()
@pytest.mark.timeout(12000)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_sp4x2_step_perf(mesh_device, reset_seeds):
    chunk_size, capacity = 2048, 262144
    max_lanes = int(os.environ.get("SP4X2_LANES", "2"))
    mixes = {
        k: v for k, v in _MIXES.items() if k in os.environ.get("SP4X2_MIXES", "shallow,u32k,mixed,deep").split(",")
    }
    reps = int(os.environ.get("G4B_REPS", "5"))
    os.environ["GEMMA4_ALLOW_SUBMESH"] = "1"
    halves = mesh_device.create_submeshes(ttnn.MeshShape(4, 4))
    assert len(halves) == 2

    deepest = max(max(mix(max_lanes)) for mix in mixes.values()) + chunk_size
    stream = _text_token_stream(_hf_model_id())[0]
    num_seqs = 2 * max_lanes
    stride = min(100_000, (len(stream) - deepest) // max(1, num_seqs - 1))
    seqs = [stream[i * stride : i * stride + deepest].clone() for i in range(num_seqs)]

    groups = []
    for g, half in enumerate(halves):
        mesh_config, _, model = build_batched_model(half, chunk_size, max_lanes, capacity)
        runner = StepRunner(half, mesh_config, model, chunk_size)
        mine = seqs[g * max_lanes : (g + 1) * max_lanes]

        def lane(slot, start, mine=mine):
            return (slot, start, mine[slot][start : start + chunk_size])

        runner.capture_all([[lane(s, 0) for s in range(b)] for b in range(1, max_lanes + 1)])
        t0 = time.time()
        for slot in range(max_lanes):
            for start in range(0, deepest - chunk_size, chunk_size):
                runner.step([lane(slot, start)])
        groups.append((half, runner, lane))
        logger.info(f"[sp4x2] half {g} ready: {mesh_config}, filled {max_lanes} slots in {time.time() - t0:.0f}s")

    def step_both(lane_sets):
        for (_, runner, _), lanes in zip(groups, lane_sets):
            runner.stage(lanes)
        t0 = time.time()
        for (half, runner, _), lanes in zip(groups, lane_sets):
            ttnn.execute_trace(half, runner.traces[runner.layout(lanes)], cq_id=0, blocking=False)
        for half, _, _ in groups:
            ttnn.synchronize_device(half)
        return (time.time() - t0) * 1000

    lanes_sdpa = os.environ.get("GEMMA4_SLIDING_LANES", "1") != "0"
    for mix_name, mix in mixes.items():
        for b in range(1, max_lanes + 1):
            prefixes = mix(b)
            lane_sets = [[lane(s, p) for s, p in enumerate(prefixes)] for _, _, lane in groups]
            step_both(lane_sets)
            both = statistics.median(step_both(lane_sets) for _ in range(reps))
            alone = groups[0][1].timed(lane_sets[0])
            logger.info(
                f"[sp4x2] PERF chunk={chunk_size} lanes_sdpa={int(lanes_sdpa)} per_half={b} total={2 * b} mix={mix_name} "
                f"prefixes={prefixes} step={both:.1f}ms one_half_alone={alone:.1f}ms "
                f"tok/s={2 * b * chunk_size * 1000 / both:.0f}"
            )
    for _, runner, _ in groups:
        runner.release()
