# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LOCAL ONLY (lanes A/B, not for commit): request-parallel Gemma4 batched prefill.

The 8x4 Galaxy is split into two CP4 x TP4 submeshes, each with its own Gemma4 instance. A step of 4 requests
x 2048 tokens gives each instance 2 requests, run with the plain per-request ring SDPA (no lanes). Both
instances replay their traces at once; the step time is the wall time until both finish. Compare with
test_batched_prefill.py::test_batched_step_perf on the full 8x4 mesh (lanes).

Needs GEMMA4_ALLOW_SUBMESH=1 and a TT_CACHE_PATH with a tensor_cache_bf16_mesh4x4 directory.
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
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric

_MIXES = {
    "shallow": lambda b: [2048] * b,
    "u32k": lambda b: [32768] * b,
}


@torch.no_grad()
@pytest.mark.timeout(5400)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_request_parallel_step_perf(mesh_device, reset_seeds):
    chunk_size, capacity = 2048, 262144
    per_group = int(os.environ.get("RP_PER_GROUP", "2"))
    reps = int(os.environ.get("G4B_REPS", "5"))
    subs = mesh_device.create_submeshes(ttnn.MeshShape(4, 4))
    assert len(subs) == 2

    deepest = max(max(mix(per_group)) for mix in _MIXES.values()) + chunk_size
    stream = _text_token_stream(_hf_model_id())[0]
    num_seqs = 2 * per_group
    stride = min(100_000, (len(stream) - deepest) // max(1, num_seqs - 1))
    seqs = [stream[i * stride : i * stride + deepest].clone() for i in range(num_seqs)]

    groups = []
    for g, sub in enumerate(subs):
        mesh_config, _, model = build_batched_model(sub, chunk_size, per_group, capacity)
        runner = StepRunner(sub, mesh_config, model, chunk_size)
        mine = seqs[g * per_group : (g + 1) * per_group]

        def lane(slot, start, mine=mine):
            return (slot, start, mine[slot][start : start + chunk_size])

        runner.capture_all([[lane(0, 0)], [lane(s, 0) for s in range(per_group)]])
        for slot in range(per_group):
            for start in range(0, deepest - chunk_size, chunk_size):
                runner.step([lane(slot, start)])
        groups.append((sub, runner, lane))
        logger.info(f"[rp] group {g} ready: {mesh_config}")

    def step_all(lane_sets):
        for (_, runner, _), lanes in zip(groups, lane_sets):
            runner.stage(lanes)
        t0 = time.time()
        for (sub, runner, _), lanes in zip(groups, lane_sets):
            ttnn.execute_trace(sub, runner.traces[runner.layout(lanes)], cq_id=0, blocking=False)
        for sub, _, _ in groups:
            ttnn.synchronize_device(sub)
        return (time.time() - t0) * 1000

    for mix_name, mix in _MIXES.items():
        prefixes = mix(per_group)
        lane_sets = [[lane(s, p) for s, p in enumerate(prefixes)] for _, _, lane in groups]
        step_all(lane_sets)
        both = statistics.median(step_all(lane_sets) for _ in range(reps))
        alone = statistics.median(groups[0][1].step(lane_sets[0]) for _ in range(reps))
        logger.info(
            f"[rp] PERF chunk={chunk_size} B={2 * per_group} (2 groups x {per_group}) mix={mix_name} "
            f"step={both:.1f}ms one_group_alone={alone:.1f}ms"
        )
    for _, runner, _ in groups:
        runner.release()
