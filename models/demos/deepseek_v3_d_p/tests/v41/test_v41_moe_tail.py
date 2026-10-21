# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The decode MoE tail (``tt/v41/moe_tail.py``) against ``post_combine_reduce`` and a torch reference on one chip's
inputs.

No weights: a random combine buffer ``[1, 1, 32, 6, 5120]`` whose rows that are off (the slot's expert outside the
chip's dispatch group, the pad rows' sentinel) hold NaN, random bf16 scores, random indices with the first ``REAL``
rows real and the rest the sentinel, and the production expert dispatch table (sharded as ``TtMoe`` shards it).
Bars: the new sum within 2 bf16 ulps of the largest term (the fp32 sum rounded once) against the fp32 torch
reference on every element, the old op within 8 (it accumulates in bf16 in L1, one rounding per slot), no NaN,
the pad rows zero. Both ops are timed eagerly; the Tracy profile is the
measure.

Run: pytest models/demos/deepseek_v3_d_p/tests/v41/test_v41_moe_tail.py -k torus-xy-8x4
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.v41.moe_tail import moe_tail

E, K, H, T, REAL = 384, 6, 5120, 32, 4
ITERS = 20

MESHES = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
] + galaxy_meshes()


def _replicated(mesh_device, host, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _time(mesh_device, fn, iters: int = ITERS) -> float:
    fn()
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=True)
def test_v41_moe_tail(mesh_device, device_params):
    sp, tp = tuple(mesh_device.shape)
    torch.manual_seed(11)
    table = ExpertMapping.create_dispatch_table(E, sp, tp)  # [tp, E + 1]
    tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, table, dispatch_axis=0)
    indices = torch.full((T, K), E, dtype=torch.int64)
    indices[:REAL] = torch.stack([torch.randperm(E)[:K] for _ in range(REAL)])
    scores = (torch.rand(T, K) * 0.5).bfloat16()
    scores[REAL:] = 0
    combine = torch.randn(T, K, H).bfloat16()
    tt_indices = _replicated(mesh_device, indices.to(torch.int32), ttnn.uint16)
    tt_scores = _replicated(mesh_device, scores, ttnn.bfloat16)

    # per chip: the rows that are on, the reference sum, the NaN-poisoned combine buffer
    per_group_combine, reference = [], []
    for g in range(tp):
        on = table[g][indices] >= 0  # [T, K]
        poisoned = combine.clone()
        poisoned[~on] = float("nan")
        per_group_combine.append(poisoned)
        terms = scores.float().unsqueeze(-1) * combine.float() * on.unsqueeze(-1)  # [T, K, H]
        reference.append((terms.sum(dim=1), terms.abs().max(dim=1).values.clamp(min=2**-14)))  # [T, H] each
    combine_host = torch.stack([c for _ in range(sp) for c in per_group_combine])  # [sp * tp, T, K, H]
    tt_combine = ttnn.from_torch(
        combine_host.reshape(sp, tp, T, K, H),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, 1)),
    )
    tt_combine = ttnn.reshape(tt_combine, (1, 1, T, K, H))

    new = moe_tail(tt_combine, tt_scores, tt_indices, tt_table)
    assert tuple(new.shape) == (1, 1, T, H) and new.layout == ttnn.TILE_LAYOUT, (tuple(new.shape), new.layout)
    old = ttnn.experimental.deepseek_prefill.post_combine_reduce(
        ttnn.reshape(tt_combine, (1, 1, T, K, H)),
        ttnn.reshape(tt_scores, (1, 1, T, K, 1)),
        ttnn.reshape(tt_indices, (1, T, K)),
        tt_table,
        expert_dim=3,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    worst_new = worst_old = 0.0
    for chip, (n, o) in enumerate(zip(ttnn.get_device_tensors(new), ttnn.get_device_tensors(old))):
        ref, scale = reference[chip % tp]
        n, o = ttnn.to_torch(n).float().reshape(T, H), ttnn.to_torch(o).float().reshape(T, H)
        assert not torch.isnan(n).any(), f"chip {chip}: NaN in the new sum"
        assert (n[REAL:] == 0).all(), f"chip {chip}: pad rows not zero"
        ulp = scale * 2**-7  # a bf16 ulp of the largest term of the sum (the sum itself may cancel)
        err_n, err_o = ((n - ref).abs() / ulp).max().item(), ((o - ref).abs() / ulp).max().item()
        worst_new, worst_old = max(worst_new, err_n), max(worst_old, err_o)
        assert err_n <= 2, f"chip {chip}: new sum {err_n:.1f} ulp from the reference"
    logger.info(f"worst error against the fp32 reference, in bf16 ulps: new {worst_new:.2f}, old {worst_old:.2f}")
    assert worst_old <= 8, worst_old

    t_new = _time(mesh_device, lambda: moe_tail(tt_combine, tt_scores, tt_indices, tt_table))
    soak = int(os.environ.get("V41_TAIL_SOAK", 0))
    if soak:  # many calls back to back: a per-call accumulation in the kernel's NoC state would hang here
        for i in range(soak):
            moe_tail(tt_combine, tt_scores, tt_indices, tt_table)
            if i % 100 == 99:
                ttnn.synchronize_device(mesh_device)
                logger.info(f"soak: {i + 1} calls")
    t_old = _time(
        mesh_device,
        lambda: ttnn.experimental.deepseek_prefill.post_combine_reduce(
            tt_combine,
            ttnn.reshape(tt_scores, (1, 1, T, K, 1)),
            ttnn.reshape(tt_indices, (1, T, K)),
            tt_table,
            expert_dim=3,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
    logger.info(f"eager us per call: post_combine_reduce {t_old:.0f} -> moe_tail {t_new:.0f}")


@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=64 * 1024 * 1024), indirect=True)
def test_v41_moe_tail_traced(mesh_device, device_params):
    """``moe_tail`` under trace: 40 calls captured (a step's worth), the capture replayed back to back
    ``V41_TAIL_REPLAYS`` times (default 200) with a device sync every 10; a hang here is the kernel's own."""
    sp, tp = tuple(mesh_device.shape)
    torch.manual_seed(11)
    table = ExpertMapping.create_dispatch_table(E, sp, tp)
    tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, table, dispatch_axis=0)
    indices = torch.full((T, K), E, dtype=torch.int64)
    indices[:REAL] = torch.stack([torch.randperm(E)[:K] for _ in range(REAL)])
    tt_indices = _replicated(mesh_device, indices.to(torch.int32), ttnn.uint16)
    tt_scores = _replicated(mesh_device, (torch.rand(T, K) * 0.5).bfloat16(), ttnn.bfloat16)
    tt_combine = ttnn.reshape(_replicated(mesh_device, torch.randn(T, K, H).bfloat16(), ttnn.bfloat16), (1, 1, T, K, H))
    calls, replays = 40, int(os.environ.get("V41_TAIL_REPLAYS", 200))
    for _ in range(2):  # compile and warm the program cache
        moe_tail(tt_combine, tt_scores, tt_indices, tt_table)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [moe_tail(tt_combine, tt_scores, tt_indices, tt_table) for _ in range(calls)]
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for i in range(replays):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        if i % 10 == 9:
            ttnn.synchronize_device(mesh_device)
            logger.info(f"traced: {(i + 1) * calls} calls in {time.perf_counter() - t0:.2f} s")
    ttnn.synchronize_device(mesh_device)
    ref = ttnn.to_torch(ttnn.get_device_tensors(outs[0])[0]).float()
    for o in outs[1:]:
        assert torch.equal(ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float(), ref)
    ttnn.release_trace(mesh_device, tid)
