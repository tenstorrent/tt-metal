# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure the actual prepared MoE input after the whole-layer trace is released."""
import statistics
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.router_topk import aten_router_topk


def measure(model, call, mesh, repeats=16):
    original = model.mlp
    captured = []

    class Capture:
        def __call__(self, value):
            captured.append(value)
            return original(value)

    model.mlp = Capture()
    try:
        output = call()
        ttnn.synchronize_device(mesh)
        del output
    finally:
        model.mlp = original
    assert len(captured) == 1, "Expected one grouped MoE execution"
    x = captured.pop()
    logits = ttnn.typecast(original.router(x), ttnn.float32)
    _, ids = aten_router_topk(logits)
    route = ttnn.to_torch(ids, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0].reshape(x.shape[-2], 10).long()
    del ids, logits
    warmed = original(x)
    eager = ttnn.to_torch(warmed, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    del warmed
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    output = original(x)
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    from ttnn.tools import trace_allocation_tracker

    trace_allocation_tracker.acknowledge_corruptible(output)
    samples = []
    try:
        for _ in range(repeats + 2):
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            samples.append((time.perf_counter() - start) * 1000)
        actual = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        assert torch.equal(eager, actual)
    finally:
        ttnn.release_trace(mesh, trace)
    return dict(
        batch=x.shape[-2],
        traced_ms=statistics.median(samples[2:]),
        samples_ms=samples[2:],
        exact_replay=True,
        selected_ids=route.tolist(),
        expert_union=route.unique().numel(),
        unique_routing_rows=torch.unique(route, dim=0).shape[0],
        scope="router, active EP experts, shared expert, score/reduction, final CCL; isolated trace of actual decoder intermediate",
    )
