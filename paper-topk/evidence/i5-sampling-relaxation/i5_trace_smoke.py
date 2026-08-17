#!/usr/bin/env python3
"""I5 trace smoke: Sampling1D decode_forward (vocab 128256, 1x1 -> split path, RELAXED
routed ttnn.topk branch) under begin/end_trace_capture + replay. Mirrors
test_sampling1d_trace_capture[1x1-topk] (BH-skipped by the models/common conftest gate).
Asserts (1) the routed call form fires inside capture, (2) capture completes with no
in-capture cache miss, (3) replayed tokens == eager tokens."""

import sys

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.sampling.sampling_1d import Sampling1D

B = 32
VOCAB = 128256

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=32 << 20)
try:
    torch.manual_seed(0)
    logits_host = torch.randn(1, 1, B, VOCAB, dtype=torch.bfloat16)
    sampler = Sampling1D(vocab_size=VOCAB, mesh_device=mesh, max_batch_size=B, allow_force_argmax=True)

    mk = lambda vals, dt: ttnn.from_torch(
        vals, device=mesh, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k = mk(torch.full((B,), 1, dtype=torch.int32), ttnn.uint32)
    p = mk(torch.full((B,), 0.0).bfloat16(), ttnn.bfloat16)
    temp = mk(torch.full((B,), 1.0).bfloat16(), ttnn.bfloat16)

    logits_tt = ttnn.from_torch(logits_host, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # record ttnn.topk kwargs to prove the relaxed branch is the one being captured
    calls = []
    real_topk = ttnn.topk

    def recording_topk(*a, **kw):
        calls.append(sorted(kw.keys()))
        return real_topk(*a, **kw)

    ttnn.topk = recording_topk

    # warmup OUTSIDE capture (JIT + program cache), as the executor does
    sampler.load_device_buffers()
    eager_tok, _ = sampler.decode_forward(logits_tt, k=k, p=p, temp=temp)
    ttnn.synchronize_device(mesh)
    eager_host = to_torch_auto_compose(eager_tok).flatten()[:B].long()
    warmup_calls = list(calls)
    calls.clear()

    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        captured_tok, _ = sampler.decode_forward(logits_tt, k=k, p=p, temp=temp)
    finally:
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
    capture_calls = list(calls)
    ttnn.topk = real_topk

    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
    replay1 = to_torch_auto_compose(captured_tok).flatten()[:B].long()
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
    replay2 = to_torch_auto_compose(captured_tok).flatten()[:B].long()
    ttnn.release_trace(mesh, trace_id)

    print("warmup topk kwargs:  ", warmup_calls)
    print("capture topk kwargs: ", capture_calls)
    assert capture_calls == [["dim", "k"], ["dim", "k"]], "relaxed branch did not fire inside capture"
    assert torch.equal(replay1, eager_host), f"replay1 != eager\n{replay1[:8]}\n{eager_host[:8]}"
    assert torch.equal(replay2, eager_host), f"replay2 != eager\n{replay2[:8]}\n{eager_host[:8]}"
    print("TRACE SMOKE PASSED: routed branch captured; replay x2 == eager")
finally:
    ttnn.close_mesh_device(mesh)
