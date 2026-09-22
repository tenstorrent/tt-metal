# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real terminal weights: native traced final norm/head versus fused norm/head."""

import json
import os
from pathlib import Path

import torch
import ttnn
from models.demos.llama31_8b_qb2.tests.test_decoder import to_host
from models.demos.llama31_8b_qb2.tests.test_megakernel import compare, copy_to
from models.demos.llama31_8b_qb2.tt.megakernel.head import FusedHead
from models.demos.llama31_8b_qb2.tt.model import LlamaModel, Checkpoint


def test_head_real_weights_replay(qb2_mesh):
    torch.set_num_threads(8)
    mesh = qb2_mesh
    model = LlamaModel(mesh, max_batch_size=1)
    body = FusedHead(model)
    embedding = Checkpoint(model.folder).load(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    source = ttnn.from_torch(
        embedding[12345].reshape(1, 1, 1, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=model.layers[0].residual_memcfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def baseline():
        norm = model.layers[0]._decode_norm(source)
        return model.lm_head(ttnn.to_memory_config(norm, model.lm_head.config.input_memcfg))

    calls = (baseline, lambda: body(source))
    for call in calls:
        call()
    ttnn.synchronize_device(mesh)
    traces, outputs, results = [], [], []
    destination = Path(os.environ["QB2_MEGAKERNEL_ARTIFACT_DIR"])
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for call in calls:
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            outputs.append(call())
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces.append(trace)
        for token in (12345, 9876, 1):
            copy_to(embedding[token].reshape(1, 1, 1, 4096), source, mesh)
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual, expected = to_host(outputs[1]), to_host(outputs[0])
            try:
                result = compare(actual, expected)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            except AssertionError:
                torch.save({"actual": actual, "expected": expected}, destination / f"head-failure-{token}.pt")
                raise
            for _ in range(8):
                ttnn.execute_trace(mesh, traces[1], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            torch.testing.assert_close(to_host(outputs[1]), expected, rtol=0, atol=0)
            results.append({"token": token, **result})
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)
    (destination / "head.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
