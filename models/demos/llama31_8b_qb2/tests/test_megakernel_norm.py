# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint RMSNorm and repeated replay against the selected native path."""

import json
import os
from pathlib import Path
import torch
from transformers import AutoConfig
import ttnn
from models.common.modules.tt_ccl import TT_CCL
from models.demos.llama31_8b_qb2.tests.test_decoder import to_host
from models.demos.llama31_8b_qb2.tests.test_megakernel import compare
from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from models.demos.llama31_8b_qb2.tt.megakernel.norm import FusedNorm
from models.demos.llama31_8b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config


def test_native_norm_real_weights(qb2_mesh):
    mesh = qb2_mesh
    torch.set_num_threads(8)
    folder = checkpoint_path()
    checkpoint = Checkpoint(folder)
    layer = LlamaDecoder.from_state_dict(
        checkpoint.load([n for n in checkpoint.index if n.startswith("model.layers.0.")]),
        hf_config=AutoConfig.from_pretrained(folder, local_files_only=True),
        layer_idx=0,
        mesh_device=mesh,
        precision_policy=load_precision_config(),
        ccl=TT_CCL(mesh),
    )
    layer.prepare_decode(1)
    body = FusedNorm(mesh, layer.residual_memcfg, layer.eps, debug=bool(os.environ.get("QB2_NORM_DEBUG")))
    embedding = checkpoint.load(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    rows = []
    for token in [12345, 9876, 1]:
        x = ttnn.from_torch(
            embedding[token].reshape(1, 1, 1, 4096),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=layer.residual_memcfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        expected = to_host(layer._decode_norm(x))
        actual = body(x)
        measured = to_host(actual)
        if body.debug:
            torch.save(
                {
                    "input": embedding[token],
                    "actual": measured,
                    "expected": expected,
                    **{str(k): to_host(v) for k, v in body.debug.items()},
                },
                Path(os.environ["QB2_MEGAKERNEL_ARTIFACT_DIR"]) / f"norm-debug-{token}.pt",
            )
        try:
            result = compare(measured, expected)
        except AssertionError:
            torch.save(
                {"actual": measured, "expected": expected},
                Path(os.environ["QB2_MEGAKERNEL_ARTIFACT_DIR"]) / f"norm-failure-{token}.pt",
            )
            raise
        rows.append({"token": token, **result})
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        body(x)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        try:
            for _ in range(8):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            torch.testing.assert_close(to_host(actual), expected, rtol=0, atol=0)
        finally:
            ttnn.release_trace(mesh, trace)
    dest = Path(os.environ["QB2_MEGAKERNEL_ARTIFACT_DIR"])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "norm.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
