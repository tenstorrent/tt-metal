# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight reduction/replay validation before folding fabric into MLP."""

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
from models.demos.llama31_8b_qb2.tt.megakernel.mlp import FusedMLP
from models.demos.llama31_8b_qb2.tt.megakernel.reduce_scatter import CompactReduceScatter
from models.demos.llama31_8b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config


def test_compact_reduce_real_weights(qb2_mesh):
    mesh = qb2_mesh
    torch.set_num_threads(8)
    folder = checkpoint_path()
    checkpoint = Checkpoint(folder)
    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    layer = LlamaDecoder.from_state_dict(
        checkpoint.load([n for n in checkpoint.index if n.startswith("model.layers.0.")]),
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh,
        precision_policy=load_precision_config(),
        ccl=TT_CCL(mesh),
    )
    layer.prepare_decode(1)
    mlp = FusedMLP([layer])
    reduction = CompactReduceScatter(mesh, layer.local_residual_memcfg)
    embedding = checkpoint.load(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    results = []
    for token in (12345, 9876):
        hidden = embedding[token].float()
        value = (hidden * torch.rsqrt(hidden.square().mean() + config.rms_norm_eps)).bfloat16().reshape(1, 1, 1, 4096)
        normalized = ttnn.from_torch(
            value,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=layer.decode_inputs["gate_up"],
        )
        local = mlp(normalized, 0)
        expected = to_host(layer._rs(local, decode=True, site="down")).clone()
        actual = reduction(local)
        results.append({"token": token, **compare(to_host(actual), expected)})
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        reduction(local)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        try:
            for _ in range(12):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            torch.testing.assert_close(to_host(reduction.output), expected, rtol=0, atol=0)
        finally:
            ttnn.release_trace(mesh, trace)
    print(json.dumps(results, indent=2), flush=True)
    if destination := os.environ.get("QB2_MEGAKERNEL_ARTIFACT_DIR"):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / "compact-reduce.json").write_text(json.dumps(results, indent=2))
