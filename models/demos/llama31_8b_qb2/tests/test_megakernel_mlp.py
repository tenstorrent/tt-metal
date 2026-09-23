# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Focused real-weight MLP stages and distinct address-table rows.

Real embedding-derived inputs support kernel bringup; this deliberately does
not replace complete decoder/model accuracy and paged-KV tests.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.common.modules.tt_ccl import TT_CCL
from models.demos.llama31_8b_qb2.tests.test_decoder import to_host
from models.demos.llama31_8b_qb2.tests.test_megakernel import compare
from models.demos.llama31_8b_qb2.tt.decoder import LlamaDecoder
from models.demos.llama31_8b_qb2.tt.megakernel.mlp import FusedMLP
from models.demos.llama31_8b_qb2.tt.model import Checkpoint, checkpoint_path
from models.demos.llama31_8b_qb2.tt.precision import load_precision_config


@pytest.mark.parametrize("gu_workers", [8, 16])
@pytest.mark.parametrize("reuse_scratch", [False, True], ids=["separate", "shared"])
def test_mlp_stages_real_weights(qb2_mesh, reuse_scratch, gu_workers, tuning=None, measure=False):
    mesh = qb2_mesh
    torch.set_num_threads(8)
    folder = checkpoint_path()
    checkpoint = Checkpoint(folder)
    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    layers = []
    workspace = None
    ccl = TT_CCL(mesh)
    for index in (0, 31):
        layer = LlamaDecoder.from_state_dict(
            checkpoint.load([name for name in checkpoint.index if name.startswith(f"model.layers.{index}.")]),
            hf_config=config,
            layer_idx=index,
            mesh_device=mesh,
            precision_policy=load_precision_config(),
            ccl=ccl,
            rope_state=layers[0] if layers else None,
        )
        workspace = layer.prepare_decode(1, workspace=workspace)
        layers.append(layer)
    assert layers[0].decode_weights["gate_up"].buffer_address() != layers[1].decode_weights["gate_up"].buffer_address()
    body = FusedMLP(layers, reuse_scratch=reuse_scratch, gu_workers=gu_workers, tuning=tuning)
    embedding = checkpoint.load(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    results = []
    timing = []
    destination = (
        Path(os.environ["QB2_MEGAKERNEL_ARTIFACT_DIR"]) if os.environ.get("QB2_MEGAKERNEL_ARTIFACT_DIR") else None
    )
    if destination:
        destination.mkdir(parents=True, exist_ok=True)

    def checked(actual, expected, *, layer, token, stage):
        a, b = to_host(actual), to_host(expected)
        try:
            result = compare(a, b)
        except AssertionError:
            if destination:
                torch.save(
                    {"actual": a, "expected": b, "layer": layer, "token": token, "stage": stage},
                    destination / f"mlp-failure-layer{layer}-{stage}-shared{reuse_scratch}-gu{gu_workers}.pt",
                )
            raise
        results.append({"layer": layer, "token": token, "stage": stage, **result})

    for row, layer in enumerate(layers):
        for token in (12345, 9876):
            hidden = embedding[token].float()
            value = (
                (hidden * torch.rsqrt(hidden.square().mean() + config.rms_norm_eps)).bfloat16().reshape(1, 1, 1, 4096)
            )
            normalized = ttnn.from_torch(
                value,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                memory_config=layer.decode_inputs["gate_up"],
            )
            packed = layer._decode_linear(normalized, "gate_up")
            interleaved = ttnn.to_memory_config(packed, ttnn.L1_MEMORY_CONFIG)
            gate, up = interleaved[:, :, :, :3584], interleaved[:, :, :, 3584:]
            product = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=gate.memory_config(),
                dtype=ttnn.bfloat16,
            )
            expected_down = layer._decode_linear(product, "down")
            body(normalized, row)
            ttnn.synchronize_device(mesh)
            # All three backing tensors remain resident after kernel CB pops.
            for stage, actual, expected in (
                ("gate_up", body.packed, packed),
                ("swiglu", body.product, product),
                ("down", body.output, expected_down),
            ):
                checked(actual, expected, layer=(0, 31)[row], token=token, stage=stage)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            body(normalized, row)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            try:
                saved = to_host(body.output).clone()
                for _ in range(5):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    torch.testing.assert_close(to_host(body.output), saved, rtol=0, atol=0)
                if measure:
                    for repeat in range(5):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(100):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        timing.append({"layer": (0, 31)[row], "token": token, "repeat": repeat,
                                       "replays": 100, "us": (time.perf_counter() - start) * 1e4})
            finally:
                ttnn.release_trace(mesh, trace)
    if destination:
        (destination / f"mlp-stage-comparison-shared{reuse_scratch}-gu{gu_workers}.json").write_text(
            json.dumps(results, indent=2)
        )
    print(json.dumps(results, indent=2), flush=True)

    return {"checks": results, "component_trace_timing": timing}
