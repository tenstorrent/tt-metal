# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory real-weight GDN full advertised context proof.

TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m pytest -s \
  models/autoports/qwen_qwen3_8_flash_next/tests/gdn_long_context.py -k 'B1-layer0'

This reduced component harness preserves the real target configuration and
weights. It streams all 262143 prefix tokens, validates every chunk against HF,
checks prefill token262143 (total262144), then restores the exact prefix and
checks changed-input traced decode at position262143. No test advances beyond
the advertised context. Memory scales with chunk size and recurrent state.
"""

import copy
import json
import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.test_gdn import CHECKPOINT, real_gdn_weights
from models.autoports.qwen_qwen3_8_flash_next.tt.gdn import GatedDeltaNet


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["layer0", "layer1"])
@pytest.mark.parametrize(
    "batch",
    [int(x) for x in os.environ.get("QWEN38_GDN_LONG_BATCHES", "1,2").split(",")],
    ids=lambda value: f"B{value}",
)
def test_gdn_full_context_stream_and_final_trace(layer_idx, batch):
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    context = config.max_position_embeddings
    weights = real_gdn_weights(layer_idx)
    reference = Qwen4ExpTextGatedDeltaNet(config, layer_idx).to(torch.bfloat16).eval()
    reference.load_state_dict(weights, strict=True)
    cache = DynamicCache(config=config)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    records = []
    artifact = (
        Path(__file__).resolve().parents[1] / "doc/functional_decoder" / f"gdn_long_layer{layer_idx}_B{batch}.json"
    )

    def metric(expected, actual, *, mode, position, length):
        metrics = pcc_metrics(expected, actual)
        score = metrics["pcc"]
        result = {"mode": mode, "position": position, "length": length, **metrics}
        records.append(result)
        print(json.dumps(result), flush=True)
        assert score >= 0.995, result
        return score

    try:
        model = GatedDeltaNet.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh,
            batch_size=batch,
            max_requests=batch + 1,
            chunk_size=128,
        )

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh)

        def lane(values):
            value = torch.zeros(batch, 32, dtype=torch.int32)
            value[:, 0] = torch.tensor(values)
            return value

        slot_ids = list(range(1, batch + 1))
        slots = upload(lane(slot_ids), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        positions = upload(lane([0] * batch), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

        def set_position(position):
            host = ttnn.from_torch(lane([position] * batch), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, positions)

        def activation(length, seed):
            return torch.randn(batch, length, 2560, generator=torch.Generator().manual_seed(seed), dtype=torch.bfloat16)

        started = time.time()
        position = 0
        while position < context - 1:
            length = min(128, context - 1 - position)
            x = activation(length, 1000 + position)
            with torch.no_grad():
                expected = reference(x, cache_params=cache)
            xd = upload(x)
            set_position(position)
            output = model.prefill_forward(xd, current_positions=positions, slots=slots)
            metric(expected, ttnn.to_torch(output), mode="prefill", position=position, length=length)
            position += length
            # Persist progress, so a failed gate retains completed measurements.
            if position % 4096 == 0 or position == context - 1:
                metric(
                    cache.layers[layer_idx].recurrent_states[0],
                    ttnn.to_torch(model.recurrent_state),
                    mode="recurrent_state",
                    position=position,
                    length=0,
                )
                artifact.write_text(
                    json.dumps(
                        {
                            "complete": False,
                            "layer": layer_idx,
                            "batch": batch,
                            "advertised_context": context,
                            "validated_prefix": position,
                            "records": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            del xd, output

        assert position == context - 1
        prefix_hf = copy.deepcopy(cache)
        prefix_recurrent = ttnn.clone(model.recurrent_bank)
        prefix_conv = ttnn.clone(model.conv_bank)

        def restore_prefix():
            ttnn.copy(prefix_recurrent, model.recurrent_bank)
            ttnn.copy(prefix_conv, model.conv_bank)

        # The final valid token is exercised through both entrypoints. Neither
        # path overwrites the persistent snapshot used for replay restoration.
        x = activation(1, 8000)
        xd = upload(x)
        set_position(context - 1)
        with torch.no_grad():
            expected = reference(x, cache_params=copy.deepcopy(prefix_hf))
        output = model.prefill_forward(xd, current_positions=positions, slots=slots)
        metric(expected, ttnn.to_torch(output), mode="full_context_prefill", position=context - 1, length=1)
        restore_prefix()
        model.decode_forward(xd, current_positions=positions, slots=slots)
        restore_prefix()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced_output = model.decode_forward(xd, current_positions=positions, slots=slots)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        replay_results = []
        for repeat in range(4):
            # Last replay repeats the first input; others change it while all
            # begin with the same full-length prefix and the same trace ID.
            x = activation(1, 8000 + repeat % 3)
            with torch.no_grad():
                expected = reference(x, cache_params=copy.deepcopy(prefix_hf))
            host = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, xd)
            restore_prefix()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(traced_output)
            metric(expected, actual, mode="full_context_traced_decode", position=context - 1, length=1)
            if repeat == 3:
                torch.testing.assert_close(actual, replay_results[0], rtol=0, atol=0)
            replay_results.append(actual.clone())
        result = {
            "complete": True,
            "layer": layer_idx,
            "batch": batch,
            "weights": "real",
            "advertised_context": context,
            "largest_prefill_context": context,
            "largest_decode_context": context,
            "long_non_aligned_prefix": context - 1,
            "trace_id": str(trace),
            "capture_count": 1,
            "replays": 4,
            "wall_seconds": time.time() - started,
            "minimum_pcc": min(r["pcc"] for r in records),
            "records": records,
        }
        artifact.write_text(json.dumps(result, indent=2) + "\n")
        print("GDN_FULL_ADVERTISED_CONTEXT_PREFILL_TRACED_DECODE_PASS", flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
