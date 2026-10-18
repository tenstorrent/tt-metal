# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real target GDN weights: chunk/tail parity and changed-input decode traces.

Run serialized by the stage owner, e.g.:
TT_METAL_TRACE_ALLOC_TRACKING=1 python_env/bin/python -m pytest -s \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_gdn.py
This test opens its own 1x1 mesh. Do not run concurrently with other device jobs.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tt.gdn import GatedDeltaNet

CHECKPOINT = Path(
    os.environ.get(
        "QWEN38_CHECKPOINT",
        "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540",
    )
)


def real_gdn_weights(layer_idx):
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    keys = {k: shard for k, shard in index.items() if k.startswith(prefix)}
    assert keys, prefix
    result = {}
    for shard in sorted(set(keys.values())):
        with safe_open(CHECKPOINT / shard, framework="pt") as handle:
            for key, filename in keys.items():
                if filename == shard:
                    result[key[len(prefix) :]] = handle.get_tensor(key)
    return result


def assert_pcc(expected, actual, label):
    metrics = pcc_metrics(expected, actual)
    score = metrics["pcc"]
    print(json.dumps({"label": label, **metrics, "threshold": 0.995}), flush=True)
    assert score >= 0.995, (label, score)


@pytest.mark.parametrize("layer_idx", [0, 1])
@pytest.mark.parametrize("batch_size", [int(x) for x in os.environ.get("QWEN38_GDN_BATCHES", "1,2").split(",")])
def test_real_gdn_chunk_tails_and_trace(layer_idx, batch_size):
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    weights = real_gdn_weights(layer_idx)
    ref = Qwen4ExpTextGatedDeltaNet(cfg, layer_idx).to(torch.bfloat16).eval()
    ref.load_state_dict(weights, strict=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        model = GatedDeltaNet.from_state_dict(
            weights,
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh,
            batch_size=batch_size,
            max_requests=batch_size + 2,
        )

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

        def positions(value):
            out = torch.zeros(batch_size, 32, dtype=torch.int32)
            out[:, 0] = value
            return out

        # Increasing lengths on one loaded instance, then a short reuse. Every
        # length includes changed activations; both output and FP32 state tested.
        lengths = [int(x) for x in os.environ.get("QWEN38_GDN_LENGTHS", "1,31,32,33,127,128,129,259,3").split(",")]
        # Full advertised context is covered by gdn_long_context.py using a
        # streaming oracle, avoiding monolithic activation/mask allocations and
        # a follow-on decode beyond the last valid position.
        for seq_len in lengths:
            torch.manual_seed(1000 + seq_len)
            x = torch.randn(batch_size, seq_len, 2560, dtype=torch.bfloat16)
            cache = DynamicCache(config=cfg)
            with torch.no_grad():
                expected = ref(x, cache_params=cache)
            model.reset_state_inplace()
            actual = model.prefill_forward(upload(x))
            assert_pcc(expected, ttnn.to_torch(actual), f"layer{layer_idx}/B{batch_size}/prefill{seq_len}")
            assert_pcc(
                cache.layers[layer_idx].recurrent_states[0],
                ttnn.to_torch(model.recurrent_state),
                f"layer{layer_idx}/B{batch_size}/state{seq_len}",
            )
            # Follow-on decode must consume the exact logical tail, not padding.
            next_x = torch.randn(batch_size, 1, 2560, dtype=torch.bfloat16)
            with torch.no_grad():
                expected_next = ref(next_x, cache_params=cache)
            next_out = model.decode_forward(upload(next_x))
            assert_pcc(expected_next, ttnn.to_torch(next_out), f"layer{layer_idx}/B{batch_size}/follow{seq_len}")

        torch.manual_seed(600)
        decode_input = upload(torch.zeros(batch_size, 1, 2560, dtype=torch.bfloat16))
        position_input = upload(positions(0), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        slot_ids = [batch_size + 1, *range(1, batch_size)]
        slot_host = positions(0)
        slot_host[:, 0] = torch.tensor(slot_ids)
        slot_input = upload(slot_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        # Compile dynamic reset path and recurrence before capture. Persistent
        # input/state buffers stay alive until trace release.
        model.decode_forward(decode_input, current_positions=position_input, slots=slot_input)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        decode_output = model.decode_forward(decode_input, current_positions=position_input, slots=slot_input)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        caches = {slot: DynamicCache(config=cfg) for slot in slot_ids}
        replay_outputs = []
        for step in range(6):
            # Step 3 starts a new request using the same trace and state buffers;
            # steps 3..5 exactly repeat 0..2, proving deterministic fresh reuse.
            torch.manual_seed(600 + step % 3)
            x = torch.randn(batch_size, 1, 2560, dtype=torch.bfloat16)
            if step == 3:
                caches = {slot: DynamicCache(config=cfg) for slot in slot_ids}
            order = slot_ids if step % 3 != 1 else list(reversed(slot_ids))
            with torch.no_grad():
                expected = torch.cat(
                    [ref(x[lane : lane + 1], cache_params=caches[slot]) for lane, slot in enumerate(order)], dim=0
                )
            host_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            host_pos = ttnn.from_torch(positions(step % 3), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_x, decode_input)
            ttnn.copy_host_to_device_tensor(host_pos, position_input)
            slot_host[:, 0] = torch.tensor(order)
            host_slot = ttnn.from_torch(slot_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_slot, slot_input)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(decode_output)
            assert_pcc(expected, actual, f"layer{layer_idx}/B{batch_size}/trace{step}")
            if step >= 3:
                torch.testing.assert_close(actual, replay_outputs[step - 3], rtol=0, atol=0)
            replay_outputs.append(actual.clone())
        print("GDN_REAL_WEIGHT_CHUNK_TAIL_TRACE_RESET_DETERMINISM_PASS", flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
