# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Text encoder trace test — verify traced output matches non-traced,
then measure per-iteration latency with Metal Trace.
"""

import time

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.z_image_turbo.tt.text_encoder.model_ttnn import TextEncoderTTNN

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CAP_TOKENS = 128
DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _to_device_int32(pt, mesh_device):
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _copy_to_persistent_int32(host_pt, persistent_tt):
    host_tt = ttnn.from_torch(
        host_pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
    )
    for shard in ttnn.get_device_tensors(persistent_tt):
        ttnn.copy_host_to_device_tensor(host_tt, shard, cq_id=0)


def _tt_to_torch(tt_tensor, mesh_device):
    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return host[: host.shape[0] // 4].float()


def pcc(a, b):
    a_flat = a.flatten().double()
    b_flat = b.flatten().double()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    den = a_centered.norm() * b_centered.norm()
    return (num / den).item() if den > 0 else 0.0


def _tokenize_prompt(prompt):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(formatted, padding="max_length", truncation=True, max_length=CAP_TOKENS, return_tensors="pt")[
        "input_ids"
    ]


@pytest.fixture(scope="function")
def device_params(request):
    return {"l1_small_size": 1 << 15, "trace_region_size": 60_000_000}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_text_encoder_trace(mesh_device):
    mesh_device.enable_program_cache()

    prompt = "a beautiful sunset over the ocean"
    input_ids_pt = _tokenize_prompt(prompt)

    model = TextEncoderTTNN(mesh_device, seq_len=CAP_TOKENS)

    # 1. Compile run (populates program cache)
    tt_ids = _to_device_int32(input_ids_pt, mesh_device)
    tt_out = model(tt_ids)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(tt_out, False)
    ttnn.deallocate(tt_ids, False)

    # 2. Non-traced golden reference
    tt_ids = _to_device_int32(input_ids_pt, mesh_device)
    tt_out = model(tt_ids)
    ttnn.synchronize_device(mesh_device)
    golden = _tt_to_torch(tt_out, mesh_device)
    ttnn.deallocate(tt_out, False)
    ttnn.deallocate(tt_ids, False)

    # 3. Capture trace
    input_buf = _to_device_int32(input_ids_pt, mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model(input_buf)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Verify traced output matches golden
    traced_result = _tt_to_torch(trace_output, mesh_device)
    correlation = pcc(golden, traced_result)
    assert correlation > 0.999, f"Trace vs golden PCC too low: {correlation:.6f}"

    # 4. Timed trace executions
    num_runs = 3
    timings = []
    for _ in range(num_runs):
        _copy_to_persistent_int32(input_ids_pt, input_buf)
        ttnn.synchronize_device(mesh_device)

        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        timings.append((time.perf_counter() - t0) * 1000)

    # Final correctness check
    final_result = _tt_to_torch(trace_output, mesh_device)
    final_pcc = pcc(golden, final_result)
    assert final_pcc > 0.999, f"Final trace vs golden PCC too low: {final_pcc:.6f}"

    avg_ms = sum(timings) / len(timings)
    print(f"\nText encoder trace: avg={avg_ms:.1f} ms over {num_runs} runs")

    # Cleanup
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(input_buf, False)
