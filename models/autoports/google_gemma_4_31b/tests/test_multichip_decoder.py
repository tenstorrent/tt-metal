# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.autoports.google_gemma_4_31b.tests.test_functional_decoder import (
    LAYER_KINDS,
    _assert_pcc,
    _layer_state,
    _rope_host,
)
from models.autoports.google_gemma_4_31b.tests.test_optimized_decoder import _paged_state as _single_paged_state
from models.autoports.google_gemma_4_31b.tt.multichip_decoder import (
    MultichipDecoder,
    PAGE_BLOCK_SIZE,
    TARGET_MESH_SHAPE,
    TP_SIZE,
    _TPOptimizedSharedMLP,
)
from models.autoports.google_gemma_4_31b.tt.optimized_decoder import DEFAULT_OPTIMIZATION_POLICY, OptimizedDecoder

PCC_THRESHOLD = 0.995


def _checkpoint_dir() -> Path:
    root = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-31B/snapshots"
    snapshots = sorted(root.glob("*"))
    if not snapshots:
        pytest.skip("real google/gemma-4-31B checkpoint is not cached")
    return snapshots[-1]


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(_checkpoint_dir(), trust_remote_code=True, local_files_only=True).text_config


def _tt_input(tensor, mesh_device):
    device_shape = (
        tensor.transpose(0, 1).unsqueeze(0)
        if tensor.shape[0] > 1 and tensor.shape[1] == 1
        else tensor.reshape(1, 1, tensor.shape[1], tensor.shape[2])
    )
    return ttnn.from_torch(
        device_shape,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=(ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None),
    )


def _rope_device(hf_config, layer_idx, max_seq_len, mesh_device, *, decode):
    cos, sin = _rope_host(hf_config, layer_idx, torch.arange(max_seq_len))
    if decode:
        cos, sin = cos.squeeze(0), sin.squeeze(0)
        layout = ttnn.ROW_MAJOR_LAYOUT
    else:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        layout = ttnn.TILE_LAYOUT
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    return tuple(
        ttnn.from_torch(
            table,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=layout,
            mesh_mapper=mapper,
        )
        for table in (cos, sin)
    )


def _to_host_first(tensor):
    device = tensor.device()
    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        tensor = ttnn.get_device_tensors(tensor)[0]
    return ttnn.to_torch(tensor).reshape(1, tensor.shape[-2], tensor.shape[-1]).float()


def _replicated_int(host, mesh_device, *, dtype):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=(ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None),
    )


def _copy_host_to_replicas(host_tensor, mesh_tensor):
    for device_tensor in ttnn.get_device_tensors(mesh_tensor):
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)


def _permuted_page_table(page_table, mesh_device):
    """Return a replicated non-identity table over the same physical blocks."""
    rows = torch.arange(page_table.shape[0] * page_table.shape[1], dtype=torch.int32).reshape(
        page_table.shape[0], page_table.shape[1]
    )
    rows = torch.roll(rows, shifts=1, dims=1)
    return _replicated_int(rows, mesh_device, dtype=ttnn.int32)


def _inputs(hf_config, layer_idx):
    torch.manual_seed(20260714 + layer_idx)
    return {
        "prefill33": torch.randn(1, 33, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        "prompt32": torch.randn(1, 32, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        "token": torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
    }


@pytest.fixture(scope="module")
def optimized_baseline(hf_config):
    """Collect the frozen single-P150 reference before enabling mesh fabric."""
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20)
    results = {}
    try:
        for layer_idx, layer_kind in LAYER_KINDS:
            values = _inputs(hf_config, layer_idx)
            state = _layer_state(layer_idx)
            decoder = OptimizedDecoder.from_state_dict(
                state, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh
            )
            cache, page_table = _single_paged_state(hf_config, layer_idx, 128, mesh)
            out = decoder.prefill_forward(
                _tt_input(values["prefill33"], mesh),
                rope_mats=_rope_device(hf_config, layer_idx, 33, mesh, decode=False),
                page_table=page_table,
                kv_cache=cache,
                valid_seq_len=33,
            )
            results[(layer_kind, "prefill33")] = _to_host_first(out).clone()

            cache, page_table = _single_paged_state(hf_config, layer_idx, 128, mesh)
            decoder.prefill_forward(
                _tt_input(values["prompt32"], mesh),
                rope_mats=_rope_device(hf_config, layer_idx, 32, mesh, decode=False),
                page_table=page_table,
                kv_cache=cache,
                valid_seq_len=32,
            )
            pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
            pos_u_host[0, 0] = 32
            out = decoder.decode_forward(
                _tt_input(values["token"], mesh),
                rope_mats=_rope_device(hf_config, layer_idx, 128, mesh, decode=True),
                page_table=page_table,
                kv_cache=cache,
                current_position=_replicated_int(pos_u_host, mesh, dtype=ttnn.uint32),
                current_position_cache=_replicated_int(torch.tensor([32], dtype=torch.int32), mesh, dtype=ttnn.int32),
            )
            results[(layer_kind, "decode32")] = _to_host_first(out).clone()

            batch_size, batch_prompt_len = 32, 33
            torch.manual_seed(20261900 + layer_idx)
            batch_prompt = (
                torch.randn(
                    batch_size,
                    batch_prompt_len,
                    hf_config.hidden_size,
                    dtype=torch.bfloat16,
                )
                * 0.02
            )
            batch_token = torch.randn(batch_size, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
            cache, page_table = _single_paged_state(
                hf_config,
                layer_idx,
                128,
                mesh,
                batch_size=batch_size,
            )
            batch_prefill = decoder.prefill_forward(
                _tt_input(
                    batch_prompt.reshape(1, batch_size * batch_prompt_len, hf_config.hidden_size),
                    mesh,
                ),
                rope_mats=_rope_device(hf_config, layer_idx, batch_prompt_len, mesh, decode=False),
                page_table=page_table,
                kv_cache=cache,
                batch_size=batch_size,
                valid_seq_len=batch_prompt_len,
            )
            assert batch_prefill.shape[-2] == batch_size * batch_prompt_len
            batch_pos_host = torch.full((1, batch_size), batch_prompt_len, dtype=torch.int32)
            batch_decoded = decoder.decode_forward(
                _tt_input(batch_token, mesh),
                rope_mats=_rope_device(hf_config, layer_idx, 128, mesh, decode=True),
                page_table=page_table,
                kv_cache=cache,
                current_position=_replicated_int(batch_pos_host, mesh, dtype=ttnn.uint32),
                current_position_cache=_replicated_int(
                    torch.full((batch_size,), batch_prompt_len, dtype=torch.int32),
                    mesh,
                    dtype=ttnn.int32,
                ),
                batch_size=batch_size,
            )
            results[(layer_kind, "batch32_decode33")] = (
                ttnn.to_torch(batch_decoded).reshape(batch_size, 1, hf_config.hidden_size).float().clone()
            )

            if layer_kind == "sliding_attention":
                for seq_len in (1025, 1057):
                    torch.manual_seed(20262000 + seq_len)
                    prompt = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
                    token = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
                    cache, page_table = _single_paged_state(hf_config, layer_idx, seq_len + 1, mesh)
                    prefill = decoder.prefill_forward(
                        _tt_input(prompt, mesh),
                        rope_mats=_rope_device(hf_config, layer_idx, seq_len, mesh, decode=False),
                        page_table=page_table,
                        kv_cache=cache,
                        valid_seq_len=seq_len,
                    )
                    assert prefill.shape[-2] == seq_len
                    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
                    pos_u_host[0, 0] = seq_len
                    decoded = decoder.decode_forward(
                        _tt_input(token, mesh),
                        rope_mats=_rope_device(hf_config, layer_idx, seq_len + 1, mesh, decode=True),
                        page_table=page_table,
                        kv_cache=cache,
                        current_position=_replicated_int(pos_u_host, mesh, dtype=ttnn.uint32),
                        current_position_cache=_replicated_int(
                            torch.tensor([seq_len], dtype=torch.int32),
                            mesh,
                            dtype=ttnn.int32,
                        ),
                    )
                    results[(layer_kind, f"decode{seq_len}")] = _to_host_first(decoded).clone()

            if "GEMMA4_MULTICHIP_BENCH" in os.environ:
                torch.manual_seed(20262100 + layer_idx)
                prefill_host = torch.randn(1, 128, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
                cache, page_table = _single_paged_state(hf_config, layer_idx, 128, mesh)
                prefill_input = _tt_input(prefill_host, mesh)
                prefill_rope = _rope_device(hf_config, layer_idx, 128, mesh, decode=False)

                def prefill_call():
                    return decoder.prefill_forward(
                        prefill_input,
                        rope_mats=prefill_rope,
                        page_table=page_table,
                        kv_cache=cache,
                        valid_seq_len=128,
                    )

                warm = prefill_call()
                ttnn.synchronize_device(mesh)
                warm.deallocate(True)
                prefill_samples = []
                for _ in range(12):
                    start = time.perf_counter_ns()
                    sample = prefill_call()
                    ttnn.synchronize_device(mesh)
                    prefill_samples.append((time.perf_counter_ns() - start) / 1_000_000)
                    sample.deallocate(True)

                baseline_token = _tt_input(values["token"], mesh)
                baseline_decode_rope = _rope_device(hf_config, layer_idx, 128, mesh, decode=True)
                baseline_pos_host = torch.zeros((1, 32), dtype=torch.int32)
                baseline_pos_host[0, 0] = 32
                baseline_pos_u = _replicated_int(baseline_pos_host, mesh, dtype=ttnn.uint32)
                baseline_pos_i = _replicated_int(
                    torch.tensor([32], dtype=torch.int32),
                    mesh,
                    dtype=ttnn.int32,
                )

                def decode_call():
                    return decoder.decode_forward(
                        baseline_token,
                        rope_mats=baseline_decode_rope,
                        page_table=page_table,
                        kv_cache=cache,
                        current_position=baseline_pos_u,
                        current_position_cache=baseline_pos_i,
                    )

                warm = decode_call()
                ttnn.synchronize_device(mesh)
                warm.deallocate(True)
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                decode_call()
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                try:
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    decode_samples = []
                    for _ in range(12):
                        start = time.perf_counter_ns()
                        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                        decode_samples.append((time.perf_counter_ns() - start) / 1_000_000)
                finally:
                    ttnn.release_trace(mesh, trace_id)
                results[(layer_kind, "baseline_latency")] = {
                    "prefill128_ms": statistics.median(prefill_samples),
                    "prefill128_samples_ms": prefill_samples,
                    "decode_ms": statistics.median(decode_samples),
                    "decode_samples_ms": decode_samples,
                }
    finally:
        ttnn.close_mesh_device(mesh)
    return results


@pytest.fixture(scope="module")
def mesh_device(optimized_baseline):
    if ttnn.get_num_devices() < TP_SIZE:
        pytest.skip("Gemma 4 multichip decoder requires four local devices")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*TARGET_MESH_SHAPE), trace_region_size=96 << 20)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def profile_mesh_device():
    """Open the target directly so Tracy does not record baseline setup ops."""
    if ttnn.get_num_devices() < TP_SIZE:
        pytest.skip("Gemma 4 multichip decoder requires four local devices")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*TARGET_MESH_SHAPE), trace_region_size=96 << 20)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


def _decoder(hf_config, mesh_device, layer_idx):
    decoder = MultichipDecoder.from_state_dict(
        _layer_state(layer_idx),
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        num_links=int(os.environ.get("GEMMA4_MC_NUM_LINKS", "2")),
        qkv_decode_output_cores=int(os.environ.get("GEMMA4_MC_QKV_OUTPUT_CORES", "32")),
    )
    assert type(decoder) is MultichipDecoder
    assert type(decoder.layer.shared_mlp) is _TPOptimizedSharedMLP
    return decoder


def _mlp_geometry_policy(num_cores):
    block_w = {
        8: (7, 21),
        12: (7, 7),
        21: (4, 4),
        24: (7, 7),
    }
    gate_up_in0_block_w, down_in0_block_w = block_w[num_cores]
    return replace(
        DEFAULT_OPTIMIZATION_POLICY,
        name=f"tp4_square_mlp_{num_cores}c",
        decode_num_cores=num_cores,
        gate_up_in0_block_w=gate_up_in0_block_w,
        down_in0_block_w=down_in0_block_w,
    )


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_prefill_matches_optimized_baseline(
    hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind
):
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    output = decoder.prefill_forward(
        _tt_input(_inputs(hf_config, layer_idx)["prefill33"], mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, 33, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=33,
    )
    reference = optimized_baseline[(layer_kind, "prefill33")]
    device_outputs = [ttnn.to_torch(t).reshape_as(reference).float() for t in ttnn.get_device_tensors(output)]
    for actual in device_outputs:
        _assert_pcc(reference, actual, PCC_THRESHOLD)
    for actual in device_outputs[1:]:
        assert torch.equal(device_outputs[0], actual)


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_paged_decode_trace_matches_optimized_baseline(
    hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind
):
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    values = _inputs(hf_config, layer_idx)
    cache, identity_page_table = decoder.init_paged_kv_cache(max_context=128)
    page_table = _permuted_page_table(identity_page_table, mesh_device)
    decoder.prefill_forward(
        _tt_input(values["prompt32"], mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, 32, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=32,
    )
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = 32
    pos_u = _replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32)
    pos_i = _replicated_int(torch.tensor([32], dtype=torch.int32), mesh_device, dtype=ttnn.int32)
    token = _tt_input(values["token"], mesh_device)
    rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=True)

    def call():
        return decoder.decode_forward(
            token,
            rope_mats=rope,
            page_table=page_table,
            kv_cache=cache,
            current_position=pos_u,
            current_position_cache=pos_i,
        )

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        first = _to_host_first(output).clone()
        _assert_pcc(optimized_baseline[(layer_kind, "decode32")], first, PCC_THRESHOLD)
        for _ in range(7):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            assert torch.equal(first, _to_host_first(output))

        changed_token_host = ttnn.from_torch(
            (values["token"] + 0.125).reshape(1, 1, 1, hf_config.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        changed_pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
        changed_pos_u_host[0, 0] = 33
        changed_pos_u = ttnn.from_torch(changed_pos_u_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        changed_pos_i = ttnn.from_torch(
            torch.tensor([33], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _copy_host_to_replicas(changed_token_host, token)
        _copy_host_to_replicas(changed_pos_u, pos_u)
        _copy_host_to_replicas(changed_pos_i, pos_i)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        changed = _to_host_first(output).clone()
        assert torch.isfinite(changed).all()
        assert not torch.equal(first, changed)

        original_token_host = ttnn.from_torch(
            values["token"].reshape(1, 1, 1, hf_config.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        original_pos_u = ttnn.from_torch(pos_u_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        original_pos_i = ttnn.from_torch(
            torch.tensor([32], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        _copy_host_to_replicas(original_token_host, token)
        _copy_host_to_replicas(original_pos_u, pos_u)
        _copy_host_to_replicas(original_pos_i, pos_i)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(first, _to_host_first(output))
        for replica in ttnn.get_device_tensors(output)[1:]:
            assert torch.equal(first, ttnn.to_torch(replica).reshape_as(first).float())
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("seq_len", [1025, 1057])
def test_multichip_sliding_nonaligned_window_wrap_matches_baseline(hf_config, mesh_device, optimized_baseline, seq_len):
    layer_idx, layer_kind = 0, "sliding_attention"
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    torch.manual_seed(20262000 + seq_len)
    prompt = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    token_host = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    cache, page_table = decoder.init_paged_kv_cache(max_context=seq_len + 1)
    output = decoder.prefill_forward(
        _tt_input(prompt, mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, seq_len, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=seq_len,
    )
    assert output.shape[-2] == seq_len
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = seq_len
    decoded = decoder.decode_forward(
        _tt_input(token_host, mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, seq_len + 1, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(
            torch.tensor([seq_len], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
        ),
    )
    reference = optimized_baseline[(layer_kind, f"decode{seq_len}")]
    for replica in ttnn.get_device_tensors(decoded):
        _assert_pcc(
            reference,
            ttnn.to_torch(replica).reshape_as(reference).float(),
            PCC_THRESHOLD,
        )


@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_batch32_nonaligned_prefill_and_traced_decode(
    hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind
):
    batch_size, prompt_len = 32, 33
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    torch.manual_seed(20261900 + layer_idx)
    prompt = torch.randn(batch_size, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    token_host = torch.randn(batch_size, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    cache, page_table = decoder.init_paged_kv_cache(max_context=128, batch_size=batch_size)
    prefill = decoder.prefill_forward(
        _tt_input(
            prompt.reshape(1, batch_size * prompt_len, hf_config.hidden_size),
            mesh_device,
        ),
        rope_mats=_rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        batch_size=batch_size,
        valid_seq_len=prompt_len,
    )
    assert prefill.shape[-2] == batch_size * prompt_len
    pos_host = torch.full((1, batch_size), prompt_len, dtype=torch.int32)
    token = _tt_input(token_host, mesh_device)
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(
            torch.full((batch_size,), prompt_len, dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
        ),
        batch_size=batch_size,
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        reference = optimized_baseline[(layer_kind, "batch32_decode33")]
        first = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape_as(reference).float()
        _assert_pcc(reference, first, PCC_THRESHOLD)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(
            first,
            ttnn.to_torch(ttnn.get_device_tensors(output)[0]).reshape_as(reference).float(),
        )
        for replica in ttnn.get_device_tensors(output)[1:]:
            assert torch.equal(first, ttnn.to_torch(replica).reshape_as(reference).float())
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize(
    "layer_idx,layer_kind,local_heads,head_dim", [(0, "sliding_attention", 4, 256), (5, "full_attention", 1, 512)]
)
def test_multichip_kv_and_layout_contract(
    hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind, local_heads, head_dim
):
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=262_144)
    expected_blocks = 16 if layer_kind == "sliding_attention" else 4096
    for tensor in cache:
        assert tuple(tensor.shape) == (expected_blocks, local_heads, PAGE_BLOCK_SIZE, head_dim)
        assert tensor.dtype == ttnn.bfloat8_b
        assert len(ttnn.get_device_tensors(tensor)) == TP_SIZE
    assert tuple(page_table.shape) == (1, expected_blocks)
    assert decoder.layer.self_attn.weights.kv_replicated is False
    assert decoder.mesh_profile["activation_contract"].startswith("replicated BF16")


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_EXACT_CONTEXT" not in os.environ,
    reason="set GEMMA4_MULTICHIP_EXACT_CONTEXT=1 for advertised-position trace",
)
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_advertised_context_traced_decode(hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind):
    max_context = hf_config.max_position_embeddings
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=max_context)
    torch.manual_seed(20262300 + layer_idx)
    token = _tt_input(
        torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        mesh_device,
    )
    pos_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_host[0, 0] = max_context - 1
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, max_context, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(
            torch.tensor([max_context - 1], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
        ),
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        first = _to_host_first(output).clone()
        assert torch.isfinite(first).all()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(first, _to_host_first(output))
        for replica in ttnn.get_device_tensors(output)[1:]:
            assert torch.equal(first, ttnn.to_torch(replica).reshape_as(first).float())
    finally:
        ttnn.release_trace(mesh_device, trace_id)


def test_multichip_runtime_source_audit():
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            MultichipDecoder._forward_device,
            MultichipDecoder._prefill_attention_tp,
            MultichipDecoder._decode_attention_tp,
            _TPOptimizedSharedMLP.__call__,
        )
    )
    for required in (
        "paged_fill_cache",
        "paged_scaled_dot_product_attention_decode",
        "ccl_allreduce",
        "local_kv_heads",
        "decode_wqkv",
    ):
        assert required in source
    for forbidden in ("ttnn.from_torch", "ttnn.to_torch", "torch."):
        assert forbidden not in source


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_MLP_SWEEP" not in os.environ,
    reason="set GEMMA4_MULTICHIP_MLP_SWEEP=1",
)
@pytest.mark.parametrize("num_cores", [8, 12, 21, 24])
def test_multichip_mlp_decode_geometry_sweep(
    hf_config,
    mesh_device,
    optimized_baseline,
    num_cores,
):
    """Measure legal TP-local 5376x5376 BFP4 geometries in the full trace."""
    layer_idx, layer_kind = LAYER_KINDS[0]
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    decoder.layer.shared_mlp.policy = _mlp_geometry_policy(num_cores)
    values = _inputs(hf_config, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    decoder.prefill_forward(
        _tt_input(values["prompt32"], mesh_device),
        rope_mats=_rope_device(hf_config, layer_idx, 32, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=32,
    )
    token = _tt_input(values["token"], mesh_device)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = 32
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(torch.tensor([32], dtype=torch.int32), mesh_device, dtype=ttnn.int32),
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        reference = optimized_baseline[(layer_kind, "decode32")]
        actuals = [ttnn.to_torch(t).reshape_as(reference).float() for t in ttnn.get_device_tensors(output)]
        for actual in actuals:
            _assert_pcc(reference, actual, PCC_THRESHOLD)
        record = {
            "num_cores": num_cores,
            "gate_up_in0_block_w": decoder.layer.shared_mlp.policy.gate_up_in0_block_w,
            "down_in0_block_w": decoder.layer.shared_mlp.policy.down_in0_block_w,
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
        }
        print("MULTICHIP_MLP_GEOMETRY " + json.dumps(record))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_MLP_PREFILL_SWEEP" not in os.environ,
    reason="set GEMMA4_MULTICHIP_MLP_PREFILL_SWEEP=1",
)
def test_multichip_mlp_prefill_placement_sweep(hf_config, profile_mesh_device):
    """Compare the interleaved prefill path with legal DRAM-sharded grids."""
    mesh_device = profile_mesh_device
    layer_idx, _ = LAYER_KINDS[0]
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    mlp = decoder.layer.shared_mlp
    hidden = hf_config.hidden_size
    rows = 128
    torch.manual_seed(20262351)
    hidden_input = _tt_input(torch.randn(1, rows, hidden, dtype=torch.bfloat16) * 0.02, mesh_device)

    def interleaved_call():
        mlp.is_decode = False
        return mlp(hidden_input)

    def explicit_1d_call(num_cores):
        policy = _mlp_geometry_policy(num_cores)
        grids = {8: (8, 1), 12: (6, 2), 21: (7, 3), 24: (8, 3)}
        grid = grids[num_cores]
        per_core_n = hidden // (ttnn.TILE_SIZE * num_cores)
        out_subblock_w = max(value for value in range(1, 9) if per_core_n % value == 0)

        def program(in0_block_w, fused_activation=None):
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=in0_block_w,
                per_core_M=rows // ttnn.TILE_SIZE,
                per_core_N=per_core_n,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                fuse_batch=True,
                fused_activation=fused_activation,
                mcast_in0=True,
            )

        up = ttnn.linear(
            hidden_input,
            mlp.up_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program(policy.gate_up_in0_block_w),
            compute_kernel_config=mlp.gate_up_compute,
        )
        gate = ttnn.linear(
            hidden_input,
            mlp.gate_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program(
                policy.gate_up_in0_block_w,
                ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
            ),
            compute_kernel_config=mlp.gate_up_compute,
        )
        activated = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate.deallocate(True)
        up.deallocate(True)
        partial = ttnn.linear(
            activated,
            mlp.down_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program(policy.down_in0_block_w),
            compute_kernel_config=mlp.down_compute,
        )
        activated.deallocate(True)
        return mlp._reduce(partial)

    def measure(call):
        warm = call()
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            output = call()
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            output.deallocate(True)
        return samples

    baseline = interleaved_call()
    baseline_devices = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(baseline)]
    baseline.deallocate(True)
    baseline_samples = measure(interleaved_call)
    candidates = []
    for num_cores in (8, 12, 21, 24):
        candidate = explicit_1d_call(num_cores)
        candidate_devices = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(candidate)]
        for expected, actual in zip(baseline_devices, candidate_devices):
            _assert_pcc(expected, actual, PCC_THRESHOLD)
        candidate.deallocate(True)
        samples = measure(lambda num_cores=num_cores: explicit_1d_call(num_cores))
        policy = _mlp_geometry_policy(num_cores)
        per_core_n = hidden // (ttnn.TILE_SIZE * num_cores)
        candidates.append(
            {
                "placement": "dram_interleaved_explicit_1d",
                "num_cores": num_cores,
                "gate_up_in0_block_w": policy.gate_up_in0_block_w,
                "down_in0_block_w": policy.down_in0_block_w,
                "per_core_M": rows // ttnn.TILE_SIZE,
                "per_core_N": per_core_n,
                "out_subblock_h": 1,
                "out_subblock_w": max(value for value in range(1, 9) if per_core_n % value == 0),
                "median_ms": statistics.median(samples),
                "samples_ms": samples,
            }
        )
    record = {
        "rows": rows,
        "baseline": {
            "placement": "dram_interleaved_auto_program",
            "median_ms": statistics.median(baseline_samples),
            "samples_ms": baseline_samples,
        },
        "candidates": candidates,
        "rejected_placement": "DRAM-sharded program hard-requires M=1; see rejected artifact",
    }
    print("MULTICHIP_MLP_PREFILL_PLACEMENT " + json.dumps(record))


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_FRACTURED_PROBE" not in os.environ,
    reason="set GEMMA4_MULTICHIP_FRACTURED_PROBE=1",
)
@pytest.mark.parametrize("num_rows", [32, 128])
def test_multichip_fractured_residual_boundary_probe(hf_config, profile_mesh_device, num_rows):
    """Compare replicated and fractured O->norm->residual->norm->gate boundaries."""
    mesh_device = profile_mesh_device
    layer_idx, _ = LAYER_KINDS[0]
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    mlp = decoder.layer.shared_mlp
    mlp.policy = _mlp_geometry_policy(24)
    hidden = hf_config.hidden_size
    torch.manual_seed(20262300 + num_rows)
    partial_host = torch.randn(TP_SIZE, 1, num_rows, hidden, dtype=torch.bfloat16) * 0.02
    residual_host = torch.randn(1, 1, num_rows, hidden, dtype=torch.bfloat16) * 0.02
    partial_replicated = ttnn.from_torch(
        partial_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    partial_fractured = ttnn.from_torch(
        partial_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    residual_replicated = ttnn.from_torch(
        residual_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    residual_fractured = ttnn.from_torch(
        residual_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    state = _layer_state(layer_idx)
    prefix = f"model.language_model.layers.{layer_idx}."

    def local_gamma(name):
        weight = state[prefix + name + ".weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
        return ttnn.from_torch(
            weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        )

    post_gamma = local_gamma("post_attention_layernorm")
    pre_ff_gamma = local_gamma("pre_feedforward_layernorm")

    def projection(full_input):
        if num_rows == 32:
            input_mem = mlp._decode_memory_config(24, hidden)
            output_mem = mlp._decode_memory_config(24, hidden)
            sharded = ttnn.to_memory_config(full_input, input_mem)
            program = mlp._decode_program_config(
                k=hidden,
                n=hidden,
                num_cores=24,
                in0_block_w=7,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
            )
            return ttnn.linear(
                sharded,
                mlp.gate_decode,
                memory_config=output_mem,
                program_config=program,
                compute_kernel_config=mlp.gate_up_compute,
            )
        return ttnn.linear(full_input, mlp.gate_prefill)

    def distributed_norm(fractured, gamma):
        stats = ttnn.rms_norm_pre_all_gather(fractured, dtype=ttnn.bfloat16)
        gathered_stats = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.rms_norm_post_all_gather(
            fractured,
            gathered_stats,
            epsilon=hf_config.rms_norm_eps,
            weight=gamma,
            dtype=ttnn.bfloat16,
        )

    def replicated_call():
        # Use the primitive directly because the production helper eagerly
        # deallocates its input after the collective.  Both candidates need a
        # stable source while their warmed timings are sampled repeatedly.
        reduced = ttnn.all_reduce(
            partial_replicated,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        post = decoder.layer.post_attention_layernorm.forward(reduced)
        residual_sum = ttnn.add(residual_replicated, post)
        normalized = decoder.layer.pre_feedforward_layernorm.forward(residual_sum)
        return projection(normalized)

    def fractured_call():
        reduced = ttnn.reduce_scatter(
            partial_fractured,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        post = distributed_norm(reduced, post_gamma)
        residual_sum = ttnn.add(residual_fractured, post)
        normalized = distributed_norm(residual_sum, pre_ff_gamma)
        gathered = ttnn.all_gather(
            normalized,
            dim=3,
            cluster_axis=1,
            num_links=2,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return projection(gathered)

    baseline = replicated_call()
    candidate = fractured_call()
    ttnn.synchronize_device(mesh_device)
    baseline_devices = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(baseline)]
    candidate_devices = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(candidate)]
    for expected, actual in zip(baseline_devices, candidate_devices):
        _assert_pcc(expected, actual, 0.995)

    def measure(call):
        if num_rows == 32:
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            output = call()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            try:
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                samples = []
                for _ in range(12):
                    start = time.perf_counter_ns()
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                    samples.append((time.perf_counter_ns() - start) / 1_000_000)
                return samples
            finally:
                ttnn.release_trace(mesh_device, trace_id)
                output.deallocate(True)
        warm = call()
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            output = call()
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            output.deallocate(True)
        return samples

    replicated_samples = measure(replicated_call)
    fractured_samples = measure(fractured_call)
    record = {
        "rows": num_rows,
        "replicated_ms": statistics.median(replicated_samples),
        "fractured_ms": statistics.median(fractured_samples),
        "fractured_over_replicated": statistics.median(fractured_samples) / statistics.median(replicated_samples),
        "replicated_samples_ms": replicated_samples,
        "fractured_samples_ms": fractured_samples,
    }
    print("MULTICHIP_FRACTURED_BOUNDARY " + json.dumps(record))


@pytest.mark.skipif("GEMMA4_MULTICHIP_BENCH" not in os.environ, reason="set GEMMA4_MULTICHIP_BENCH=1")
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
def test_multichip_warmed_latency(hf_config, mesh_device, optimized_baseline, layer_idx, layer_kind):
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    values = _inputs(hf_config, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    torch.manual_seed(20262100 + layer_idx)
    prefill128_host = torch.randn(1, 128, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02
    prefill128 = _tt_input(prefill128_host, mesh_device)
    prefill128_rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=False)

    def prefill_call():
        return decoder.prefill_forward(
            prefill128,
            rope_mats=prefill128_rope,
            page_table=page_table,
            kv_cache=cache,
            valid_seq_len=128,
        )

    warm = prefill_call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    prefill_samples = []
    for _ in range(12):
        start = time.perf_counter_ns()
        sample = prefill_call()
        ttnn.synchronize_device(mesh_device)
        prefill_samples.append((time.perf_counter_ns() - start) / 1_000_000)
        sample.deallocate(True)

    prompt = _tt_input(values["prompt32"], mesh_device)
    decoder.prefill_forward(
        prompt,
        rope_mats=_rope_device(hf_config, layer_idx, 32, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=32,
    )
    token = _tt_input(values["token"], mesh_device)
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = 32
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(torch.tensor([32]), mesh_device, dtype=ttnn.int32),
    )
    warm = decoder.decode_forward(token, **kwargs)
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decoder.decode_forward(token, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        samples = []
        for _ in range(12):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        baseline = optimized_baseline[(layer_kind, "baseline_latency")]
        multichip_prefill = statistics.median(prefill_samples)
        multichip_decode = statistics.median(samples)
        record = {
            "kind": layer_kind,
            "baseline_prefill128_ms": baseline["prefill128_ms"],
            "multichip_prefill128_ms": multichip_prefill,
            "prefill_speedup": baseline["prefill128_ms"] / multichip_prefill,
            "prefill_tp4_efficiency": baseline["prefill128_ms"] / multichip_prefill / TP_SIZE,
            "baseline_decode_ms": baseline["decode_ms"],
            "multichip_decode_ms": multichip_decode,
            "decode_speedup": baseline["decode_ms"] / multichip_decode,
            "decode_tp4_efficiency": baseline["decode_ms"] / multichip_decode / TP_SIZE,
            "multichip_prefill128_samples_ms": prefill_samples,
            "multichip_decode_samples_ms": samples,
        }
        print("MULTICHIP_WARMED " + json.dumps(record))
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif(
    "GEMMA4_MULTICHIP_PROFILE" not in os.environ,
    reason="set GEMMA4_MULTICHIP_PROFILE=1 and run under Tracy",
)
@pytest.mark.parametrize("layer_idx,layer_kind", LAYER_KINDS)
@pytest.mark.parametrize("perf_mode", ["prefill", "decode"])
def test_multichip_profile(hf_config, profile_mesh_device, layer_idx, layer_kind, perf_mode):
    from tracy import signpost

    mesh_device = profile_mesh_device
    decoder = _decoder(hf_config, mesh_device, layer_idx)
    cache, page_table = decoder.init_paged_kv_cache(max_context=128)
    torch.manual_seed(20262200 + layer_idx)
    if perf_mode == "prefill":
        hidden = _tt_input(
            torch.randn(1, 128, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
            mesh_device,
        )
        rope = _rope_device(hf_config, layer_idx, 128, mesh_device, decode=False)

        def call():
            return decoder.prefill_forward(
                hidden,
                rope_mats=rope,
                page_table=page_table,
                kv_cache=cache,
                valid_seq_len=128,
            )

        warm = call()
        ttnn.synchronize_device(mesh_device)
        warm.deallocate(True)
        signpost(f"MC_{layer_kind}_PREFILL")
        output = call()
        ttnn.synchronize_device(mesh_device)
        signpost(f"MC_{layer_kind}_PREFILL_END")
        output.deallocate(True)
        return

    prompt_len = 32
    prompt = _tt_input(
        torch.randn(1, prompt_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        mesh_device,
    )
    decoder.prefill_forward(
        prompt,
        rope_mats=_rope_device(hf_config, layer_idx, prompt_len, mesh_device, decode=False),
        page_table=page_table,
        kv_cache=cache,
        valid_seq_len=prompt_len,
    )
    token = _tt_input(
        torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.02,
        mesh_device,
    )
    pos_u_host = torch.zeros((1, 32), dtype=torch.int32)
    pos_u_host[0, 0] = prompt_len
    kwargs = dict(
        rope_mats=_rope_device(hf_config, layer_idx, 128, mesh_device, decode=True),
        page_table=page_table,
        kv_cache=cache,
        current_position=_replicated_int(pos_u_host, mesh_device, dtype=ttnn.uint32),
        current_position_cache=_replicated_int(
            torch.tensor([prompt_len], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
        ),
    )

    def call():
        return decoder.decode_forward(token, **kwargs)

    warm = call()
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    call()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        signpost(f"MC_{layer_kind}_DECODE")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost(f"MC_{layer_kind}_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)
