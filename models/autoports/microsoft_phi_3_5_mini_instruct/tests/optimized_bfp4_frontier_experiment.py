# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-weight BFP4 precision-frontier experiment for optimized Phi decode.

This is deliberately candidate-only: the production decoder is changed only
through its public precision policy.  Attention therefore uses the compatible
automatic matmul topology, while down projection retains the selected
DRAM-sharded LoFi decode topology.
"""

import statistics
import time

import pytest
import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _page_table,
    _positions,
    _real_state,
    _reference_decode_zero_prefix,
    _to_torch_decode,
    _to_tt_decode,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizationPolicy, OptimizedDecoder


def _pcc(reference, actual):
    x, y = reference.float().flatten(), actual.float().flatten()
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def _trace_mean_ms(mesh_device, trace_id, iterations=100):
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return statistics.mean(samples)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_real_weight_bfp4_isolated_projection_frontier(mesh_device, batch):
    config = _config()
    state = _real_state()
    prefix = f"model.layers.{LAYER_IDX}."
    roles = {
        "qkv": ("self_attn.qkv_proj.weight", config.hidden_size),
        "output": ("self_attn.o_proj.weight", config.hidden_size),
        "down": ("mlp.down_proj.weight", config.intermediate_size),
    }
    generator = torch.Generator().manual_seed(9810 + batch)
    for role, (suffix, k) in roles.items():
        weight = state[prefix + suffix].transpose(-2, -1).contiguous()
        logical = (torch.randn(1, 1, batch, k, generator=generator) * 0.2).to(torch.bfloat16)
        padded = torch.zeros(1, 1, 32, k, dtype=torch.bfloat16)
        padded[:, :, :batch, :] = logical
        reference = logical.float() @ weight.float()
        activation = ttnn.from_torch(
            padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_weight = ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def projection():
            return ttnn.linear(activation, tt_weight, dtype=ttnn.bfloat16)

        output = projection()
        pcc = _pcc(reference, ttnn.to_torch(output)[:, :, :batch, :])
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        projection()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            mean_ms = _trace_mean_ms(mesh_device, trace_id)
        finally:
            ttnn.release_trace(mesh_device, trace_id)
        print(f"BFP4_ISOLATED batch={batch} role={role} pcc={pcc:.8f} mean_ms={mean_ms:.6f}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_real_weight_bfp4_cumulative_decoder_frontier(mesh_device, batch):
    config = _config()
    state = _real_state()
    hidden = (
        torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(9820 + batch)) * 0.2
    ).to(torch.bfloat16)
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    positions_list = [33] if batch == 1 else list(range(1, batch + 1))
    positions = _positions(positions_list, mesh_device)
    page_table = _page_table(batch, 64, mesh_device, permute=True)
    reference = _reference_decode_zero_prefix(config, state, hidden, positions_list, use_long=False)
    policies = {
        "bfp8_attention_down": OptimizationPolicy(
            attention_weight_dtype=ttnn.bfloat8_b,
            mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
            mlp_down_weight_dtype=ttnn.bfloat8_b,
        ),
        "attention_bfp4": OptimizationPolicy(
            attention_weight_dtype=ttnn.bfloat4_b,
            mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
            mlp_down_weight_dtype=ttnn.bfloat8_b,
        ),
        "down_bfp4": OptimizationPolicy(
            attention_weight_dtype=ttnn.bfloat8_b,
            mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
            mlp_down_weight_dtype=ttnn.bfloat4_b,
        ),
        "attention_down_bfp4": OptimizationPolicy(
            attention_weight_dtype=ttnn.bfloat4_b,
            mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
            mlp_down_weight_dtype=ttnn.bfloat4_b,
        ),
    }
    decodes = {}
    outputs = {}
    traces = {}
    resources = {}
    try:
        # Allocate and compile every candidate before trace capture so later
        # allocations cannot disturb trace-owned buffers.
        for name, policy in policies.items():
            decoder = OptimizedDecoder.from_state_dict(
                state,
                hf_config=config,
                layer_idx=LAYER_IDX,
                mesh_device=mesh_device,
                batch=batch,
                max_context=64,
                optimization_policy=policy,
            )
            key_cache, value_cache = decoder.create_paged_kv_cache()
            resources[name] = (decoder, key_cache, value_cache)

            def decode(d=decoder, k=key_cache, v=value_cache):
                return d.decode_forward(
                    tt_hidden,
                    key_cache=k,
                    value_cache=v,
                    page_table=page_table,
                    current_positions=positions,
                    use_long_rope=False,
                )

            outputs[name] = decode()
            ttnn.synchronize_device(mesh_device)
            decodes[name] = decode

        for name, decode in decodes.items():
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            outputs[name] = decode()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            traces[name] = trace_id

        for name, trace_id in traces.items():
            mean_ms = _trace_mean_ms(mesh_device, trace_id)
            actual = _to_torch_decode(outputs[name])
            print(
                f"BFP4_CUMULATIVE batch={batch} candidate={name} "
                f"pcc={_pcc(reference, actual):.8f} mean_ms={mean_ms:.6f}"
            )
    finally:
        for trace_id in traces.values():
            ttnn.release_trace(mesh_device, trace_id)
