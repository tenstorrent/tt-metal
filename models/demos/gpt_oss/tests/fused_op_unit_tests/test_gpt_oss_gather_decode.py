# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole expert decode regression without downloading model weights."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.experts_throughput import ThroughputExpertConfig, ThroughputExperts


@run_for_blackhole("Requires a Blackhole Galaxy")
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": fabric,
            "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 16_777_216,
            "l1_small_size": 16_384,
        }
        for fabric in (ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ttnn.FabricConfig.FABRIC_2D)
    ],
    ids=["torus", "mesh"],
    indirect=True,
)
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16, ttnn.bfloat4_b], ids=["bf16", "bfp4"])
@pytest.mark.parametrize("fused_gate_up", [False, True], ids=["separate", "fused"])
def test_blackhole_gather_decode(mesh_device, weight_dtype, fused_gate_up):
    width, num_experts, total_tokens = 2880, 128, 128
    diagonal = torch.eye(width, dtype=torch.bfloat16)
    state = {
        "gate_proj": (diagonal * 0.25).expand(num_experts, width, width),
        "up_proj": (diagonal * 0.5).expand(num_experts, width, width),
        "down_proj": (diagonal * 0.75).expand(num_experts, width, width),
        "down_proj_bias": (torch.arange(num_experts).reshape(num_experts, 1).expand(num_experts, width) / 1024).to(
            torch.bfloat16
        ),
    }
    config = ThroughputExpertConfig(width, num_experts, width, 4, 32, use_fused_gate_up=fused_gate_up)
    experts = ThroughputExperts(
        mesh_device,
        config,
        state,
        ccl_manager=CCLManager(mesh_device, num_links=2),
        mesh_config=None,
        weight_dtype=weight_dtype,
    )
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(4, 8))
    input_dtypes = (ttnn.bfloat16, ttnn.uint16, ttnn.bfloat16)

    def make_inputs(seed):
        torch.manual_seed(seed)
        hidden = (torch.randn(total_tokens, 1, 1, width) * 0.25).to(torch.bfloat16)
        indices = (torch.arange(total_tokens).reshape(-1, 1) + torch.tensor([0, 33, 66, 99]) + seed) % num_experts
        scores = torch.tensor([0.125, 0.25, 0.25, 0.375], dtype=torch.bfloat16).expand(total_tokens, 4).clone()
        scores[::7, 0] = 0  # Zero-weight experts must not contribute their bias.
        x = hidden.float().reshape(total_tokens, width)
        gate = x * 0.25
        reference = gate * torch.sigmoid(gate * 1.702) * (x * 0.5 + 1) * 0.75
        reference *= scores.float().sum(-1, keepdim=True)
        reference += ((indices.float() / 1024) * scores.float()).sum(-1, keepdim=True)
        return (hidden, indices, scores), reference

    def host_tensors(values):
        return [
            ttnn.from_torch(value, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            for value, dtype in zip(values, input_dtypes)
        ]

    values, reference = make_inputs(58044)
    inputs = [ttnn.to_device(value, mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG) for value in host_tensors(values)]

    def run():
        # The expert layer consumes inputs. Keep these source buffers alive so
        # traced clones read newly uploaded tokens and routing on every replay.
        return experts.forward_decode(*(ttnn.clone(value) for value in inputs))

    def check(output, expected):
        ttnn.synchronize_device(mesh_device)
        shards = ttnn.get_device_tensors(output)
        assert len(shards) == 32
        for device_id, shard in enumerate(shards):
            actual = ttnn.to_torch(shard).reshape(32, width).float()
            row = device_id // 8
            torch.testing.assert_close(actual, expected[row * 32 : (row + 1) * 32], rtol=0.04, atol=0.004)

    def replace_inputs(seed):
        values, reference = make_inputs(seed)
        for source, destination in zip(host_tensors(values), inputs):
            ttnn.copy_host_to_device_tensor(source, destination)
        return reference

    mesh_device.enable_program_cache()
    output = run()
    check(output, reference)
    ttnn.deallocate(output)
    reference = replace_inputs(58045)
    output = run()
    check(output, reference)
    ttnn.deallocate(output)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = run()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        for seed in (58046, 58047):
            reference = replace_inputs(seed)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            check(output, reference)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
