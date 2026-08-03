# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-weight explicit prefill matmul program-config experiment."""

import statistics
import time

import pytest
import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _real_state,
)


def _pcc(reference, actual):
    x, y = reference.float().flatten(), actual.float().flatten()
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def _mean_ms(mesh_device, op, iterations):
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        op()
        ttnn.synchronize_device(mesh_device)
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    return statistics.mean(samples)


def _trace_ms(mesh_device, op, iterations=100):
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        return _mean_ms(
            mesh_device,
            lambda: ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False),
            iterations,
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_explicit_prefill_projection_configs(mesh_device, batch):
    config = _config()
    state = _real_state()
    prefix = f"model.layers.{LAYER_IDX}."
    roles = {
        "qkv": ("self_attn.qkv_proj.weight", config.hidden_size),
        "output": ("self_attn.o_proj.weight", config.hidden_size),
        "gate_up": ("mlp.gate_up_proj.weight", config.hidden_size),
        "down": ("mlp.down_proj.weight", config.intermediate_size),
    }
    generator = torch.Generator().manual_seed(9900 + batch)
    for role, (suffix, k) in roles.items():
        weight = state[prefix + suffix].transpose(-2, -1).contiguous()
        n = weight.shape[-1]
        activation = (torch.randn(1, batch, 128, k, generator=generator) * 0.2).to(torch.bfloat16)
        reference = activation.float() @ weight.float()
        tt_activation = ttnn.from_torch(
            activation,
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

        def automatic():
            return ttnn.linear(tt_activation, tt_weight, dtype=ttnn.bfloat16)

        output = automatic()
        print(
            f"PREFILL_CONFIG batch={batch} role={role} candidate=automatic "
            f"pcc={_pcc(reference, ttnn.to_torch(output)):.8f} "
            f"warmed_ms={_mean_ms(mesh_device, automatic, 20):.6f} "
            f"trace_ms={_trace_ms(mesh_device, automatic):.6f}"
        )
        m_tiles = batch * 4
        per_core_m = (m_tiles + 7) // 8
        per_core_n = (n // 32) // 8
        out_subblock_w = 4 if per_core_n % 4 == 0 else 3
        for block_w in (4, 8):
            program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=per_core_m,
                per_core_N=per_core_n,
                transpose_mcast=False,
                fused_activation=None,
            )

            def explicit():
                return ttnn.linear(
                    tt_activation,
                    tt_weight,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=program_config,
                )

            try:
                output = explicit()
                print(
                    f"PREFILL_CONFIG batch={batch} role={role} candidate=explicit "
                    f"in0_block_w={block_w} grid=8x8 per_core_m={per_core_m} per_core_n={per_core_n} "
                    f"pcc={_pcc(reference, ttnn.to_torch(output)):.8f} "
                    f"warmed_ms={_mean_ms(mesh_device, explicit, 20):.6f} "
                    f"trace_ms={_trace_ms(mesh_device, explicit):.6f}"
                )
            except RuntimeError as error:
                print(
                    f"PREFILL_REJECT batch={batch} role={role} in0_block_w={block_w} "
                    f"grid=8x8 per_core_m={per_core_m} per_core_n={per_core_n} error={error}"
                )
