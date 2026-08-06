# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device-liveness coverage for the fused routed-expert FFN.

This deliberately bypasses gate, dispatch, extract, combine, the model wrapper,
and host-reference/PCC work.  It launches exactly one
``unified_routed_expert_ffn`` program in the direct-write configuration used by
Kimi's routed-expert composite: the input/output is a 256-row shared dispatch
buffer, while the expert has 64 live tokens at offset zero.

The dimensions match the reduced four-chip Kimi device smoke.  Keep this test
small and deterministic so it is useful with watcher and LLK assertions when
debugging the reader/compute/writer circular-buffer protocol.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

_MESH_PARAMS = [
    pytest.param(
        (4, 1),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
        marks=[
            pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            # A failing device liveness test must return control to the safe
            # runner so it can collect triage and clean up the QuietBox. The
            # first JIT build takes longer than the device operation itself.
            pytest.mark.timeout(60),
        ],
        id="quietbox-linear-4",
    )
]


def _device_tensor(mesh_device, host_tensor, dtype, layout):
    return ttnn.from_torch(
        host_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _expert_metadata(mesh_device, host_tensor):
    """Place one metadata row on each EP chip, as the production mapper does."""
    return ttnn.from_torch(
        host_tensor,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None)),
    )


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_ffn is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
def test_unified_routed_expert_ffn_kimi_direct_write_liveness(mesh_device, device_params):
    """One direct-write routed-expert FFN completes on every QuietBox chip."""
    max_tokens = 256
    live_tokens = 64
    emb_dim = 1024
    hidden_dim = 256

    x = _device_tensor(
        mesh_device,
        torch.zeros((max_tokens, emb_dim), dtype=torch.float32),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
    )
    gate_proj = _device_tensor(
        mesh_device,
        torch.zeros((emb_dim, hidden_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    up_proj = _device_tensor(
        mesh_device,
        torch.zeros((emb_dim, hidden_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    down_proj = _device_tensor(
        mesh_device,
        torch.zeros((hidden_dim, emb_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    counts = _device_tensor(
        mesh_device,
        torch.tensor([[live_tokens]], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    global_expert_idx_table = _device_tensor(
        mesh_device,
        torch.tensor([[0]], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    expert_region_offsets = _device_tensor(
        mesh_device,
        torch.tensor([[0]], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    output = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
        x,
        gate_proj,
        up_proj,
        down_proj,
        counts,
        global_expert_idx_table,
        local_expert_id=0,
        output=x,
        expert_region_offsets=expert_region_offsets,
        input_m_tiles=max_tokens // 32,
        read_x_at_offset=True,
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    ttnn.synchronize_device(mesh_device)

    assert output.shape == x.shape


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_ffn is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
def test_unified_routed_expert_moe_kimi_composite_liveness(mesh_device, device_params):
    """The 16-expert direct-write schedule drains its shared buffer.

    Eight experts each own one tile and the other eight are empty.  This is the
    smallest tile-aligned schedule that exercises the composite's complete
    local-expert loop while remaining inside Kimi's 256-row dispatch buffer.
    """
    max_tokens = 256
    emb_dim = 1024
    hidden_dim = 256
    experts_per_chip = 16
    active_experts = 8

    x = _device_tensor(
        mesh_device,
        torch.zeros((max_tokens, emb_dim), dtype=torch.float32),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
    )
    gate_proj = _device_tensor(
        mesh_device,
        torch.zeros((emb_dim, hidden_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    up_proj = _device_tensor(
        mesh_device,
        torch.zeros((emb_dim, hidden_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    down_proj = _device_tensor(
        mesh_device,
        torch.zeros((hidden_dim, emb_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    counts = _device_tensor(
        mesh_device,
        torch.tensor([[32] * active_experts + [0] * (experts_per_chip - active_experts)], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    global_expert_idx_table = _device_tensor(
        mesh_device,
        torch.tensor([list(range(experts_per_chip))], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    expert_region_offsets = _device_tensor(
        mesh_device,
        torch.tensor(
            [[32 * i for i in range(active_experts)] + [0] * (experts_per_chip - active_experts)], dtype=torch.int32
        ),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    output = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        x,
        expert_region_offsets,
        counts,
        global_expert_idx_table,
        [gate_proj] * experts_per_chip,
        [up_proj] * experts_per_chip,
        [down_proj] * experts_per_chip,
        max_dispatched_tokens_per_expert=max_tokens,
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    ttnn.synchronize_device(mesh_device)

    assert output.shape == x.shape


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_ffn is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
def test_unified_routed_expert_moe_kimi_row_major_liveness(mesh_device, device_params):
    """The production ROW_MAJOR-bf16 routed-expert path drains its output.

    Dispatch produces this layout in the Kimi smoke.  Unlike the TILE test
    above, the composite must tilize the input inside the FFN and allocate a
    separate TILE bf8 output buffer before its 16 direct-write FFNs run.
    """
    max_tokens = 256
    emb_dim = 1024
    hidden_dim = 256
    experts_per_chip = 16
    active_experts = 8

    x = _device_tensor(
        mesh_device,
        torch.zeros((max_tokens, emb_dim), dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    gate_proj = _device_tensor(
        mesh_device,
        torch.zeros((emb_dim, hidden_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    up_proj = _device_tensor(
        mesh_device,
        torch.zeros((emb_dim, hidden_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    down_proj = _device_tensor(
        mesh_device,
        torch.zeros((hidden_dim, emb_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    counts = _device_tensor(
        mesh_device,
        torch.tensor([[32] * active_experts + [0] * (experts_per_chip - active_experts)], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    global_expert_idx_table = _device_tensor(
        mesh_device,
        torch.tensor([list(range(experts_per_chip))], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    expert_region_offsets = _device_tensor(
        mesh_device,
        torch.tensor(
            [[32 * i for i in range(active_experts)] + [0] * (experts_per_chip - active_experts)], dtype=torch.int32
        ),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    output = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        x,
        expert_region_offsets,
        counts,
        global_expert_idx_table,
        [gate_proj] * experts_per_chip,
        [up_proj] * experts_per_chip,
        [down_proj] * experts_per_chip,
        max_dispatched_tokens_per_expert=max_tokens,
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    ttnn.synchronize_device(mesh_device)

    assert output.shape == x.shape


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_ffn is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
def test_unified_routed_expert_moe_kimi_captured_routing_liveness(mesh_device, device_params):
    """Reproduce Kimi's captured per-chip routing metadata without dispatch.

    These are the exact count and region-offset tables from the failing Kimi
    debug smoke. Each local expert has fewer than one tile of live tokens, but
    all 16 experts consume a tile-aligned output region. The final region starts
    at row 480, so the shared buffer must contain at least 512 rows.
    """
    max_tokens = 512
    emb_dim = 1024
    hidden_dim = 256
    experts_per_chip = 16
    counts_values = [
        4,
        9,
        4,
        3,
        4,
        4,
        3,
        6,
        6,
        2,
        4,
        3,
        4,
        9,
        2,
        6,
        2,
        0,
        3,
        8,
        9,
        1,
        6,
        4,
        5,
        2,
        4,
        5,
        4,
        3,
        0,
        9,
        4,
        5,
        5,
        0,
        5,
        1,
        6,
        2,
        3,
        7,
        4,
        5,
        8,
        0,
        4,
        3,
        1,
        3,
        5,
        0,
        8,
        4,
        6,
        4,
        5,
        7,
        2,
        3,
        2,
        3,
        1,
        2,
    ]
    offsets_values = [
        0,
        32,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
        0,
        32,
        32,
        64,
        96,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        416,
        0,
        32,
        64,
        96,
        96,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        384,
        416,
        0,
        32,
        64,
        96,
        96,
        128,
        160,
        192,
        224,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
    ]

    x = _device_tensor(
        mesh_device,
        torch.zeros((max_tokens, emb_dim), dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    gate_proj = _device_tensor(
        mesh_device,
        torch.zeros((emb_dim, hidden_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    up_proj = _device_tensor(
        mesh_device,
        torch.zeros((emb_dim, hidden_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    down_proj = _device_tensor(
        mesh_device,
        torch.zeros((hidden_dim, emb_dim), dtype=torch.float32),
        ttnn.bfloat4_b,
        ttnn.TILE_LAYOUT,
    )
    counts = _device_tensor(
        mesh_device,
        torch.tensor([counts_values], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    expert_region_offsets = _device_tensor(
        mesh_device,
        torch.tensor([offsets_values], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    global_expert_idx_table = _expert_metadata(
        mesh_device,
        torch.arange(experts_per_chip * mesh_device.get_num_devices(), dtype=torch.int32).reshape(
            mesh_device.get_num_devices(), experts_per_chip
        ),
    )

    output = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        x,
        expert_region_offsets,
        counts,
        global_expert_idx_table,
        [gate_proj] * experts_per_chip,
        [up_proj] * experts_per_chip,
        [down_proj] * experts_per_chip,
        max_dispatched_tokens_per_expert=max_tokens,
        activation=ttnn.RoutedExpertActivation.Silu,
    )
    ttnn.synchronize_device(mesh_device)

    assert output.shape == x.shape
