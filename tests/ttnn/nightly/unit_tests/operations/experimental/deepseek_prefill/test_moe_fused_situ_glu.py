# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import (
    create_program_descriptor,
    make_mailbox,
    weight_memory_configs as descriptor_weight_memory_configs,
)

EMB = 3584
HIDDEN = 3072
COUNT = 32
GRID = ttnn.CoreCoord(11, 8)
GLOBAL_EXPERT_ID = 137
GRID_CASES = (
    (ttnn.CoreCoord(11, 8), "8x11"),
    (ttnn.CoreCoord(12, 8), "8x12"),
)


def _to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def _situ_glu(gate, up):
    gate = gate.float()
    up = up.float()
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(up / 25.0))


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
@pytest.mark.parametrize("weight_scale", [2.0e-2, 2.0e-1], ids=["model-scale", "saturation-scale"])
def test_moe_fused_situ_glu_3584x3072_rm_bf16_bfp4(device, weight_scale):
    """Requested Kimi K3 shape: RM bf16 input, bfp4 weights, SiTU-GLU in the fused kernel."""
    torch.manual_seed(20260819)
    x_host = torch.randn((1, 1, COUNT, EMB), dtype=torch.bfloat16)
    host_weights = [
        torch.randn(shape, dtype=torch.bfloat16) * weight_scale
        for shape in ((EMB, HIDDEN), (EMB, HIDDEN), (HIDDEN, EMB))
    ]

    x = _to_device(x_host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    gate_up_memory_config, down_memory_config = weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
    weights = [
        _to_device(weight, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device, memory_config)
        for weight, memory_config in zip(
            host_weights,
            (gate_up_memory_config, gate_up_memory_config, down_memory_config),
        )
    ]

    counts_host = torch.zeros(384, dtype=torch.int32)
    counts_host[GLOBAL_EXPERT_ID] = COUNT
    counts = _to_device(counts_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    offsets = _to_device(torch.zeros(384, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    expert_ids = _to_device(
        torch.tensor([GLOBAL_EXPERT_ID], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )

    output = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
        x,
        *weights,
        counts,
        expert_ids,
        0,
        input_m_tiles=COUNT // 32,
        core_grid=GRID,
        activation=ttnn.RoutedExpertActivation.SituGlu,
    )
    actual = ttnn.to_torch(output)[0, 0, :COUNT].float()
    composite_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        x,
        offsets,
        counts,
        expert_ids,
        [weights[0]],
        [weights[1]],
        [weights[2]],
        COUNT,
        activation=ttnn.RoutedExpertActivation.SituGlu,
        implementation=ttnn.RoutedExpertImplementation.MoeFusedSwiGlu,
    )
    composite_actual = ttnn.to_torch(composite_output)[0, 0, :COUNT].float()
    gate = x_host[0, 0].float() @ host_weights[0].float()
    up = x_host[0, 0].float() @ host_weights[1].float()
    reference = _situ_glu(gate, up) @ host_weights[2].float()

    assert torch.isfinite(actual).all()
    assert torch.isfinite(composite_actual).all()
    assert_with_pcc(reference, actual, pcc=0.97)
    assert_with_pcc(reference, composite_actual, pcc=0.97)
    assert_with_pcc(actual, composite_actual, pcc=0.999)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_tensix_config": ttnn.FabricTensixConfig.MUX,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
@pytest.mark.parametrize(
    "emb,hidden,activation",
    [
        (3584, 3072, ttnn.RoutedExpertActivation.Silu),
        (3584, 3072, ttnn.RoutedExpertActivation.SituGlu),
        (6144, 2048, ttnn.RoutedExpertActivation.Silu),
        (7168, 2048, ttnn.RoutedExpertActivation.Silu),
    ],
    ids=["3584x3072-silu", "3584x3072-situ-glu", "6144x2048-silu", "7168x2048-silu"],
)
@pytest.mark.parametrize("core_grid", [case[0] for case in GRID_CASES], ids=[case[1] for case in GRID_CASES])
def test_moe_fused_swiglu_python_descriptor_grouped_m(mesh_device, emb, hidden, activation, core_grid):
    """Validate both supported grids in the Python descriptor against the C++ implementation."""
    device = mesh_device
    available_grid = device.compute_with_storage_grid_size()
    if core_grid.x > available_grid.x or core_grid.y > available_grid.y:
        pytest.skip(
            f"requested {core_grid.y}x{core_grid.x} grid exceeds available "
            f"{available_grid.y}x{available_grid.x} grid"
        )
    count = 1024  # Four 256-token blocks: enough to exercise the grouped-M down schedule.
    torch.manual_seed(20260819)
    x_host = torch.randn((1, 1, count, emb), dtype=torch.bfloat16)
    host_weights = [
        torch.randn(shape, dtype=torch.bfloat16) * 2.0e-2 for shape in ((emb, hidden), (emb, hidden), (hidden, emb))
    ]
    x = _to_device(x_host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    gate_up_memory_config, down_memory_config = descriptor_weight_memory_configs(
        device, emb, hidden, core_grid=core_grid
    )
    weights = [
        _to_device(weight, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device, memory_config)
        for weight, memory_config in zip(
            host_weights,
            (gate_up_memory_config, gate_up_memory_config, down_memory_config),
        )
    ]
    counts_host = torch.zeros(384, dtype=torch.int32)
    counts_host[GLOBAL_EXPERT_ID] = count
    counts = _to_device(counts_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    expert_ids = _to_device(
        torch.tensor([GLOBAL_EXPERT_ID], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )
    output = ttnn.empty(
        x.shape,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mailbox = make_mailbox(device, core_grid.x * core_grid.y)
    descriptor = create_program_descriptor(
        x,
        *weights,
        counts,
        expert_ids,
        output,
        mailbox,
        local_expert_id=0,
        input_m_tiles=count // 32,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        core_grid=core_grid,
        situ_glu=activation == ttnn.RoutedExpertActivation.SituGlu,
    )
    ttnn.generic_op([x, *weights, counts, expert_ids, output, mailbox], descriptor)
    actual = ttnn.to_torch(output)[0, 0, :count].float()
    reference = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
        x,
        *weights,
        counts,
        expert_ids,
        0,
        input_m_tiles=count // 32,
        core_grid=core_grid,
        activation=activation,
    )
    reference = ttnn.to_torch(reference)[0, 0, :count].float()
    assert torch.isfinite(actual).all()
    assert_with_pcc(reference, actual, pcc=0.999)
