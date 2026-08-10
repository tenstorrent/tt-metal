# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The 8-row x 12-column Blackhole grid exposed by ROW dispatch with fabric MUX."""

import pytest
import torch

import ttnn

from tests.ttnn.unit_tests.operations.moe_fused_swiglu import test_moe_fused_swiglu_grid as grid_test
from tests.ttnn.unit_tests.operations.moe_fused_swiglu import test_moe_fused_swiglu as golden_test
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import (
    create_program_descriptor,
    make_mailbox,
)


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
def test_row_dispatch_grid_probe(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    print(f"ROW_DISPATCH_GRID={grid.x}x{grid.y}", flush=True)
    assert grid.x >= 12 and grid.y >= 8


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
def test_row_dispatch_8x12_correctness(mesh_device):
    grid_test.test_grid(mesh_device, 12, 8, 2048, "8 rows x 12 columns with ROW dispatch")


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
@pytest.mark.parametrize("count", [96, 288], ids=["short_block", "full_then_short"])
def test_row_dispatch_8x12_tail_correctness(mesh_device, count):
    emb, capacity = 6144, 1024
    x_rows, (w_gate, w_up, w_down), tt_x, tt_w, tt_counts, tt_idx = golden_test._build_inputs(
        emb, capacity, count, "bf16_rm", mesh_device
    )
    out = golden_test.moe_fused_swiglu(
        tt_x,
        tt_w[0],
        tt_w[1],
        tt_w[2],
        tt_counts,
        tt_idx,
        golden_test.LOCAL_EXPERT_ID,
        core_grid=(12, 8),
    )
    expected = golden_test._reference(x_rows, w_gate, w_up, w_down)
    actual = ttnn.to_torch(out)[0, 0, :count, :].to(torch.float32)
    golden_test.assert_with_pcc(expected, actual, golden_test.PCC_GATE)


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
def test_row_dispatch_8x12_generic_matches_cpp_bitwise(mesh_device):
    """The retained generic factory and production C++ factory agree on the 12-column geometry."""
    emb, capacity, count = 7168, 1024, 288
    _, _, tt_x, tt_w, tt_counts, tt_idx = golden_test._build_inputs(emb, capacity, count, "bf16_rm", mesh_device)
    standard = golden_test.moe_fused_swiglu(
        tt_x,
        tt_w[0],
        tt_w[1],
        tt_w[2],
        tt_counts,
        tt_idx,
        golden_test.LOCAL_EXPERT_ID,
        core_grid=(12, 8),
    )

    generic = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, capacity, emb]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    available = mesh_device.compute_with_storage_grid_size()
    mailbox = make_mailbox(mesh_device, int(available.x) * int(available.y))
    descriptor = create_program_descriptor(
        tt_x,
        tt_w[0],
        tt_w[1],
        tt_w[2],
        tt_counts,
        tt_idx,
        generic,
        mailbox,
        local_expert_id=golden_test.LOCAL_EXPERT_ID,
        input_m_tiles=capacity // golden_test.TILE,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
            bfp8_pack_precise=True,
        ),
        core_grid=(12, 8),
    )
    ttnn.generic_op([tt_x, *tt_w, tt_counts, tt_idx, mailbox, generic], descriptor)

    standard_host = ttnn.to_torch(standard)[0, 0, :count]
    generic_host = ttnn.to_torch(generic)[0, 0, :count]
    assert torch.equal(standard_host, generic_host), (
        "12x8 standard/generic mismatch: " f"max abs {(standard_host - generic_host).abs().max().item()}"
    )


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
def test_row_dispatch_8x12_transposed_generic_correctness(mesh_device):
    """Use physical columns for K and physical rows for hidden groups on the same 12x8 rectangle."""
    emb, capacity, count = 7168, 1024, 288
    x_rows, (w_gate, w_up, w_down), tt_x, tt_w, tt_counts, tt_idx = golden_test._build_inputs(
        emb, capacity, count, "bf16_rm", mesh_device
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, capacity, emb]),
        ttnn.bfloat8_b,
        ttnn.TILE_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    available = mesh_device.compute_with_storage_grid_size()
    mailbox = make_mailbox(mesh_device, int(available.x) * int(available.y))
    descriptor = create_program_descriptor(
        tt_x,
        tt_w[0],
        tt_w[1],
        tt_w[2],
        tt_counts,
        tt_idx,
        output,
        mailbox,
        local_expert_id=golden_test.LOCAL_EXPERT_ID,
        input_m_tiles=capacity // golden_test.TILE,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
            bfp8_pack_precise=True,
        ),
        core_grid=(12, 8),
        transpose_grid_axes=True,
    )
    ttnn.generic_op([tt_x, *tt_w, tt_counts, tt_idx, mailbox, output], descriptor)

    expected = golden_test._reference(x_rows, w_gate, w_up, w_down)
    actual = ttnn.to_torch(output)[0, 0, :count, :].to(torch.float32)
    golden_test.assert_with_pcc(expected, actual, golden_test.PCC_GATE)
