# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Isolated repro for the untilization step that crashes/hangs at
models/demos/deepseek_v3_d_p/tt/moe/tt_moe.py:524

    expert_outputs_rm = ttnn.to_layout(expert_outputs, ttnn.ROW_MAJOR_LAYOUT)

Reproduces the exact tensor properties observed in pdb at that point:
    shape       = [1, 1, 32000, 7168]
    layout      = TILE_LAYOUT
    dtype       = BFLOAT8_B
    memory_cfg  = INTERLEAVED, DRAM

Run with the same wormhole 4x2 mesh as the failing MoE test:
    pytest models/demos/deepseek_v3_d_p/tests/pcc/test_expert_outputs_untilize.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 320, 320),  # PASS
        (1, 1, 32000, 7168),  # FAILS
    ],
    ids=["1x1x320x320", "1x1x32000x7168"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_expert_outputs_untilize(mesh_device, device_params, shape):
    """Untilize a [1,1,32000,7168] BFLOAT8_B INTERLEAVED-DRAM tensor on each device."""
    mesh_device.disable_and_clear_program_cache()

    torch.manual_seed(0)

    # Build a host tensor matching the per-device shape, replicated to every chip.
    host_tensor = torch.randn(shape, dtype=torch.bfloat16)

    tt_tensor = ttnn.from_torch(
        host_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info(
        f"[pre-untilize] shape={tt_tensor.shape} layout={tt_tensor.layout} "
        f"dtype={tt_tensor.dtype} mem={tt_tensor.memory_config()}"
    )

    # Memory snapshot before the suspect call.
    for btype_name, btype in (
        ("L1", ttnn.BufferType.L1),
        ("DRAM", ttnn.BufferType.DRAM),
    ):
        v = ttnn.get_memory_view(mesh_device, btype)
        logger.info(
            f"[pre-untilize][{btype_name}] banks={v.num_banks} "
            f"per_bank_alloc={v.total_bytes_allocated_per_bank} "
            f"per_bank_free={v.total_bytes_free_per_bank} "
            f"per_bank_largest_contig_free={v.largest_contiguous_bytes_free_per_bank}"
        )

    # The suspect call.
    tt_rm = ttnn.to_layout(tt_tensor, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.synchronize_device(mesh_device)

    logger.info(
        f"[post-untilize] shape={tt_rm.shape} layout={tt_rm.layout} " f"dtype={tt_rm.dtype} mem={tt_rm.memory_config()}"
    )

    assert tt_rm.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tuple(tt_rm.shape) == shape
