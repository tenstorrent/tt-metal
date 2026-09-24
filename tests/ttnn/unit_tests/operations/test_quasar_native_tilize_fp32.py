# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolates whether the UNPACKER_0 ILLEGAL_FORMAT_CONVERSION fault (seen in the mainline
TilizeDeviceOperation default factory for FLOAT32/UINT8 inputs on Quasar) is an architecture-level
issue with enable_32_bit_dest+UnpackToDest, or specific to the mainline factory's Gen1->Gen2 config
copy. Calls ttnn.experimental.quasar.tilize (the "canonical" Quasar-native factory, which builds its
ComputeGen2Config via ttnn::to_compute_hardware_config rather than a manual field copy) directly on a
FLOAT32 row-major input, bypassing from_torch's TILE-layout auto-tilize entirely.
"""

import torch
import ttnn
from loguru import logger


def test_quasar_experimental_tilize_fp32(mesh_device):
    torch.manual_seed(0)
    t = torch.randn(1, 1, 32, 32, dtype=torch.float32)

    logger.info("[native-tilize-fp32] upload row-major fp32 (1,1,32,32)")
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    logger.info("[native-tilize-fp32] ttnn.experimental.quasar.tilize begin")
    tt = ttnn.experimental.quasar.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
    logger.info("[native-tilize-fp32] tilize done; reading back")
    out = ttnn.to_torch(tt).float()
    logger.info("[native-tilize-fp32] readback complete")

    assert torch.isfinite(out).all()
    assert torch.allclose(out.reshape(t.shape), t, atol=1e-3, rtol=1e-3)
