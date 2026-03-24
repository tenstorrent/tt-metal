#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.utility_functions import comp_allclose, comp_pcc

# SEQUENCE_LENGTHS = [128, 1024, 2048, 4096, 8192]
SEQUENCE_LENGTHS = [8192]


def require_single_device(device) -> None:
    if hasattr(device, "get_num_devices") and device.get_num_devices() != 1:
        pytest.skip("BGE-M3 PCC tests currently target single-device execution")


def make_lazy_weight(
    source: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> LazyWeight:
    return LazyWeight(
        source=source,
        dtype=dtype,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )


def to_ttnn_tensor(
    tensor: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )


def to_ttnn_ids(input_ids: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        input_ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def to_torch(tt_tensor: ttnn.Tensor, expected_shape: tuple[int, ...]) -> torch.Tensor:
    output = to_torch_auto_compose(tt_tensor).to(torch.float32)
    assert tuple(output.shape) == expected_shape, f"Expected output shape {expected_shape}, got {tuple(output.shape)}"
    return output


def assert_pcc(reference: torch.Tensor, candidate: torch.Tensor, threshold: float) -> None:
    passing, pcc_message = comp_pcc(reference, candidate, threshold)
    allclose, allclose_message = comp_allclose(reference, candidate)
    assert passing, f"PCC check failed: {pcc_message}; {allclose_message}; allclose={allclose}"
