# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import ttnn
from loguru import logger
from models.utility_functions import comp_pcc


def allocate_tensor_on_device_like(
    t: ttnn.Tensor, *, device: ttnn.Device, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    return ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, memory_config=memory_config)


def from_torch_fast(
    t: torch.Tensor,
    *,
    device: ttnn.Device | None = None,
    layout: ttnn.Layout | None = None,
    dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    to_host: bool = False,
) -> ttnn.Tensor:
    # ttnn.to_layout does not support changing the datatype or memory_config if the layout already matches. ttnn.clone
    # does not support changing the datatype if the input is not tiled. An option could be to tilize the input before
    # changing the datatype and then untilize again, but it was not tested if this would be faster than converting the
    # datatype on the host.
    if device is None or layout is None or layout == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.from_torch(t, device=device, layout=layout, dtype=dtype)

    tensor = ttnn.from_torch(t, device=device)

    new = ttnn.to_layout(tensor, layout, dtype=dtype, memory_config=memory_config)
    ttnn.deallocate(tensor)
    tensor = new

    if to_host:
        new = tensor.cpu()
        ttnn.deallocate(tensor)
        tensor = new

    return tensor


def to_memory_config(
    t: ttnn.Tensor, memory_config: ttnn.MemoryConfig, *, dtype: ttnn.DataType | None = None, deallocate: bool = False
) -> ttnn.Tensor:
    result = ttnn.to_memory_config(t, memory_config, dtype=dtype)

    result_is_same = result.memory_config() == t.memory_config() and result.buffer_address() == t.buffer_address()
    if deallocate and not result_is_same:
        ttnn.deallocate(t)

    return result


def assert_quality(
    a: ttnn.Tensor | torch.Tensor,
    b: ttnn.Tensor | torch.Tensor,
    *,
    pcc: float | None = None,
    mse: float | None = None,
) -> None:
    if isinstance(a, ttnn.Tensor):
        a = ttnn.to_torch(a)
    if isinstance(b, ttnn.Tensor):
        b = ttnn.to_torch(b)

    a = a.to(torch.float32)
    b = b.to(torch.float32)

    _, pcc_calculated = comp_pcc(a, b)
    mse_calculated = torch.nn.functional.mse_loss(a, b).item()

    logger.info(f"PCC={pcc_calculated:.6f}, MSE={mse_calculated:.6f}")
    if pcc is not None:
        assert pcc_calculated >= pcc, f"PCC={pcc_calculated:.6f}"
    if mse is not None:
        assert mse_calculated <= mse, f"MSE={mse_calculated:.6f}"
