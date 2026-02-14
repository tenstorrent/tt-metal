# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for DeepSeek V3 B1 model (prefill harness and decode flow).

Uses HostInterface in loopback mode as a mock decoder; validates prefill
(token-by-token with outputs discarded) and autoregressive decode (input → output).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.model import DeepSeekV3, page_size_bytes


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("prompt_length", [1, 8, 64, 128])
@pytest.mark.parametrize("num_decode_steps", [1, 16, 32])
@pytest.mark.parametrize(
    "h2d_mode",
    [
        ttnn.H2DMode.HOST_PUSH,
    ],
)
def test_prefill_and_decode(
    mesh_device: Any,
    batch_size: int,
    prompt_length: int,
    num_decode_steps: int,
    h2d_mode: ttnn.H2DMode,
) -> None:
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    fifo_size: int = page_size_bytes(batch_size)

    device_coord: ttnn.MeshCoordinate = ttnn.MeshCoordinate(0, 0)
    core_coord: ttnn.CoreCoord = ttnn.CoreCoord(0, 0)
    socket_core: ttnn.MeshCoreCoord = ttnn.MeshCoreCoord(device_coord, core_coord)

    logger.info("Creating sockets and DeepSeekV3 model")
    h2d_socket = ttnn.H2DSocket(mesh_device, socket_core, ttnn.BufferType.L1, fifo_size, h2d_mode)
    d2h_socket = ttnn.D2HSocket(mesh_device, socket_core, fifo_size)
    model = DeepSeekV3(h2d_socket, d2h_socket, batch_size, h2d_mode)
    model.start()

    logger.info(
        f"B={batch_size}, prefill {prompt_length} tokens, then {num_decode_steps} decode steps; " f"h2d_mode={h2d_mode}"
    )

    # Phase 1: Prefill — (B, 1) tokens one-by-one
    prompt_tokens: list[torch.Tensor] = [
        torch.full((batch_size, 1), i, dtype=torch.int32) for i in range(prompt_length)
    ]
    model.prefill(prompt_tokens)
    assert model.position == prompt_length, f"Position after prefill: expected {prompt_length}, got {model.position}"

    # Phase 2: Decode — each step (B, 1) in, (B, 1) out; loopback => output == input
    for step in range(num_decode_steps):
        token_id: int = prompt_length + step
        torch_input: torch.Tensor = torch.full((batch_size, 1), token_id, dtype=torch.int32)
        input_tensor: ttnn.Tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = model.decode_step(input_tensor)
        result_torch = ttnn.to_torch(output_tensor)

        # Output is padded to PCIe alignment; valid data is first batch_size elements
        result_valid = result_torch.reshape(-1)[:batch_size].reshape(batch_size, 1)
        assert torch.equal(
            torch_input, result_valid
        ), f"Decode step {step} loopback mismatch: expected {torch_input}, got {result_valid}"

    assert (
        model.position == prompt_length + num_decode_steps
    ), f"Position after decode: expected {prompt_length + num_decode_steps}, got {model.position}"

    model.stop()
    logger.info("Prefill and decode test passed")
