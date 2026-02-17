# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for DeepSeek V3 B1 model (prefill harness and decode flow).

Uses HostInterface in loopback mode as a mock decoder; validates prefill
(token-by-token with outputs discarded) and autoregressive decode (input → output).
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.model import TOKEN_ID_BYTES, DeepSeekV3, page_size_bytes, to_padded_input


@pytest.mark.parametrize("loopback_mode", [True, False])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("prompt_length", [1, 8, 64, 128])
@pytest.mark.parametrize("num_decode_steps", [1, 16, 32])
def test_prefill_and_decode(
    mesh_device: ttnn.MeshDevice,
    loopback_mode: bool,
    batch_size: int,
    prompt_length: int,
    num_decode_steps: int,
) -> None:
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")
    if not loopback_mode:
        pytest.skip("Non-loopback mode is not currently supported")

    fifo_size = page_size_bytes(batch_size)

    device_coord: ttnn.MeshCoordinate = ttnn.MeshCoordinate(0, 0)
    core_coord: ttnn.CoreCoord = ttnn.CoreCoord(0, 0)
    socket_core: ttnn.MeshCoreCoord = ttnn.MeshCoreCoord(device_coord, core_coord)

    logger.info("Creating sockets and DeepSeekV3 model (prefill=DEVICE_PULL, decode=HOST_PUSH)")
    h2d_socket_prefill = ttnn.H2DSocket(
        mesh_device, socket_core, ttnn.BufferType.L1, fifo_size, ttnn.H2DMode.DEVICE_PULL
    )
    h2d_socket_decode = ttnn.H2DSocket(mesh_device, socket_core, ttnn.BufferType.L1, fifo_size, ttnn.H2DMode.HOST_PUSH)
    d2h_socket = ttnn.D2HSocket(mesh_device, socket_core, fifo_size)
    model = DeepSeekV3(h2d_socket_prefill, h2d_socket_decode, d2h_socket, batch_size, loopback_mode=loopback_mode)
    model.start()

    logger.info(f"B={batch_size}, prefill {prompt_length} tokens, then {num_decode_steps} decode steps")

    # Phase 1: Prefill - list of padded ttnn.Tensor tokens
    page_size_datums = page_size_bytes(batch_size) // TOKEN_ID_BYTES
    prompt_tokens: list[ttnn.Tensor] = [
        to_padded_input(
            torch.full((batch_size, 1), i, dtype=torch.int32),
            batch_size,
            page_size_datums,
        )
        for i in range(prompt_length)
    ]
    model.prefill(prompt_tokens)
    assert model.position == prompt_length, f"Position after prefill: expected {prompt_length}, got {model.position}"

    # Phase 2: Decode - each step (B, 1) in, (B, 1) out; loopback => output == input
    for step in range(num_decode_steps):
        token_id = prompt_length + step
        torch_input = torch.full((batch_size, 1), token_id, dtype=torch.int32)
        input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
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
