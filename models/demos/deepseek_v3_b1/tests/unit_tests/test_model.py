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
from models.demos.deepseek_v3_b1.demo.runtime import TokenCodec, create_model


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

    logger.info("Creating DeepSeekV3 model via shared runtime helper")
    token_codec = TokenCodec(batch_size=batch_size)
    model = create_model(mesh_device=mesh_device, batch_size=batch_size, loopback_mode=loopback_mode)
    model.start()

    logger.info(f"B={batch_size}, prefill {prompt_length} tokens, then {num_decode_steps} decode steps")

    try:
        # Phase 1: Prefill - list of padded ttnn.Tensor tokens
        prompt_token_ids = list(range(prompt_length))
        prompt_inputs = token_codec.make_prefill_inputs(prompt_token_ids)
        model.prefill(prompt_inputs)
        assert (
            model.position == prompt_length
        ), f"Position after prefill: expected {prompt_length}, got {model.position}"

        # Phase 2: Decode - loopback mode echoes token IDs.
        for step in range(num_decode_steps):
            token_id = prompt_length + step
            torch_input = torch.full((batch_size, 1), token_id, dtype=torch.int32)
            input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = model.decode_step(input_tensor)
            result_token_id = token_codec.extract_token_id(output_tensor)
            assert (
                result_token_id == token_id
            ), f"Decode step {step} loopback mismatch: expected token {token_id}, got {result_token_id}"

        assert (
            model.position == prompt_length + num_decode_steps
        ), f"Position after decode: expected {prompt_length + num_decode_steps}, got {model.position}"
    finally:
        model.stop()

    logger.info("Prefill and decode test passed")
