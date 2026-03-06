# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for DeepSeek V3 B1 model (prefill harness and decode flow).

Uses create_model context manager (HostInterface loopback); validates prefill
(token-by-token with outputs discarded) and autoregressive decode (input → output).
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.runtime import TokenCodec, create_model


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("prompt_length", [1, 64, 128])
@pytest.mark.parametrize("num_decode_steps", [1, 16, 32])
def test_prefill_and_decode(
    mesh_device: ttnn.MeshDevice,
    batch_size: int,
    prompt_length: int,
    num_decode_steps: int,
) -> None:
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    logger.info("Creating DeepSeekV3 model via shared runtime helper")
    token_codec = TokenCodec(batch_size=batch_size)
    logger.info(f"B={batch_size}, prefill {prompt_length} tokens, then {num_decode_steps} decode steps")

    with create_model(mesh_device=mesh_device, batch_size=batch_size) as model:
        # Phase 1: Prefill - list of raw torch tokens (B, 1)
        prompt_token_ids = list(range(prompt_length))
        prompt_inputs = token_codec.make_prefill_inputs(prompt_token_ids)
        model.prefill(prompt_inputs)
        assert (
            model.position == prompt_length
        ), f"Position after prefill: expected {prompt_length}, got {model.position}"

        # Phase 2: Decode - loopback mode echoes token IDs.
        for step in range(num_decode_steps):
            token_id = prompt_length + step
            input_tensor = torch.full((batch_size, 1), token_id, dtype=torch.int32)
            output_token_id = model.decode_step(input_tensor)
            assert output_token_id is not None
            result_token_id = int(output_token_id)
            assert (
                result_token_id == token_id
            ), f"Decode step {step} loopback mismatch: expected token {token_id}, got {result_token_id}"

        assert (
            model.position == prompt_length + num_decode_steps
        ), f"Position after decode: expected {prompt_length + num_decode_steps}, got {model.position}"

    logger.info("Prefill and decode test passed")
