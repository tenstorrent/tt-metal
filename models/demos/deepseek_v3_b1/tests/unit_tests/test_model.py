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
from models.demos.deepseek_v3_b1.model import DeepSeekV3


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

    logger.info("Prefill and decode test passed")


def create_event_logging_model(*, pipeline_depth: int) -> tuple[DeepSeekV3, list[str]]:
    event_log: list[str] = []

    def write_fn(token: object) -> None:
        del token
        event_log.append("write")

    def read_fn(output_tensor: ttnn.Tensor) -> None:
        del output_tensor
        event_log.append("read")

    model = DeepSeekV3(write_fn=write_fn, read_fn=read_fn, batch_size=1, pipeline_depth=pipeline_depth)
    return model, event_log


def test_prefill_starts_readback_once_pipeline_is_saturated() -> None:
    model, event_log = create_event_logging_model(pipeline_depth=4)

    _ = model.prefill([object() for _ in range(6)])

    assert event_log[:5] == ["write", "write", "write", "write", "read"]
    assert event_log.count("write") == 6
    assert event_log.count("read") == 6
    assert model.position == 6


def test_prefill_drains_tail_when_prompt_shorter_than_pipeline_depth() -> None:
    model, event_log = create_event_logging_model(pipeline_depth=8)

    _ = model.prefill([object() for _ in range(3)])

    assert event_log == ["write", "write", "write", "read", "read", "read"]
    assert model.position == 3
