# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA recurrent chunk scan."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    BF16_ALLOWED,
    CHUNK_SIZE,
    PROTOCOL_NAMES,
    assert_runtime_contract,
    device_protocol,
    host_protocol,
    initial_state,
    one_core_height_sharded,
    recurrent_oracle,
    run_recurrent,
    to_device,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 2_000_000}], indirect=True),
]


@pytest.mark.parametrize(
    ("batch_heads", "num_chunks", "key_dim", "value_dim", "bf16_names"),
    [
        pytest.param(2, 2, 32, 32, frozenset(), id="direct-fp32"),
        pytest.param(2, 3, 32, 64, BF16_ALLOWED, id="direct-all-allowed-bf16"),
        pytest.param(6, 2, 64, 32, frozenset({"kd", "q_decay", "final_decay"}), id="grouped-batch-heads"),
    ],
)
def test_recurrent_chunk_scan_contract_cache_trace_and_determinism(
    device: ttnn.Device,
    batch_heads: int,
    num_chunks: int,
    key_dim: int,
    value_dim: int,
    bf16_names: frozenset[str],
) -> None:
    host_inputs = host_protocol(batch_heads, num_chunks, key_dim, value_dim, bf16_names=bf16_names)
    host_state = initial_state(batch_heads, key_dim, value_dim)
    expected = recurrent_oracle(host_inputs, host_state)
    inputs = device_protocol(host_inputs, device)
    state = to_device(host_state, device)

    assert_runtime_contract(
        device,
        (*inputs, state),
        lambda: run_recurrent(inputs, state),
        expected,
        names=("token_output", "final_state"),
        dtypes=(ttnn.bfloat16, ttnn.float32),
        shapes=((batch_heads, num_chunks, CHUNK_SIZE, value_dim), (batch_heads, key_dim, value_dim)),
    )


@pytest.mark.parametrize("host_index", range(7))
def test_recurrent_chunk_scan_rejects_host_protocol_inputs(
    device: ttnn.Device, expect_error: Callable, host_index: int
) -> None:
    host_inputs = host_protocol(2, 2, 32, 32)
    inputs = list(device_protocol(host_inputs, device))
    host = host_inputs[host_index]
    dtype = ttnn.bfloat16 if host.dtype == torch.bfloat16 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    state = to_device(initial_state(2, 32, 32), device)
    with expect_error(RuntimeError, f"{PROTOCOL_NAMES[host_index]} must be an allocated device tensor"):
        run_recurrent(tuple(inputs), state)


def test_recurrent_chunk_scan_rejects_host_initial_state(device: ttnn.Device, expect_error: Callable) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    state = ttnn.from_torch(initial_state(2, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "initial_state must be an allocated device tensor"):
        run_recurrent(inputs, state)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("v_beta_dtype", "v_beta must be FLOAT32 or BFLOAT16"),
        ("intra_dtype", "intra must be FLOAT32"),
        ("t_inv_dtype", "t_inv must be FLOAT32"),
        ("layout", "kd must use TILE layout"),
        ("rank", "v_beta must be rank 4"),
        ("shape", "kd shape mismatch"),
        ("chunk", "v_beta shape mismatch"),
        ("key_alignment", "K and V must be positive and tile aligned"),
        ("value_alignment", "K and V must be positive and tile aligned"),
        ("sharded", "v_beta must use interleaved memory"),
        ("state_dtype", "initial_state must be FLOAT32"),
        ("state_rank", "initial_state must be rank 3"),
        ("state_shape", "initial_state shape mismatch"),
        ("output_sharded", "output memory must be interleaved"),
    ],
)
def test_recurrent_chunk_scan_rejects_invalid_inputs(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    host_inputs = list(host_protocol(2, 2, 32, 32))
    inputs = list(device_protocol(host_inputs, device))
    host_state = initial_state(2, 32, 32)
    state = to_device(host_state, device)
    memory_config = None
    if case == "v_beta_dtype":
        inputs[0] = to_device(host_inputs[0], device, dtype=ttnn.bfloat8_b)
    elif case == "intra_dtype":
        inputs[3] = to_device(host_inputs[3], device, dtype=ttnn.bfloat16)
    elif case == "t_inv_dtype":
        inputs[6] = to_device(host_inputs[6], device, dtype=ttnn.bfloat16)
    elif case == "layout":
        inputs[1] = to_device(host_inputs[1], device, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "rank":
        inputs[0] = to_device(host_inputs[0].reshape(2, 64, 32), device)
    elif case == "shape":
        inputs[1] = to_device(host_inputs[1][:, :1], device)
    elif case == "chunk":
        inputs[0] = to_device(torch.randn(2, 2, 64, 32), device)
    elif case == "key_alignment":
        inputs[1] = to_device(torch.randn(2, 2, 32, 48), device)
    elif case == "value_alignment":
        inputs[0] = to_device(torch.randn(2, 2, 32, 48), device)
    elif case == "sharded":
        inputs[0] = to_device(host_inputs[0], device, memory_config=one_core_height_sharded((128, 32)))
    elif case == "state_dtype":
        state = to_device(host_state, device, dtype=ttnn.bfloat16)
    elif case == "state_rank":
        state = to_device(host_state.reshape(2, 1, 32, 32), device)
    elif case == "state_shape":
        state = to_device(host_state[:, :, :16], device)
    elif case == "output_sharded":
        memory_config = one_core_height_sharded((128, 32))
    with expect_error(RuntimeError, message):
        run_recurrent(tuple(inputs), state, memory_config=memory_config)


def test_recurrent_chunk_scan_requires_initial_state(device: ttnn.Device, expect_error: Callable) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.recurrent_chunk_scan(*inputs)


@pytest.mark.parametrize(
    "removed_keyword", ["chunk_size", "state_only", "identity_tile", "summary_pair", "output_bf16"]
)
def test_recurrent_chunk_scan_does_not_expose_prototype_modes(
    device: ttnn.Device, expect_error: Callable, removed_keyword: str
) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    state = to_device(initial_state(2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.recurrent_chunk_scan(*inputs, state, **{removed_keyword: True})
