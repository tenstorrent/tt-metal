# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA recurrence summaries."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    BF16_ALLOWED,
    PROTOCOL_NAMES,
    assert_runtime_contract,
    assert_summary_reconstructs_state,
    device_protocol,
    group_summary_height_sharded,
    host_protocol,
    run_summary,
    summary_oracle,
    to_device,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 2_000_000}], indirect=True),
]


@pytest.mark.parametrize(
    ("batch_heads", "num_chunks", "dim", "bf16_names"),
    [
        pytest.param(2, 2, 32, frozenset(), id="direct-fp32"),
        pytest.param(4, 2, 64, BF16_ALLOWED, id="direct-all-allowed-bf16"),
        pytest.param(8, 1, 32, frozenset({"v_beta", "kd", "final_decay"}), id="grouped-batch-heads"),
    ],
)
def test_summarize_chunk_recurrence_contract_cache_trace_and_semantics(
    device: ttnn.Device,
    batch_heads: int,
    num_chunks: int,
    dim: int,
    bf16_names: frozenset[str],
) -> None:
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, bf16_names=bf16_names, seed=811)
    expected = summary_oracle(host_inputs)
    inputs = device_protocol(host_inputs, device)

    first = assert_runtime_contract(
        device,
        inputs,
        lambda: run_summary(inputs),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.float32, ttnn.float32),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
    )
    assert_summary_reconstructs_state(host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]))


def test_summarize_chunk_recurrence_height_sharded_l1_output(device: ttnn.Device) -> None:
    batch_heads, num_chunks, dim = 4, 2, 32
    host_inputs = host_protocol(batch_heads, num_chunks, dim, dim, seed=812)
    expected = summary_oracle(host_inputs)
    inputs = device_protocol(host_inputs, device)
    output_memory = group_summary_height_sharded(device, batch_heads, dim)

    first = assert_runtime_contract(
        device,
        inputs,
        lambda: run_summary(inputs, memory_config=output_memory),
        expected,
        names=("affine_a", "affine_b"),
        dtypes=(ttnn.float32, ttnn.float32),
        shapes=((batch_heads, dim, dim), (batch_heads, dim, dim)),
        expected_memory_config=output_memory,
    )
    assert_summary_reconstructs_state(host_inputs, ttnn.to_torch(first[0]), ttnn.to_torch(first[1]))


@pytest.mark.parametrize("host_index", range(7))
def test_summarize_chunk_recurrence_rejects_host_protocol_inputs(
    device: ttnn.Device, expect_error: Callable, host_index: int
) -> None:
    host_inputs = host_protocol(2, 2, 32, 32)
    inputs = list(device_protocol(host_inputs, device))
    host = host_inputs[host_index]
    dtype = ttnn.bfloat16 if host.dtype == torch.bfloat16 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, f"{PROTOCOL_NAMES[host_index]} must be an allocated device tensor"):
        run_summary(tuple(inputs))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("key_value_mismatch", "K must equal V"),
        ("q_decay_dtype", "q_decay must be FLOAT32 or BFLOAT16"),
        ("intra_dtype", "intra must be FLOAT32"),
    ],
)
def test_summarize_chunk_recurrence_rejects_invalid_inputs(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    host_inputs = list(host_protocol(2, 2, 32, 32))
    inputs = list(device_protocol(host_inputs, device))
    memory_config = None
    if case == "key_value_mismatch":
        host_inputs = list(host_protocol(2, 2, 32, 64))
        inputs = list(device_protocol(host_inputs, device))
    elif case == "q_decay_dtype":
        inputs[2] = to_device(host_inputs[2], device, dtype=ttnn.bfloat8_b)
    elif case == "intra_dtype":
        inputs[3] = to_device(host_inputs[3], device, dtype=ttnn.bfloat16)
    with expect_error(RuntimeError, message):
        run_summary(tuple(inputs), memory_config=memory_config)


@pytest.mark.parametrize(
    "removed_keyword",
    ["chunk_size", "initial_state", "state_only", "identity_tile", "summary_pair", "output_bf16", "raw_seed"],
)
def test_summarize_chunk_recurrence_does_not_expose_prototype_modes(
    device: ttnn.Device, expect_error: Callable, removed_keyword: str
) -> None:
    inputs = device_protocol(host_protocol(2, 2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.summarize_chunk_recurrence(*inputs, **{removed_keyword: True})
