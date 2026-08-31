# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small valid KDA workload used by the Tracy performance-model report test."""

import pytest

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import test_affine_exclusive_scan as affine
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import test_prepare_chunk_recurrence as prepare
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import test_qkv_causal_conv1d_silu as qkv
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import test_reduce_affine_transforms as reduce
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import test_sigmoid_gated_rms_norm as rms
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    device_protocol,
    host_protocol,
    initial_state,
    run_recurrent,
    run_summary,
    to_device,
)

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device({"l1_small_size": 24576})]


def test_kda_performance_model_workload(device: ttnn.Device) -> None:
    _, rms_inputs = rms._device_inputs(device)
    rms._run(*rms_inputs)

    _, (qkv_input, history, taps) = qkv._device_inputs(device)
    qkv._run(qkv_input, history, taps, channel_chunk_size=sum(qkv._DEFAULT_WIDTHS))

    reduce_case = reduce._SMALL_CASE
    reduce_host = reduce._host_inputs(
        reduce_case.batch_heads,
        reduce_case.groups_per_head,
        reduce_case.key_dim,
        reduce_case.value_dim,
    )
    reduce_inputs = tuple(reduce._to_device(tensor, device) for tensor in reduce_host)
    reduce._run(*reduce_inputs, reduce_case.groups_per_head)

    affine_case = affine._UNIT_CASE
    affine_host = affine._host_inputs(
        affine_case.batch_heads,
        affine_case.groups_per_head,
        affine_case.key_dim,
        affine_case.value_dim,
    )
    affine_inputs = tuple(affine._to_device(tensor, device) for tensor in affine_host)
    affine._run(*affine_inputs, affine_case.groups_per_head)

    prepare_host = prepare._host_inputs(num_heads=2, num_chunks=1, key_dim=32, value_dim=32)
    prepare._run(prepare._device_inputs(prepare_host, device), num_heads=2)

    protocol_host = host_protocol(batch_heads=2, num_chunks=1, key_dim=32, value_dim=32)
    protocol = device_protocol(protocol_host, device)
    state = to_device(initial_state(batch_heads=2, key_dim=32, value_dim=32), device)
    run_recurrent(protocol, state)
    run_summary(protocol)
