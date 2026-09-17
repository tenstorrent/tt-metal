# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Changing actual_start values use one capture, including transitions into/out of local splits."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.layer.test_offset import (
    _assert_matches_reference,
    _build_layer,
    _mla_row_permutation,
    _reference_case,
    _to_sp_input,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_actual_start
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical


@pytest.mark.parametrize(
    "mesh_device,sp_axis,tp_axis", [((1, 8), 0, 1), ((2, 4), 0, 1), ((2, 4), 1, 0)], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 16 * 1024 * 1024}],
    indirect=True,
)
def test_changing_offset_single_capture(mesh_device, sp_axis, tp_axis, device_params):
    p = tuple(mesh_device.shape)[sp_axis]
    local_rows = 2560
    config, weights, hidden, expected_output, expected_state = _reference_case(p * local_rows)
    layer = _build_layer(mesh_device, config, weights, sp_axis, tp_axis, summary_group_chunks=20)
    actual_start = make_actual_start(mesh_device, 0)
    hidden_tt = _to_sp_input(hidden, mesh_device, sp_axis)
    initial = layer.allocate_state(batch_size=1)
    trace = None
    output = state = None

    def snapshots(tensors):
        return [ttnn.to_torch(s).clone() for t in tensors for s in ttnn.get_device_tensors(t)]

    initial_snapshot = snapshots((initial.recurrent, initial.convolution))
    try:
        for _ in range(2):
            warm, carry = layer.forward(hidden_tt, initial, actual_start)
            ttnn.synchronize_device(mesh_device)
            for t in (warm, carry.recurrent, carry.convolution):
                ttnn.deallocate(t)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output, state = layer.forward(hidden_tt, initial, actual_start)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        actual_start_values = [0, 32, 640, 1312, 2528, local_rows, local_rows + 1312, 2 * p * local_rows + 32, 0]
        for actual_start_value in actual_start_values:
            permutation = _mla_row_permutation(actual_start_value, p, local_rows)
            source = _to_sp_input(hidden[:, permutation, :], mesh_device, sp_axis)
            ttnn.copy(source, hidden_tt)
            ttnn.deallocate(source)
            source = make_actual_start(mesh_device, actual_start_value)
            ttnn.copy(source, actual_start)
            ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            _assert_matches_reference(
                output_tt=output,
                state=state,
                permutation=permutation,
                expected_output=expected_output,
                expected_state=expected_state,
                mesh_device=mesh_device,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                config=config,
                label=f"dynamic SP{p} actual_start_value={actual_start_value}",
                state_linf_threshold=0.65,
            )
            before = snapshots((output, state.recurrent, state.convolution))
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            for i, (a, b) in enumerate(
                zip(before, snapshots((output, state.recurrent, state.convolution)), strict=True)
            ):
                assert_bit_identical(
                    a, b, name=f"dynamic replay SP{p} actual_start_value={actual_start_value} shard={i}"
                )
            for i, (a, b) in enumerate(
                zip(initial_snapshot, snapshots((initial.recurrent, initial.convolution)), strict=True)
            ):
                assert_bit_identical(a, b, name=f"immutable input carry shard={i}")
        # A cache owner advances carries outside the captured layer. Reuse the
        # same input addresses and verify that the layer leaves those inputs intact.
        reference_state = expected_state
        for chunk, actual_start_value in enumerate((p * local_rows + 32, 2 * p * local_rows + 640), 1):
            ttnn.copy(state.recurrent, initial.recurrent)
            ttnn.copy(state.convolution, initial.convolution)
            carry_before = snapshots((initial.recurrent, initial.convolution))
            assert all(torch.count_nonzero(t) for t in carry_before)
            expected, reference_state = kda_forward_reference(hidden, weights, config, reference_state)
            permutation = _mla_row_permutation(actual_start_value, p, local_rows)
            source = _to_sp_input(hidden[:, permutation, :], mesh_device, sp_axis)
            ttnn.copy(source, hidden_tt)
            ttnn.deallocate(source)
            source = make_actual_start(mesh_device, actual_start_value)
            ttnn.copy(source, actual_start)
            ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            _assert_matches_reference(
                output_tt=output,
                state=state,
                permutation=permutation,
                expected_output=expected.bfloat16(),
                expected_state=reference_state,
                mesh_device=mesh_device,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                config=config,
                label=f"dynamic continuation SP{p} chunk={chunk}",
                state_linf_threshold=0.65,
            )
            for i, (a, b) in enumerate(
                zip(carry_before, snapshots((initial.recurrent, initial.convolution)), strict=True)
            ):
                assert_bit_identical(a, b, name=f"immutable continuation carry shard={i}")
    finally:
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        for t in (actual_start, hidden_tt, initial.recurrent, initial.convolution):
            ttnn.deallocate(t)
        if output is not None:
            for t in (output, state.recurrent, state.convolution):
                ttnn.deallocate(t)
