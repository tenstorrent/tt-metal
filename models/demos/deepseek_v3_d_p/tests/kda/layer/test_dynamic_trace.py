# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Changing actual_start values use one capture, including transitions into/out of local splits."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    assert_matches_reference,
    build_layer,
    deallocate_state,
    mla_row_permutation,
    reference_case,
    to_sp_input,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start


@pytest.mark.parametrize(
    "mesh_device,sp_axis,tp_axis,local_rows",
    [
        ((1, 8), 0, 1, 2560),
        ((2, 4), 0, 1, 2560),
        ((2, 4), 1, 0, 2560),
        ((2, 4), 0, 1, 640),
        ((4, 2), 0, 1, 640),
        ((4, 2), 0, 1, 2560),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_changing_actual_start_single_capture(mesh_device, sp_axis, tp_axis, device_params, local_rows):
    p = tuple(mesh_device.shape)[sp_axis]
    config, weights, hidden, expected_output, expected_state = reference_case(p * local_rows)
    layer = build_layer(
        mesh_device, config, weights, sp_axis, tp_axis, summary_group_chunks=20, active_seq_len=hidden.shape[1]
    )
    actual_start = make_actual_start(mesh_device, 0)
    hidden_tt = to_sp_input(hidden, mesh_device, sp_axis)
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
        actual_start_values = [
            0,
            32,
            640,
            1312,
            2528,
            local_rows,
            local_rows + 1312,
            (p - 1) * local_rows + 320,
            2 * p * local_rows + 32,
            0,
        ]
        for actual_start_value in actual_start_values:
            permutation = mla_row_permutation(actual_start_value, p, local_rows)
            source = to_sp_input(hidden[:, permutation, :], mesh_device, sp_axis)
            ttnn.copy(source, hidden_tt)
            ttnn.deallocate(source)
            source = make_actual_start(mesh_device, actual_start_value)
            ttnn.copy(source, actual_start)
            ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            assert_matches_reference(
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
                state_linf_threshold=0.65 if local_rows == 2560 else 0.6,
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
            permutation = mla_row_permutation(actual_start_value, p, local_rows)
            source = to_sp_input(hidden[:, permutation, :], mesh_device, sp_axis)
            ttnn.copy(source, hidden_tt)
            ttnn.deallocate(source)
            source = make_actual_start(mesh_device, actual_start_value)
            ttnn.copy(source, actual_start)
            ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            assert_matches_reference(
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
                state_linf_threshold=0.65 if local_rows == 2560 else 0.6,
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


@pytest.mark.parametrize(
    "mesh_device,tp_axis,device_params",
    [
        pytest.param((1, 8), 1, fabric_1d_device_params(), id="SP1xTP8"),
        pytest.param((8, 1), 0, fabric_1d_device_params(), id="SP1xTP8-transposed"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("local_rows", [640, 2560])
def test_production_local_trace(mesh_device, tp_axis, device_params, local_rows) -> None:
    axis = 1 - tp_axis
    sp = tuple(mesh_device.shape)[axis]
    # Use the established production K128 PCC policy for SP1; the small SP2
    # oracle's recurrent peak-error envelope is not calibrated for this geometry.
    from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import load_or_compute_cpu_reference
    from models.demos.deepseek_v3_d_p.tests.kda.utils import make_kimi_k3_device_case, make_synthetic_kimi_k3_test_case

    case = make_synthetic_kimi_k3_test_case(sequence=local_rows)
    config, hidden = case.config, case.hidden
    expected_output, expected_state, _ = load_or_compute_cpu_reference(case)
    expected_output = expected_output.bfloat16()
    layer, unused = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=tp_axis, cache_weights=False)
    ttnn.deallocate(unused)
    for actual_start in (0, local_rows, 32, (sp - 1) * local_rows + 320):
        actual_start_tt = make_actual_start(mesh_device, actual_start)
        permutation = mla_row_permutation(actual_start, sp, local_rows)
        hidden_tt = to_sp_input(hidden[:, permutation, :], mesh_device, axis)
        state = layer.allocate_state(batch_size=1)
        for _ in range(2):
            output, next_state = layer.forward(hidden_tt, state, actual_start_tt)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(output)
            deallocate_state(next_state)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        output, next_state = layer.forward(hidden_tt, state, actual_start_tt)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)

        try:
            for _ in range(2):
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                assert_matches_reference(
                    output_tt=output,
                    state=next_state,
                    permutation=permutation,
                    expected_output=expected_output,
                    expected_state=expected_state,
                    mesh_device=mesh_device,
                    sp_axis=axis,
                    tp_axis=tp_axis,
                    config=config,
                    label=f"compact SP{sp} start={actual_start}",
                    state_linf_threshold=None,
                    pcc_threshold=0.9995,
                )
        finally:
            ttnn.release_trace(mesh_device, trace)
            ttnn.deallocate(output)
            deallocate_state(next_state)
            deallocate_state(state)
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(actual_start_tt)
