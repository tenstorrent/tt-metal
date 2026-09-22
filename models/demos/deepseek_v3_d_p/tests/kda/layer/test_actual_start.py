# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA correctness when MLA supplies an offset-rotated, block-cyclic activation.

The layer must be equivalent to processing ``[actual_start, actual_start + G)`` in
natural chronological order, so every case here compares against the natural-order
reference after undoing MLA's row permutation. Output and BOTH replacement carries
are checked, because an ordering bug can leave the output plausible while
corrupting the state that feeds the next chunk.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    assert_matches_reference,
    build_layer,
    collect_mesh_accuracy_and_determinism_results,
    mla_row_permutation,
    reference_case,
    to_sp_input,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]

SEQUENCE = 1280


@pytest.mark.parametrize(
    "tensor_parallel_axis,sequence,actual_starts,summary_group_chunks,state_peak",
    [
        pytest.param(1, 5120, (0, 1280, 1312), 20, 0.65, id="SP2-four-groups-representative"),
        pytest.param(
            1,
            5120,
            tuple(start for start in range(0, 5120, 32) if start not in (0, 1280, 1312)),
            20,
            0.65,
            marks=pytest.mark.extended_actual_start,
            id="SP2-four-groups-exhaustive-remainder",
        ),
        pytest.param(0, 10240, (0, 2560, 32, 640, 1312, 2528, 6432), 20, 0.65, id="SP4-four-groups"),
        pytest.param(
            1,
            1280,
            tuple(rank * 640 + split for rank in range(2) for split in (0, 32, 320, 608)),
            5,
            0.6,
            id="SP2-boundaries-and-splits",
        ),
        pytest.param(
            0,
            1280,
            tuple(rank * 320 + split for rank in range(4) for split in (0, 32, 160, 288)),
            5,
            0.6,
            id="SP4-boundaries-and-splits",
        ),
    ],
)
def test_actual_start_matches_natural_order(
    mesh_device, tensor_parallel_axis, sequence, actual_starts, summary_group_chunks, state_peak
):
    """Accuracy over every retained rank, group-boundary and split equivalence class."""
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    local_rows = sequence // sp_size
    config, weights, hidden, expected_output, expected_state = reference_case(sequence)
    layer = build_layer(
        mesh_device,
        config,
        weights,
        sp_axis,
        tensor_parallel_axis,
        active_seq_len=sequence,
        summary_group_chunks=summary_group_chunks,
    )
    for actual_start in actual_starts:
        permutation = mla_row_permutation(actual_start, sp_size, local_rows)
        hidden_tt = to_sp_input(hidden[:, permutation, :], mesh_device, sp_axis)
        initial = layer.allocate_state()
        actual_start_tt = make_actual_start(mesh_device, actual_start)
        output = state = None
        try:
            with ttnn.manage_config("throw_exception_on_fallback", True):
                output, state = layer.forward(hidden_tt, initial, actual_start_tt)
            assert_matches_reference(
                output_tt=output,
                state=state,
                permutation=permutation,
                expected_output=expected_output,
                expected_state=expected_state,
                mesh_device=mesh_device,
                sp_axis=sp_axis,
                tp_axis=tensor_parallel_axis,
                config=config,
                label=f"SP{sp_size} T{sequence} start={actual_start}",
                state_linf_threshold=state_peak,
            )
        finally:
            for tensor in (hidden_tt, actual_start_tt, initial.recurrent, initial.convolution):
                ttnn.deallocate(tensor)
            if output is not None:
                for tensor in (output, state.recurrent, state.convolution):
                    ttnn.deallocate(tensor)


@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
@pytest.mark.parametrize("split", [False, True], ids=["baseline", "midpoint-rank1"])
def test_actual_start_is_deterministic_and_matches_reference(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    split: bool,
) -> None:
    """Each start must stay bit-identical run to run and equal to the reference."""
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    local_rows = SEQUENCE // sp_size

    config, weights, hidden, expected_output, expected_state = reference_case()
    layer = build_layer(mesh_device, config, weights, sp_axis, tensor_parallel_axis, active_seq_len=hidden.shape[1])
    actual_start = local_rows + local_rows // 2 if split else 0
    permutation = mla_row_permutation(actual_start, sp_size, local_rows)
    hidden_tt = to_sp_input(hidden[:, permutation, :], mesh_device, sp_axis)

    actual_start_tt = make_actual_start(mesh_device, actual_start)

    def run():
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output_tt, state = layer.forward(hidden_tt, layer.allocate_state(batch_size=1), actual_start_tt)
        return output_tt, state.recurrent, state.convolution

    (output_tt, recurrent_tt, convolution_tt), mismatch_markers = collect_mesh_accuracy_and_determinism_results(run)
    assert all(marker.item() == 0 for marker in mismatch_markers), "offset-aware KDA is not bit-identical"

    assert_matches_reference(
        output_tt=output_tt,
        state=KdaState(recurrent=recurrent_tt, convolution=convolution_tt),
        permutation=permutation,
        expected_output=expected_output,
        expected_state=expected_state,
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tensor_parallel_axis,
        config=config,
        label=f"tp_axis={tensor_parallel_axis} start={actual_start}",
    )


def test_split_trace_replay_is_bit_identical(mesh_device: ttnn.MeshDevice) -> None:
    """Two replays of the complete split layer must produce identical outputs and carries."""
    tensor_parallel_axis = 1
    sp_axis = 0
    sp_size = 2
    local_rows = SEQUENCE // sp_size
    actual_start = local_rows + local_rows // 2
    config, weights, hidden, expected_output, expected_state = reference_case()
    layer = build_layer(mesh_device, config, weights, sp_axis, tensor_parallel_axis, active_seq_len=hidden.shape[1])
    permutation = mla_row_permutation(actual_start, sp_size, local_rows)
    hidden_tt = to_sp_input(hidden[:, permutation, :], mesh_device, sp_axis)

    input_state = layer.allocate_state(batch_size=1)
    actual_start_tt = make_actual_start(mesh_device, actual_start)
    trace_id = None
    capturing = False
    output_tt = state = None
    try:
        for _ in range(2):
            warm_output, warm_state = layer.forward(hidden_tt, input_state, actual_start_tt)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(warm_output)
            ttnn.deallocate(warm_state.recurrent)
            ttnn.deallocate(warm_state.convolution)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        capturing = True
        output_tt, state = layer.forward(hidden_tt, input_state, actual_start_tt)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        capturing = False
        first = None
        for replay in range(3):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            assert_matches_reference(
                output_tt=output_tt,
                state=state,
                permutation=permutation,
                expected_output=expected_output,
                expected_state=expected_state,
                mesh_device=mesh_device,
                sp_axis=sp_axis,
                tp_axis=tensor_parallel_axis,
                config=config,
                label=f"split trace replay={replay}",
            )
            current = [
                ttnn.to_torch(shard).clone()
                for tensor in (output_tt, state.recurrent, state.convolution)
                for shard in ttnn.get_device_tensors(tensor)
            ]
            if first is None:
                first = current
            else:
                for index, (expected, actual) in enumerate(zip(first, current, strict=True)):
                    assert_bit_identical(expected, actual, name=f"split trace replay={replay} shard={index}")
    finally:
        try:
            if capturing:
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        finally:
            if trace_id is not None:
                ttnn.release_trace(mesh_device, trace_id)
            if output_tt is not None:
                ttnn.deallocate(output_tt)
            if state is not None:
                ttnn.deallocate(state.recurrent)
                ttnn.deallocate(state.convolution)
            ttnn.deallocate(input_state.recurrent)
            ttnn.deallocate(input_state.convolution)
            ttnn.deallocate(hidden_tt)


def test_split_offset_continuation_preserves_nonzero_caller_carries(mesh_device: ttnn.MeshDevice) -> None:
    config, weights, _, _, _ = reference_case()
    layer = build_layer(mesh_device, config, weights, 0, 1, active_seq_len=SEQUENCE)
    hidden = torch.randn(1, 3 * SEQUENCE, config.hidden_size, generator=torch.Generator().manual_seed(941)).to(
        torch.bfloat16
    )
    state = layer.allocate_state(batch_size=1)
    reference_state = None
    for chunk in range(3):
        actual_start = SEQUENCE // 2 + 32 + chunk * SEQUENCE
        part = hidden[:, chunk * SEQUENCE : (chunk + 1) * SEQUENCE]
        expected_output, reference_state = kda_forward_reference(part, weights, config, reference_state)
        permutation = mla_row_permutation(actual_start, 2, SEQUENCE // 2)
        input_tt = to_sp_input(part[:, permutation, :], mesh_device, 0)
        previous = state
        before = [
            ttnn.to_torch(shard).clone()
            for tensor in (previous.recurrent, previous.convolution)
            for shard in ttnn.get_device_tensors(tensor)
        ]
        if chunk:
            assert all(torch.count_nonzero(tensor) for tensor in before), "continuation needs nonzero caller carries"
        output = None
        try:
            with ttnn.manage_config("throw_exception_on_fallback", True):
                output, state = layer.forward(input_tt, previous, make_actual_start(layer.device, actual_start))
            assert_matches_reference(
                output_tt=output,
                state=state,
                permutation=permutation,
                expected_output=expected_output.to(torch.bfloat16),
                expected_state=reference_state,
                mesh_device=mesh_device,
                sp_axis=0,
                tp_axis=1,
                config=config,
                label=f"split continuation chunk={chunk} start={actual_start}",
            )
            after = [
                ttnn.to_torch(shard)
                for tensor in (previous.recurrent, previous.convolution)
                for shard in ttnn.get_device_tensors(tensor)
            ]
            for index, (old, new) in enumerate(zip(before, after, strict=True)):
                assert_bit_identical(old, new, name=f"caller carry shard={index}")
            for old, new in ((previous.recurrent, state.recurrent), (previous.convolution, state.convolution)):
                assert all(
                    a.buffer_address() != b.buffer_address()
                    for a, b in zip(ttnn.get_device_tensors(old), ttnn.get_device_tensors(new), strict=True)
                )
        finally:
            if output is not None:
                ttnn.deallocate(output)
            ttnn.deallocate(input_tt)
            ttnn.deallocate(previous.recurrent)
            ttnn.deallocate(previous.convolution)
    ttnn.deallocate(state.recurrent)
    ttnn.deallocate(state.convolution)
