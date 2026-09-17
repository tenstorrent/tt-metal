# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runtime aligned padding with changing ownership, empty ranks, and one capture."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    assert_matches_reference as _assert_matches_reference,
    build_layer as _build_layer,
    mla_row_permutation as _mla_row_permutation,
    reference_case as _reference_case,
    to_sp_input as _to_sp_input,
)
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    reconstruct_convolution_at_sp_rank,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start

pytestmark = run_for_blackhole()


@pytest.mark.parametrize(
    "mesh_device,sp_axis,tp_axis", [((1, 8), 0, 1), ((2, 4), 0, 1), ((2, 4), 1, 0)], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 16 * 1024 * 1024}],
    indirect=True,
)
def test_padding_changing_interval_single_capture(mesh_device, sp_axis, tp_axis, device_params):
    p = tuple(mesh_device.shape)[sp_axis]
    local_rows = 256
    sequence = p * local_rows
    config, weights, hidden, _, reference_initial = _reference_case(sequence)
    layer = _build_layer(
        mesh_device, config, weights, sp_axis, tp_axis, summary_group_chunks=2, active_seq_len=sequence
    )
    start_tt = make_actual_start(mesh_device, 0)
    end_tt = make_actual_start(mesh_device, sequence)
    hidden_tt = _to_sp_input(hidden, mesh_device, sp_axis)
    zero = layer.allocate_state()
    warm_output, initial = layer.forward(hidden_tt, zero, start_tt)
    ttnn.synchronize_device(mesh_device)
    for tensor in (warm_output, zero.recurrent, zero.convolution):
        ttnn.deallocate(tensor)

    def snapshots(tensors):
        return [ttnn.to_torch(shard).clone() for tensor in tensors for shard in ttnn.get_device_tensors(tensor)]

    initial_snapshot = snapshots((initial.recurrent, initial.convolution))
    # Use the actual nonzero input carry for the independent CPU transition;
    # error from the setup invocation is not padding error in the tested call.
    history = reconstruct_convolution_at_sp_rank(
        initial.convolution,
        mesh_device,
        sp_axis,
        tp_axis,
        0,
        config.num_heads // tuple(mesh_device.shape)[tp_axis] * config.head_k_dim,
    ).float()
    reference_initial = KDAReferenceState(
        recurrent=reconstruct_state_at_sp_rank(initial.recurrent, mesh_device, sp_axis, tp_axis, 0).float(),
        q_convolution=history[..., : config.q_dim],
        k_convolution=history[..., config.q_dim : config.q_dim + config.k_dim],
        v_convolution=history[..., config.q_dim + config.k_dim :],
    )
    trimmed_carries = {}
    if p == 1:
        for length in (32, 96, 128, 224, sequence):
            trimmed_hidden = _to_sp_input(hidden[:, :length], mesh_device, sp_axis)
            trimmed_layer = _build_layer(
                mesh_device,
                config,
                weights,
                sp_axis,
                tp_axis,
                active_seq_len=length,
                summary_group_chunks=1,
            )
            trimmed_out, trimmed_state = trimmed_layer.forward(trimmed_hidden, initial, start_tt)
            trimmed_carries[length] = snapshots((trimmed_state.recurrent, trimmed_state.convolution))
            for tensor in (trimmed_hidden, trimmed_out, trimmed_state.recurrent, trimmed_state.convolution):
                ttnn.deallocate(tensor)
    full_carries = {}
    for baseline_start in (0, 2 * sequence + 32):
        permutation = _mla_row_permutation(baseline_start, p, local_rows)
        baseline_input = _to_sp_input(hidden[:, permutation], mesh_device, sp_axis)
        baseline_bound = make_actual_start(mesh_device, baseline_start)
        baseline_out, baseline_state = layer.forward(baseline_input, initial, baseline_bound)
        full_carries[baseline_start] = snapshots((baseline_state.recurrent, baseline_state.convolution))
        for tensor in (
            baseline_input,
            baseline_bound,
            baseline_out,
            baseline_state.recurrent,
            baseline_state.convolution,
        ):
            ttnn.deallocate(tensor)
    for _ in range(2):
        warm, carry = layer.forward(hidden_tt, initial, start_tt, end_tt)
        ttnn.synchronize_device(mesh_device)
        for tensor in (warm, carry.recurrent, carry.convolution):
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output, state = layer.forward(hidden_tt, initial, start_tt, end_tt)
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    # One chunk, partial/full groups, empty ranks, boundary crossings, separated tail,
    # and return to full length exercise changing bounds without recapture.
    cases = [(0, sequence), (32, 32), (96, 96), (32, 128), (32, 224)]
    if p > 1:
        cases += [(32, 256), (local_rows + 96, sequence - 64), (96, sequence - 32)]
    cases += [(2 * sequence + 32, sequence), (0, 32), (0, sequence)]
    try:
        for start, length in cases:
            expected, expected_state = kda_forward_reference(hidden[:, :length], weights, config, reference_initial)
            permutation = _mla_row_permutation(start, p, local_rows)
            for destination, value in ((start_tt, start), (end_tt, start + length)):
                source = make_actual_start(mesh_device, value)
                ttnn.copy(source, destination)
                ttnn.deallocate(source)
            previous = None
            for zero_tail in (False, True, True):
                payload = hidden.clone()
                if zero_tail:
                    payload[:, length:] = 0
                source = _to_sp_input(payload[:, permutation], mesh_device, sp_axis)
                ttnn.copy(source, hidden_tt)
                ttnn.deallocate(source)
                ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                if p == 1:
                    for padded, trimmed in zip(
                        snapshots((state.recurrent, state.convolution)),
                        trimmed_carries[length],
                        strict=True,
                    ):
                        assert_bit_identical(padded, trimmed, name="physically trimmed carry baseline")
                if length == sequence:
                    for padded, baseline in zip(
                        snapshots((state.recurrent, state.convolution)),
                        full_carries[start],
                        strict=True,
                    ):
                        assert_bit_identical(padded, baseline, name="full length unbounded carry baseline")
                _assert_matches_reference(
                    output_tt=output,
                    state=state,
                    permutation=permutation,
                    expected_output=expected.bfloat16(),
                    expected_state=expected_state,
                    mesh_device=mesh_device,
                    sp_axis=sp_axis,
                    tp_axis=tp_axis,
                    config=config,
                    label=f"padding SP{p} start={start} length={length}",
                    # Short SP1 and full-length offset baselines can exceed
                    # the inherited peak-error gate without padding. These
                    # cases require exact unpadded carries plus CPU PCC/RMSE.
                    state_linf_threshold=None if p == 1 or length == sequence else 0.60,
                )
                physical = reconstruct_sp_tp_tensor(output, mesh_device, sp_axis, tp_axis, tp_dim=2, sp_dim=1)
                valid_output = physical[:, torch.argsort(permutation)][:, :length].clone()
                current = [valid_output, *snapshots((state.recurrent, state.convolution))]
                if previous is not None:
                    for a, b in zip(previous, current, strict=True):
                        assert_bit_identical(a, b, name="padding invariance/repeated replay")
                previous = current
            for a, b in zip(initial_snapshot, snapshots((initial.recurrent, initial.convolution)), strict=True):
                assert_bit_identical(a, b, name="immutable padding input carry")
    finally:
        ttnn.release_trace(mesh_device, trace)
        for tensor in (
            start_tt,
            end_tt,
            hidden_tt,
            initial.recurrent,
            initial.convolution,
            output,
            state.recurrent,
            state.convolution,
        ):
            ttnn.deallocate(tensor)
