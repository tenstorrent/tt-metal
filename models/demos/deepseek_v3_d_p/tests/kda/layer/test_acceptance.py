# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic CI and local real-weight acceptance for the Kimi-K3 KDA layer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import load_or_compute_cpu_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    assert_matches_reference,
    check_kimi_k3_accuracy,
    collect_mesh_accuracy_and_determinism_results,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
    make_synthetic_kimi_k3_test_case,
    mla_row_permutation,
    to_sp_input,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.timeout(900)]

_SEQUENCE = 5120
_PCC_THRESHOLD = 0.9995

# Valid interval [start, start + length) of one prefill call, from the local rows L
# per SP rank and the capacity C. Full intervals place the sequence on every rank
# boundary class; partial ones end where the end-specific paths are taken.
_INTERVALS = {
    # Natural placement.
    "full-rank0": lambda L, C: (0, C),
    # The first rank holds the head and, after the wrap, the tail.
    "full-split-rank0": lambda L, C: (32, C),
    # The sequence starts on a rank boundary away from rank 0.
    "full-boundary-rank1": lambda L, C: (L, C),
    # Split start on a rank other than 0.
    "full-split-rank1": lambda L, C: (L + 32, C),
    # The last chunk holds 15 valid rows: its padded rows are identity steps.
    "partial-chunk": lambda L, C: (0, C - 17),
    # One valid row on rank 1: the convolution carry continues rank 0's history.
    "one-row-on-rank1": lambda L, C: (0, L + 1),
    # Two valid rows in rank 0's separated tail: the carry continues the history
    # of the rank before it.
    "two-row-tail": lambda L, C: (32, C - 30),
    # A one-token prompt: the carry continues the incoming layer carry.
    "one-token": lambda L, C: (0, 1),
}


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param(
            (2, 4),
            1,
            fabric_1d_device_params(),
            id="SP2xTP4-fabric-1d",
        ),
        pytest.param((2, 4), 0, fabric_1d_device_params(), id="SP4xTP2-fabric-1d"),
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("interval", list(_INTERVALS))
def test_synthetic_kimi_k3_accuracy_and_determinism(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    device_params: dict,
    interval: str,
) -> None:
    """Gate K3 dimensions on one valid interval against Torch; repeat runs must match exactly.

    Padded rows hold NaN, so any read of them reaches the checked results. The
    reference runs on the valid prefix only. Repeated runs compare by value on device
    (``ttnn.ne``), on valid output rows and on both carries.
    """
    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    sp_size = mesh_shape[sequence_parallel_axis]
    layout = f"SP{sp_size}xTP{mesh_shape[tensor_parallel_axis]}"
    local_rows = _SEQUENCE // sp_size
    actual_start, length = _INTERVALS[interval](local_rows, _SEQUENCE)
    case = make_synthetic_kimi_k3_test_case(sequence=_SEQUENCE)
    golden_output, golden_state, reference_seconds = load_or_compute_cpu_reference(
        replace(case, hidden=case.hidden[:, :length].clone())
    )
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )

    permutation = mla_row_permutation(actual_start, sp_size, local_rows)
    padded = case.hidden.clone()
    padded[:, length:] = float("nan")
    ttnn.deallocate(hidden_tt)
    hidden_tt = to_sp_input(padded[:, permutation, :], mesh_device, sequence_parallel_axis)
    valid_rows = to_sp_input(
        (permutation < length).to(torch.bfloat16).reshape(1, -1, 1), mesh_device, sequence_parallel_axis
    )
    actual_start_tt = make_actual_start(layer.device, actual_start)
    actual_end_tt = make_actual_start(layer.device, actual_start + length)

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        initial_state = layer.allocate_state(batch_size=1)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output, state = layer.forward(hidden_tt, initial_state, actual_start_tt, actual_end_tt)
        return output, state.recurrent, state.convolution

    (output, recurrent, convolution), mismatch_markers = collect_mesh_accuracy_and_determinism_results(
        run, defined=(valid_rows, None, None)
    )
    state = KdaState(recurrent=recurrent, convolution=convolution)
    try:
        assert_matches_reference(
            output_tt=output,
            state=state,
            permutation=permutation,
            expected_output=golden_output.bfloat16(),
            expected_state=golden_state,
            mesh_device=mesh_device,
            sp_axis=sequence_parallel_axis,
            tp_axis=tensor_parallel_axis,
            config=case.config,
            label=f"Synthetic Kimi-K3 T{_SEQUENCE} {layout} {interval} start={actual_start} length={length}",
            state_linf_threshold=None,
            pcc_threshold=_PCC_THRESHOLD,
        )
        assert all(
            marker.item() == 0 for marker in mismatch_markers
        ), "synthetic Kimi-K3 valid outputs and states differ across three runs"
        print(
            "KDA_SYNTHETIC_ACCURACY_DETERMINISM="
            + json.dumps(
                {
                    "layout": layout,
                    "sequence": _SEQUENCE,
                    "weights": "deterministic synthetic",
                    "reference": "independent pure-Torch FP32 CPU reference",
                    "cpu_reference_seconds": reference_seconds,
                    "interval": interval,
                    "actual_start": actual_start,
                    "valid_length": length,
                    "padding": "NaN",
                    "pcc_threshold": _PCC_THRESHOLD,
                    "determinism_repetitions": 3,
                    "repeat_comparison": "ttnn.ne on valid rows and carries",
                    "repeats_equal": True,
                },
                sort_keys=True,
            )
        )
    finally:
        for tensor in (output, recurrent, convolution, hidden_tt, valid_rows, actual_start_tt, actual_end_tt):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params,sequence",
    [
        pytest.param((1, 8), 1, fabric_1d_device_params(), 128, id="SP1xTP8"),
        pytest.param((2, 4), 1, fabric_1d_device_params(), 128, id="SP2xTP4"),
        pytest.param((2, 4), 0, fabric_1d_device_params(), 128, id="SP4xTP2"),
        pytest.param(
            (8, 4),
            1,
            torus_xy_device_params(),
            512,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_kimi_k3_layer_1_real_weights_accuracy(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_checkpoint_dir: Path,
    sequence: int,
) -> None:
    """Real weights on valid intervals from start 0; NaN padding cannot reach the results.

    Lengths: the full capacity; a partial last chunk on the last rank (one valid row
    on SP4); two valid rows on rank 1 (SP > 1), whose carry continues rank 0's
    history; and a one-token prompt.
    """
    case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=sequence)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    mesh_shape = tuple(mesh_device.shape)
    local_rows = sequence // mesh_shape[sequence_parallel_axis]
    # Existing eight-device layouts retain their shortest common tile-aligned T=128; the
    # Galaxy case uses T=512 so SP8 has two local chunks. Keep production K3 tuning
    # except for this local grouping constraint.
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        summary_group_chunks=local_rows // ttnn.TILE_SIZE,
    )
    ttnn.deallocate(hidden_tt)
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
    lengths = dict.fromkeys(n for n in (sequence, sequence - 31, local_rows + 2, 1) if n <= sequence)
    start_tt = make_actual_start(layer.device, 0)
    for length in lengths:
        golden_output, golden_state = kda_forward_reference(case.hidden[:, :length], case.state_dict, case.config)
        padded = case.hidden.clone()
        padded[:, length:] = float("nan")
        hidden_tt = to_sp_input(padded, mesh_device, sequence_parallel_axis)
        end_tt = make_actual_start(layer.device, length)
        state = layer.allocate_state(batch_size=1)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output, state = layer.forward(hidden_tt, state, start_tt, end_tt)
        ttnn.synchronize_device(mesh_device)
        check_kimi_k3_accuracy(
            f"Kimi-K3 layer 1 {layout} length={length}",
            case,
            golden_output,
            golden_state,
            state,
            output,
            mesh_device,
            tensor_parallel_axis,
            pcc_threshold=0.9995,
            valid_length=length,
        )
        for tensor in (hidden_tt, end_tt, output, state.recurrent, state.convolution):
            ttnn.deallocate(tensor)
    ttnn.deallocate(start_tt)
