# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-layer correctness gates for the isolated halo experiments."""

from functools import partial

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.layer.test_offset import (
    _assert_matches_reference,
    _build_layer,
    _mla_row_permutation,
    _reference_case,
    _to_sp_input,
)
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_carry_experiments import composed_exchange
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _deallocate_state
from models.demos.deepseek_v3_d_p.tt.kda import kda as kda_module


@pytest.mark.parametrize(
    "mesh_device,tp_axis,device_params",
    [
        pytest.param((1, 8), 1, fabric_1d_device_params(trace_region_size=8 * 1024 * 1024), id="SP1xTP8"),
        pytest.param((2, 4), 1, fabric_1d_device_params(trace_region_size=8 * 1024 * 1024), id="SP2xTP4"),
        pytest.param((4, 2), 1, fabric_1d_device_params(trace_region_size=8 * 1024 * 1024), id="SP4xTP2"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_halo_candidates_in_layer(mesh_device, tp_axis, device_params, monkeypatch):
    torch.set_num_threads(1)
    axis = 1 - tp_axis
    sp = tuple(mesh_device.shape)[axis]
    local_rows = 640
    if sp == 1:
        # Use the established production K128 PCC policy for SP1; the small SP2
        # oracle's recurrent peak-error envelope is not calibrated for this geometry.
        from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
        from models.demos.deepseek_v3_d_p.tests.kda.utils import (
            make_kimi_k3_device_case,
            make_synthetic_kimi_k3_test_case,
        )

        case = make_synthetic_kimi_k3_test_case(sequence=local_rows)
        config, hidden = case.config, case.hidden
        expected_output, expected_state = kda_forward_reference(hidden, case.state_dict, config)
        expected_output = expected_output.bfloat16()
        layer, unused = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=tp_axis, cache_weights=False)
        ttnn.deallocate(unused)
    else:
        config, weights, hidden, expected_output, expected_state = _reference_case(sp * local_rows)
        layer = _build_layer(mesh_device, config, weights, axis, tp_axis, summary_group_chunks=20)
    for actual_start in (0, local_rows, 32, (sp - 1) * local_rows + 320):
        permutation = _mla_row_permutation(actual_start, sp, local_rows)
        hidden_tt = _to_sp_input(hidden[:, permutation, :], mesh_device, axis)
        for variant in ("baseline", "shared", "separate_padded", "separate_compact", "separate_broadcast"):
            with monkeypatch.context() as patch:
                if variant != "baseline":
                    patch.setattr(
                        kda_module, "exchange_split_convolution_carry", partial(composed_exchange, variant=variant)
                    )
                scans = []
                original_scan = ttnn.experimental.kda.recurrent_chunk_scan

                def counted_scan(*args, **kwargs):
                    scans.append(kwargs["groups_per_head"])
                    return original_scan(*args, **kwargs)

                patch.setattr(ttnn.experimental.kda, "recurrent_chunk_scan", counted_scan)
                state = layer.allocate_state(batch_size=1)
                for _ in range(2):
                    output, next_state = layer.forward(hidden_tt, state, actual_start)
                    ttnn.synchronize_device(mesh_device)
                    ttnn.deallocate(output)
                    _deallocate_state(next_state)
                trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                output, next_state = layer.forward(hidden_tt, state, actual_start)
                ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
                assert len(scans) == 3, scans  # two warm forwards plus capture
                assert len(set(scans)) == 1, scans
                if variant == "baseline":
                    baseline_groups = scans[0]
                assert scans[0] == baseline_groups

                try:
                    for _ in range(2):
                        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
                        _assert_matches_reference(
                            output_tt=output,
                            state=next_state,
                            permutation=permutation,
                            expected_output=expected_output,
                            expected_state=expected_state,
                            mesh_device=mesh_device,
                            sp_axis=axis,
                            tp_axis=tp_axis,
                            config=config,
                            label=f"{variant} SP{sp} start={actual_start}",
                            state_linf_threshold=None if sp == 1 else 0.6,
                            pcc_threshold=0.9995 if sp == 1 else 0.999,
                        )
                finally:
                    ttnn.release_trace(mesh_device, trace)
                    ttnn.deallocate(output)
                    _deallocate_state(next_state)
                    _deallocate_state(state)
        ttnn.deallocate(hidden_tt)
