# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only, skip_for_wormhole
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sfpu_generic_moe_gate_topk import (
    invoke_with_perf,
    test_sfpu_generic_moe_gate_biased_scores,
    test_sfpu_generic_moe_gate_biased_scores_multiple_faces,
    test_sfpu_generic_moe_gate_partial_face,
    test_sfpu_generic_moe_gate_topk,
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    num_selected_experts=[4, 8, *range(9, 17)],
    full_sort=[True, False],
    normalize=[True, False],
    zero_tail=[False, True],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_generic_moe_gate_topk(
    perf_report,
    num_selected_experts,
    full_sort,
    normalize,
    zero_tail,
    run_types,
    loop_factor,
    is_perf,
):
    invoke_with_perf(
        test_sfpu_generic_moe_gate_topk,
        num_selected_experts,
        full_sort,
        normalize,
        zero_tail,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    num_total_experts=[16, 32, 48, 64, 80, 112, 128, 144, 192, 240],
    num_selected_experts=[4, 8, 12, 16],
    normalize=[False, True],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_generic_moe_gate_partial_face(
    perf_report,
    num_total_experts,
    num_selected_experts,
    normalize,
    run_types,
    loop_factor,
    is_perf,
):
    invoke_with_perf(
        test_sfpu_generic_moe_gate_partial_face,
        num_total_experts,
        num_selected_experts,
        normalize,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    num_total_experts=[80, 128, 256],
    num_selected_experts=[4, 12],
    normalize=[False, True],
    full_sort=[False, True],
    scores_include_bias=[True],
    generate_indices=[False, True],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_generic_moe_gate_biased_scores(
    perf_report,
    num_total_experts,
    num_selected_experts,
    normalize,
    full_sort,
    scores_include_bias,
    generate_indices,
    run_types,
    loop_factor,
    is_perf,
):
    invoke_with_perf(
        test_sfpu_generic_moe_gate_biased_scores,
        num_total_experts,
        num_selected_experts,
        normalize,
        full_sort,
        scores_include_bias,
        generate_indices,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    num_total_experts=[512, 1024],
    num_selected_experts=[4, 12],
    normalize=[False, True],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sfpu_generic_moe_gate_biased_scores_multiple_faces(
    perf_report,
    num_total_experts,
    num_selected_experts,
    normalize,
    run_types,
    loop_factor,
    is_perf,
):
    invoke_with_perf(
        test_sfpu_generic_moe_gate_biased_scores_multiple_faces,
        num_total_experts,
        num_selected_experts,
        normalize,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
