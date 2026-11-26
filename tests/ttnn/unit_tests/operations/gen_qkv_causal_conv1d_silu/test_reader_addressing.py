# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — reader addressing generality (DO NOT DELETE).

input_window = column_window: the convolution channels are columns [offset, offset + C) of a wider
input (W > C and/or offset > 0), read where they lie (RM: byte offset inside the token-row page;
TILE: page r * W_t + offset / 32 + c).
history_layout = channel_paged: history [1, C/P, 3P], row r of channel c = page c / P, element
r * P + c % P; a channel group straddling pages issues one read per contiguous run per row.

Covers:
- the four golden cells this refinement unlocks (conv1d-012, -013, -018, -019);
- pinned plans that force the t0 > 0 halo paths (RM DRAM halo rows, TILE extra untilized tile-row)
  and the later-block L1 halo under a column window, plus a channel-paged history read by a group
  whose channel slice straddles pages on more than one core;
- a width W that is not a multiple of 32 (both layouts: TILE pages stride by the padded width);
- C == 3P, where [1, 3, C] (token-paged) and [1, C/P, 3P] (channel-paged) share a shape: both
  layouts back to back must each be read correctly (page_channels keys the program cache).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.gen_qkv_causal_conv1d_silu import gen_qkv_causal_conv1d_silu, get_last_plan, plan_limits
from eval.golden_tests.gen_qkv_causal_conv1d_silu.helpers import FP32_DEST_BOUND, encode_tail
from eval.metrics import check_output
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    _BUFFER,
    _EXPECTED_OUTPUT_MEMORY,
    _MF,
    _OUTPUT_MEMORY,
    _RM,
    _TILE,
    _actual_start,
    _cancelling_operands,
    _case,
    _compute_config,
    _random_operands,
    _tolerance,
    pytorch_reference,
)


def _run_and_check(device, case):
    (_, seq, width), widths, chunk, page_channels, offset = case["inputs"]
    channels = sum(widths)
    make = _cancelling_operands if case["cancelling"] else _random_operands
    full, conv_input, history, taps = make(seq, width, channels, offset)
    history_tail = encode_tail(history, page_channels)

    input_tt = ttnn.from_torch(
        full, dtype=ttnn.bfloat16, layout=case["layout"], device=device, memory_config=_BUFFER[case["in_buffer"]]
    )
    history_tt = ttnn.from_torch(
        history_tail,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_BUFFER[case["history_buffer"]],
    )
    taps_tt = [
        ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_BUFFER[case["taps_buffer"]]
        )
        for t in taps
    ]
    outputs = gen_qkv_causal_conv1d_silu(
        input_tt,
        history_tt,
        *taps_tt,
        *widths,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=chunk),
        actual_start=_actual_start(device),
        predecessor_carry=history_tt,
        memory_config=_OUTPUT_MEMORY[case["output_memory"]],
        compute_kernel_config=_compute_config(device, case),
        input_column_offset=offset,
        page_channels=page_channels,
    )

    (q_ref, k_ref, v_ref), sums = pytorch_reference(conv_input, history, taps, widths)
    tolerance = _tolerance(case)
    for name, out, ref, width_, terms in zip("qkv", outputs, (q_ref, k_ref, v_ref), widths, sums):
        check_output(
            out, ref, shape=[1, seq, width_], dtype=ttnn.bfloat16, expected_layout=ttnn.TILE_LAYOUT, tolerance=tolerance
        )
        assert out.memory_config() == _EXPECTED_OUTPUT_MEMORY[case["output_memory"]], f"{name}: memory_config"
        if case["fp32"] and case["fidelity"] == _MF.HiFi4:
            actual = ttnn.to_torch(out).double()
            relative, absolute = FP32_DEST_BOUND
            ratio = (actual - ref).abs() / (relative * ref.abs() + absolute * terms)
            assert float(ratio.max()) <= 1.0, f"{name}: fp32 accumulation bound exceeded, {float(ratio.max()):.3f}"
    assert torch.equal(ttnn.to_torch(input_tt).reshape(full.shape), full), "input was modified"
    assert torch.equal(ttnn.to_torch(history_tt).reshape(history_tail.shape), history_tail), "history was modified"
    return outputs


# The golden cells this refinement unlocks (same inputs tuple, placements and compute config).
GOLDEN_CASES = [
    _case("conv1d-012_channel_paged_P160", ((1, 256, 2560), (512, 512, 1536), 256, 160, 0), _RM),
    _case("conv1d-013_column_window", ((1, 64, 2176), (1024, 512, 512), 64, None, 32), _RM, fidelity=_MF.HiFi2),
    _case(
        "conv1d-018_tile_l1_paged_window",
        ((1, 32, 480), (128, 128, 128), 96, 32, 64),
        _TILE,
        in_buffer="l1",
        history_buffer="l1",
        output_memory="dram_interleaved",
        fp32=True,
        fidelity=_MF.HiFi2,
    ),
    _case(
        "conv1d-019_tile_l1_paged_wide_input",
        ((1, 2080, 448), (128, 128, 128), 384, 32, 0),
        _TILE,
        in_buffer="l1",
        output_memory="l1_interleaved",
        fp32=True,
    ),
]


@pytest.mark.parametrize("case", GOLDEN_CASES)
def test_golden_cells(device, case):
    _run_and_check(device, case)


# Pinned plans: every halo source under a column window / channel-paged history.
_RM_WINDOW = _case("rm_window_paged", ((1, 128, 2240), (512, 384, 1024), 128, 160, 96), _RM)
_TILE_WINDOW = _case("tile_window_paged", ((1, 128, 2240), (512, 384, 1024), 128, 160, 96), _TILE)
_PINNED = [
    pytest.param(_RM_WINDOW, dict(max_block_rows=1, max_cores_per_group=1), 1, id="rm_l1_halo_later_blocks"),
    pytest.param(_RM_WINDOW, dict(max_block_rows=1, max_cores_per_group=2), 2, id="rm_dram_halo_t0_gt_0"),
    pytest.param(_TILE_WINDOW, dict(max_block_rows=1, max_cores_per_group=1), 1, id="tile_l1_halo_later_blocks"),
    pytest.param(_TILE_WINDOW, dict(max_block_rows=1, max_cores_per_group=2), 2, id="tile_extra_row_halo_t0_gt_0"),
    pytest.param(_TILE_WINDOW, dict(max_block_rows=2, max_cores_per_group=1), 1, id="tile_multirow_block"),
]


@pytest.mark.parametrize("case,limits,max_cpg", _PINNED)
def test_pinned_halo_paths(device, case, limits, max_cpg):
    with plan_limits(**limits):
        _run_and_check(device, case.values[0])
        plan = get_last_plan()
    assert 1 <= plan.cores_per_group <= max_cpg


@pytest.mark.parametrize("layout", [_RM, _TILE], ids=["rm", "tile"])
def test_width_not_multiple_of_32(device, layout):
    """W = 2200: RM pages are 4400 B (the read offset 2 * (offset + c0) stays 64 B aligned); TILE
    rows stride by the padded width, ceil(W / 32) tiles."""
    case = _case("w2200", ((1, 64, 2200), (512, 512, 1024), 256, 64, 128), layout).values[0]
    _run_and_check(device, case)


def test_token_and_channel_paged_same_shape(device):
    """C = 96 = 3P with P = 32: both history layouts are [1, 3, 96]; each must be read as itself."""
    token = _case("token", ((1, 64, 96), (32, 32, 32), 96, None, 0), _RM).values[0]
    channel = _case("channel", ((1, 64, 96), (32, 32, 32), 96, 32, 0), _RM).values[0]
    _run_and_check(device, token)
    _run_and_check(device, channel)
    _run_and_check(device, token)
