# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Rank-stamped TopK with a NARROW tag field (TOPK_TAG_BITS < 16).

The rank-stamped engine tags each value word's low bits with a sign-conditioned local rank so the
plain unstable network sorts ties by index. With bf16 values the whole low half is free and the
default field is 16 bits. fp32 keys whose low mantissa bits are known to be zero (TF32-unpacked
words: 13 zero bits) can only spare a narrow field; the MoE gate's k=32 chains use 6 bits.

A narrow field changes the LLK mechanics: the stamp clears the stale field with an SFPAND against
a programmed mask instead of SFPLOADI LOWER 0, the complement and clear masks (LREG12 / LREG14) are
derived from the width, the merge's right-run complement 2K-1 must fit the field, and the strip
clears only the field. This test runs the whole rank-stamped chain (stamp, local sort, re-stamping
merge, rebuild, strip) with 6- and 8-bit fields on the adversarial tie classes and demands the
exact result of the 16-bit engine: the canonical stable-argsort golden, indices and value bits.
The tag never touches bits above the field, so with bf16 values every result must be bit-identical
to the wide-tag engine; the fp32-key contract (field inside the TF32 zero bits) is pinned at the
op level by the MoE gate tie-order suites.
"""
import pytest
import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, TopKSortDirection, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    INPUT_DIMENSIONS,
    TILE_COUNT,
    TOPK,
)
from helpers.utils import _RECORD_TEST_ORDER
from test_topk import (
    NUM_STAGES,
    _adversarial_value_bits,
    _canon_stable_golden,
    _extract_topk_values_and_indices,
    _hex_row,
    _model_write_path_bits,
    prepare_input_tensor_for_topk,
    transform_result_tensor_to_right_form,
)

pytestmark = [skip_for_quasar]

# Every class here plants exact ties, so the tag order is what decides the result.
TIE_CLASSES = ["tie_straddle_k", "neg_ties", "mixed_sign_ties", "signed_zero"]
INPUT_DIMENSIONS_2D = [
    32,
    128,
]  # one 2-tile slab per stage, the rank-stamped harness width
K = 32


@parametrize(
    sort_direction=[TopKSortDirection.Descending, TopKSortDirection.Ascending],
    tag_bits=[6, 8],
    stimuli_class=TIE_CLASSES,
)
def test_topk_narrow_rank_tag(
    sort_direction: TopKSortDirection, tag_bits: int, stimuli_class: str
):
    formats = input_output_formats([DataFormat.Float16_b])[0]
    descending = sort_direction == TopKSortDirection.Descending
    torch.manual_seed(0)

    num_rows, num_cols = INPUT_DIMENSIONS_2D
    w_values = num_cols // NUM_STAGES

    filler_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_DIMENSIONS_2D,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_DIMENSIONS_2D,
        spec_A=filler_spec,
        spec_B=filler_spec,
    )

    row_bits = _adversarial_value_bits(stimuli_class, num_rows, w_values)
    src_A = src_A.clone().view(num_rows, num_cols)
    src_A[:, :w_values] = row_bits.view(torch.bfloat16)
    src_A = src_A.flatten()
    src_A = prepare_input_tensor_for_topk(src_A, formats, INPUT_DIMENSIONS_2D)

    configuration = TestConfig(
        test_name="sources/topk_test.cpp",
        formats=formats,
        templates=[
            DEST_SYNC(),
            TOPK(
                topk_k=K,
                topk_matrix_width=num_cols,
                topk_sort_direction=sort_direction,
                topk_stable_sort=False,
                topk_fused_stable=False,
                topk_rank_stamped=True,
                topk_tag_bits=tag_bits,
            ),
        ],
        runtimes=[
            INPUT_DIMENSIONS(num_rows // 32, num_cols // 32),
            TILE_COUNT(tile_cnt_A),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=DestAccumulation.Yes,  # tagged keys are 32-bit words
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = transform_result_tensor_to_right_form(
        res_tensor, formats, K, INPUT_DIMENSIONS_2D
    )

    if _RECORD_TEST_ORDER:
        return

    res_value_bits, res_indices = _extract_topk_values_and_indices(
        res_tensor, formats, num_rows, K
    )
    as_written_bits = _model_write_path_bits(row_bits)

    for row in range(num_rows):
        exp_bits, exp_idx = _canon_stable_golden(as_written_bits[row], K, descending)
        if torch.equal(res_indices[row], exp_idx) and torch.equal(
            res_value_bits[row], exp_bits
        ):
            continue
        pytest.fail(
            "\n".join(
                [
                    f"narrow-tag rank-stamped topk mismatch: class={stimuli_class} tag_bits={tag_bits} "
                    f"direction={sort_direction.name} row={row}",
                    f"  expected indices:    {exp_idx.tolist()}",
                    f"  expected value bits: {_hex_row(exp_bits)}",
                    f"  result indices:      {res_indices[row].tolist()}",
                    f"  result value bits:   {_hex_row(res_value_bits[row])}",
                ]
            )
        )
