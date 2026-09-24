# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Rank-stamped TopK with a narrow tag field (TOPK_TAG_BITS < 16).

bf16 values leave the whole low half free (16-bit tags); fp32 keys unpacked as TF32 have 13 zero
mantissa bits, and the MoE gate uses a 6-bit field there. A narrow field changes the LLK mechanics
(SFPAND clear against a programmed mask, width-derived LREG12 / LREG14 masks, the merge's 2K-1
complement must fit). This runs the whole chain (stamp, local sort, merge, rebuild, strip) with 6-
and 8-bit fields on the adversarial tie classes and demands the exact canonical stable golden,
indices and value bits. The fp32-key contract itself is pinned at op level by the MoE gate suites.
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
# One 2-tile slab per stage: the width the rank-stamped harness supports.
INPUT_DIMENSIONS_2D = [32, 128]
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
