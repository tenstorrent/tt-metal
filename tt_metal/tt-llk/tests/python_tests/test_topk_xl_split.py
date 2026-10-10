# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
The split-thread topk_xl chunk (sources/topk_xl_split_test.cpp: SFPU on PACK, copy and transposes on MATH, two
chunks in flight) against the single-thread fused end-to-end kernel (sources/topk_xl_test.cpp) on the same
stimuli: the packed results must match bit for bit, and both must pass test_topk_xl's golden check.
"""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TOPK_XL
from test_topk_xl import (
    ELEMENTS_PER_TILE,
    FORMATS,
    _build_input,
    _check,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

FP32_FORMATS = InputOutputFormat(DataFormat.Float32, DataFormat.UInt32)

# (K, num_chunks, tail_elements, num_rows, mode, fp32 input, seg_base)
CASES = [
    (512, 1, 512, 1, "positive", False, 0),
    (512, 2, 512, 1, "positive", False, 0),
    (512, 3, 512, 1, "positive", False, 0),
    (512, 4, 256, 1, "positive", False, 0),
    (512, 5, 512, 2, "positive", False, 0),
    (512, 8, 512, 1, "signed", False, 0),
    (512, 8, 512, 1, "random", False, 0),
    (512, 2, 512, 1, "zeros_win", False, 0),
    (512, 4, 512, 1, "positive", True, 0),
    (512, 4, 512, 1, "positive", False, 32 * 512),
    (512, 32, 512, 1, "positive", False, 0),
    (512, 32, 100, 2, "planted:512", False, 0),
    (2048, 2, 2048, 1, "positive", False, 0),
    (2048, 3, 2048, 1, "positive", False, 0),
    (2048, 4, 1500, 1, "positive", False, 0),
    (2048, 5, 2048, 2, "positive", False, 0),
    (2048, 8, 2048, 1, "signed", False, 0),
    (2048, 8, 2048, 1, "random", False, 0),
    (2048, 2, 2048, 1, "zeros_win", False, 0),
    (2048, 4, 2048, 1, "positive", True, 0),
    (2048, 25, 2048, 1, "planted:2048", False, 0),
    (2048, 32, 700, 2, "planted:2048", False, 0),
]
CASE_IDS = [
    f"k{k}-chunks{c}-tail{t}-rows{r}-{m.replace(':', '')}-{'fp32' if f else 'bf16'}-seg{s}"
    for k, c, t, r, m, f, s in CASES
]


def _config(test_name, case):
    K, num_chunks, tail, num_rows, mode, fp32, seg_base = case
    formats = FP32_FORMATS if fp32 else FORMATS
    tiles_per_seq = (K + ELEMENTS_PER_TILE - 1) // ELEMENTS_PER_TILE
    src_A, rows = _build_input(K, num_chunks, tail, num_rows, mode, fp32)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=format_dict[formats.input_format])
    config = TestConfig(
        test_name=test_name,
        formats=formats,
        templates=[
            DEST_SYNC(DestSync.Full),
            TOPK_XL(
                k=K,
                num_chunks=num_chunks,
                tail_elements=tail,
                num_rows=num_rows,
                fused_e2e=True,
                seg_base=seg_base,
            ),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=num_rows * num_chunks * tiles_per_seq,
            tile_count_B=1,
            tile_count_res=num_rows * 2 * tiles_per_seq,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=fp32,
    )
    return config, rows, formats


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_topk_xl_split(case):
    K, num_chunks, tail, num_rows, mode, fp32, seg_base = case
    single, rows, formats = _config("sources/topk_xl_test.cpp", case)
    split, _, _ = _config("sources/topk_xl_split_test.cpp", case)
    # A compile-only run stops at the first run(), so build both first.
    single.prepare()
    split.prepare()

    split_result = split.run().result
    single_result = single.run().result

    out_format = format_dict[formats.output_format]
    split_words = torch.tensor(split_result, dtype=out_format)
    single_words = torch.tensor(single_result, dtype=out_format)
    differ = (split_words != single_words).nonzero().flatten()
    assert differ.numel() == 0, (
        f"split result differs from the single-thread kernel in {differ.numel()} words, "
        f"first at {differ[:8].tolist()}"
    )

    _check(
        split_result,
        K,
        rows,
        compare_index_set=not mode.startswith("random"),
        index_offset=seg_base,
        value_format=formats.input_format,
    )
