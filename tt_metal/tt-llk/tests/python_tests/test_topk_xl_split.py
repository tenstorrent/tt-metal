# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The split-thread topk_xl chunk against the single-thread fused kernel: packed results match bit for bit."""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    TopKSortDirection,
    format_dict,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TOPK_XL
from test_topk_xl import (
    ELEMENTS_PER_TILE,
    FORMATS,
    _build_input,
    _check,
    _extract_hw_topk,
    _positional_rank,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]

FP32_FORMATS = InputOutputFormat(DataFormat.Float32, DataFormat.UInt32)

# (K, num_chunks, tail_elements, num_rows, mode, fp32 input, seg_base); the last chunk's rebuild descends
DESCENDING_CASES = [
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
    (512, 4, 512, 3, "random", False, 0),
    (512, 6, 300, 4, "signed", False, 0),
    (1024, 2, 1024, 1, "positive", False, 0),
    (1024, 3, 1024, 1, "positive", False, 0),
    (1024, 4, 700, 1, "positive", False, 0),
    (1024, 5, 1024, 2, "positive", False, 0),
    (1024, 8, 1024, 1, "signed", False, 0),
    (1024, 8, 1024, 1, "random", False, 0),
    (1024, 2, 1024, 1, "zeros_win", False, 0),
    (1024, 4, 1024, 1, "positive", True, 0),
    (1024, 4, 1024, 1, "positive", False, 32 * 1024),
    (1024, 32, 1024, 1, "planted:1024", False, 0),
    (1024, 32, 300, 2, "planted:1024", False, 0),
    (1024, 2, 1024, 3, "random", False, 0),
    (1024, 4, 700, 3, "signed", False, 0),
    (1024, 6, 1024, 4, "random", True, 0),
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
    (2048, 2, 2048, 3, "random", False, 0),
    (2048, 2, 1500, 4, "signed", False, 0),
    (2048, 4, 2048, 3, "random", False, 0),
    (2048, 6, 1000, 4, "random", True, 0),
    (2048, 24, 2048, 3, "planted:2048", False, 0),
]
# The last chunk's rebuild ascends, as for a mirrored segment of an op; two chunks put it in the paired head.
ASCENDING_CASES = [
    (512, 2, 512, 1, "positive", False, 0),
    (512, 5, 512, 2, "positive", False, 0),
    (1024, 2, 1024, 1, "positive", False, 0),
    (1024, 4, 700, 2, "positive", False, 0),
    (1024, 32, 1024, 1, "planted:1024", False, 0),
    (2048, 2, 2048, 1, "positive", False, 0),
    (2048, 5, 2048, 2, "positive", False, 0),
    (2048, 24, 1500, 1, "planted:2048", False, 0),
]
CASES = [(*c, False) for c in DESCENDING_CASES] + [(*c, True) for c in ASCENDING_CASES]
CASE_IDS = [
    f"k{k}-chunks{c}-tail{t}-rows{r}-{m.replace(':', '')}-{'fp32' if f else 'bf16'}-seg{s}"
    f"{'-asc' if a else ''}"
    for k, c, t, r, m, f, s, a in CASES
]


def _config(test_name, case, ascending=None):
    K, num_chunks, tail, num_rows, mode, fp32, seg_base, case_ascending = case
    ascending = case_ascending if ascending is None else ascending
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
                sort_direction=(
                    TopKSortDirection.Ascending
                    if ascending
                    else TopKSortDirection.Descending
                ),
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
    K, _, _, _, mode, _, seg_base, ascending = case
    single, rows, formats = _config("sources/topk_xl_test.cpp", case)
    split, _, _ = _config("sources/topk_xl_split_test.cpp", case)
    # A compile-only run stops at the first run(), so build both first.
    single.prepare()
    split.prepare()
    if ascending:
        descending, _, _ = _config(
            "sources/topk_xl_split_test.cpp", case, ascending=False
        )
        descending.prepare()

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

    if ascending:
        # The ascending rebuild only reorders the survivors: each lane holds the descending run's rank from the other end.
        descending_result = descending.run().result
        for r, ((_, asc_val), (_, desc_val)) in enumerate(
            zip(
                _extract_hw_topk(split_result, K, rows.shape[0]),
                _extract_hw_topk(descending_result, K, rows.shape[0]),
            )
        ):
            mismatch = int(
                torch.count_nonzero(
                    _positional_rank(asc_val, descending=False)
                    != _positional_rank(desc_val, descending=True)
                )
            )
            assert mismatch == 0, (
                f"row {r}: ascending final rebuild is not the mirror of the descending one "
                f"in {mismatch}/{K} lanes"
            )
