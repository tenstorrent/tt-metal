# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MATH_ISOLATE perf for the topk_xl index splits (Blackhole).

Drives sources/topk_xl_split_perf.cpp: one TILE_LOOP "tile" is one call of the split
on one K-element slot, so mean(MATH_ISOLATE) of the TILE_LOOP marker is cycles per
split call. `topk_split` picks the entry point:

  RowMajor    _topk_xl_separate_indices_row_major_<K>             (Classic op path)
  Global      _topk_xl_separate_indices_row_major_global_<K>      (fused end-to-end)
  GlobalBase  _topk_xl_separate_indices_row_major_global_base_<K> (segmented fusion,
              seg_base = 32 * K)
  Separate    _topk_xl_separate_indices_<K, gid>                  (control)

Correctness of the same entry points is covered by test_topk_xl.py.
"""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PerfRunType, TopKXLSplit
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, TILE_COUNT, TOPK_XL_SPLIT

pytestmark = [skip_for_wormhole, skip_for_quasar]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    topk_xl_k=[512, 1024, 2048],
    topk_split=list(TopKXLSplit),
    loop_factor=[64],
)
def test_perf_topk_xl_split(perf_report, formats, topk_xl_k, topk_split, loop_factor):
    tile_count = 1
    configuration = PerfConfig(
        "sources/topk_xl_split_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[TOPK_XL_SPLIT(topk_xl_k=topk_xl_k, topk_split=topk_split)],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(loop_factor)],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        dest_acc=DestAccumulation.Yes,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    configuration.run(perf_report)
