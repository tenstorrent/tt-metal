# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""custom_mm perf: split_acc against plain accumulation, and the DEST layouts (default, dense, compact).

The ct = 1 cases are the decode shapes the gpt_oss fused-decode and deepseek DRAM-streaming
matmuls run: every k-tile accumulates into the same two DEST blocks, so the MVMUL
read-after-write stall is exposed unless split_acc spreads the writes over four blocks.
"""

from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN_FACE_DIMS,
    LOOP_FACTOR,
    NUM_FACES,
    TemplateParameter,
)


@dataclass
class CUSTOM_MM_LAYOUT(TemplateParameter):
    split_acc: bool = False
    dense_packing: bool = False
    compact_packing: bool = False
    num_calls: int = 1

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool CUSTOM_MM_SPLIT_ACC = {str(self.split_acc).lower()};",
                f"constexpr bool CUSTOM_MM_DENSE = {str(self.dense_packing).lower()};",
                f"constexpr bool CUSTOM_MM_COMPACT = {str(self.compact_packing).lower()};",
                f"constexpr std::uint32_t CUSTOM_MM_NUM_CALLS = {self.num_calls};",
            ]
        )


def _run(perf_report, M, kt, ct, in1_format, dest_acc, layout, loop_factor):
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b, in1_format)
    configuration = PerfConfig(
        "sources/custom_mm_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
        ],
        templates=[CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt), layout],
        runtimes=[
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(in0_face_r_dim=M),
            LOOP_FACTOR(loop_factor),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format_B,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
    )
    configuration.run(perf_report)


IN1_FORMATS = [DataFormat.Bfp4_b, DataFormat.Bfp8_b, DataFormat.Float16_b]


@pytest.mark.perf
@pytest.mark.parametrize("split_acc", [False, True])
@pytest.mark.parametrize("kt", [8, 32, 128])
@pytest.mark.parametrize("in1_format", IN1_FORMATS)
def test_perf_custom_mm_ct1(perf_report, in1_format, kt, split_acc):
    # 32-bit DEST, as the gpt_oss callers run; the 16-bit one measured the same.
    _run(
        perf_report,
        1,
        kt,
        1,
        in1_format,
        DestAccumulation.Yes,
        CUSTOM_MM_LAYOUT(split_acc=split_acc),
        loop_factor=max(1, 512 // kt),
    )


@pytest.mark.perf
@pytest.mark.parametrize("split_acc", [False, True])
@pytest.mark.parametrize("in1_format", [DataFormat.Bfp4_b, DataFormat.Bfp8_b])
def test_perf_custom_mm_ct1_multi_call(perf_report, in1_format, split_acc):
    """K over 4 calls into one DEST, finalized only on the last, as the K-subblocked DRAM-streaming matmuls do."""
    _run(
        perf_report,
        1,
        32,
        1,
        in1_format,
        DestAccumulation.Yes,
        CUSTOM_MM_LAYOUT(split_acc=split_acc, num_calls=4),
        loop_factor=16,
    )


# Each layout at ct = 8, which all of them fit in a 16-bit SyncHalf DEST, and at its widest:
# 8 tiles at 64 rows, 16 at 32 rows (dense, with or without split_acc), 32 at 16 rows (compact).
LAYOUT_WIDTHS = [
    ("default", CUSTOM_MM_LAYOUT(), 8),
    ("dense", CUSTOM_MM_LAYOUT(dense_packing=True), 8),
    ("dense", CUSTOM_MM_LAYOUT(dense_packing=True), 16),
    ("dense_split", CUSTOM_MM_LAYOUT(split_acc=True, dense_packing=True), 8),
    ("dense_split", CUSTOM_MM_LAYOUT(split_acc=True, dense_packing=True), 16),
    ("compact", CUSTOM_MM_LAYOUT(compact_packing=True), 8),
    ("compact", CUSTOM_MM_LAYOUT(compact_packing=True), 16),
    ("compact", CUSTOM_MM_LAYOUT(compact_packing=True), 32),
]


@pytest.mark.perf
@pytest.mark.parametrize(
    "layout,ct",
    [
        pytest.param(layout, ct, id=f"{name}-ct{ct}")
        for name, layout, ct in LAYOUT_WIDTHS
    ],
)
@pytest.mark.parametrize("kt", [4, 16])
@pytest.mark.parametrize("in1_format", [DataFormat.Bfp8_b, DataFormat.Float16_b])
def test_perf_custom_mm_layouts(perf_report, in1_format, kt, layout, ct):
    _run(
        perf_report, 8, kt, ct, in1_format, DestAccumulation.No, layout, loop_factor=16
    )
