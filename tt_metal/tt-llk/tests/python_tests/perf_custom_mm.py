# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep of the Blackhole custom_mm LLK pair (sources/custom_mm_perf.cpp, the perf twin of test_custom_mm.py);
the per-tile figures are per weight (in1) tile."""

from dataclasses import dataclass

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN_FACE_DIMS,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    TemplateParameter,
)

pytestmark = [skip_for_wormhole, skip_for_quasar]


@dataclass
class CUSTOM_MM_PERF_FLAGS(TemplateParameter):
    """Compile-time knobs of sources/custom_mm_perf.cpp (finalize requires split_acc)."""

    split_acc: bool = False
    finalize: bool = False
    dense_layout: bool = True
    clear_src: bool = True

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool SPLIT_ACC = {str(self.split_acc).lower()};",
                f"constexpr bool FINALIZE = {str(self.finalize).lower()};",
                f"constexpr bool DENSE_PACKING = {str(self.dense_layout).lower()};",
                f"constexpr bool CLEAR_SRC = {str(self.clear_src).lower()};",
            ]
        )


BF16 = DataFormat.Float16_b

# (ct, kt): a one-tile call, a one-column K loop, a square call, a short wide call and the DeepSeek decode call.
CALL_SHAPES = [(1, 1), (1, 16), (4, 4), (8, 2), (8, 16)]


def loop_factor(kt, ct):
    # Short calls loop longer so the TILE_LOOP zone stays well above the zone overhead.
    return 256 if kt * ct < 8 else 64


@pytest.mark.perf
@parametrize(
    in1_format=[DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Bfp4_b],
    in0_face_r_dim=[1, 8],
    call_shape=CALL_SHAPES,
    split_acc=[False, True],
)
def test_perf_custom_mm(perf_report, in1_format, in0_face_r_dim, call_shape, split_acc):
    ct, kt = call_shape
    configuration = PerfConfig(
        "sources/custom_mm_perf.cpp",
        InputOutputFormat(BF16, BF16, in1_format),
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
            CUSTOM_MM_PERF_FLAGS(
                split_acc=split_acc,
                finalize=split_acc,
                dense_layout=True,
                clear_src=True,
            ),
        ],
        runtimes=[
            # in0 and the result use 2 faces of M rows; in1 uses 4 full faces.
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(in0_face_r_dim=in0_face_r_dim),
            TILE_COUNT(kt * ct),
            LOOP_FACTOR(loop_factor(kt, ct)),
        ],
        variant_stimuli=StimuliConfig(
            None,
            BF16,
            None,
            in1_format,
            BF16,
            tile_count_A=kt,
            tile_count_B=kt * ct,
            tile_count_res=ct,
        ),
        dest_acc=DestAccumulation.No,
    )
    configuration.run(perf_report)
