# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# SFPU binary bcast with the data tile at a nonzero dest index and the bcast / result tiles below it (negative
# relative dest offsets), plus back-to-back calls with different layouts; and an SFPU BCAST_ROW call between
# transpose_dest init and the transpose, which must leave the FPU replay program intact.
from dataclasses import dataclass

import pytest
import test_eltwise_binary_sfpu as F
import test_transpose_dest as T
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.llk_params import DestAccumulation, MathOperation, Transpose, format_dict
from helpers.param_config import parametrize
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import TemplateParameter

# name: (data, bcast, out, pre_data); pre_data >= 0 runs a first in-place call on that tile.
LAYOUTS = {
    "L012": (0, 1, 2, -1),
    "L210": (2, 1, 0, -1),
    "L303": (3, 0, 3, -1),
    "L132p0": (1, 3, 2, 0),
}


@dataclass
class BCAST_LAYOUT(TemplateParameter):
    data: int = 0
    bcast: int = 1
    out: int = 2
    pre_data: int = -1
    runtime: int = 0

    def convert_to_cpp(self) -> str:
        return (
            f"#define BCAST_L_DATA {self.data}\n#define BCAST_L_BCAST {self.bcast}\n"
            f"#define BCAST_L_OUT {self.out}\n#define BCAST_L_PRE_DATA ({self.pre_data})\n"
            f"#define BCAST_L_RUNTIME {self.runtime}"
        )


@dataclass
class REPRO_ROW(TemplateParameter):
    scratch: int = -1

    def convert_to_cpp(self) -> str:
        return f"#define REPRO_ROW_SCRATCH ({self.scratch})"


def _skip_quasar():
    if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR:
        pytest.skip("Wormhole / Blackhole SFPU bcast layout coverage")


_OPS = [MathOperation.SfpuElwadd, MathOperation.SfpuElwsub, MathOperation.SfpuElwmul]
_FMT_ACC = [
    (DataFormat.Float16_b, DestAccumulation.No),
    (DataFormat.Float16_b, DestAccumulation.Yes),
    (DataFormat.Float32, DestAccumulation.Yes),
]
LAYOUT_CASES = [
    (f, a, LlkBroadcastType.Row, op, l)
    for l in LAYOUTS
    for f, a in _FMT_ACC
    for op in _OPS
] + [
    (f, a, LlkBroadcastType.Column, op, l)
    for l in ("L210", "L303")
    for f, a in (_FMT_ACC[0], _FMT_ACC[2])
    for op in _OPS
]


@parametrize(case=LAYOUT_CASES)
def test_sfpu_binary_bcast_layout(case):
    _skip_quasar()
    fmt, dest_acc, bcast_dim, mathop, layout = case[0]
    formats = InputOutputFormat(fmt, fmt)
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = F.generate_stimuli(
        stimuli_format_A=fmt,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=fmt,
        input_dimensions_B=input_dimensions,
    )
    golden_tensor = F._golden_sfpu_binary_bcast(
        src_A, src_B, bcast_dim, F._BCAST_BINARY_OPS[mathop], fmt
    )
    num_blocks, num_tiles_in_block = F.get_num_blocks_and_num_tiles_in_block(
        F.DestSync.Half, dest_acc, formats, input_dimensions, F.TILE_DIMENSIONS
    )
    d, b, o, p = LAYOUTS[layout]
    kwargs = {
        "test_name": "sources/sfpu_binary_bcast_layout_test.cpp",
        "formats": formats,
        "templates": [
            F.generate_input_dim(input_dimensions, input_dimensions),
            F.MATH_OP(mathop=mathop),
            F.APPROX_MODE(F.ApproximationMode.No),
            F.ITERATIONS(32),
            F.BROADCAST_TYPE(LlkBroadcastType.None_),
            F.SFPU_BCAST_DIM(bcast_dim),
            F.SFPU_DST_ROUNDING_MODE(),
            F.SIGN_MAGNITUDE_FORMAT(False),
            BCAST_LAYOUT(d, b, o, p),
        ],
        "runtimes": [
            F.TILE_COUNT(tile_cnt_A + tile_cnt_B),
            F.NUM_BLOCKS(num_blocks),
            F.NUM_TILES_IN_BLOCK(num_tiles_in_block),
            F.LOOP_FACTOR(1),
            F.NUM_FACES(),
            F.ZERO_POINT(0),
            F.OUT_ZERO_POINT(0),
        ],
        "variant_stimuli": F.StimuliConfig(
            F.tilize_block(src_A, input_dimensions, fmt).flatten(),
            fmt,
            F.tilize_block(src_B, input_dimensions, fmt).flatten(),
            fmt,
            fmt,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": F.unpacks_to_dest(fmt, dest_acc),
        "compile_time_formats": True,
    }
    cfg = F.create_test_or_perf_config(
        is_perf=False,
        run_types=F._resolve_perf_args(False, None, None),
        test_config_kwargs=kwargs,
    )
    res_from_L1 = cfg.run().result
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return
    res = torch.tensor(res_from_L1, dtype=format_dict[fmt]).flatten()
    assert len(res) == len(golden_tensor)
    F.assert_against_contract(mathop, formats, dest_acc, golden_tensor, res)


REPRO_CASES = [
    # (format, dest_acc, math_transpose_faces, unpack_to_dest)
    (DataFormat.Float16_b, DestAccumulation.No, Transpose.Yes, False),
    (DataFormat.Float32, DestAccumulation.Yes, Transpose.Yes, True),
    (DataFormat.Float32, DestAccumulation.Yes, Transpose.No, True),
]


@parametrize(case=REPRO_CASES)
def test_transpose_dest_after_sfpu_bcast_row(case):
    _skip_quasar()
    fmt, dest_acc, mtf, u2d = case[0]
    formats = InputOutputFormat(fmt, fmt)
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = T.generate_stimuli(
        stimuli_format_A=fmt,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=fmt,
        input_dimensions_B=input_dimensions,
    )
    datacopy = T.get_golden_generator(T.DataCopyGolden)(
        src_A, fmt, num_faces=4, input_dimensions=input_dimensions
    )
    golden = []
    if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
        tm = T.get_golden_generator(T.TransposeGolden)
        golden = tm.transpose_faces_multi_tile(
            datacopy,
            fmt,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=input_dimensions,
        )
        golden = tm.transpose_within_faces_multi_tile(
            golden,
            fmt,
            num_tiles=tile_cnt_A,
            untilize=False,
            input_dimensions=input_dimensions,
        )
    cfg = TestConfig(
        "sources/transpose_dest_sfpu_bcast_row_test.cpp",
        formats,
        templates=[T.MATH_TRANSPOSE_FACES(mtf), REPRO_ROW(1)],
        runtimes=[
            T.UNPACK_TRANS_FACES(
                Transpose.Yes
                if (dest_acc == DestAccumulation.Yes and mtf == Transpose.No)
                else Transpose.No
            ),
            T.TILE_COUNT(tile_cnt_A),
            T.NUM_FACES(),
            T.NUM_BLOCKS(1),
            T.NUM_TILES_IN_BLOCK(1),
        ],
        variant_stimuli=T.StimuliConfig(
            src_A,
            fmt,
            src_B,
            fmt,
            fmt,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=u2d,
    )
    res_from_L1 = cfg.run().result
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return
    res = T.result_tensor(res_from_L1, fmt, golden)
    if u2d and fmt.is_32_bit():
        assert torch.equal(
            res, golden
        ), f"{int((res != golden).sum())}/{res.numel()} elements differ"
    else:
        assert T.passed_test(golden, res, fmt)
