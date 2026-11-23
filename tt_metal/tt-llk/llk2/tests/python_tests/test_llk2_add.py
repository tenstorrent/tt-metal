# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 add and sub through the Compute API, plain and with each broadcast, against the goldens of
test_eltwise_binary.py. run_eltwise_binary is shared with test_llk2_mul.py.

a is [rt, ct] tiles. b has a's shape; or with a row broadcast is one row of ct tiles (a bias) whose row 0 is added to
every row of a's matching tile column, as the matmul kernel's fused bias does; or with a column broadcast one column of
rt tiles whose column 0 is added across every column of a's matching tile row (SDPA's per-row statistics); or with a
scalar broadcast one tile whose datum [0, 0] is added everywhere. Shapes and formats are compile time in LLK 2.0: the
tile dims reach the kernel as constexprs (LLK2_ADD_PARAMS), the formats with compile_time_formats=True.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from llk2_common import DEST_HALF_TILES, TILE_DATUMS, TILE_DIM, blocks_in_pop_order
from llk2_paths import cpp_source, include_dirs

TILE_GRIDS = [(1, 1), (1, 3), (2, 2), (2, 4), (4, 1)]  # (rt, ct)

# llk2::Broadcast values (resolver/plan.h)
LLK2_BROADCAST = {
    BroadcastType.None_: 0,
    BroadcastType.Row: 1,
    BroadcastType.Column: 2,
    BroadcastType.Scalar: 3,
}

# The kernel's op (OP in llk2_add_compute_api_test.cpp)
LLK2_OP = {MathOperation.Elwadd: 0, MathOperation.Elwsub: 1, MathOperation.Elwmul: 2}


@dataclass
class LLK2_ADD_PARAMS(TemplateParameter):
    rt_dim: int
    ct_dim: int
    broadcast: BroadcastType
    math_op: MathOperation
    sections: int = 2
    accumulate: bool = False
    # 0: whole tensors; 1: one tile row at a time; 2: one tile at a time; 3: one half of the tile columns at a time
    # (tensor.h's views; 3 is a window of a wider tile grid)
    views: int = 0

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t VIEWS = {self.views};\n"
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr std::uint32_t BCAST = {LLK2_BROADCAST[self.broadcast]};\n"
            f"constexpr std::uint32_t OP = {LLK2_OP[self.math_op]};\n"
            f"#define DEST_SECTIONS {self.sections}\n"
            f"constexpr bool ACCUMULATE = {'true' if self.accumulate else 'false'};\n"
        )


def in1_tile_grid(broadcast, rt, ct):
    """b's [rows, cols] in tiles for a of [rt, ct] tiles."""
    return {
        BroadcastType.None_: (rt, ct),
        BroadcastType.Row: (1, ct),
        BroadcastType.Column: (rt, 1),
        BroadcastType.Scalar: (1, 1),
    }[broadcast]


def in1_tile_for(broadcast, r, c, in1_cols):
    """Index of the tile of b that output tile (r, c) reads."""
    br = 0 if broadcast in (BroadcastType.Row, BroadcastType.Scalar) else r
    bc = 0 if broadcast in (BroadcastType.Column, BroadcastType.Scalar) else c
    return br * in1_cols + bc


def run_eltwise_binary(
    math_op,
    broadcast,
    formats,
    dest_acc,
    kernel_fidelity,
    rt,
    ct,
    sections=2,
    accumulate=False,
    views=0,
):
    """Runs one add, sub or mul of a [rt, ct]-tile a and a b shaped for the broadcast on the simulator or the device,
    and checks it against the harness goldens. mul runs (and its golden takes) the kernel's fidelity. accumulate (mul
    only): the kernel computes (a + b) + a * b, the mul adding into the add's result in Dest. views: the kernel runs
    the op once per tile row (1) or per tile (2) on views of the tensors; the result is the same. views 3: once per
    half of a's tile columns (a window of a wider tile grid); the output holds the left half's tiles, then the right
    half's.
    """
    assert not accumulate or math_op == MathOperation.Elwmul, "accumulate: mul only"
    in1_rows, in1_cols = in1_tile_grid(broadcast, rt, ct)
    dims_A = [rt * TILE_DIM, ct * TILE_DIM]
    dims_B = [in1_rows * TILE_DIM, in1_cols * TILE_DIM]
    tile_cnt = rt * ct

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format_B,
        input_dimensions_B=dims_B,
    )
    tilized_A = tilize_block(
        src_A, dimensions=dims_A, stimuli_format=formats.input_format
    ).flatten()
    tilized_B = tilize_block(
        src_B, dimensions=dims_B, stimuli_format=formats.input_format_B
    ).flatten()

    # Golden b: one tile per output tile. With a broadcast, output tile (r, c) takes the tile of b it reads, broadcast.
    bcast = broadcast != BroadcastType.None_
    golden_B = tilized_B
    if bcast:
        b_tiles = tilized_B.view(in1_rows * in1_cols, TILE_DATUMS)
        per_output = torch.cat(
            [
                b_tiles[in1_tile_for(broadcast, r, c, in1_cols)]
                for r in range(rt)
                for c in range(ct)
            ]
        )
        golden_B = get_golden_generator(BroadcastGolden)(
            broadcast,
            per_output,
            formats.input_format_B,
            tile_cnt=tile_cnt,
        )
    golden_fidelity = (
        kernel_fidelity if math_op == MathOperation.Elwmul else MathFidelity.LoFi
    )

    def eltwise_golden(op, fidelity):
        return get_golden_generator(EltwiseBinaryGolden)(
            op,
            tilized_A,
            golden_B,
            formats.output_format,
            fidelity,
            input_format=formats.input_format,
            input_format_B=(
                None if bcast else formats.input_format_B
            ),  # the broadcast golden already quantized b
            dest_acc=dest_acc,
        )

    golden = eltwise_golden(math_op, golden_fidelity)
    if accumulate:
        golden = (
            eltwise_golden(MathOperation.Elwadd, MathFidelity.LoFi).to(torch.float32)
            + golden.to(torch.float32)
        ).to(format_dict[formats.output_format])

    if views == 3:
        golden = blocks_in_pop_order(golden, rt, ct, rt, ct // 2, [(0, 0), (0, 1)])

    configuration = TestConfig(
        cpp_source("llk2_add_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(kernel_fidelity),
            DEST_SYNC(),
            LLK2_ADD_PARAMS(
                rt,
                ct,
                broadcast,
                math_op,
                sections,
                accumulate,
                views,
            ),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A,
            formats.input_format,
            tilized_B,
            formats.input_format_B,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    torch_format = format_dict[formats.output_format]
    assert passed_test(
        golden, torch.tensor(res, dtype=torch_format), formats.output_format
    ), "Assert against golden failed"


def _combinations():
    combos = []
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32], same=True
    ):
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
            if (
                fmt.input_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                continue
            for rt, ct in TILE_GRIDS:
                if rt * ct <= DEST_HALF_TILES[dest_acc]:
                    combos.append((fmt, dest_acc, (rt, ct)))
    return combos


@parametrize(
    # The kernel's fidelity (kernel_settings): add and sub run one phase whatever it is; HiFi4 is what a matmul
    # kernel with a fused bias sets.
    kernel_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub],
    broadcast=[BroadcastType.None_, BroadcastType.Row],
    format_dest_acc_and_grid=_combinations(),
)
def test_llk2_add(kernel_fidelity, math_op, broadcast, format_dest_acc_and_grid):
    formats, dest_acc, (rt, ct) = format_dest_acc_and_grid
    run_eltwise_binary(math_op, broadcast, formats, dest_acc, kernel_fidelity, rt, ct)


# Column and scalar broadcast (SDPA: scores minus the row max). Float16_b with a 16-bit Dest, the SDPA target, on a
# few grids, plus one 32-bit Dest case; [1, 16] is one SDPA score row (one Col tile across 16), in a one-section ring.
BCAST_CASES = [
    (DestAccumulation.No, (1, 1), 2),
    (DestAccumulation.No, (1, 4), 2),
    (DestAccumulation.No, (2, 4), 2),
    (DestAccumulation.No, (1, 16), 1),
    (DestAccumulation.Yes, (2, 2), 2),
]


@parametrize(
    kernel_fidelity=[MathFidelity.HiFi2],
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub],
    broadcast=[BroadcastType.Column, BroadcastType.Scalar],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    case=BCAST_CASES,
)
def test_llk2_add_bcast(kernel_fidelity, math_op, broadcast, formats, case):
    dest_acc, (rt, ct), sections = case
    run_eltwise_binary(
        math_op, broadcast, formats, dest_acc, kernel_fidelity, rt, ct, sections
    )


# The grids SDPA runs its column- and scalar-broadcast ops on: [1, 8] is its P block (one row max tile under 8 score
# tiles), [2, 4] its U rescale (two runs), [3, 1] its L rescale (every run one tile), [1, 16] a one-section score row
# (tiles 8..15 in Dest's upper half).
SDPA_GRID_CASES = [
    (DestAccumulation.No, (1, 1), 2),
    (DestAccumulation.No, (1, 8), 2),
    (DestAccumulation.No, (2, 4), 2),
    (DestAccumulation.No, (3, 1), 2),
    (DestAccumulation.No, (1, 16), 1),
    (DestAccumulation.Yes, (2, 2), 2),
]


@parametrize(
    kernel_fidelity=[MathFidelity.HiFi2],
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    broadcast=[BroadcastType.Column, BroadcastType.Scalar],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    case=SDPA_GRID_CASES,
)
def test_llk2_eltwise_sdpa_grids(kernel_fidelity, math_op, broadcast, formats, case):
    dest_acc, (rt, ct), sections = case
    run_eltwise_binary(
        math_op,
        broadcast,
        formats,
        dest_acc,
        kernel_fidelity,
        rt,
        ct,
        sections,
    )


# The same grids with mul's other forms: one phase (LoFi), four phases, and .accumulate ((a + b) + a * b).
@parametrize(
    kernel_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4],
    broadcast=[BroadcastType.None_, BroadcastType.Row, BroadcastType.Column],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    case=[
        (DestAccumulation.No, (2, 4), 2, False),
        (DestAccumulation.No, (2, 4), 2, True),
        (DestAccumulation.Yes, (2, 2), 2, True),
    ],
)
def test_llk2_mul_sdpa_grids(kernel_fidelity, broadcast, formats, case):
    dest_acc, (rt, ct), sections, accumulate = case
    run_eltwise_binary(
        MathOperation.Elwmul,
        broadcast,
        formats,
        dest_acc,
        kernel_fidelity,
        rt,
        ct,
        sections,
        accumulate=accumulate,
    )


# A 32-bit Dest with the formats SDPA has there: Float32 scores and sums beside Float16_b statistics, in either
# operand, packed as Float32 or Float16_b; and block-float data. InputOutputFormat(in0, out, in1). A Float32 operand
# goes through its Src register as TF32.
FP32_DEST_FORMATS = [
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b, DataFormat.Float32),
    InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float32, DataFormat.Float16_b),
]


@parametrize(
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    broadcast=[BroadcastType.None_, BroadcastType.Column],
    formats=FP32_DEST_FORMATS,
    grid=[(1, 4)],
)
def test_llk2_eltwise_fp32_dest(math_op, broadcast, formats, grid):
    rt, ct = grid
    run_eltwise_binary(
        math_op, broadcast, formats, DestAccumulation.Yes, MathFidelity.HiFi2, rt, ct
    )


# mul with and without .accumulate there ((a + b) + a * b): SDPA's U rescale added into P V.
@parametrize(
    kernel_fidelity=[MathFidelity.HiFi2],
    broadcast=[BroadcastType.None_, BroadcastType.Column],
    formats=FP32_DEST_FORMATS[:3],
    accumulate=[False, True],
)
def test_llk2_mul_fp32_dest(kernel_fidelity, broadcast, formats, accumulate):
    run_eltwise_binary(
        MathOperation.Elwmul,
        broadcast,
        formats,
        DestAccumulation.Yes,
        kernel_fidelity,
        2,
        2,
        accumulate=accumulate,
    )


# Views (tensor/tensor.h: row, cols<N>): the same op once per tile row or once per tile, each on views of a, b and
# the output. Same goldens as the whole-tensor op: a view at a wrong address or of a wrong size reads or writes the
# wrong tiles. [2, 4] has several rows and columns, [3, 1] is a column of tiles, [1, 8] one long row.
@parametrize(
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    broadcast=[
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    grid=[(2, 4), (3, 1), (1, 8)],
    views=[1, 2],
)
def test_llk2_eltwise_views(math_op, broadcast, formats, grid, views):
    rt, ct = grid
    run_eltwise_binary(
        math_op,
        broadcast,
        formats,
        DestAccumulation.No,
        MathFidelity.HiFi2,
        rt,
        ct,
        2,
        views=views,
    )


# Windows (tensor/tensor.h: cols<N> of a tensor with several tile rows): the op once per half of a's tile columns.
# A window's tile rows are the parent's width apart, not its own: an op that steps by the window's width reads the
# wrong tiles from the second tile row on. [2, 4]: two windows of [2, 2]; [4, 2]: two windows of [4, 1].
@parametrize(
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    broadcast=[
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    grid=[(2, 4), (4, 2)],
)
def test_llk2_eltwise_windows(math_op, broadcast, formats, grid):
    rt, ct = grid
    run_eltwise_binary(
        math_op,
        broadcast,
        formats,
        DestAccumulation.No,
        MathFidelity.HiFi2,
        rt,
        ct,
        2,
        views=3,
    )
