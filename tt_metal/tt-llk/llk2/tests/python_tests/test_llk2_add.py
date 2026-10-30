# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 add and sub through the Compute API, plain and with a row broadcast, against the goldens of
test_eltwise_binary.py.

a is [rt, ct] tiles. b has a's shape, or with a row broadcast is one row of ct tiles (a bias) whose row 0 is added to
every row of a's matching tile column, as the matmul kernel's fused bias does. Shapes and formats are compile time in
LLK 2.0: the tile dims reach the kernel as constexprs (LLK2_ADD_PARAMS), the formats with compile_time_formats=True.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
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
from llk2_common import DEST_HALF_TILES, TILE_DIM
from llk2_paths import cpp_source, include_dirs

TILE_GRIDS = [(1, 1), (1, 3), (2, 2), (2, 4), (4, 1)]  # (rt, ct)


@dataclass
class LLK2_ADD_PARAMS(TemplateParameter):
    rt_dim: int
    ct_dim: int
    bcast_row: bool
    is_sub: bool

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr bool BCAST_ROW = {str(self.bcast_row).lower()};\n"
            f"constexpr bool IS_SUB = {str(self.is_sub).lower()};"
        )


def _combinations():
    combos = []
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32], same=True
    ):
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
            if fmt.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
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
    bcast_row = broadcast == BroadcastType.Row
    dims_A = [rt * TILE_DIM, ct * TILE_DIM]
    dims_B = [TILE_DIM, ct * TILE_DIM] if bcast_row else dims_A
    tile_cnt = rt * ct

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims_B,
    )
    tilized_A = tilize_block(src_A, dimensions=dims_A, stimuli_format=formats.input_format).flatten()
    tilized_B = tilize_block(src_B, dimensions=dims_B, stimuli_format=formats.input_format).flatten()

    # Golden b: one tile per output tile. With a row broadcast, output tile (r, c) takes bias tile c, broadcast.
    golden_B = tilized_B
    if bcast_row:
        golden_B = get_golden_generator(BroadcastGolden)(
            BroadcastType.Row,
            tilized_B.repeat(rt),
            formats.input_format,
            tile_cnt=tile_cnt,
        )
    golden = get_golden_generator(EltwiseBinaryGolden)(
        math_op,
        tilized_A,
        golden_B,
        formats.output_format,
        MathFidelity.LoFi,
        input_format=formats.input_format,
        input_format_B=None if bcast_row else formats.input_format,  # the broadcast golden already quantized b
        dest_acc=dest_acc,
    )

    configuration = TestConfig(
        cpp_source("llk2_add_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(kernel_fidelity),
            DEST_SYNC(),
            LLK2_ADD_PARAMS(rt, ct, bcast_row, math_op == MathOperation.Elwsub),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A,
            formats.input_format,
            tilized_B,
            formats.input_format,
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
    assert len(res) == len(golden), "Result tensor and golden tensor are not of the same length"
    torch_format = format_dict[formats.output_format]
    assert passed_test(golden, torch.tensor(res, dtype=torch_format), formats.output_format), "Assert against golden failed"
