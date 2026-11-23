# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 pack modifiers through the Compute API: ReLU, L1 accumulation (Off, On, Runtime) and sum_tiles, behind a
matmul.

The kernel runs the matmul and packs it PASSES times, one Dest section per pass: the first pack overwrites, later
ones add into L1 (l1_acc). Goldens: the harness's MatmulGolden, then PackGolden's ReLU and L1 accumulation.

test_llk2_pack_l1_acc_onto covers .l1_acc behind a copy (llk2_pack_onto_compute_api_test.cpp): a tensor in a format of
its own added onto another in L1.

test_llk2_pack_sum_tiles covers pack_modifiers{.sum_tiles = true}: every tile of the matmul's result goes to one L1
tile, the first as l1_acc says and the rest added by the packer's L1 accumulation, so the tile ends as the
elementwise sum of the result's tiles (SDPA's softmax row sum: the exp block packed a second time into the row's
sum tile, as the 1.0 kernel does). Its golden adds the result's tiles in pack order with PackGolden.accumulate_l1.
The kernel runs the overwriting pack twice, so a first tile that added to what L1 held would double the result.
"""

from dataclasses import dataclass

import torch
from helpers.bfp_format_utils import bfp4b_to_float16b, bfp8b_to_float16b
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden, PackGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    PackerReluType,
    format_dict,
)
from helpers.matmul_sweep import (
    generate_matmul_dimension_combinations,
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    TemplateParameter,
)
from helpers.utils import passed_test
from llk2_common import (
    DEST_HALF_TILES,
    TILE_DATUMS,
    TILE_DIM,
    as_packed,
    lattice_only,
    tilize_operands,
    unwrap_case,
)
from llk2_paths import cpp_source, include_dirs

ACCUMULATING_PASSES = 3  # > 2: the Dest sections wrap around

# Output tile grids (rows, columns) of 5 to 8 tiles, which only a 16-bit Dest section holds: a multi-tile pack is
# one MOP run over all its tiles, so the larger tile counts are packed with every modifier too.
LARGE_GRIDS = [(1, 5), (2, 3), (7, 1), (1, 8), (2, 4), (8, 1)]


def _dimension_combinations(dest_acc):
    """Matmul operand dimensions: every output grid of up to 4 tiles, and with a 16-bit Dest the LARGE_GRIDS."""
    dims = generate_matmul_dimension_combinations(
        DEST_HALF_TILES[DestAccumulation.Yes], kt_dims=[1, 2]
    )
    if dest_acc == DestAccumulation.No:
        assert all(rt * ct <= DEST_HALF_TILES[dest_acc] for rt, ct in LARGE_GRIDS)
        dims += [
            ([rt * TILE_DIM, TILE_DIM], [TILE_DIM, ct * TILE_DIM])
            for rt, ct in LARGE_GRIDS
        ]
    return dims


@dataclass
class LLK2_PACK_TEST(TemplateParameter):
    rt_dim: int
    ct_dim: int
    kt_dim: int
    relu: bool
    passes: int
    l1_acc_runtime: bool  # every pass packs with l1_acc = Toggle::Runtime, accumulating from pass 1 on
    # every pack has sum_tiles: the output is one tile, the sum of the result's tiles
    sum_tiles: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr std::uint32_t KT_DIM = {self.kt_dim};\n"
            f"constexpr bool PACK_RELU = {'true' if self.relu else 'false'};\n"
            f"constexpr std::uint32_t PASSES = {self.passes};\n"
            f"constexpr bool L1_ACC_RUNTIME = {'true' if self.l1_acc_runtime else 'false'};\n"
            f"constexpr bool SUM_TILES = {'true' if self.sum_tiles else 'false'};"
        )


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
            for relu in (False, True):
                # (l1_acc, passes): Off packs once; On overwrites in pass 0 and accumulates after (two packs, Off and
                # On); Runtime is one pack whose run-time value is false in pass 0 and true after, so Runtime with one
                # pass checks the false value against Off and with ACCUMULATING_PASSES both values against On.
                for l1_acc, passes in (
                    ("Off", 1),
                    ("On", ACCUMULATING_PASSES),
                    ("Runtime", 1),
                    ("Runtime", ACCUMULATING_PASSES),
                ):
                    # L1 adds Float32, Float16_b and Float16 only (Packers/Accumulation.md); the Resolver rejects
                    # On and Runtime into Bfp8_b and Bfp4_b.
                    if l1_acc != "Off" and fmt.output_format in (
                        DataFormat.Bfp8_b,
                        DataFormat.Bfp4_b,
                    ):
                        continue
                    for dims in _dimension_combinations(dest_acc):
                        combos.append((fmt, dest_acc, relu, l1_acc, passes, dims))
    return combos


# sum_tiles: [rt, ct] result tiles into one L1 tile. [1, 8] is SDPA's exp block (a whole 16-bit Dest section),
# [2, 4] the same section as two tile rows; a 32-bit Dest section holds 4 tiles. [1, 1] is the degenerate fold: a
# plain pack.
SUM_GRIDS = {
    DestAccumulation.No: [(1, 1), (1, 2), (1, 8), (2, 4)],
    DestAccumulation.Yes: [(1, 1), (1, 4), (2, 2)],
}


def _sum_combinations():
    combos = []
    # L1 adds Float32, Float16_b and Float16 only, and sum_tiles always accumulates: no Bfp8_b output.
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32], same=True
    ):
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
            if (
                fmt.input_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                continue
            for relu in (False, True):
                # As test_llk2_pack: Off is one pass (the sum over what L1 held); On and Runtime with
                # ACCUMULATING_PASSES add the sum again in every later pass (the first tile accumulates too).
                for l1_acc, passes in (
                    ("Off", 1),
                    ("On", ACCUMULATING_PASSES),
                    ("Runtime", 1),
                    ("Runtime", ACCUMULATING_PASSES),
                ):
                    for rt, ct in SUM_GRIDS[dest_acc]:
                        # kt = 1
                        dims = ([rt * TILE_DIM, TILE_DIM], [TILE_DIM, ct * TILE_DIM])
                        combos.append((fmt, dest_acc, relu, l1_acc, passes, dims))
    return combos


@parametrize(case=_sum_combinations())
def test_llk2_pack_sum_tiles(case):
    run_pack(case, sum_tiles=True)


@parametrize(case=_combinations())
def test_llk2_pack(case):
    run_pack(case)


# A 32-bit Dest packed, and summed, into the other float format in L1: Float32 data into Float16_b tiles, Float16_b
# data into Float32 ones (a sum is rounded to the L1 format after every tile). The matmul that fills Dest runs HiFi4
# here: the harness golden models a LoFi product of Float32 operands by the output format's precision, which for
# signed data is off by more than the comparison allows, whatever the pack does.
FP32_DEST_MIXED_FORMATS = [
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]


@parametrize(
    formats=FP32_DEST_MIXED_FORMATS,
    relu=[False, True],
    l1_acc=[("Off", 1), ("Runtime", ACCUMULATING_PASSES)],
    grid=[(1, 4)],
    sum_tiles=[False, True],
)
def test_llk2_pack_fp32_dest_mixed(formats, relu, l1_acc, grid, sum_tiles):
    rt, ct = grid
    dims = ([rt * TILE_DIM, TILE_DIM], [TILE_DIM, ct * TILE_DIM])
    case = (formats, DestAccumulation.Yes, relu, l1_acc[0], l1_acc[1], dims)
    run_pack(case, sum_tiles, MathFidelity.HiFi4)


def run_pack(case, sum_tiles=False, math_fidelity=MathFidelity.LoFi):
    formats, dest_acc, relu, l1_acc, passes, (dims_A, dims_B) = unwrap_case(case)
    torch_format = format_dict[formats.output_format]

    # With ReLU, negative values so it has something to clamp. Otherwise [0, 1] as test_llk2_matmul.py: signed sums
    # near 0, added up PASSES times in BF16, can miss the relative tolerance by one rounding step.
    spec = StimuliSpec.uniform(low=-1.0 if relu else 0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims_B,
        spec_A=spec,
        spec_B=spec,
    )
    matmul_dims = generate_tile_dims((dims_A, dims_B))

    golden = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=dims_A,
        input_B_dimensions=dims_B,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    if relu:
        pack_src = infer_data_formats(
            input_format=formats.input_format,
            output_format=formats.output_format,
            is_fp32_dest_acc_en=dest_acc,
        ).pack_src
        relu_config = PackGolden.generate_relu_config(
            PackerReluType.ZeroRelu, relu_threshold=0.0, intermediate_format=pack_src
        )
        golden = PackGolden.apply_relu(golden, relu_config, pack_src)
    else:
        golden = as_packed(golden, formats.output_format)
    if sum_tiles:
        # Tile by tile in pack order, every pass: L1 rounds to the output format after each addition.
        tiles = list(golden.view(matmul_dims.output_tile_cnt, TILE_DATUMS))
        golden = PackGolden.accumulate_l1(tiles * passes, formats.output_format)
    elif passes > 1:
        golden = PackGolden.accumulate_l1([golden] * passes, formats.output_format)

    tilized_A, tilized_B = tilize_operands(
        src_A, dims_A, src_B, dims_B, formats.input_format
    )

    configuration = TestConfig(
        cpp_source("llk2_pack_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            LLK2_PACK_TEST(
                matmul_dims.rt_dim,
                matmul_dims.ct_dim,
                matmul_dims.kt_dim,
                relu,
                passes,
                l1_acc == "Runtime",
                sum_tiles,
            ),
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
            tile_count_res=1 if sum_tiles else matmul_dims.output_tile_cnt,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    res_tensor = torch.tensor(res, dtype=torch_format)
    if relu:
        assert (res_tensor >= 0).all(), "ReLU left negative values"
    assert passed_test(
        golden, res_tensor, formats.output_format, **lattice_only(formats.output_format)
    ), "Assert against golden failed"


# ---- .l1_acc behind a copy: a tensor added onto another in L1 -------------------------------------------------------
# x += y through Dest (copy, then pack with .l1_acc), y in Bfp4_b, Bfp8_b or Float16_b and x in Float16_b (16-bit
# Dest) or Float32 (32-bit Dest): the packer's L1 accumulation behind an op other than matmul, from a block-float
# source. y holds 0, a large negative value and blocks of other values (an additive mask's contents). The golden is x
# plus y as its L1 format holds it, rounded to x's format. Both stimuli are in tile order: the sum is datum by datum.
LARGE_NEGATIVE = -1.0e9


@dataclass
class LLK2_PACK_ONTO(TemplateParameter):
    tiles: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TILES = {self.tiles};"


@parametrize(
    added_format=[DataFormat.Bfp4_b, DataFormat.Bfp8_b, DataFormat.Float16_b],
    format_and_dest=[
        (DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float32, DestAccumulation.Yes),
        (DataFormat.Float16_b, DestAccumulation.Yes),
    ],
    tiles=[1, 4],
)
def test_llk2_pack_l1_acc_onto(added_format, format_and_dest, tiles):
    x_format, dest_acc = format_and_dest
    x_type = format_dict[x_format]
    generator = torch.Generator().manual_seed(tiles)
    n = tiles * TILE_DATUMS

    x = (torch.randn(n, generator=generator) * 8).to(x_type)
    # Blocks of 16 datums (one exponent each in a block-float format): all 0, all large negative, a boundary block (0
    # then large negative), and small values.
    added = torch.zeros(n // 16, 16)
    added[1::4] = LARGE_NEGATIVE
    added[2::4, 8:] = LARGE_NEGATIVE
    added[3::4] = torch.randn(added[3::4].shape, generator=generator) * 2
    added = added.flatten().to(torch.bfloat16)

    stored = {
        DataFormat.Bfp4_b: bfp4b_to_float16b,
        DataFormat.Bfp8_b: bfp8b_to_float16b,
    }.get(added_format, lambda values: values)(added)
    golden = (x.float() + stored.float()).to(x_type)

    formats = InputOutputFormat(x_format, x_format, added_format)
    to_dest = x_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    configuration = TestConfig(
        cpp_source("llk2_pack_onto_compute_api_test.cpp"),
        formats,
        templates=[DEST_SYNC(), LLK2_PACK_ONTO(tiles)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            x,
            x_format,
            added,
            added_format,
            x_format,
            tile_count_A=tiles,
            tile_count_B=tiles,
            tile_count_res=tiles,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=to_dest,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )
    res = torch.tensor(configuration.run().result, dtype=x_type).float()
    assert len(res) == len(golden), "Result and golden are not of the same length"
    # One step of x's format where the sum is rounded (BF16: 2^-7 relative), and of the large negative value.
    err = (res - golden.float()).abs()
    bound = golden.float().abs() * 2.0**-7 + 1e-6
    bad = (err > bound).nonzero().flatten()
    assert (
        bad.numel() == 0
    ), f"off at {bad.numel()} datums, first {bad[:8].tolist()}: got {res[bad[:8]].tolist()}, want {golden.float()[bad[:8]].tolist()}, added {stored.float()[bad[:8]].tolist()}"
