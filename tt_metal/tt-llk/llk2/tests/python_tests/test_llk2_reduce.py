# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 reduce through the Compute API: a row max or row sum over [rt, ct] tiles into [rt, 1] tiles, unseeded or
seeded with a previous result (SDPA's running max, m_new = max(m_prev, rowmax(S))).

Every row of the input is reduced over all ct tiles of its tile row into column 0 of the output tile; every other
datum of the output is 0, as LLK 1.0's reduce leaves it after the pack reduce mask. The scaler is one tile of 1.0. The
seed is a previous result: its values in column 0, and the other columns filled with noise, which the reduce must
ignore. Max is exact, so it is compared bit for bit; the sum accumulates in a BF16 Dest and gets a tolerance.

test_llk2_reduce_window reduces a view of a wider tensor: ct tile columns from column col0 of a [rt, parent_ct]
grid (llk2::cols), whose tile rows are parent_ct tiles apart. The columns outside the view hold values above every
row max, which a reduce that walked the parent's tile rows densely would pick up. SDPA's narrow last K chunk.

Shapes, the kind and the seed are compile time in LLK 2.0 and reach the kernel as constexprs (LLK2_REDUCE_PARAMS),
the formats with compile_time_formats=True.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize_block
from llk2_common import TILE_DIM
from llk2_paths import cpp_source, include_dirs

TILE_GRIDS = [
    (1, 1),
    (1, 4),
    (1, 16),
    (2, 4),
]  # (rt, ct): ct input tiles per output tile
# (rt, ct, parent_ct, col0): ct tile columns from column col0 of a [rt, parent_ct] grid. [3, 8] of [3, 16] is the
# row max of SDPA's last K chunk (a group of 3 Q rows, 8 valid key tiles of 16).
WINDOW_GRIDS = [
    (3, 8, 16, 0),
    (2, 2, 4, 0),
    (2, 2, 4, 2),
    (4, 1, 3, 1),
    (1, 4, 8, 4),
]
LARGE_NEGATIVE = -3.0e38  # -inf-like: SDPA's masked scores and its first running max
OUTSIDE = 500.0  # outside a window: above every row max and every seed


@dataclass
class LLK2_REDUCE_PARAMS(TemplateParameter):
    rt_dim: int
    ct_dim: int
    is_sum: bool
    seeded: bool
    parent_ct: int = (
        0  # tile columns of the tensor in buffer A; 0: ct_dim (the input is the whole tensor)
    )
    col0: int = 0  # the input's first tile column in it

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr std::uint32_t PARENT_CT = {self.parent_ct or self.ct_dim};\n"
            f"constexpr std::uint32_t WINDOW_COL0 = {self.col0};\n"
            f"constexpr bool IS_SUM = {str(self.is_sum).lower()};\n"
            f"constexpr bool SEEDED = {str(self.seeded).lower()};"
        )


def _src_precision(values, fmt, is_sum=False):
    """values as the pool reads them when unpacked from fmt. BF16: as they are. Float32: a Src register holds its top
    19 bits (TF32), which a max compares; a sum's multiplier reads the top 16 (BF16 precision, cut towards zero).
    """
    if fmt == DataFormat.Float32:
        dropped = 0xFFFF if is_sum else 0x1FFF
        bits = values.to(torch.float32).contiguous().view(torch.int32) & ~dropped
        return bits.view(torch.float32)
    return values.to(torch.bfloat16).float()


def _data(rows, cols, is_sum, generator):
    """The input: normal noise, with rows that are all negative, and (for max) rows of -inf-like values."""
    data = torch.randn(rows, cols, generator=generator) * 4
    data[
        1::5
    ] -= 50  # whole rows below zero: a max into a zeroed (not -inf) Dest would read 0
    if not is_sum:
        data[2::7] = LARGE_NEGATIVE  # whole rows at the -inf-like value
        data[3::7, ::3] = LARGE_NEGATIVE  # rows with some -inf-like datums
    return data


def _seed(rows, is_sum, generator):
    """A previous result, [rows, 32]: values in column 0 (some above the new row maxima, some below, some -inf-like),
    noise elsewhere that the reduce must ignore."""
    seed = torch.randn(rows, TILE_DIM, generator=generator) * 1000  # noise
    seed[:, 0] = torch.randn(rows, generator=generator) * 8
    if not is_sum:
        seed[::4, 0] = LARGE_NEGATIVE  # SDPA's first running max
        seed[1::4, 0] = 100.0  # above every new row max: the seed wins
    return seed


def _dest_sum(data, start, dest_acc):
    """The row sums as Dest accumulates them, and how large the running sum got: from start, each pool adds one face
    row's 16 datums and the Dest row rounds to its width (BF16, or FP32 with a 32-bit Dest), two pools per tile (its
    left and right faces).
    """
    dest_type = torch.float32 if dest_acc == DestAccumulation.Yes else torch.bfloat16
    acc = start.to(dest_type)
    largest = acc.float().abs()
    for col in range(0, data.shape[1], 16):
        acc = (acc.float() + data[:, col : col + 16].float().sum(dim=1)).to(dest_type)
        largest = torch.maximum(largest, acc.float().abs())
    return acc.float(), largest


@parametrize(
    kind=["max", "sum"],
    seeded=[False, True],
    grid=TILE_GRIDS,
    formats=input_output_formats([DataFormat.Float16_b], same=True),
)
def test_llk2_reduce(kind, seeded, grid, formats):
    rt, ct = grid
    run_reduce(kind, seeded, rt, ct, ct, 0, formats)


@parametrize(
    kind=["max", "sum"],
    seeded=[False, True],
    grid=WINDOW_GRIDS,
    formats=input_output_formats([DataFormat.Float16_b], same=True),
)
def test_llk2_reduce_window(kind, seeded, grid, formats):
    rt, ct, parent_ct, col0 = grid
    run_reduce(kind, seeded, rt, ct, parent_ct, col0, formats)


# 32-bit Dest: (input format, output format). BF16 and Float32 (TF32 in the Src registers) data; SDPA's scores are
# Float32 there and its row max BF16.
FP32_DEST_FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b),
]


@parametrize(
    kind=["max", "sum"],
    seeded=[False, True],
    grid=[
        (1, 1),
        (1, 4),
        (2, 4),
        (4, 2),
    ],  # at most 4 output tiles: a 32-bit Dest section
    formats=FP32_DEST_FORMATS,
)
def test_llk2_reduce_fp32_dest(kind, seeded, grid, formats):
    rt, ct = grid
    run_reduce(kind, seeded, rt, ct, ct, 0, formats, DestAccumulation.Yes)


@parametrize(
    kind=["max", "sum"],
    seeded=[False, True],
    grid=[(3, 8, 16, 0), (2, 2, 4, 2)],
    formats=FP32_DEST_FORMATS[2:],
)
def test_llk2_reduce_window_fp32_dest(kind, seeded, grid, formats):
    rt, ct, parent_ct, col0 = grid
    run_reduce(kind, seeded, rt, ct, parent_ct, col0, formats, DestAccumulation.Yes)


def run_reduce(
    kind, seeded, rt, ct, parent_ct, col0, formats, dest_acc=DestAccumulation.No
):
    """One reduce of ct tile columns, from column col0, of a [rt, parent_ct] grid against the golden."""
    is_sum = kind == "sum"
    rows, cols = rt * TILE_DIM, ct * TILE_DIM
    generator = torch.Generator().manual_seed(
        rt * 1000 + ct * 10 + is_sum * 2 + seeded + parent_ct * 100000 + col0 * 10000000
    )

    # The data and the seed as the Src registers hold them: the reduction of those values is the golden.
    data = _src_precision(
        _data(rows, cols, is_sum, generator), formats.input_format, is_sum
    )
    # What buffer A holds: the data in its columns, and outside them a value every reduce over them would show.
    parent = torch.full((rows, parent_ct * TILE_DIM), OUTSIDE)
    parent[:, col0 * TILE_DIM : (col0 + ct) * TILE_DIM] = data
    scaler = torch.ones(TILE_DIM, TILE_DIM)
    if is_sum and dest_acc == DestAccumulation.Yes:
        # A sum in a 32-bit Dest is a product with the scaler: it must be a reduce scaler tile, the factor in row 0
        # of every face and 0 elsewhere (what tt-metal's writers make).
        scaler = torch.zeros(TILE_DIM, TILE_DIM)
        scaler[0::16] = 1.0
    seed = _src_precision(_seed(rows, is_sum, generator), formats.input_format)
    if is_sum and dest_acc == DestAccumulation.Yes:
        # A sum by product adds into the seed where it is: the seed is a reduce result, 0 outside column 0.
        seed[:, 1:] = 0.0
    out_type = format_dict[formats.output_format]

    # Golden: the reduction of every row into column 0, every other datum 0.
    if is_sum:
        reduced, largest = _dest_sum(
            data, seed[:, 0] if seeded else torch.zeros(rows), dest_acc
        )
    else:
        reduced = data.float().max(dim=1).values
        if seeded:
            reduced = torch.maximum(reduced, seed[:, 0].float())
    golden = torch.zeros(rows, TILE_DIM)
    golden[:, 0] = reduced
    golden = tilize_block(
        golden.to(out_type),
        dimensions=[rows, TILE_DIM],
        stimuli_format=formats.output_format,
    ).flatten()

    tilized_data = tilize_block(
        parent,
        dimensions=[rows, parent_ct * TILE_DIM],
        stimuli_format=formats.input_format,
    ).flatten()
    tilized_scaler = tilize_block(
        scaler, dimensions=[TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
    ).flatten()
    tilized_seed = tilize_block(
        seed, dimensions=[rows, TILE_DIM], stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        cpp_source("llk2_reduce_compute_api_test.cpp"),
        formats,
        templates=[
            # HiFi2: the SDPA target. Max runs one phase whatever it is; the sum runs two (exact with a 1.0 scaler).
            MATH_FIDELITY(MathFidelity.HiFi2),
            DEST_SYNC(),
            LLK2_REDUCE_PARAMS(rt, ct, is_sum, seeded, parent_ct, col0),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_data,
            formats.input_format,
            tilized_scaler,
            formats.input_format,
            formats.output_format,
            tile_count_A=rt * parent_ct,
            tile_count_B=1,
            tile_count_res=rt,
            buffer_C=tilized_seed,
            stimuli_C_format=formats.input_format,
            tile_count_C=rt,
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
    res_tensor = torch.tensor(res, dtype=torch_format)
    if is_sum:
        # Every datum but column 0 is exactly 0. Column 0 against the Dest-accurate golden: the hardware rounds once
        # per pool and fidelity phase, not exactly where the golden does. Measured on ttsim: at most ~5 BF16 ulps of the
        # largest running sum (1x16 tiles); allow 2 + roundings / 8 of them. A lost tile or face is far outside it.
        ulp = 2.0**-7
        roundings = 2 * ct * 2  # pools per row x HiFi2 phases
        bound = torch.zeros(rows, TILE_DIM)
        bound[:, 0] = (2 + roundings / 8) * largest * ulp
        if dest_acc == DestAccumulation.Yes:
            # A sum in a 32-bit Dest is a product (resolver/reduce.h): the multiplier cuts every datum on its way in,
            # so the error follows the sum of the magnitudes, not the running sum. Measured on the board: at most
            # 2^-11 of that sum (HiFi2); allow 2^-9 of it on top. A lost tile or face is still far outside.
            bound[:, 0] += data.float().abs().sum(dim=1) * 2.0**-9
        bound = tilize_block(
            bound, dimensions=[rows, TILE_DIM], stimuli_format=DataFormat.Float32
        ).flatten()
        err = (res_tensor.float() - golden.float()).abs()
        bad = (err > bound).nonzero().flatten()
        worst = (err / bound.clamp(min=1e-30)).max().item()
        assert (
            bad.numel() == 0
        ), f"sum off at {bad.numel()} datums (worst {worst:.1f} x the bound), first {bad[:8].tolist()}: got {res_tensor[bad[:8]].tolist()}, want {golden[bad[:8]].tolist()}"
    elif formats.input_format == DataFormat.Float32 and out_type == torch.bfloat16:
        # A TF32 max packed to BF16: the packer drops the low mantissa bits, the golden rounds them. One BF16 ulp.
        err = (res_tensor.float() - golden.float()).abs()
        bad = (err > golden.float().abs() * 2.0**-7).nonzero().flatten()
        assert (
            bad.numel() == 0
        ), f"max off at {bad.numel()} datums, first {bad[:8].tolist()}: got {res_tensor[bad[:8]].tolist()}, want {golden[bad[:8]].tolist()}"
    else:
        # Max is exact: bit for bit.
        mismatch = (res_tensor.float() != golden.float()).nonzero().flatten()
        assert (
            mismatch.numel() == 0
        ), f"max differs at {mismatch.numel()} datums, first {mismatch[:8].tolist()}: got {res_tensor[mismatch[:8]].tolist()}, want {golden[mismatch[:8]].tolist()}"
