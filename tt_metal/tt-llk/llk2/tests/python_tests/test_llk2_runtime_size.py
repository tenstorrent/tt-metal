# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 with a tensor size known only at run time (tensor/tensor.h, "run-time sizes"): one build per bound, every
count up to the bound a run-time parameter of the same three ELFs.

test_llk2_runtime_size          add / sub / mul and pack with a run-time number of tile columns of one tile row, or of
                                tile rows (llk2_runtime_size_compute_api_test.cpp, KIND 0 and 1)
test_llk2_runtime_k             matmul with a run-time K (KIND 2)
test_llk2_runtime_views         run-time lengths through views (cols_up_to<N>(t, c0, n)): matmul with a run-time number of
                                output tile columns (KIND 3), a sum pack of a run-time number of tiles (KIND 4) and
                                a row max over a run-time row length (KIND 5), a column-broadcast sub of a
                                run-time number of tiles (KIND 6) and the fast exp on the pack thread over a
                                run-time number of tiles (KIND 7), each also with a bound of 1 where pack takes it,
                                and matmul with a run-time K and several in0 tile rows (KIND 8: SDPA's P V)
test_llk2_eltwise_kernel        the tt-metal kernel kernels/eltwise_binary_no_bcast.cpp (blocks of a run-time size),
                                the file itself, for a run-time number of tiles; with 4 tiles per cycle and with 1
                                (what ttnn passes for tensors that are not sharded: a bound of 1)
The goldens are the harness's, for the tensors of the actual size.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    MatmulGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize, runtime
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    RuntimeParameter,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from llk2_common import TILE_DATUMS, TILE_DIM, tilize_operands
from llk2_paths import cpp_source, include_dirs, kernel_test_include_dirs

# The kernel's op (OP in the test sources; binary_op in the kernels)
LLK2_OP = {MathOperation.Elwadd: 0, MathOperation.Elwsub: 1, MathOperation.Elwmul: 2}


@dataclass
class LLK2_RUNTIME_SIZE(TemplateParameter):
    # 0: run-time tile columns, 1: run-time tile rows, 2: matmul with a run-time K, 3: matmul with a run-time N through
    # views, 4: a sum pack of a run-time number of tiles, 5: reduce over a run-time row length, 6: sub with a column
    # broadcast over a run-time number of tiles, 7: the fast exp over a run-time number of tiles, 8: matmul with a
    # run-time K through views, in0 of several tile rows
    kind: int
    rt_dim: int  # the bound (kind 1), else the tile rows
    ct_dim: int  # the bound (kind 0), else the tile columns
    kt_dim: int = 1  # the bound of K (kind 2)
    op: int = 0

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t KIND = {self.kind};\n"
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr std::uint32_t KT_DIM = {self.kt_dim};\n"
            f"constexpr std::uint32_t OP = {self.op};\n"
        )


@dataclass
class LLK2_COUNT(RuntimeParameter):
    count: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t COUNT = {self.count};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t COUNT;", "I"


@dataclass
class LLK2_ELTWISE_KERNEL(TemplateParameter):
    tiles_per_cycle: int
    max_tiles: int
    op: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t TILES_PER_CYCLE = {self.tiles_per_cycle};\n"
            f"constexpr std::uint32_t MAX_TILES = {self.max_tiles};\n"
            f"constexpr std::uint32_t OP = {self.op};\n"
        )


@dataclass
class LLK2_NUM_TILES(RuntimeParameter):
    num_tiles: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t NUM_TILES = {self.num_tiles};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t NUM_TILES;", "I"


def _eltwise_stimuli_and_golden(math_op, formats, dest_acc, fidelity, rows, cols):
    """a and b of [rows, cols] tiles, tilized, and the harness golden of a <op> b."""
    dims = [rows * TILE_DIM, cols * TILE_DIM]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    tilized_A, tilized_B = tilize_operands(
        src_A, dims, src_B, dims, formats.input_format
    )
    golden = get_golden_generator(EltwiseBinaryGolden)(
        math_op,
        tilized_A,
        tilized_B,
        formats.output_format,
        fidelity if math_op == MathOperation.Elwmul else MathFidelity.LoFi,
        input_format=formats.input_format,
        input_format_B=formats.input_format,
        dest_acc=dest_acc,
    )
    return tilized_A, tile_cnt_A, tilized_B, tile_cnt_B, golden


def _run(
    source,
    formats,
    dest_acc,
    fidelity,
    templates,
    runtimes,
    a,
    tiles_a,
    b,
    tiles_b,
    golden,
    tiles_res,
    dirs,
    tolerance=None,
):
    configuration = TestConfig(
        cpp_source(source),
        formats,
        templates=[MATH_FIDELITY(fidelity), DEST_SYNC()] + templates,
        runtimes=runtimes,
        variant_stimuli=StimuliConfig(
            a,
            formats.input_format,
            b,
            formats.input_format,
            formats.output_format,
            tile_count_A=tiles_a,
            tile_count_B=tiles_b,
            tile_count_res=tiles_res,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=dirs,
    )
    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    assert passed_test(
        golden,
        torch.tensor(res, dtype=format_dict[formats.output_format]),
        formats.output_format,
        **(tolerance or {}),
    ), "Assert against golden failed"


def _formats():
    """(formats, dest_acc): Float16_b and Float32 pack as one block (one MOP run), Bfp8_b tile by tile."""
    fp16b, bfp8b, fp32 = input_output_formats(
        [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32], same=True
    )
    return [
        (fp16b, DestAccumulation.No),
        (bfp8b, DestAccumulation.No),
        (fp32, DestAccumulation.Yes),
    ]


# (kind, bound rows, bound cols): columns of one row up to a 16-bit Dest section; rows of 2 tiles; a bound of 1
_GRIDS = [(0, 1, 4), (1, 2, 2), (0, 1, 1)]


def _size_cases():
    cases = []
    for formats, dest_acc in _formats():
        for kind, rt, ct in _GRIDS:
            for op in (
                MathOperation.Elwadd,
                MathOperation.Elwsub,
                MathOperation.Elwmul,
            ):
                cases.append((formats, dest_acc, kind, rt, ct, op))
    return cases


@parametrize(case=_size_cases(), count=runtime([1, 2, 3, 4]))
def test_llk2_runtime_size(case, count):
    formats, dest_acc, kind, rt, ct, op = case
    bound = rt if kind == 1 else ct
    if count > bound:
        count = bound  # the bound of this build: the same run again
    rows, cols = (count, ct) if kind == 1 else (rt, count)
    fidelity = MathFidelity.HiFi2
    a, tiles_a, b, tiles_b, golden = _eltwise_stimuli_and_golden(
        op, formats, dest_acc, fidelity, rows, cols
    )
    _run(
        "llk2_runtime_size_compute_api_test.cpp",
        formats,
        dest_acc,
        fidelity,
        [LLK2_RUNTIME_SIZE(kind, rt, ct, 1, LLK2_OP[op])],
        [LLK2_COUNT(count)],
        a,
        tiles_a,
        b,
        tiles_b,
        golden,
        rows * cols,
        include_dirs(),
    )


@parametrize(case=_formats(), ct=[1, 4], count=runtime([1, 2, 3]))
def test_llk2_runtime_k(case, ct, count):
    formats, dest_acc = case
    kt_bound, fidelity = 3, MathFidelity.HiFi2
    dims_A, dims_B = [TILE_DIM, count * TILE_DIM], [count * TILE_DIM, ct * TILE_DIM]
    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tiles_a, src_B, tiles_b = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims_B,
        spec_A=spec,
        spec_B=spec,
    )
    golden = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        fidelity,
        input_A_dimensions=dims_A,
        input_B_dimensions=dims_B,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    a, b = tilize_operands(src_A, dims_A, src_B, dims_B, formats.input_format)
    _run(
        "llk2_runtime_size_compute_api_test.cpp",
        formats,
        dest_acc,
        fidelity,
        [LLK2_RUNTIME_SIZE(2, 1, ct, kt_bound)],
        [LLK2_COUNT(count)],
        a,
        tiles_a,
        b,
        tiles_b,
        golden,
        ct,
        include_dirs(),
    )


@parametrize(
    case=_formats(),
    tiles_per_cycle=[1, 4],  # 4: what a 32-bit Dest section holds
    op=[MathOperation.Elwadd, MathOperation.Elwmul],
    num_tiles=runtime([1, 4, 7, 8, 11]),
)
def test_llk2_eltwise_kernel(case, tiles_per_cycle, op, num_tiles):
    formats, dest_acc = case
    fidelity = MathFidelity.HiFi4  # the host's LLK2_MATH_FIDELITY for this kernel
    a, tiles_a, b, tiles_b, golden = _eltwise_stimuli_and_golden(
        op, formats, dest_acc, fidelity, 1, num_tiles
    )
    _run(
        "llk2_eltwise_kernel_compute_api_test.cpp",
        formats,
        dest_acc,
        fidelity,
        [LLK2_ELTWISE_KERNEL(tiles_per_cycle, 16, LLK2_OP[op])],
        [LLK2_NUM_TILES(num_tiles)],
        a,
        tiles_a,
        b,
        tiles_b,
        golden,
        num_tiles,
        kernel_test_include_dirs(),
    )


def _run_view_case(kind, formats, dest_acc, rt, ct, kt, count):
    """One run of KIND 3 to 7 of llk2_runtime_size_compute_api_test.cpp with COUNT = count."""
    fidelity = MathFidelity.HiFi2
    tolerance = {}
    out_fmt = format_dict[formats.output_format]
    if kind == 3:
        # a [1, kt] x b [kt, count], b laid out in the first count tile columns of a [kt, ct] grid
        dims_A, dims_B = [TILE_DIM, kt * TILE_DIM], [kt * TILE_DIM, count * TILE_DIM]
        spec = StimuliSpec.uniform(low=0.0, high=1.0)
        src_A, tiles_a, src_B, _ = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=dims_A,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=dims_B,
            spec_A=spec,
            spec_B=spec,
        )
        golden = get_golden_generator(MatmulGolden)(
            src_A,
            src_B,
            formats.output_format,
            fidelity,
            input_A_dimensions=dims_A,
            input_B_dimensions=dims_B,
            tilize=True,
            input_A_format=formats.input_format,
            input_B_format=formats.input_format,
        )
        a, b_used = tilize_operands(src_A, dims_A, src_B, dims_B, formats.input_format)
        grid = torch.zeros(kt, ct, TILE_DATUMS, dtype=b_used.dtype)
        grid[:, :count] = b_used.view(kt, count, TILE_DATUMS)
        b, tiles_b, tiles_res = grid.flatten(), kt * ct, count
    elif kind == 8:
        # a [rt, count] x b [count, ct]: a laid out in the first count tile columns of a [rt, kt] grid, b in the first
        # count tile rows of a [kt, ct] grid; count is K, at most kt
        dims_A, dims_B = [rt * TILE_DIM, count * TILE_DIM], [
            count * TILE_DIM,
            ct * TILE_DIM,
        ]
        spec = StimuliSpec.uniform(low=0.0, high=1.0)
        src_A, _, src_B, _ = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=dims_A,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=dims_B,
            spec_A=spec,
            spec_B=spec,
        )
        golden = get_golden_generator(MatmulGolden)(
            src_A,
            src_B,
            formats.output_format,
            fidelity,
            input_A_dimensions=dims_A,
            input_B_dimensions=dims_B,
            tilize=True,
            input_A_format=formats.input_format,
            input_B_format=formats.input_format,
        )
        a_used, b_used = tilize_operands(
            src_A, dims_A, src_B, dims_B, formats.input_format
        )
        # outside the K tiles used: values a product over them would show
        a_grid = torch.full((rt, kt, TILE_DATUMS), 100.0, dtype=a_used.dtype)
        a_grid[:, :count] = a_used.view(rt, count, TILE_DATUMS)
        b_pad = torch.full(
            ((kt - count) * ct * TILE_DATUMS,), 100.0, dtype=b_used.dtype
        )
        a, b = a_grid.flatten(), torch.cat([b_used, b_pad])
        tiles_a, tiles_b, tiles_res = rt * kt, kt * ct, rt * ct
    elif kind == 4:
        a_used, _, b_used, _, tiles_golden = _eltwise_stimuli_and_golden(
            MathOperation.Elwadd, formats, dest_acc, fidelity, 1, count
        )
        pad = torch.zeros((ct - count) * TILE_DATUMS, dtype=a_used.dtype)
        a, b = torch.cat([a_used, pad]), torch.cat([b_used, pad])
        tiles_a = tiles_b = ct
        # the packer adds tile after tile into L1, rounding to the L1 format each time
        acc = None
        for t in torch.as_tensor(tiles_golden).view(count, TILE_DATUMS):
            acc = (
                t.to(out_fmt) if acc is None else (acc.float() + t.float()).to(out_fmt)
            )
        golden, tiles_res = acc, 1
    elif kind == 6:
        # a [1, count] minus column 0 of the one tile m, as the broadcast tests of test_llk2_add.py take the golden
        dims_A, dims_B = [TILE_DIM, count * TILE_DIM], [TILE_DIM, TILE_DIM]
        src_A, _, src_B, _ = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=dims_A,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=dims_B,
        )
        a_used, m = tilize_operands(src_A, dims_A, src_B, dims_B, formats.input_format)
        m_per_tile = get_golden_generator(BroadcastGolden)(
            BroadcastType.Column,
            torch.cat([m] * count),
            formats.input_format,
            tile_cnt=count,
        )
        golden = get_golden_generator(EltwiseBinaryGolden)(
            MathOperation.Elwsub,
            a_used,
            m_per_tile,
            formats.output_format,
            MathFidelity.LoFi,
            input_format=formats.input_format,
            input_format_B=None,  # the broadcast golden already quantized m
            dest_acc=dest_acc,
        )
        pad = torch.zeros((ct - count) * TILE_DATUMS, dtype=a_used.dtype)
        a, b = torch.cat([a_used, pad]), m
        tiles_a, tiles_b, tiles_res = ct, 1, count
    elif kind == 7:
        # the fast exp's range (test_llk2_vec.py, the SDPA score exp): x in [-16, 0.5]; b is zeros
        generator = torch.Generator().manual_seed(100 * ct + count)
        x = (
            torch.empty(count * TILE_DATUMS)
            .uniform_(-16.0, 0.5, generator=generator)
            .to(torch.bfloat16)
        )
        pad = torch.full(((ct - count) * TILE_DATUMS,), 1000.0, dtype=torch.bfloat16)
        a, b = torch.cat([x, pad]), torch.zeros(ct * TILE_DATUMS, dtype=torch.bfloat16)
        golden = torch.exp(x.to(torch.float32))
        tiles_a = tiles_b = ct
        tiles_res = count
        tolerance = {"custom_atol": 1e-12, "custom_rtol": 5e-2}  # the fast exp's
    else:
        rows = rt * TILE_DIM
        generator = torch.Generator().manual_seed(1000 * rt + 10 * ct + count)
        data = (torch.randn(rows, ct * TILE_DIM, generator=generator) * 4).to(
            torch.bfloat16
        )
        data[1::5] -= 50  # whole rows below zero
        # outside the first count tile columns: values a max over them would show
        data[:, count * TILE_DIM :] = 1000.0
        reduced = torch.zeros(rows, TILE_DIM)
        reduced[:, 0] = data[:, : count * TILE_DIM].float().max(dim=1).values
        golden = tilize_block(
            reduced.to(torch.bfloat16),
            dimensions=[rows, TILE_DIM],
            stimuli_format=formats.output_format,
        ).flatten()
        a = tilize_block(
            data, dimensions=[rows, ct * TILE_DIM], stimuli_format=formats.input_format
        ).flatten()
        b = tilize_block(
            torch.ones(TILE_DIM, TILE_DIM, dtype=torch.bfloat16),
            dimensions=[TILE_DIM, TILE_DIM],
            stimuli_format=formats.input_format,
        ).flatten()
        tiles_a, tiles_b, tiles_res = rt * ct, 1, rt
    _run(
        "llk2_runtime_size_compute_api_test.cpp",
        formats,
        dest_acc,
        fidelity,
        [LLK2_RUNTIME_SIZE(kind, rt, ct, kt)],
        [LLK2_COUNT(count)],
        a,
        tiles_a,
        b,
        tiles_b,
        golden,
        tiles_res,
        include_dirs(),
        tolerance,
    )


def _view_cases():
    fp16b, fp32 = input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32], same=True
    )
    return [
        # (kind, formats, dest_acc, rt, ct, kt)
        (3, fp16b, DestAccumulation.No, 1, 4, 2),
        (3, fp32, DestAccumulation.Yes, 1, 4, 2),
        (4, fp16b, DestAccumulation.No, 1, 4, 1),
        (4, fp32, DestAccumulation.Yes, 1, 4, 1),
        (
            4,
            fp16b,
            DestAccumulation.No,
            1,
            1,
            1,
        ),  # a bound of one tile: the count can only be 1
        (5, fp16b, DestAccumulation.No, 2, 4, 1),
        (6, fp16b, DestAccumulation.No, 1, 4, 1),
        (6, fp32, DestAccumulation.Yes, 1, 4, 1),
        (6, fp16b, DestAccumulation.No, 1, 1, 1),  # a bound of 1
        (7, fp16b, DestAccumulation.No, 1, 4, 1),  # the exp walks the block
        (7, fp16b, DestAccumulation.No, 1, 1, 1),  # a bound of 1: tile by tile
        (3, fp16b, DestAccumulation.No, 1, 1, 2),  # a bound of 1
        (
            8,
            fp16b,
            DestAccumulation.No,
            2,
            2,
            4,
        ),  # in0 held (ct >= rt): [2, K] x [K, 2], K up to 4
        (8, fp32, DestAccumulation.Yes, 2, 2, 4),
        (8, fp16b, DestAccumulation.No, 3, 1, 4),  # in1 held: [3, K] x [K, 1]
        # a 32-bit Dest: the reduce over a run-time row length, and the exp's block walk over a run-time tile count
        (5, fp16b, DestAccumulation.Yes, 2, 2, 1),
        (7, fp16b, DestAccumulation.Yes, 1, 4, 1),
    ]


@parametrize(case=_view_cases(), count=runtime([1, 2, 3, 4]))
def test_llk2_runtime_views(case, count):
    kind, formats, dest_acc, rt, ct, kt = case
    # above the bound of this build: the bound again
    _run_view_case(
        kind, formats, dest_acc, rt, ct, kt, min(count, kt if kind == 8 else ct)
    )
