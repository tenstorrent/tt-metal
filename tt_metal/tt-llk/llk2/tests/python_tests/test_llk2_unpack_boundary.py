# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 back-to-back ops whose unpack configuration differs in what the unpackers do with a tile.

The kernel (llk2_unpack_boundary_compute_api_test.cpp) runs two ops alternately, ROUNDS times, with nothing between
them: a matmul that transposes in1 and a plain matmul or add; an add with a row bias (the unpacker reads half a tile)
and a plain add or a transposing matmul. Every op boundary rewrites unpacker state right behind the previous op's
UNPACRs (llk_lib/unpack.h's configure waits for the unpackers only in front of the words they read while data moves).
Every result of every round must match its golden, and every round must repeat the first bit for bit.

math_spin > 0 keeps the math thread away from its Src banks before every op, so the previous op's last unpacks stand
still in the unpackers while the next configure is issued. test_llk2_reconfig.py covers ops that differ in format;
test_llk2_unpack_latch_probe.py measures, word by word, which configuration an unpack in flight reads.
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
from llk2_common import TILE_DATUMS, TILE_DIM, tilize_operands
from llk2_paths import cpp_source, include_dirs
from test_llk2_matmul import transpose_each_tile

SIDE = 2  # a, b and every result are [SIDE, SIDE] tiles
TILES = SIDE * SIDE
ROUNDS = 6
FIDELITY = MathFidelity.HiFi2

# The kernel's SEQ -> (first op, second op)
SEQUENCES = {
    0: ("matmul_transposed", "matmul"),
    1: ("matmul_transposed", "add"),
    2: ("add_row_bias", "add"),
    3: ("add_row_bias", "matmul_transposed"),
}


@dataclass
class LLK2_UNPACK_BOUNDARY(TemplateParameter):
    seq: int
    math_spin: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t SEQ = {self.seq};\n"
            f"constexpr std::uint32_t ROUNDS = {ROUNDS};\n"
            f"constexpr std::uint32_t MATH_SPIN = {self.math_spin};"
        )


def _same(x, y):
    """Bit-for-bit equality of two float tensors, NaNs included."""
    x, y = x.to(torch.float32), y.to(torch.float32)
    return bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all())


@parametrize(seq=sorted(SEQUENCES), math_spin=[0, 3000])
def test_llk2_unpack_boundary(seq, math_spin):
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    fmt = formats.input_format
    dims = [SIDE * TILE_DIM, SIDE * TILE_DIM]
    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=fmt,
        input_dimensions_A=dims,
        stimuli_format_B=fmt,
        input_dimensions_B=dims,
        spec_A=spec,
        spec_B=spec,
    )
    tilized_A, tilized_B = tilize_operands(src_A, dims, src_B, dims, fmt)

    def matmul_golden(in1):
        return get_golden_generator(MatmulGolden)(
            src_A,
            in1,
            formats.output_format,
            FIDELITY,
            input_A_dimensions=dims,
            input_B_dimensions=dims,
            tilize=True,
            input_A_format=fmt,
            input_B_format=fmt,
        )

    def add_golden(golden_B, quantized):
        return get_golden_generator(EltwiseBinaryGolden)(
            MathOperation.Elwadd,
            tilized_A,
            golden_B,
            formats.output_format,
            MathFidelity.LoFi,
            input_format=fmt,
            input_format_B=None if quantized else fmt,
            dest_acc=DestAccumulation.No,
        )

    def row_bias():
        # Output tile (r, c) reads b's tile (0, c), its row 0 down every row.
        b_tiles = tilized_B.view(TILES, TILE_DATUMS)
        per_output = torch.cat([b_tiles[c] for _ in range(SIDE) for c in range(SIDE)])
        return get_golden_generator(BroadcastGolden)(
            BroadcastType.Row, per_output, fmt, tile_cnt=TILES
        )

    goldens = {
        "matmul_transposed": lambda: matmul_golden(transpose_each_tile(src_B, dims)),
        "matmul": lambda: matmul_golden(src_B),
        "add": lambda: add_golden(tilized_B, False),
        "add_row_bias": lambda: add_golden(row_bias(), True),
    }
    torch_format = format_dict[formats.output_format]
    pair = [
        torch.as_tensor(goldens[name]()).to(torch_format).flatten()
        for name in SEQUENCES[seq]
    ]

    configuration = TestConfig(
        cpp_source("llk2_unpack_boundary_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(FIDELITY),
            DEST_SYNC(),
            LLK2_UNPACK_BOUNDARY(seq, math_spin),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A,
            fmt,
            tilized_B,
            fmt,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=2 * ROUNDS * TILES,
        ),
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    op_len = TILES * TILE_DATUMS
    assert len(res) == 2 * ROUNDS * op_len, "Result tensor does not hold every round"
    results = torch.tensor(res, dtype=torch_format).view(ROUNDS, 2, op_len)
    for r in range(ROUNDS):
        for j, name in enumerate(SEQUENCES[seq]):
            assert passed_test(
                pair[j], results[r][j], formats.output_format
            ), f"round {r}, {name}: golden mismatch"
            assert _same(
                results[r][j], results[0][j]
            ), f"round {r}, {name}: differs from round 0"
