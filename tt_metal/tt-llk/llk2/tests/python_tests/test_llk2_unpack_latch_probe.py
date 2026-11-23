# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hardware probe: when does a Blackhole unpacker read its configuration?

llk_lib/unpack.h's configure writes the unpackers' words behind the previous op's UNPACRs and waits for the
unpackers only in front of the words the hardware reads while data moves. This probe measures which words those
are. The kernel (llk2_unpack_latch_probe_test.cpp) runs ROUNDS adds of two four-tile operands; behind each add's
last UNPACR it issues ONE write that changes a field the unpackers use, a different number of cycles later in every
round (round * NOP_STEP NOPs), then waits for the unpackers. The last rounds are the control: their write lands on idle
unpackers. A second mode keeps the math thread away from its Src banks, so the last tile's unpack stands still in
the unpacker while the write lands.

  LATCHED   every round must be exact: the unpacker took the field before a write issued behind its UNPACR could
            land, so such a write needs no wait for the unpackers.
  LIVE      the early rounds must be corrupted and the control rounds exact: the unpacker reads the field while data
            moves, and unpack.h must wait for the unpackers in front of it. This also shows that the probe's writes
            do land on unpacks in flight.

If a LATCHED case fails, the word needs a wait in unpack_configure; if a LIVE case stops failing, its wait can go.
Set LLK2_PROBE_LOG=<file> to get one line per variant with the rounds that fail.

Not here: SrcAUnsigned / SrcBUnsigned, which the unpackers also read while data moves (tt-isa-documentation,
Unpackers/FormatConversion.md) but which change nothing for the BF16 data of this kernel:
test_llk2_unpack_signedness.py covers them with integer matmuls.
"""

import os
from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import EltwiseBinaryGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from llk2_common import TILE_DATUMS, TILE_DIM
from llk2_paths import cpp_source, include_dirs

ROUNDS = 32
CONTROL_ROUNDS = 4  # free-running mode: the last rounds' writes land on idle unpackers

# (tiles, NOPs more per round, math spins before each add)
FREE = (
    4,
    4,
    0,
)  # the write moves from "right behind the last UNPACR" (round 0) to 124 cycles and more behind it
BLOCKED = (
    3,
    1,
    3000,
)  # math holds both banks: the third tile's unpack stands still while the write lands

# POISON values of the kernel -> (name, the unpacker reads it while data moves)
LATCHED = {
    0: "none",
    1: "unp0 Out_data_format",
    2: "unp0 InDataFormat",
    3: "unp0 Haloize_mode",
    4: "unp0 Tile_x_dim",
    5: "unp0 ZDim",
    6: "unp0 ADC X end",
    7: "unp0 ADC ch0 Z",
    8: "unp0 Src write base",
    9: "unp0 Throttle_mode",
    10: "unp0 Unpack_if_sel",
    13: "unp0 ADC ch1 Y, Z",
    21: "unp1 Out_data_format",
    22: "unp1 InDataFormat",
    23: "unp1 XDim",
    24: "unp1 ZDim",
    25: "unp1 ADC X end",
    26: "unp1 ADC ch0 Z",
    27: "unp1 Throttle_mode",
}
LIVE = {
    12: "SRCA_SET SetOvrdWithAddr",
}


@dataclass
class LLK2_LATCH_PROBE(TemplateParameter):
    poison: int
    tiles: int
    nop_step: int
    math_spin: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t POISON = {self.poison};\n"
            f"constexpr std::uint32_t TILES = {self.tiles};\n"
            f"constexpr std::uint32_t ROUNDS = {ROUNDS};\n"
            f"constexpr std::uint32_t NOP_STEP = {self.nop_step};\n"
            f"constexpr std::uint32_t MATH_SPIN = {self.math_spin};"
        )


def _same(x, y):
    """Bit-for-bit equality of two float tensors, NaNs included."""
    x, y = x.to(torch.float32), y.to(torch.float32)
    return bool((((x == y) | (torch.isnan(x) & torch.isnan(y)))).all())


def run_probe(poison, mode):
    """Runs the kernel with one poison. Returns (rounds that fail the golden, rounds that differ bit for bit from the
    first round that passes it)."""
    tiles, nop_step, math_spin = mode
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    dims = [TILE_DIM, tiles * TILE_DIM]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    tilized_A = tilize_block(
        src_A, dimensions=dims, stimuli_format=formats.input_format
    ).flatten()
    tilized_B = tilize_block(
        src_B, dimensions=dims, stimuli_format=formats.input_format
    ).flatten()
    golden = get_golden_generator(EltwiseBinaryGolden)(
        MathOperation.Elwadd,
        tilized_A,
        tilized_B,
        formats.output_format,
        MathFidelity.LoFi,
        input_format=formats.input_format,
        input_format_B=formats.input_format,
        dest_acc=DestAccumulation.No,
    )

    configuration = TestConfig(
        cpp_source("llk2_unpack_latch_probe_test.cpp"),
        formats,
        templates=[DEST_SYNC(), LLK2_LATCH_PROBE(poison, tiles, nop_step, math_spin)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A,
            formats.input_format,
            tilized_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tiles * ROUNDS,
        ),
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )
    res = configuration.run().result
    round_len = tiles * TILE_DATUMS
    assert len(res) == ROUNDS * round_len, "Result tensor does not hold every round"
    torch_format = format_dict[formats.output_format]
    rounds = torch.tensor(res, dtype=torch_format).view(ROUNDS, round_len)

    wrong = [
        r
        for r in range(ROUNDS)
        if not passed_test(golden, rounds[r], formats.output_format, print_errors=False)
    ]
    right = [r for r in range(ROUNDS) if r not in wrong]
    differing = [r for r in right if not _same(rounds[r], rounds[right[0]])]

    log = os.environ.get("LLK2_PROBE_LOG")
    if log:
        name = {**LATCHED, **LIVE}[poison]
        with open(log, "a") as f:
            f.write(
                f"mode {mode} poison {poison:2d} {name:28s} wrong_rounds={wrong} right_but_differing={differing}\n"
            )
    return wrong, differing


@parametrize(poison=sorted(LATCHED), mode=[FREE, BLOCKED])
def test_llk2_unpack_latched_field(poison, mode):
    wrong, differing = run_probe(poison, mode)
    assert not wrong and not differing, (
        f"{LATCHED[poison]}: a write behind the UNPACRs changed rounds {wrong + differing}: the unpacker reads this "
        "field while data moves, so unpack_configure must wait for the unpackers in front of it"
    )


@parametrize(poison=sorted(LIVE), mode=[FREE, BLOCKED])
def test_llk2_unpack_live_field(poison, mode):
    wrong, differing = run_probe(poison, mode)
    assert (
        not differing
    ), f"{LIVE[poison]}: rounds {differing} pass the golden but differ from the others"
    if mode == BLOCKED:
        expected = wrong == list(
            range(ROUNDS)
        )  # every round's write lands on the unpack that stands still
    else:
        expected = wrong and all(
            r < ROUNDS - CONTROL_ROUNDS for r in wrong
        )  # the early rounds only
    assert expected, (
        f"{LIVE[poison]}: wrong rounds {wrong}: either the probe's writes no longer land on unpacks in flight, or "
        "the unpacker does not read this field while data moves"
    )
