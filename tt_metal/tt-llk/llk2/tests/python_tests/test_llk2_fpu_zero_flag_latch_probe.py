# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hardware probe: can ALU_ACC_CTRL_Zero_Flag_disabled_src change what the FPU moves, and if so, when is it read?

llk_lib/reduce.h writes that flag right behind its last pool and its last MOVB2D with no wait for the FPU. The kernel
(llk2_fpu_zero_flag_latch_probe_test.cpp) copies tiles half full of BF16 denormals through SrcA (MOVA2D flushes a
SrcA datum whose low byte is 0 unless the flag is set) in three forms per repetition:

  Default   the flag clear
  FlagOn    the flag set before the MOVA2Ds
  Behind    the flag set rep * NOP_STEP NOPs behind the last MOVA2D, with no wait for the FPU

Behind must equal Default bit for bit in every repetition. Measured on the p100a: FlagOn equals Default too: a BF16
denormal does not survive the way L1 -> SrcA -> Dest -> L1 whatever the flag (the packed result holds 0 in its
place; which unit drops it was not measured). So with these formats no packed result shows the flag, and the timing
of a flag write cannot show either. The test logs how many datums the flag keeps (LLK2_PROBE_LOG=<file>); if that
ever becomes non-zero, Behind == Default is then the measurement of when the FPU reads the flag, and a failure means
reduce.h must wait for the FPU in front of its flag writes.
"""

import os
from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter
from helpers.tilize_untilize import tilize_block
from llk2_common import TILE_DATUMS, TILE_DIM, unwrap_case
from llk2_paths import cpp_source, include_dirs

REPS = 12


@dataclass
class LLK2_FPU_LATCH_PROBE(TemplateParameter):
    tiles: int
    nop_step: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t TILES = {self.tiles};\n"
            f"constexpr std::uint32_t REPS = {REPS};\n"
            f"constexpr std::uint32_t NOP_STEP = {self.nop_step};"
        )


def _same(x, y):
    x, y = x.to(torch.float32), y.to(torch.float32)
    return bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all())


@parametrize(tiles=[1, 4])
def test_llk2_fpu_zero_flag_latched(tiles):
    tiles = unwrap_case(tiles)
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    generator = torch.Generator().manual_seed(tiles)
    cols = tiles * TILE_DIM
    # Half BF16 denormals (exponent field 0, mantissa 1..127, either sign), half ordinary values.
    bits = torch.randint(1, 0x80, (TILE_DIM, cols), generator=generator) | (
        torch.randint(0, 2, (TILE_DIM, cols), generator=generator) << 15
    )
    denormal = (bits.to(torch.int32) << 16).view(torch.float32)
    normal = torch.randn(TILE_DIM, cols, generator=generator) * 4
    data = torch.where(
        torch.rand(TILE_DIM, cols, generator=generator) < 0.5, denormal, normal
    ).to(torch.bfloat16)
    tilized = tilize_block(
        data, dimensions=[TILE_DIM, cols], stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        cpp_source("llk2_fpu_zero_flag_latch_probe_test.cpp"),
        formats,
        templates=[DEST_SYNC(), LLK2_FPU_LATCH_PROBE(tiles, 1)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized,
            formats.input_format,
            tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=tiles,
            tile_count_B=tiles,
            tile_count_res=3 * REPS * tiles,
        ),
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )
    res = configuration.run().result
    n = tiles * TILE_DATUMS
    assert len(res) == 3 * REPS * n, "Result tensor does not hold every copy"
    out = torch.tensor(res, dtype=format_dict[formats.output_format]).view(REPS, 3, n)
    default, flag_on = out[0][0], out[0][1]
    kept = int(((flag_on.float() != 0) & (default.float() == 0)).sum())
    behind_differs = [rep for rep in range(REPS) if not _same(out[rep][2], default)]
    unstable = [
        rep
        for rep in range(REPS)
        if not (_same(out[rep][0], default) and _same(out[rep][1], flag_on))
    ]

    log = os.environ.get("LLK2_PROBE_LOG")
    if log:
        with open(log, "a") as f:
            f.write(
                f"MOVA2D tiles={tiles}: datums the set flag keeps={kept} of {n}, behind_differs_in_reps={behind_differs}, "
                f"unstable_reps={unstable}, default==input: {_same(default, tilized)}, flag_on==input: {_same(flag_on, tilized)}\n"
            )

    assert not unstable, f"Default or FlagOn changed between repetitions: {unstable}"
    assert not behind_differs, (
        f"A flag write behind the last MOVA2D changed the copy (reps {behind_differs}): the FPU reads the Src zero flag "
        "after the instruction has left the Wait Gate, so reduce.h must wait for the FPU in front of its flag writes"
    )
