# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Hardware probe: does the timing of reduce's Src zero-flag writes matter?

llk_lib/reduce.h sets ALU_ACC_CTRL_Zero_Flag_disabled_src right behind its last pool and restores it right behind its
last MOVB2D, with no wait for the FPU. The kernel (llk2_reduce_zero_flag_probe_test.cpp) runs the same reduce in four
forms, REPS times each, on the same inputs:

  0  Library        reduce_execute as it is
  1  Waited         with a wait for the FPU in front of each flag write (cannot race)
  2  PoolsFlagOn    the flag already 1 while the pools run
  3  MovesFlagOff   the moves run with the default flag

Library must equal Waited bit for bit in every repetition: that is the test. Forms 2 and 3 say whether the flag can
change a result at all on this data; they are logged (LLK2_PROBE_LOG=<file>), not asserted. The "special" data holds
what the flag is about: zeros and BF16 denormals (exponent field 0, where a set flag keeps a datum that a clear one
flushes) next to very large values, in the input and in the scaler.
"""

import os
from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
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
from llk2_common import TILE_DATUMS, TILE_DIM
from llk2_paths import cpp_source, include_dirs

REPS = 8
FORMS = ["Library", "Waited", "PoolsFlagOn", "MovesFlagOff"]


@dataclass
class LLK2_ZERO_FLAG_PROBE(TemplateParameter):
    kt_dim: int
    is_sum: bool

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t KT_DIM = {self.kt_dim};\n"
            f"constexpr std::uint32_t REPS = {REPS};\n"
            f"constexpr bool IS_SUM = {str(self.is_sum).lower()};"
        )


def _bf16_from_bits(bits):
    """BF16 values from their 16-bit patterns (as float32 with the low half zero)."""
    return (bits.to(torch.int32) << 16).view(torch.float32)


def _inputs(kt, special, generator):
    cols = kt * TILE_DIM
    data = (
        (torch.randn(TILE_DIM, cols, generator=generator) * 4)
        .to(torch.bfloat16)
        .float()
    )
    scaler = torch.ones(TILE_DIM, TILE_DIM)
    if special:
        denormal = _bf16_from_bits(
            torch.randint(1, 0x80, (TILE_DIM, cols), generator=generator)
        )
        pick = torch.rand(TILE_DIM, cols, generator=generator)
        data = torch.where(pick < 0.25, torch.zeros(()), data)
        data = torch.where((pick >= 0.25) & (pick < 0.5), denormal, data)
        data = torch.where((pick >= 0.5) & (pick < 0.6), torch.full((), 1.0e30), data)
        data[5::8] = denormal[
            5::8
        ]  # whole rows of denormals: their row max and row sum are denormal or 0
        data[6::8] = 0.0
        # Rows whose sum is below the smallest normal: 1.5 * 2^-126 - 2^-126 (and nothing else).
        data[7::8] = 0.0
        data[7::8, 0] = _bf16_from_bits(torch.tensor([0x00C0] * len(data[7::8])))
        data[7::8, 1] = _bf16_from_bits(torch.tensor([0x8080] * len(data[7::8])))
        # The scaler's first rows are what the pools read: zeros, denormals and ones.
        row = torch.ones(TILE_DIM)
        row[1::4] = 0.0
        row[2::4] = _bf16_from_bits(torch.tensor([0x0040] * len(row[2::4])))
        scaler[0:4] = row
    return data.to(torch.bfloat16), scaler.to(torch.bfloat16)


def _same(x, y):
    x, y = x.to(torch.float32), y.to(torch.float32)
    return bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all())


@parametrize(
    kind=["max", "sum"],
    fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    kt=[1, 8],
    special=[False, True],
)
def test_llk2_reduce_zero_flag_timing(kind, fidelity, kt, special):
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    is_sum = kind == "sum"
    generator = torch.Generator().manual_seed(kt * 10 + is_sum * 2 + special)
    data, scaler = _inputs(kt, special, generator)
    tilized_data = tilize_block(
        data, dimensions=[TILE_DIM, kt * TILE_DIM], stimuli_format=formats.input_format
    ).flatten()
    tilized_scaler = tilize_block(
        scaler, dimensions=[TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        cpp_source("llk2_reduce_zero_flag_probe_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(fidelity),
            DEST_SYNC(),
            LLK2_ZERO_FLAG_PROBE(kt, is_sum),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_data,
            formats.input_format,
            tilized_scaler,
            formats.input_format,
            formats.output_format,
            tile_count_A=kt,
            tile_count_B=1,
            tile_count_res=REPS * len(FORMS),
        ),
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )
    res = configuration.run().result
    assert (
        len(res) == REPS * len(FORMS) * TILE_DATUMS
    ), "Result tensor does not hold every reduce"
    out = torch.tensor(res, dtype=format_dict[formats.output_format]).view(
        REPS, len(FORMS), TILE_DATUMS
    )

    waited = out[0][1]
    unstable = [
        (rep, FORMS[f])
        for rep in range(REPS)
        for f in range(len(FORMS))
        if not _same(out[rep][f], out[0][f])
    ]
    library_differs = [rep for rep in range(REPS) if not _same(out[rep][0], waited)]
    pools_sensitive = not _same(out[0][2], waited)
    moves_sensitive = not _same(out[0][3], waited)

    log = os.environ.get("LLK2_PROBE_LOG")
    if log:
        with open(log, "a") as f:
            f.write(
                f"{kind} {fidelity.name} kt={kt} special={special}: library_differs_in_reps={library_differs} "
                f"flag_changes_pools={pools_sensitive} flag_changes_moves={moves_sensitive} unstable={unstable} "
                f"nonzero_out={int((waited.float() != 0).sum())}\n"
            )

    if not special and not is_sum:
        # Plain data, max: the result is exact, so the probe itself is checked against the row maxima.
        golden = torch.zeros(TILE_DIM, TILE_DIM)
        golden[:, 0] = data.float().max(dim=1).values
        golden = tilize_block(
            golden.to(torch.bfloat16),
            dimensions=[TILE_DIM, TILE_DIM],
            stimuli_format=formats.output_format,
        ).flatten()
        assert _same(
            waited, golden
        ), "The waited form does not give the row maxima: the probe itself is broken"
    assert (
        not unstable
    ), f"The same form gave different results in different repetitions: {unstable}"
    assert not library_differs, (
        f"reduce_execute differs from the form that waits for the FPU in front of its flag writes (reps {library_differs}): "
        "an FPU instruction read the Src zero flag written behind it, so reduce must wait"
    )
