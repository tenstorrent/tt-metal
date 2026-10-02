# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wormhole: a math ZEROSRC with write_mode redirects a concurrent unpacker SrcB clear.

An unpacker UNPACR_NOP source clear is meant to hit the unpacker's own bank. On
Wormhole, if it lands in the same cycle as a math-thread ZEROSRC with
write_mode=1, it clears the bank the Matrix Unit is reading instead.

The kernel publishes one face to SrcA (all zeros) and SrcB. Math, holding the
SrcB bank, issues a burst of ZEROSRC(SrcA) and then ELWADDs SrcA onto SrcB, so the
packed face equals SrcB exactly unless math's SrcB bank was cleared. The unpacker
issues a burst of SrcB clears at the same time. Arms (unpacker clear, math burst):

* ``no_clears``        -- none, write_mode=1: SrcB must survive.
* ``default_wait``     -- Matrix-Unit-bank wait, write_mode=1: no clear can fire
                          while math holds the bank, SrcB must survive.
* ``own_bank_alone``   -- own-bank wait, no math burst: the clears run against the
                          free bank only, SrcB must survive.
* ``own_bank_wm0``     -- own-bank wait, write_mode=0: same instruction and timing
                          with only the write_mode bit clear, SrcB must survive.
* ``own_bank_wm1``     -- own-bank wait, write_mode=1 (the reduce-row MAX form):
                          a coincident clear is redirected onto math's SrcB bank.
"""

import torch
from conftest import skip_for_blackhole, skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    NUM_FACES,
    TILE_COUNT,
    ZEROSRC_COLLISION,
    generate_input_dim,
)

FACE_ELEMENTS = 16 * 16


def _srcb_pattern() -> torch.Tensor:
    # Every datum has a non-zero low byte, so the source zero-flag flush (which
    # zeroes datums whose low byte is 0) cannot produce a zero on its own.
    bits = torch.tensor(
        [0x3F80 | (2 * (i % 64) + 1) for i in range(1024)], dtype=torch.int16
    )
    return bits.view(torch.bfloat16)


# arm -> (unpacker clear_mode, math ZEROSRC count, math write_mode, expect SrcB wiped)
ARMS = {
    "no_clears": ("none", 4096, 1, False),
    "default_wait": ("default", 4096, 1, False),
    "own_bank_alone": ("own_bank", 0, 1, False),
    "own_bank_wm0": ("own_bank", 4096, 0, False),
    "own_bank_wm1": ("own_bank", 4096, 1, True),
}


@skip_for_blackhole
@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    arm=list(ARMS),
    repeat=list(range(5)),
)
def test_srcb_zerosrc_write_mode_collision(formats, arm, repeat):
    clear_mode, math_zerosrc, math_write_mode, expect_wiped = ARMS[arm]
    input_dimensions = [32, 32]
    num_faces = 1

    _, tile_cnt_A, _, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    src_A = torch.zeros(1024, dtype=format_dict[formats.input_format])
    src_B = _srcb_pattern()
    golden = src_B[:FACE_ELEMENTS].to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/srcb_zerosrc_write_mode_collision_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            ZEROSRC_COLLISION(
                clear_mode=clear_mode,
                math_zerosrc=math_zerosrc,
                math_write_mode=math_write_mode,
            ),
        ],
        runtimes=[TILE_COUNT(1), NUM_FACES(num_faces)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
            num_faces=num_faces,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )

    res = torch.tensor(
        configuration.run().result[:FACE_ELEMENTS],
        dtype=format_dict[formats.output_format],
    )
    zeroed = int((res == 0).sum())
    mismatched = int((res != golden).sum())
    print(
        f"\n[collision] arm={arm} repeat={repeat}: "
        f"{zeroed}/{FACE_ELEMENTS} zero, {mismatched}/{FACE_ELEMENTS} mismatched"
    )

    if expect_wiped:
        assert zeroed > 0, (
            "Expected an unpacker SrcB clear to collide with a math ZEROSRC "
            "(write_mode=1) and wipe the Matrix Unit's SrcB bank, but SrcB survived"
        )
    else:
        assert mismatched == 0, (
            f"SrcB was corrupted in arm {arm}: "
            f"{zeroed} zero, {mismatched} mismatched"
        )
