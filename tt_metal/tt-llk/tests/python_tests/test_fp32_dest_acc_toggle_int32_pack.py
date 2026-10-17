# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Packing a 32-bit integer output after a mid-kernel dest-acc enable / disable pair.

The kernel is built with a 16-bit Dest and an Int32 output, so the packer's full configure selects
the 32-bit Dest read for the Int32 pack source. MATH enables FP32 dest-acc, copies the Float16_b tile
into the 32-bit Dest, converts it in place to Int32 with the production typecast primitive, and
disables dest-acc again; PACK then packs the tile. The disable must keep the packer's 32-bit Dest
read, which the Int32 pack source needs whatever the accumulation mode: the packed words must equal
the integer-converted stimuli.
"""

import numpy as np
import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig

# 16-bit Dest is the point of the test: with a 32-bit Dest the disable path is never taken.
_DEST_ACC = DestAccumulation.No
_FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Int32)
# One tile: the defect does not depend on the tile count.
_INPUT_DIMENSIONS = [32, 32]
# Whole numbers below this are exact in bf16, and their fp32 -> int32 conversion is the value itself.
_WHOLE_NUMBER_LIMIT = 201


def _whole_numbers(size, dtype, generator):
    return torch.randint(0, _WHOLE_NUMBER_LIMIT, (size,), generator=generator).to(dtype)


@pytest.mark.skipif(
    get_chip_architecture()
    not in (ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE),
    reason="_llk_set_fp32_dest_acc_ exists on Blackhole and Wormhole only.",
)
def test_fp32_dest_acc_toggle_int32_pack():
    spec = StimuliSpec(distribution=_whole_numbers, seed=0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=_FORMATS.input_format,
        input_dimensions_A=_INPUT_DIMENSIONS,
        stimuli_format_B=_FORMATS.input_format,
        input_dimensions_B=_INPUT_DIMENSIONS,
        spec_A=spec,
        spec_B=spec,
    )

    configuration = TestConfig(
        "sources/fp32_dest_acc_toggle_int32_pack_test.cpp",
        _FORMATS,
        variant_stimuli=StimuliConfig(
            src_A,
            _FORMATS.input_format,
            src_B,
            _FORMATS.input_format,
            _FORMATS.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=_DEST_ACC,
    )
    # The kernel converts the tile to Int32 in Dest, so the packer reads Int32, not the Float16_b the
    # inference derives for a 16-bit Dest.
    for fmt_config in configuration.formats_config:
        fmt_config.pack_src = DataFormat.Int32

    configuration.run()

    got = np.frombuffer(
        configuration.variant_stimuli.collect_raw_result_bytes(
            TestConfig.TENSIX_LOCATION
        ),
        dtype=np.uint32,
    )
    expected = np.frombuffer(
        StimuliConfig.get_packer(DataFormat.Int32)(src_A.flatten().to(torch.int32)),
        dtype=np.uint32,
    )
    assert len(got) == len(
        expected
    ), f"Result buffer holds {len(got)} words, the stimuli {len(expected)}"
    mismatch = np.nonzero(got != expected)[0].tolist()
    detail = ", ".join(
        f"[{i}] expected 0x{expected[i]:08X} got 0x{got[i]:08X}" for i in mismatch[:8]
    )
    assert not mismatch, (
        "Int32 pack after a dest-acc enable / disable pair is wrong in "
        f"{len(mismatch)} of {len(expected)} datums: {detail}"
    )
