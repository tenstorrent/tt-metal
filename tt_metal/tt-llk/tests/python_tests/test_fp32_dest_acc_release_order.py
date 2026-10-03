# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression test for the mid-kernel FP32 dest-acc handshake (``_llk_set_fp32_dest_acc_``).

When the handshake returns on UNPACK and PACK, the dest-acc config MATH just programmed must
already be in effect. Each released thread reads the field back after every enable and disable,
over many cycles.

The kernel is built with LLK_ASSERT compiled out, as tt-metal builds kernels by default: a
compiled-in assert consumes the handshake's mailbox reads and would hide a handshake whose reads
are not consumed on their own.
"""

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig

# Must match sources/fp32_dest_acc_release_order_test.cpp
CYCLES = 256


@pytest.mark.skipif(
    get_chip_architecture()
    not in (ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE),
    reason="_llk_set_fp32_dest_acc_ exists on Blackhole and Wormhole only.",
)
def test_fp32_dest_acc_release_order():
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/fp32_dest_acc_release_order_test.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
    )
    vals = [int(v) for v in configuration.run().result]
    cycles, pack_enabled, pack_disabled, unpack_enabled, unpack_disabled = vals[:5]

    assert cycles == CYCLES, f"kernel reported {cycles} cycles, expected {CYCLES}"
    seen = {
        "PACK after enable": pack_enabled,
        "PACK after disable": pack_disabled,
        "UNPACK after enable": unpack_enabled,
        "UNPACK after disable": unpack_disabled,
    }
    stale = {name: count for name, count in seen.items() if count != CYCLES}
    assert not stale, (
        "_llk_set_fp32_dest_acc_ returned before MATH's dest-acc config was in effect: "
        + ", ".join(f"{name} {count}/{CYCLES}" for name, count in stale.items())
    )
