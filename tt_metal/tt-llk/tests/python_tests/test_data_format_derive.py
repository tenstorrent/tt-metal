# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Build-only: compile-time checks on the Blackhole ``data_format_derive.h`` rules.

``data_format_derive.h`` (llk_api) derives register formats from L1 formats for the 2.0
compute API and mirrors the host rule in ``jit_build/data_format.cpp``. Its two-operand
exponent-family rule must treat Float32 and the integer formats the host skips (UInt32,
UInt16, UInt8, Int32) as wildcards that adopt the other operand's family and default to
the 8-bit "b" family; otherwise a Float32 operand next to an integer operand unpacks to
the 5-bit-exponent Float16 (silently saturating |x| > 65504) and an integer next to a
b-family float is rejected as mixed exponent widths. Int8 and Fp8_e4m3 are deliberately not
wildcards and are pinned too. The checks are static_asserts in
``sources/data_format_derive_test.cpp``, so this test only builds the driver.
"""

from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig


@blackhole_only
def test_two_operand_exponent_family_rules():
    # The driver is header-only; one bf16 variant is enough to build it.
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    configuration = TestConfig(
        "sources/data_format_derive_test.cpp",
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
        dest_acc=DestAccumulation.No,
    )
    # Build only: a failing static_assert fails the build here.
    configuration.prepare()

    math_elf = (
        TestConfig.ARTEFACTS_DIR
        / configuration.test_name
        / configuration.variant_id
        / "elf"
        / "math.elf"
    )
    assert math_elf.is_file(), f"math ELF was not built: {math_elf}"
