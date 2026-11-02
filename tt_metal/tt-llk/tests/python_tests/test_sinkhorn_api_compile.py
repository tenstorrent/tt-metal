# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Build-only: the Blackhole sinkhorn compute-API wrapper header must compile.

``api/compute/experimental/sinkhorn.h`` includes
``llk_api/experimental/llk_sfpu/llk_math_sinkhorn.h`` into every TRISC_MATH kernel that
includes it. ``_llk_math_sinkhorn_4x4_init_`` there is a non-template inline function,
so its body is checked at header inclusion whether or not ``sinkhorn_4x4_init()`` is
ever called. No in-tree kernel includes the header; of the tests,
``sinkhorn_lconst_neg1_regression_test.cpp`` does, but dispatches only a one-face,
full-rectangle ``sinkhorn_4x4``. This test builds a driver that
adds the rest: the init call, the row-max-sub entry point, the four-face forwarder the
compute API defaults to, the ``SINGLE_SUBMAT`` arm and both arms of the padding zeroing
(whole-vector store and the per-lane masked store). It asserts the math ELF was produced
and never runs the kernel.
"""

from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig


@blackhole_only
def test_sinkhorn_api_header_compiles():
    # The wrapper's init is format-agnostic; one bf16 variant is enough to build the driver.
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    configuration = TestConfig(
        "sources/sinkhorn_api_compile_test.cpp",
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
        # Pinned, not swept: the wrapper static_asserts DST_ACCUM_MODE == false (its
        # strip-pair store rounds into a 16-bit DEST).
        dest_acc=DestAccumulation.No,
    )
    # Build only: a compile error in the wrapper header fails the build here.
    configuration.prepare()

    math_elf = (
        TestConfig.ARTEFACTS_DIR
        / configuration.test_name
        / configuration.variant_id
        / "elf"
        / "math.elf"
    )
    assert math_elf.is_file(), f"math ELF was not built: {math_elf}"
