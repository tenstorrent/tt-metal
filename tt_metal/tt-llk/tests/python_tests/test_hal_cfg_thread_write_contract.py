# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
The HAL's thread-CFG write paths refuse the CFG state ID at compile time.

``hal::cfg::write<Access::TensixCfgUnit, ...>`` on a thread-CFG field emits a bare SETC16. For
the state ID that is not enough: SETC16.md requires a state-ID write to be padded so that nothing
issued right behind it depends on the new bank, and the HAL's MMIO bank resolution reads
``ckernel::cfg_state_id``, which such a write would leave stale. The one writer that does both is
``ckernel::flip_cfg_state_id``, so every public spelling of a thread-CFG write must reject the
register instead of emitting it, and must keep accepting ordinary thread registers.

Both tests compile ``sources/hal_cfg_thread_write_contract_test.cpp`` and assert on the compiler's
verdict; nothing runs on the device.
"""

import re

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import HAL_CFG_THREAD_WRITE_PROBE
from loguru import logger

pytestmark = [blackhole_only]

STATE_ID = "hal::cfg::CfgStateId::StateID"
ORDINARY_REGISTER = "hal::cfg::DestTargetRegCfgMath::Offset"
SPELLINGS = list(HAL_CFG_THREAD_WRITE_PROBE.SPELLINGS)

# The guard's message names the writer to use instead.
REJECTION = "flip_cfg_state_id"


def _configuration(field, spelling):
    formats = input_output_formats([DataFormat.Int32])[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    return TestConfig(
        "sources/hal_cfg_thread_write_contract_test.cpp",
        formats,
        templates=[HAL_CFG_THREAD_WRITE_PROBE(field, spelling)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
            twos_complement=True,
        ),
    )


def _compile(configuration):
    """Build the variant in every mode.

    The compile is the assertion, so it has to happen in the consumer pass as well:
    ``prepare()`` skips the build under ``--compile-consumer``, which would turn the
    rejection test into DID NOT RAISE there. ``build_elfs()`` is the step ``prepare()``
    runs under ``--compile-producer``.
    """
    configuration.generate_variant_hash()
    configuration.build_elfs()


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_thread_cfg_write_compiles(spelling):
    """Every spelling accepts an ordinary thread register, so the rejection below is
    about the register and not about a spelling that no longer compiles."""
    _compile(_configuration(ORDINARY_REGISTER, spelling))


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_state_id_write_is_rejected(spelling):
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        RuntimeError
    ) as excinfo:
        _compile(_configuration(STATE_ID, spelling))

    message = str(excinfo.value)
    # The compiler's own diagnostic, not just the echoed command line.
    diagnostic = re.search(
        r"hal_cfg_thread_write_contract_test\.cpp.*?error: static assertion failed: ([^\n]*)",
        message,
        re.DOTALL,
    )
    assert (
        diagnostic
    ), f"{spelling}: the state-ID write did not fail at the HAL's static_assert:\n{message}"
    logger.info(f"{spelling}: {diagnostic.group(0).splitlines()[-1]}")
    assert REJECTION in diagnostic.group(
        1
    ), f"{spelling}: the rejection does not point at the writer to use instead:\n{message}"
