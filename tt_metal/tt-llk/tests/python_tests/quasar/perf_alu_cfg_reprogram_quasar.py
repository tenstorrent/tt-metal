# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark: tracked vs unconditional ALU format (re)configure on the math thread.

Compares _configure_alu_cfg_ (early return + per-word diff against a cached copy) with a
baseline that stores every ALU_FORMAT_SPEC word on each call. TILE_CNT is 1, so the
post-processed TILE_LOOP figure is cycles per configure call. See
sources/quasar/alu_cfg_reprogram_quasar_test.cpp for the mode and pattern definitions.
"""

from dataclasses import dataclass
from enum import Enum

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    TILE_COUNT,
    TemplateParameter,
)


class AluCfgMode(Enum):
    Tracked = 0
    Unconditional = 1
    TrackedNoDiff = 2


class AluCfgPattern(Enum):
    Same = 0
    AltFormats = 1
    AltType = 2


@dataclass
class ALU_CFG_MODE(TemplateParameter):
    alu_cfg_mode: AluCfgMode = AluCfgMode.Tracked

    def convert_to_cpp(self) -> str:
        return f"constexpr int ALU_CFG_MODE = {self.alu_cfg_mode.value};"


@dataclass
class ALU_CFG_PATTERN(TemplateParameter):
    alu_cfg_pattern: AluCfgPattern = AluCfgPattern.Same

    def convert_to_cpp(self) -> str:
        return f"constexpr int ALU_CFG_PATTERN = {self.alu_cfg_pattern.value};"


@dataclass
class ALU_CFG_ALT_FORMAT(TemplateParameter):
    alu_cfg_alt_format: DataFormat = DataFormat.Float16

    def convert_to_cpp(self) -> str:
        return f"constexpr auto ALU_CFG_ALT_FORMAT = DataFormat::{self.alu_cfg_alt_format.name};"


# Calls per TILE_LOOP; large enough that zone overhead is noise at the per-call level.
ALU_CFG_CALLS = 4096


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    alt_format=[DataFormat.Float16],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mode=[AluCfgMode.Tracked, AluCfgMode.Unconditional, AluCfgMode.TrackedNoDiff],
    pattern=[AluCfgPattern.Same, AluCfgPattern.AltFormats, AluCfgPattern.AltType],
    loop_factor=[ALU_CFG_CALLS],
)
def test_perf_alu_cfg_reprogram_quasar(
    perf_report, formats, alt_format, dest_acc, mode, pattern, loop_factor
):
    configuration = create_test_or_perf_config(
        is_perf=True,
        # Only math does work. L1_TO_L1 times unpack start -> pack end and ignores the math zone,
        # so it would report the empty unpack/pack zones; MATH_ISOLATE times math's own TILE_LOOP.
        run_types=[PerfRunType.MATH_ISOLATE],
        test_config_kwargs={
            "test_name": "sources/quasar/alu_cfg_reprogram_quasar_test.cpp",
            "formats": formats,
            "templates": [
                ALU_CFG_MODE(mode),
                ALU_CFG_PATTERN(pattern),
                ALU_CFG_ALT_FORMAT(alt_format),
            ],
            "runtimes": [
                TILE_COUNT(1),
                LOOP_FACTOR(loop_factor),
            ],
            "dest_acc": dest_acc,
        },
    )
    configuration.run(perf_report)
