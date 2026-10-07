# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""BH hardware reproducer, not a performance benchmark or full LLK sweep.

From tests/python_tests, with matching SFPI headers/compiler installed:
    CHIP_ARCH=blackhole ../.venv/bin/python -m pytest test_raw_lreg_device.py -s -q

Requires the sfprawlreg_effect builtin. Do not use --compile-producer for
hardware validation. Schemes 0/1 are diagnostic controls: XFAIL means actual
wrong output was observed, not that correctness passed. Scheme 2 must pass.
Only all-active lanes are covered; partial predicates need separate tests.
"""
from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter


@dataclass
class RawLregCase(TemplateParameter):
    scheme: int
    mmio: int

    def convert_to_cpp(self):
        return f"constexpr unsigned SCHEME = {self.scheme}; constexpr bool USE_MMIO = {self.mmio};"


@pytest.mark.parametrize("scheme", [0, 1, 2])
@pytest.mark.parametrize("mmio", [0, 1])
def test_raw_lreg_device(scheme, mmio):
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole-specific raw instruction encoding")
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    source = torch.full((1024,), 2.0, dtype=torch.bfloat16)
    config = TestConfig(
        "sources/sfpu_raw_lreg_device.cpp", formats,
        templates=[RawLregCase(scheme, mmio)], runtimes=[],
        variant_stimuli=StimuliConfig(
            source, formats.input_format, torch.zeros_like(source),
            formats.input_format, formats.output_format,
            tile_count_A=1, tile_count_B=1, tile_count_res=2,
        ),
        dest_acc=DestAccumulation.No, unpack_to_dest=False,
        disable_format_inference=True, compile_time_formats=True,
    )
    result = torch.tensor(config.run().result, dtype=torch.bfloat16)
    assert result.numel() == 2048
    assert torch.all(result[:1024] == 2.0), "input tile calibration failed"
    observed = result[1024:]
    assert torch.all((observed == 1.0) | (observed == 2.0)), "unexpected output: inspect harness/encoding"
    bad = int((observed != 1.0).sum())
    print(f"RAW_LREG_DEVICE scheme={scheme} mmio={mmio} mismatches={bad}/1024 values={torch.unique(observed).tolist()}")
    if scheme == 2:
        assert bad == 0, "effect-annotated raw L0 was corrupted"
    elif bad:
        pytest.xfail("observed raw L0 corruption without effect interval protection")
