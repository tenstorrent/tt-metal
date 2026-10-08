# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""BH hardware reproducer, not a performance benchmark or full LLK sweep.

From tests/python_tests, with matching SFPI headers/compiler installed:
    CHIP_ARCH=blackhole ../.venv/bin/python -m pytest test_raw_lreg_device.py -s -q

Requires the sfprawlreg_effect builtin. Do not use --compile-producer for
hardware validation. Schemes 0/1 are diagnostic controls: XFAIL means actual
wrong output was observed, not that correctness passed. Schemes 2/3 must pass.
Scheme 3 threads a C++ value; it does not implement the proposed sfpvalue API.
Also covers partial predicates, forced register relocation, and dead raw outputs.
Registers are tested individually, not all simultaneously live. Existing harness
sentinel clearing and repeated execution are enabled for every device case.
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
    lreg: int
    live_before: bool
    partial: bool = False
    force_move: bool = False
    dead_output: bool = False

    def convert_to_cpp(self):
        return (f"constexpr unsigned SCHEME = {self.scheme}; constexpr bool USE_MMIO = {self.mmio}; "
                f"constexpr unsigned LREG = {self.lreg}; constexpr bool LIVE_BEFORE = {int(self.live_before)}; "
                f"constexpr bool PARTIAL = {int(self.partial)}; constexpr bool FORCE_MOVE = {int(self.force_move)}; "
                f"constexpr bool DEAD_OUTPUT = {int(self.dead_output)};")


def _run_case(scheme, mmio, lreg, live_before, partial=False, force_move=False, dead_output=False):
    if TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole-specific raw instruction encoding")
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    source = torch.full((1024,), 2.0, dtype=torch.bfloat16)
    if partial:
        source[(torch.arange(1024) % 4) >= 2] = -2.0
    config = TestConfig(
        "sources/sfpu_raw_lreg_device.cpp", formats,
        templates=[RawLregCase(scheme, mmio, lreg, live_before, partial, force_move, dead_output)], runtimes=[],
        variant_stimuli=StimuliConfig(
            source, formats.input_format, torch.zeros_like(source),
            formats.input_format, formats.output_format,
            tile_count_A=1, tile_count_B=1, tile_count_res=1 if dead_output else 2,
        ),
        dest_acc=DestAccumulation.No, unpack_to_dest=False,
        disable_format_inference=True, compile_time_formats=True,
    )
    result = torch.tensor(config.run().result, dtype=torch.bfloat16)
    assert result.numel() == (1024 if dead_output else 2048)
    input_bad = int((result[:1024] != source).sum())
    if not live_before:
        assert input_bad == 0, "input tile calibration failed"
    observed = result[1024:]
    # SFPSTORE's inverse address map, in packed face order (not untilized).
    expected = torch.empty((64, 16), dtype=torch.bfloat16)
    for row in range(32):
        first = 4 * (row // 2)
        expected[first:first + 4, row % 2::2] = 1.0 + row / 128.0
    expected = expected.flatten()
    if partial:
        expected = torch.where(source < 0, expected, 3.0)
    bad = 0 if dead_output else int((observed != expected).sum())
    print(f"RAW_LREG_DEVICE scheme={scheme} mmio={mmio} lreg={lreg} live_before={live_before} "
          f"partial={partial} force_move={force_move} dead_output={dead_output} "
          f"mismatches={bad}/1024 input_mismatches={input_bad}/1024")
    if scheme == 2 or (scheme == 3 and not dead_output):
        assert bad == 0 and input_bad == 0, "protected raw/typed value was corrupted"
    elif bad or input_bad:
        pytest.xfail("observed raw/typed value corruption in diagnostic control")


@pytest.fixture
def cleared_output(monkeypatch):
    # Existing harness repeats clear output to a sentinel before each run.
    monkeypatch.setattr(TestConfig, "BIT_EXACT_RUNS", max(2, TestConfig.BIT_EXACT_RUNS))


@pytest.mark.parametrize("scheme", [0, 1, 2, 3], ids=["none", "pairs", "effects", "threaded"])
@pytest.mark.parametrize("mmio", [0, 1], ids=["TTI", "TT"])
@pytest.mark.parametrize("lreg", range(8))
@pytest.mark.parametrize("live_before", [False, True], ids=["gap", "live-before"])
def test_raw_lreg_device(scheme, mmio, lreg, live_before, cleared_output):
    _run_case(scheme, mmio, lreg, live_before)


@pytest.mark.parametrize("scheme,force_move", [(2, False), (3, False), (3, True)],
                         ids=["effects", "threaded", "threaded-relocate"])
@pytest.mark.parametrize("mmio", [0, 1], ids=["TTI", "TT"])
@pytest.mark.parametrize("lreg", range(8))
def test_raw_lreg_partial(scheme, force_move, mmio, lreg, cleared_output):
    _run_case(scheme, mmio, lreg, False, partial=True, force_move=force_move)


@pytest.mark.parametrize("scheme", [0, 1, 2, 3], ids=["none", "pairs", "effects", "discarded-read"])
@pytest.mark.parametrize("mmio", [0, 1], ids=["TTI", "TT"])
@pytest.mark.parametrize("lreg", range(8))
def test_raw_lreg_dead_output(scheme, mmio, lreg, cleared_output):
    _run_case(scheme, mmio, lreg, True, dead_output=True)
