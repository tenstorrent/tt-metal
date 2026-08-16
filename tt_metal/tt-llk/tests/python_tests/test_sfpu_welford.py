# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Actual silicon correctness validation for handwritten and vFloat Welford."""

from dataclasses import dataclass

import torch
import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter
from helpers.tilize_untilize import tilize_block, untilize_block


@dataclass
class WELFORD_IMPL(TemplateParameter):
    value: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t WELFORD_IMPL = {self.value}u;"


IMPLEMENTATIONS = {
    "HANDWRITTEN_REPLAY": 1,
    "HANDWRITTEN_DIRECT": 0,
    "VFLOAT_DIRECT": 2,
    "VFLOAT_RESCUE": 3,
    "VFLOAT_MANUAL_EARLY_FOLD": 4,
}


def _run(implementation: str):
    torch.manual_seed(20260814)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    input_2d = torch.empty((32, 32), dtype=torch.bfloat16).uniform_(-4.0, 4.0)
    source = tilize_block(input_2d.flatten(), [32, 32], stimuli_format=formats.input_format).flatten()
    config = TestConfig(
        "sources/sfpu_welford_test.cpp",
        formats,
        templates=[WELFORD_IMPL(IMPLEMENTATIONS[implementation])],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            source, formats.input_format, torch.zeros_like(source), formats.input_format,
            formats.output_format, tile_count_A=1, tile_count_B=1, tile_count_res=2,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    raw = torch.tensor(config.run().result, dtype=torch.bfloat16)
    # Welford stores vectors in the first face, at even locations. Untilize returns
    # the packed tile layout; these locations contain the 32 independent columns.
    tiles = untilize_block(raw, formats.output_format, [64, 32])
    # Each 16-column face stores its even lanes then its odd lanes.  Reassemble
    # the two face pairs into logical column order before comparing with torch.
    def unpack_vector(face_rows):
        lanes = tiles[face_rows, :16:2].reshape(2, 2, 8)
        return lanes.transpose(1, 2).flatten().float()

    mean = unpack_vector(slice(0, 4))
    m2 = unpack_vector(slice(32, 36))
    return input_2d, mean, m2, config


@pytest.mark.parametrize("name", IMPLEMENTATIONS)
def test_sfpu_welford(name):
    input_2d, mean, m2, _ = _run(name)
    expected_mean = input_2d.float().mean(dim=0)
    expected_m2 = ((input_2d.float() - expected_mean) ** 2).sum(dim=0)
    for label, actual, expected, atol in (("mean", mean, expected_mean, 2e-2), ("m2", m2, expected_m2, 3e-2)):
        error = (actual - expected).abs()
        passing = torch.isclose(actual, expected, rtol=2e-2, atol=atol)
        print(
            f"WELFORD_STATS impl={name} value={label} n={actual.numel()} "
            f"pass={passing.sum().item()}/{actual.numel()} "
            f"max_abs={error.max().item():.7g} mean_abs={error.mean().item():.7g}"
        )
    assert torch.allclose(mean, expected_mean, rtol=2e-2, atol=2e-2), name
    assert torch.allclose(m2, expected_m2, rtol=3e-2, atol=3e-2), name
