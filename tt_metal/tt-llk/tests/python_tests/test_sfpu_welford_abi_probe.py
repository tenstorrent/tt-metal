# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""BH differential probe: raw Welford ABI sequence versus vFloat lowering."""

from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter
from helpers.tilize_untilize import tilize_block, untilize_block


@dataclass
class UIntTemplate(TemplateParameter):
    name: str
    value: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t {self.name} = {self.value}u;"


def _run(impl: int, stage: int):
    torch.manual_seed(20260814)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    input_2d = torch.empty((32, 32), dtype=torch.bfloat16).uniform_(-4.0, 4.0)
    source = tilize_block(input_2d.flatten(), [32, 32], stimuli_format=formats.input_format).flatten()
    config = TestConfig(
        "sources/sfpu_welford_abi_probe.cpp", formats,
        templates=[UIntTemplate("PROBE_IMPL", impl), UIntTemplate("PROBE_STAGE", stage)],
        runtimes=[],
        variant_stimuli=StimuliConfig(source, formats.input_format, torch.zeros_like(source), formats.input_format,
            formats.output_format, tile_count_A=1, tile_count_B=1, tile_count_res=9),
        dest_acc=DestAccumulation.No, unpack_to_dest=False,
        disable_format_inference=True, compile_time_formats=True,
    )
    raw = torch.tensor(config.run().result, dtype=torch.bfloat16)
    tiles = untilize_block(raw, formats.output_format, [9 * 32, 32])
    # The Welford store format puts the 32 lanes at even positions in the
    # first four physical rows, exactly like the existing Welford validator.
    def unpack_vector(tile):
        lanes = tiles[tile * 32 : tile * 32 + 4, :16:2].reshape(2, 2, 8)
        return lanes.transpose(1, 2).flatten().float()
    return torch.stack([unpack_vector(tile) for tile in range(9)])


@pytest.mark.parametrize("stage", range(1, 7))
def test_sfpu_welford_abi_probe(stage):
    raw = _run(0, stage)
    vfloat = _run(1, stage)
    names = ["LREG0", "LREG1", "LREG2", "LREG3", "LREG4", "LREG5", "LREG6", "LREG7", "LREG11"]
    differing = []
    for index, name in enumerate(names):
        error = (raw[index] - vfloat[index]).abs()
        equal = torch.equal(raw[index], vfloat[index])
        print(f"ABI_PROBE stage={stage} reg={name} exact={equal} max_abs={error.max().item():.7g}")
        if not equal:
            differing.append(name)
    print(f"ABI_PROBE stage={stage} differing={','.join(differing) or 'none'}")
    assert not differing, f"earliest divergent ABI state at stage={stage}: {differing}"
