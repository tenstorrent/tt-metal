# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Non-overlapping full-tile Welford LREG4/LREG5 trace on BH."""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter
from helpers.tilize_untilize import tilize_block, untilize_block


@dataclass
class TraceImpl(TemplateParameter):
    value: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TRACE_IMPL = {self.value}u;"


def _decode_vector(tiles, tile, slot):
    rows = tiles[tile * 32 + slot * 4 : tile * 32 + slot * 4 + 4, :16:2]
    return rows.reshape(2, 2, 8).transpose(1, 2).flatten().float()


def _run(impl):
    torch.manual_seed(20260814)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    input_2d = torch.empty((32, 32), dtype=torch.bfloat16).uniform_(-4.0, 4.0)
    source = tilize_block(input_2d.flatten(), [32, 32], stimuli_format=formats.input_format).flatten()
    config = TestConfig(
        "sources/sfpu_welford_full_trace.cpp", formats, templates=[TraceImpl(impl)], runtimes=[],
        variant_stimuli=StimuliConfig(source, formats.input_format, torch.zeros_like(source), formats.input_format,
            formats.output_format, tile_count_A=1, tile_count_B=1, tile_count_res=7),
        dest_acc=DestAccumulation.No, unpack_to_dest=False, disable_format_inference=True, compile_time_formats=True,
    )
    raw = torch.tensor(config.run().result, dtype=torch.bfloat16)
    tiles = untilize_block(raw, formats.output_format, [7 * 32, 32])
    for tile in range(7):
        print(f"TRACE_TILE impl={impl} tile={tile} nonzero={(tiles[tile*32:(tile+1)*32] != 0).sum().item()} max_abs={tiles[tile*32:(tile+1)*32].float().abs().max().item():.7g}")
    # Result buffers are packed from Dst indices 1..7, so host tile 0 maps
    # to device trace tile 1 (not device input tile 0).
    trace_mean = torch.stack([_decode_vector(tiles, row // 16, row % 16) for row in range(32)])
    trace_m2 = torch.stack([_decode_vector(tiles, 2 + row // 16, row % 16) for row in range(32)])
    final_mean = _decode_vector(tiles, 4, 0)
    final_m2 = _decode_vector(tiles, 5, 0)
    return input_2d, trace_mean, trace_m2, final_mean, final_m2


def test_sfpu_welford_full_trace():
    input_2d, raw_mean, raw_m2, raw_final_mean, raw_final_m2 = _run(0)
    _, vf_mean, vf_m2, vf_final_mean, vf_final_m2 = _run(1)
    expected_mean = torch.stack([input_2d[:row + 1].float().mean(dim=0) for row in range(32)])
    expected_m2 = torch.stack([((input_2d[:row + 1].float() - expected_mean[row]) ** 2).sum(dim=0) for row in range(32)])
    # Capture integrity: traced terminal state and regular terminal store agree.
    for name, traced, final in (("raw_mean", raw_mean[-1], raw_final_mean), ("raw_m2", raw_m2[-1], raw_final_m2),
                                ("vf_mean", vf_mean[-1], vf_final_mean), ("vf_m2", vf_m2[-1], vf_final_m2)):
        print(f"TRACE_INTEGRITY {name} exact={torch.equal(traced, final)} max_abs={(traced-final).abs().max().item():.7g}")
    assert torch.allclose(raw_mean, expected_mean, rtol=2e-2, atol=2e-2)
    assert torch.allclose(raw_m2, expected_m2, rtol=3e-2, atol=3e-2)
    first = None
    for row in range(32):
        mean_error = (vf_mean[row] - raw_mean[row]).abs()
        m2_error = (vf_m2[row] - raw_m2[row]).abs()
        same = torch.equal(vf_mean[row], raw_mean[row]) and torch.equal(vf_m2[row], raw_m2[row])
        print(f"FULL_TRACE n={row+1} exact={same} mean_max_abs={mean_error.max().item():.7g} m2_max_abs={m2_error.max().item():.7g}")
        if first is None and not same:
            first = row + 1
    print(f"FULL_TRACE first_divergence={first}")
