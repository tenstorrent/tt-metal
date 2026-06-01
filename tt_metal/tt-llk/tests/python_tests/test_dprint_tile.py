# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig


@pytest.fixture(scope="module", autouse=True)
def _force_device_print_enabled():
    prev = TestConfig.DEVICE_PRINT_ENABLED
    TestConfig.DEVICE_PRINT_ENABLED = True
    yield
    TestConfig.DEVICE_PRINT_ENABLED = prev


# SliceRange::hw0_32_8 = {h0=0, h1=32, hs=8, w0=0, w1=32, ws=8}.
# Picks 4 rows × 4 cols = 16 values out of the 32×32 tile.
_H_VALUES = (0, 8, 16, 24)
_W_VALUES = (0, 8, 16, 24)


def _tilized_index(h: int, w: int) -> int:
    """Map a logical (h, w) inside a 32×32 tile to the L1 face-major index."""
    face_r, face_c = h // 16, w // 16
    return (face_r * 2 + face_c) * 256 + (h % 16) * 16 + (w % 16)


@parametrize(formats=input_output_formats([DataFormat.Float16_b], same=True))
def test_dprint_tile(formats):
    if get_chip_architecture() == ChipArchitecture.QUASAR:
        pytest.skip("llk_dprint::TileSlice not validated on Quasar yet")

    formats = formats[0]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    configuration = TestConfig(
        "sources/dprint_tile_test.cpp",
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

    outcome = configuration.run()
    full = "".join(outcome.device_print_lines)

    # Expected values: src_A is laid out in tilized L1 order (face-major).
    expected: list[float] = []
    for h in _H_VALUES:
        for w in _W_VALUES:
            expected.append(float(src_A[_tilized_index(h, w)]))

    # The host renderer emits "v0 v1 v2 v3\n" per row; collect every float-like
    # token from the device-print output.
    decoded: list[float] = []
    for line in outcome.device_print_lines:
        body = line.split("] ", 1)[-1].strip()
        for tok in body.split():
            try:
                decoded.append(float(tok))
            except ValueError:
                continue

    assert len(decoded) >= len(expected), (
        f"Decoded {len(decoded)} floats from TileSlice output, "
        f"expected at least {len(expected)}"
    )
    # bf16 -> fp32 widening is bit-exact, so decoded values must match the
    # stimulus at the chosen slice positions.
    assert (
        decoded[: len(expected)] == expected
    ), "Decoded TileSlice values do not match the bf16 stimulus at the sliced positions"
