# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blaze gated-reduce fusion: arithmetic, DEST isolation and final-only rounding.

Full backing tiles make writes outside the selected tiny-tile rows observable.
Each dispatch uses two adjacent pairs, a pair between guard tiles, and a partial
batch across three DEST acquisitions. The compute-API test also uses real tiny CBs.
"""

import struct

import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import GatedReduceGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    GATED_REDUCE_PARAMS,
    GATED_REDUCE_SCALARS,
    TILE_COUNT,
)

pytestmark = blackhole_only

TILES = 12
ELEMENTS = 1024
BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
BF16_TO_FP32 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)
FP32 = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _run(
    formats, dest_acc, gate, up, flags, rows=32, sync=DestSync.Half, rounding=False
):
    generator = torch.Generator().manual_seed(3448)
    source = torch.empty((TILES, ELEMENTS)).uniform_(-8, 8, generator=generator)
    # Exact zeros, signs, and both sides of the clamp boundary in every face.
    edges = torch.tensor([-8, -1.5, -1.25, -1, -0.0, 0, 1, 1.25, 1.5, 8])
    source.view(TILES, 4, 256)[:, :, : len(edges)] = edges
    scale, out_scale, limit, alpha = 0.703125, -1.3125, 1.25, 1.702
    if rounding:
        scale, out_scale = 1.00390625, 1.0078125
        source.fill_(0.5)
        # At these positive gates sigmoid is saturated. The exact products are
        # safely away from BF16 midpoints and differ from truncation and staged BF16.
        for gate_tile in (0, 2, 5, 10):
            source[gate_tile].fill_(24.0)
            source[gate_tile + 1].fill_(1.0390625)
    source = source.to(format_dict[formats.input_format])
    expected = source.float().clone()
    active = torch.zeros_like(source, dtype=torch.bool)
    golden = get_golden_generator(GatedReduceGolden)
    for block, pairs in enumerate(((0, 2), (1,), (2,))):
        input_scale, output_scale = (
            (out_scale, scale) if block == 2 else (scale, out_scale)
        )
        for pair in pairs:
            index = block * 4 + pair
            result = golden(
                source[index],
                source[index + 1],
                gate,
                up,
                flags,
                input_scale,
                output_scale,
                limit,
                alpha,
                dest_acc,
            )
            face_mask = active[index].view(4, 16, 16)
            face_mask[:2, : min(rows, 16)] = True
            if rows == 32:
                face_mask[2:] = True
            expected[index, active[index]] = result[active[index]].float()

    config = TestConfig(
        "sources/sfpu_gated_reduce_test.cpp",
        formats,
        templates=[GATED_REDUCE_PARAMS(gate, up, flags, rows), DEST_SYNC(sync)],
        runtimes=[
            TILE_COUNT(TILES),
            GATED_REDUCE_SCALARS(*map(_bits, (scale, out_scale, limit, alpha))),
        ],
        variant_stimuli=StimuliConfig(
            source.flatten(),
            formats.input_format,
            torch.zeros(ELEMENTS, dtype=source.dtype),
            formats.input_format,
            formats.output_format,
            tile_count_A=TILES,
            tile_count_B=1,
            tile_count_res=TILES,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )
    actual = (
        torch.tensor(config.run().result, dtype=format_dict[formats.output_format])
        .float()
        .reshape_as(expected)
    )
    expected = expected.to(format_dict[formats.output_format]).float()
    assert torch.equal(
        actual[~active], expected[~active]
    ), "Up/guard tiles or inactive gate rows were modified"
    # Elementwise comparisons catch one-sided clamp mistakes even on near-zero
    # negative gates; PCC alone would hide these behind the large positive outputs.
    rtol, atol = (0.012, 2e-5) if dest_acc == DestAccumulation.No else (2e-4, 2e-6)
    torch.testing.assert_close(actual[active], expected[active], rtol=rtol, atol=atol)
    if rounding:
        torch.testing.assert_close(actual[active], expected[active], rtol=0, atol=0)


@parametrize(
    formats=[BF16],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    gate=["Silu", "ClampedSilu"],
    up=["Identity", "Clamp"],
    flags=list(range(8)),
)
def test_gated_reduce_modes(formats, dest_acc, gate, up, flags):
    _run(formats, dest_acc, gate, up, flags)


@parametrize(
    formats=[BF16_TO_FP32],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    rows=[4, 8, 16, 32],
    sync=[DestSync.Half, DestSync.Full],
)
def test_gated_reduce_geometry(formats, dest_acc, rows, sync):
    _run(formats, dest_acc, "ClampedSilu", "Clamp", 7, rows, sync)


@parametrize(formats=[FP32], gate=["Silu", "ClampedSilu"])
def test_gated_reduce_fp32_input(formats, gate):
    _run(formats, DestAccumulation.Yes, gate, "Clamp", 7)


def test_gated_reduce_round_once():
    _run(BF16, DestAccumulation.No, "Silu", "Identity", 7, rounding=True)
