# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    TemplateParameter,
)
from ttexalens.tt_exalens_lib import read_words_from_device

pytestmark = [blackhole_only]


@dataclass
class BoundsCase(TemplateParameter):
    start_a: int
    tiles_a: int
    start_b: int
    tiles_b: int

    def convert_to_cpp(self):
        return "\n".join(
            [
                "constexpr bool PRESERVE_SRC_ZERO_FLAG = false;",
                f"constexpr unsigned BOUNDS_A_START = {self.start_a};",
                f"constexpr unsigned BOUNDS_A_TILES = {self.tiles_a};",
                f"constexpr unsigned BOUNDS_B_START = {self.start_b};",
                f"constexpr unsigned BOUNDS_B_TILES = {self.tiles_b};",
            ]
        )


@pytest.mark.parametrize(
    "start_a,tiles_a,start_b,tiles_b,valid",
    [
        (
            0,
            2,
            0,
            1,
            True,
        ),  # SrcA ends exactly at its CB limit; only one SrcB tile is needed.
        (1, 3, 1, 2, True),  # Nonzero indices, still ending exactly at both CB limits.
        (
            1,
            2,
            0,
            1,
            False,
        ),  # First SrcA tile is valid, but its two-tile span crosses the limit.
        (0, 2, 1, 1, False),  # SrcB's single-tile check must remain in force.
    ],
)
def test_sub_bcast_col_bounds(monkeypatch, start_a, tiles_a, start_b, tiles_b, valid):
    monkeypatch.setattr(
        TestConfig,
        "INCLUDES",
        [
            *TestConfig.INCLUDES,
            "-I../../hw/ckernels/blackhole/metal/llk_io",
        ],
    )
    # Extra physical storage makes the invalid logical CB span safe on unfixed code.
    a = torch.arange(1, 5, dtype=torch.bfloat16).repeat_interleave(1024) * 10
    b = torch.full((2048,), 3.0, dtype=torch.bfloat16)
    cfg = TestConfig(
        "sources/sub_bcast_col_bounds_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        templates=[
            BoundsCase(start_a, tiles_a, start_b, tiles_b),
            MATH_OP(MathOperation.Elwsub),
            MATH_FIDELITY(MathFidelity.LoFi),
            BROADCAST_TYPE(BroadcastType.Column),
        ],
        runtimes=[
            NUM_BLOCKS(1),
            NUM_TILES_IN_BLOCK(2),
            NUM_FACES(4),
            TEST_FACE_DIMS(face_r_dim=16),
        ],
        variant_stimuli=StimuliConfig(
            a,
            DataFormat.Float16_b,
            b,
            DataFormat.Float16_b,
            DataFormat.Float16_b,
            tile_count_A=4,
            tile_count_B=2,
            tile_count_res=2,
        ),
        dest_acc=DestAccumulation.No,
    )
    result = torch.as_tensor(cfg.run().result)
    failures = read_words_from_device(
        location=TestConfig.TENSIX_LOCATION,
        addr=cfg.variant_stimuli.buf_a_addr + 3 * 2048,
        word_count=1,
    )[0]
    assert failures == (0 if valid else 1), "wrapper checked the wrong CB span"
    torch.testing.assert_close(
        result.float(),
        a[start_a * 1024 : (start_a + 2) * 1024].float() - 3,
        rtol=0,
        atol=0,
    )
