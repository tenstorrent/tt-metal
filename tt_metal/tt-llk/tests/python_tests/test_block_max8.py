# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct SFPU coverage with eight occupied DST slots and alternating DST halves."""

from dataclasses import dataclass

import pytest
import torch
from tt_llk_harness import (
    DataFormat,
    DestAccumulation,
    StimuliConfig,
)
from tt_llk_harness import TestConfig as LLKTestConfig
from tt_llk_harness import (
    blackhole_only,
    input_output_formats,
    params,
    tilize_block,
)


@dataclass
class ValidScores(params.RuntimeParameter):
    """Pass the valid prefix and optional DST sign-bit probe to the driver."""

    valid_scores: int
    inject_negative_zero: int

    def convert_to_cpp(self):
        """Provide the equivalent constant for harness template mode."""
        return (
            f"constexpr unsigned VALID_SCORES = {self.valid_scores};"
            f"constexpr unsigned INJECT_NEGATIVE_ZERO = {self.inject_negative_zero};"
        )

    def convert_to_struct_fields(self):
        """Serialize the valid prefix and probe flag as unsigned runtime arguments."""
        return "unsigned VALID_SCORES; unsigned INJECT_NEGATIVE_ZERO;", "II"


@blackhole_only
@pytest.mark.parametrize("dst_index", [0, 3, 7])
@pytest.mark.parametrize(
    "valid", [0, 1, 7, 8, 9, 15, 16, 17, 511, 512, 513, 1023, 1024]
)
@pytest.mark.parametrize(
    "pattern", ["random", "winners", "fractional", "negative_zero"]
)
def test_block_max8(dst_index, valid, pattern):
    """Check all pooled blocks and preserve every other tile in both DST halves.

    Three batches reuse the first half after exercising the second. Directed
    winners cover every position within each eight-score block. Ordinary patterns
    put large positive values in invalid tails to detect masking omissions.
    The negative-zero pattern is injected directly into DST because the input
    unpack/datacopy path can canonicalize its sign before the SFPU sees it.
    Exact negative-zero output bits are encoded as -1 in DST before packing,
    which otherwise canonicalizes zeros; positive zero cannot pass this probe.
    """
    generator = torch.Generator().manual_seed(4869)
    logical = torch.randint(-256, 0, (24, 32, 32), generator=generator).bfloat16()
    targets = logical[dst_index::8].reshape(3, 128, 8).clone()
    if pattern == "winners":
        targets.fill_(-256)
        blocks = torch.arange(128)
        targets[:, blocks, blocks % 8] = (blocks - 128).bfloat16()
    elif pattern == "fractional":
        targets *= 0.125
        targets[:, 0, :] = torch.tensor(
            [0.0, -0.0, 768.0, -768.0, 1.5, -1.5, 0.125, -0.125], dtype=torch.bfloat16
        )
        targets[:, 1, :] = float("-inf")
        targets[:, 2, 0] = float("inf")
    elif pattern == "negative_zero":
        targets.fill_(-0.0)
    targets.reshape(3, 1024)[:, valid:] = 2048
    logical[dst_index::8] = targets.reshape(3, 32, 32)
    masked = targets.reshape(3, 1024).clone()
    masked[:, valid:] = float("-inf")
    expected = masked.reshape(3, 128, 8).amax(-1)
    if pattern == "negative_zero":
        expected[expected.view(torch.int16) == -32768] = -1.0
    src = tilize_block(
        logical.reshape(-1), [24 * 32, 32], DataFormat.Float16_b
    ).flatten()
    formats = input_output_formats([DataFormat.Float16_b])[0]
    config = LLKTestConfig(
        "sources/block_max8_test.cpp",
        formats,
        runtimes=[
            params.TILE_COUNT(24),
            params.DEST_INDEX(dst_index),
            ValidScores(valid, int(pattern == "negative_zero")),
        ],
        variant_stimuli=StimuliConfig(
            src,
            formats.input_format,
            torch.zeros_like(src),
            formats.input_format,
            formats.output_format,
            tile_count_A=24,
            tile_count_B=24,
            tile_count_res=24,
        ),
        dest_acc=DestAccumulation.No,
    )
    for _ in range(2):
        actual = torch.as_tensor(config.run().result, dtype=torch.bfloat16).reshape(
            3, 8, 1024
        )
        assert torch.equal(
            actual[:, dst_index, :128].view(torch.int16), expected.view(torch.int16)
        ), "Pooled values or compacted order differ"
        untouched = [i for i in range(8) if i != dst_index]
        assert torch.equal(
            actual[:, untouched], src.reshape(3, 8, 1024)[:, untouched]
        ), "Another DST tile changed"
