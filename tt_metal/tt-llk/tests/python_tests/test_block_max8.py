# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct SFPU and FPU (GMPOOL) coverage with eight occupied DST slots and alternating DST halves."""

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

DST_INDICES = [0, 3, 7]
VALID_COUNTS = [0, 1, 7, 8, 9, 15, 16, 17, 511, 512, 513, 1023, 1024]
PATTERNS = ["random", "winners", "fractional", "negative_zero"]
# Host markers that block_max8_fpu_test.cpp replaces in DST with BF16 denormals.
DENORMAL_MARKERS = {0.375: 0x007F, -0.375: -0x7FFF}


# Call k of an acquire pools slot (dst_index + CALL_STRIDE * k) % 8, as in block_max8_fpu_test.cpp.
CALL_STRIDE = 3


def _run_block_max8(
    source, dst_index, valid, pattern, negative_zero_result, fpu_calls=None
):
    """Check all pooled blocks and preserve every other tile in both DST halves.

    Three batches reuse the first half after exercising the second. Directed
    winners cover every position within each eight-score block. Ordinary patterns
    put large positive values in invalid tails to detect masking omissions.
    The negative-zero and denormal patterns are injected directly into DST
    because the input unpack/datacopy path can canonicalize or flush them before
    the kernel sees them. The denormal pattern (FPU kernel only) makes some block
    maxima positive or negative denormals, which GMPOOL returns as +0.0.
    negative_zero_result is the expected output for a block whose maximum is
    -0.0. Both kernels encode an exact -0.0 result as -1 before packing, which
    otherwise canonicalizes zeros, so -1.0 checks that the sign survived and
    0.0 checks that it was flushed. fpu_calls (FPU kernel only) is the number
    of calls per DST acquire, each on a different slot.
    """
    pooled = (
        [dst_index]
        if fpu_calls is None
        else [(dst_index + CALL_STRIDE * k) % 8 for k in range(fpu_calls)]
    )
    generator = torch.Generator().manual_seed(4869)
    logical = torch.randint(-256, 0, (24, 32, 32), generator=generator).bfloat16()
    expected = {}
    for slot in pooled:
        targets = logical[slot::8].reshape(3, 128, 8).clone()
        if pattern == "winners":
            targets.fill_(-256)
            blocks = torch.arange(128)
            # Offset by slot so that results swapped between slots differ.
            targets[:, blocks, blocks % 8] = (blocks - 128 - slot).bfloat16()
        elif pattern == "fractional":
            targets *= 0.125
            targets[:, 0, :] = torch.tensor(
                [0.0, -0.0, 768.0, -768.0, 1.5, -1.5, 0.125, -0.125],
                dtype=torch.bfloat16,
            )
            targets[:, 1, :] = float("-inf")
            targets[:, 2, 0] = float("inf")
        elif pattern == "negative_zero":
            targets.fill_(-0.0)
        elif pattern == "denormal":
            blocks = torch.arange(128)
            targets[:, blocks[0::3], 0] = 0.375
            targets[:, blocks[1::3], 7] = -0.375
        targets.reshape(3, 1024)[:, valid:] = 2048
        logical[slot::8] = targets.reshape(3, 32, 32)
        masked = targets.reshape(3, 1024).clone()
        if pattern == "denormal":
            for marker, bits in DENORMAL_MARKERS.items():
                masked[masked == marker] = (
                    torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16).item()
                )
        masked[:, valid:] = float("-inf")
        expected[slot] = masked.reshape(3, 128, 8).amax(-1)
        if pattern == "negative_zero":
            expected[slot][
                expected[slot].view(torch.int16) == -32768
            ] = negative_zero_result
        if pattern == "denormal":
            expected[slot][(expected[slot].view(torch.int16) & 0x7F80) == 0] = 0.0
    src = tilize_block(
        logical.reshape(-1), [24 * 32, 32], DataFormat.Float16_b
    ).flatten()
    formats = input_output_formats([DataFormat.Float16_b])[0]
    runtimes = [
        params.TILE_COUNT(24),
        params.DEST_INDEX(dst_index),
        params.BLOCK_MAX8_PARAMS(valid, int(pattern == "negative_zero")),
    ]
    if fpu_calls is not None:
        runtimes.append(
            params.BLOCK_MAX8_FPU_PARAMS(fpu_calls, int(pattern == "denormal"))
        )
    config = LLKTestConfig(
        source,
        formats,
        runtimes=runtimes,
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
        for slot in pooled:
            assert torch.equal(
                actual[:, slot, :128].view(torch.int16),
                expected[slot].view(torch.int16),
            ), f"Pooled values or compacted order differ in slot {slot}"
        untouched = [i for i in range(8) if i not in pooled]
        assert torch.equal(
            actual[:, untouched], src.reshape(3, 8, 1024)[:, untouched]
        ), "Another DST tile changed"


@blackhole_only
@pytest.mark.parametrize("dst_index", DST_INDICES)
@pytest.mark.parametrize("valid", VALID_COUNTS)
@pytest.mark.parametrize("pattern", PATTERNS)
def test_block_max8(dst_index, valid, pattern):
    """SFPU variant: a negative-zero block maximum keeps its sign (-1 probe; +0 cannot pass)."""
    _run_block_max8(
        "sources/block_max8_test.cpp",
        dst_index,
        valid,
        pattern,
        negative_zero_result=-1.0,
    )


@blackhole_only
@pytest.mark.parametrize("dst_index", DST_INDICES)
@pytest.mark.parametrize("valid", VALID_COUNTS)
@pytest.mark.parametrize("pattern", PATTERNS + ["denormal"])
@pytest.mark.parametrize("fpu_calls", [1, 3])
def test_block_max8_fpu(dst_index, valid, pattern, fpu_calls):
    """FPU (GMPOOL) variant: GMPOOL flushes exponent-zero inputs, so a negative-zero or denormal maximum returns +0.

    Three calls per DST acquire chain FPU calls on different slots in one acquire.
    """
    _run_block_max8(
        "sources/block_max8_fpu_test.cpp",
        dst_index,
        valid,
        pattern,
        negative_zero_result=0.0,
        fpu_calls=fpu_calls,
    )
