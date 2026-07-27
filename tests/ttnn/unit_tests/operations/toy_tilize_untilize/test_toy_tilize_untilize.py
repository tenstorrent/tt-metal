# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from ttnn.operations.toy_tilize_untilize import toy_tilize_untilize


def pcc(a, b):
    """Pearson correlation coefficient between two tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


# Heights: 1, small prime, non-aligned <32, aligned, prime >32, non-aligned >32, aligned >64, prime >64
HEIGHTS = [1, 7, 16, 32, 37, 48, 64, 97]

# Widths: 1-tile minimum (16 bf16 = 32B DRAM align), small, prime, aligned, non-aligned,
#         multi-tile aligned, prime >tile, 4-tile aligned, >256 (exceeds DEST), large prime
WIDTHS = [16, 17, 32, 48, 64, 67, 96, 128, 288, 331]

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
]

GRANULARITIES = [
    pytest.param(False, id="tile"),
    pytest.param(True, id="row"),
]

# Build cross-product shape list with readable IDs
SHAPES = [pytest.param((h, w), id=f"H{h}_W{w}") for h in HEIGHTS for w in WIDTHS]


@pytest.mark.parametrize("use_row_granularity", GRANULARITIES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_toy_tilize_untilize(device, shape, dtype, use_row_granularity):
    """Identity test: tilize then untilize should return original data."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = toy_tilize_untilize(ttnn_input, use_row_granularity=use_row_granularity)

    assert list(ttnn_output.shape) == list(shape)
    torch_output = ttnn.to_torch(ttnn_output)

    if dtype == ttnn.bfloat16:
        # bf16 tilize+untilize is purely data reordering — must be exact
        assert torch.equal(
            torch_output, torch_input
        ), f"bf16 mismatch. Max diff: {(torch_output - torch_input).abs().max()}"
    else:
        # fp32 tilize+untilize with fp32 dest accumulation — should be near-lossless
        correlation = pcc(torch_output, torch_input)
        assert correlation > 0.999, f"fp32 PCC too low: {correlation:.6f}"


# Streaming tilize (compute_kernel_lib::tilize_stream): per-tile pack + push_back(1).
# This proves the shared helper's streaming entry point produces a bit-identical tiled
# result (identity holds exactly for bf16) through the real ttnn generic_op JIT path,
# and does not regress the atomic path. Streaming is symmetric-only, so
# use_row_granularity is always False here.
#
# NOTE: The toy op interleaves tilize->untilize per block, and untilize consumes a whole
# tile-row, so cb_tilized must hold the full W-wide block here (streaming's small-output-CB
# win cannot be exercised through this identity op). The definitive proof that streaming is
# bit-exact to the atomic path AND to a host golden at W=4 and W=128 tiles with the output CB
# sized to just 2 tiles lives in the tt_metal/programming_examples/streaming_tilize device
# example, which drives the helper directly with a tile-granular consumer. Widths here (4 and
# 32 tiles) are chosen to stay within the single-core L1 budget of the block-sized toy CBs.
STREAM_SHAPES = [pytest.param((h, w), id=f"H{h}_W{w}") for h in [32, 64] for w in [128, 1024]]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", STREAM_SHAPES)
def test_toy_tilize_untilize_streaming(device, shape, dtype):
    """Streaming-tilize identity test: tilize_stream then untilize returns original data."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(shape, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Streaming path (the helper entry point under test).
    ttnn_stream = toy_tilize_untilize(ttnn_input, use_streaming_tilize=True)
    # Atomic path (must still pass unchanged) — used as a bit-exact reference.
    ttnn_atomic = toy_tilize_untilize(ttnn_input, use_streaming_tilize=False)

    assert list(ttnn_stream.shape) == list(shape)
    torch_stream = ttnn.to_torch(ttnn_stream)
    torch_atomic = ttnn.to_torch(ttnn_atomic)

    if dtype == ttnn.bfloat16:
        # bf16 identity is pure data reordering: streaming must match input AND the
        # atomic path exactly (same regular-tilize LLK datapath, different pack granularity).
        assert torch.equal(
            torch_stream, torch_input
        ), f"streaming bf16 mismatch vs input. Max diff: {(torch_stream - torch_input).abs().max()}"
        assert torch.equal(
            torch_stream, torch_atomic
        ), f"streaming bf16 mismatch vs atomic. Max diff: {(torch_stream - torch_atomic).abs().max()}"
    else:
        correlation = pcc(torch_stream, torch_input)
        assert correlation > 0.999, f"streaming fp32 PCC too low: {correlation:.6f}"
