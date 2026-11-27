# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 13: split output (`chunks`) on the interleaved placement.

The program writes each chunk directly (kernels/out_chunks.hpp): the writer, and the reader's tail-write
share, route every output tile column to its chunk. Cases pin a chunk boundary inside a block row (chunk
width 1 tile), every epilogue (swiglu: chunks of the halved width; addcmul: full-width residual read by
global column), M = 1, folded leading dims, a batched weight, and an L1 output. Accuracy gate:
helpers.TOLERANCES via run_genify_matmul (the golden runner splits the reference the same way).
"""

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul


def _run(inputs, epilogue, chunks, device, *, dtype=ttnn.bfloat16, weight_dtype=ttnn.bfloat16, acc=True, **extras):
    run_genify_matmul(
        inputs,
        dtype=dtype,
        weight_dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=acc,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
        extras={"chunks": chunks, **extras},
    )


@pytest.mark.parametrize("epilogue", ["none", "bias", "bias_relu", "swiglu", "addcmul"])
@pytest.mark.parametrize("chunks", [2, 3, 16])
def test_chunks_epilogues(epilogue, chunks, device):
    # N_out = 1536 for every epilogue (swiglu: B is 3072 wide): chunk widths 24 / 16 / 3 tiles.
    n = 3072 if epilogue == "swiglu" else 1536
    _run(((1, 384, 1536), (1536, n)), epilogue, chunks, device)


@pytest.mark.parametrize(
    "inputs,chunks",
    [
        (((1, 384, 1536), (1536, 1024)), 32),  # chunk width 1 tile: every block row straddles chunks
        (((1, 2048), (2048, 1536)), 6),  # M = 1 (m_non_aligned), 1-D grid
        (((2, 3, 64, 128), (128, 192)), 3),  # folded leading dims
        (((1, 8, 64, 128), (1, 8, 128, 256)), 4),  # batched weight (R1d)
        (((1, 2048, 1024), (1024, 4096)), 4),  # large M x N, multi-block walk
    ],
    ids=["width1", "m1", "folded", "batched", "large"],
)
def test_chunks_shapes(inputs, chunks, device):
    _run(inputs, "none", chunks, device)


def test_chunks_bias_fp32_a_bf16_out(device):
    """time_mod's format mix (fp32 A, bf16 bias / out) at a small size."""
    _run(
        ((1, 1024), (1024, 2176)),
        "bias",
        17,
        device,
        dtype=ttnn.float32,
        output_dtype=ttnn.bfloat16,
        bias_dtype=ttnn.bfloat16,
    )


def test_chunks_l1_out(device):
    _run(((1, 1, 128, 1024), (1024, 512)), "bias", 4, device, act_buffer="l1", out_buffer="l1")


def test_chunks_contract(device, expect_error):
    a = ttnn.from_torch(torch.randn(1, 64, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.randn(128, 192), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for bad in (0, -1, 4, 5, 2.0):  # 192 / 4 = 48 (not a tile multiple), 192 % 5 != 0
        with expect_error(ValueError, "chunks"):
            genify_matmul(a, b, chunks=bad)
    out = genify_matmul(a, b, chunks=1)
    assert isinstance(out, ttnn.Tensor)
    out = genify_matmul(a, b, chunks=2)
    assert isinstance(out, list) and len(out) == 2 and list(out[0].shape) == [1, 64, 96]
