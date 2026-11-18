# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 14: A as a list of K segments (concat(A_i, dim=-1) @ B, never materialized).

The reader (and R1b's A-streaming writer) map each K tile to its segment through kernels/a_segments.hpp.
Cases pin: a segment boundary inside a K-block, one-tile segments (every K-block row straddles segments),
every epilogue, each R1-family operand source (R1 per-core reads, R1a line injectors, R1b writer stream,
R1d batched weight), A in L1, segments + chunks together, and the ValueError contract. Accuracy gate:
helpers.TOLERANCES via run_genify_matmul (the golden runner splits A with torch.split the same way).
"""

import pytest
import torch
import ttnn

from eval.golden_tests.genify_matmul.helpers import run_genify_matmul
from ttnn.operations.genify_matmul import genify_matmul
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

HIFI2 = {"math_fidelity": ttnn.MathFidelity.HiFi2}


def _run(inputs, epilogue, split, device, *, dtype=ttnn.bfloat16, **extras):
    run_genify_matmul(
        inputs,
        dtype=dtype,
        weight_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
        extras={"a_split_k": split, **extras},
    )


def _mode(device, a_shape, b_shape, split, epilogue="none"):
    """The operand-source mode the plan picks for the segmented call (0 R1, 1 R1a, 2 R1b)."""
    segs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(a_shape[:-1]) + [k]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        for k in split
    ]
    n_out = b_shape[-1] // 2 if epilogue == "swiglu" else b_shape[-1]
    b = ttnn.allocate_tensor_on_device(ttnn.Shape(list(b_shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    o = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(a_shape[:-1]) + [n_out]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device
    )
    geom, grid, blk, _ = pd.plan(segs, b, o, True, device, epilogue=epilogue, math_fidelity=ttnn.MathFidelity.HiFi2)
    assert geom.tensor_k_tiles == sum(split) // 32
    for t in (*segs, b, o):
        ttnn.deallocate(t)
    return pd.operand_mode(geom, grid, blk)


@pytest.mark.parametrize("epilogue", ["none", "bias", "bias_relu", "swiglu", "addcmul"])
def test_segments_epilogues(epilogue, device):
    n = 2048 if epilogue == "swiglu" else 1024
    _run(((1, 384, 1536), (1536, n)), epilogue, [512, 1024], device)


@pytest.mark.parametrize(
    "inputs,split",
    [
        (((1, 384, 1536), (1536, 1024)), [32, 1472, 32]),  # one-tile edge segments
        (((1, 256, 256), (256, 512)), [32] * 8),  # every K tile its own segment
        (((1, 384, 1536), (1536, 1024)), [1536]),  # one segment == the single-tensor call
        (((1, 2048), (2048, 1536)), [640, 1408]),  # M = 1 (m_non_aligned)
        (((2, 3, 64, 256), (256, 192)), [96, 160]),  # folded leading dims
        (((1, 8, 64, 128), (1, 8, 128, 256)), [64, 64]),  # batched weight (R1d)
        (((1, 2048, 1024), (1024, 4096)), [320, 704]),  # large M x N, multi-block walk
    ],
    ids=["edges", "per_tile", "one", "m1", "folded", "batched", "large"],
)
def test_segments_shapes(inputs, split, device):
    _run(inputs, "none", split, device)


def test_segments_r1a(device):
    a_shape, b_shape, split = (1, 1024, 2048), (2048, 1536), [672, 1376]
    assert _mode(device, a_shape, b_shape, split) == 1, "expected the R1a line injectors"
    _run((a_shape, b_shape), "none", split, device, **HIFI2)


@pytest.mark.parametrize("epilogue", ["none", "swiglu"])
def test_segments_r1b(epilogue, device):
    a_shape, split = (1, 256, 6144), [1536, 4608]
    b_shape = (6144, 6144 if epilogue == "none" else 12288)
    assert _mode(device, a_shape, b_shape, split, epilogue) == 2, "expected the R1b writer A stream"
    _run((a_shape, b_shape), epilogue, split, device, **HIFI2)


def test_segments_l1_a(device):
    _run(((1, 1, 128, 1024), (1024, 512)), "bias", [256, 768], device, act_buffer="l1", out_buffer="l1")


def test_segments_with_chunks(device):
    _run(((1, 384, 1536), (1536, 1536)), "bias", [512, 1024], device, chunks=3)


def test_segments_fp32_a(device):
    _run(((1, 1024), (1024, 2048)), "none", [256, 768], device, dtype=ttnn.float32)


def test_segments_contract(device, expect_error):
    def seg(shape, dtype=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            torch.randn(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )

    b = seg((128, 192))
    with expect_error(ValueError, "sum to K"):
        genify_matmul([seg((1, 64, 64)), seg((1, 64, 32))], b)
    with expect_error(ValueError, "multiple of"):
        genify_matmul([seg((1, 64, 48)), seg((1, 64, 80))], b)
    with expect_error(ValueError, "leading dims"):
        genify_matmul([seg((1, 64, 64)), seg((1, 96, 64))], b)
    with expect_error(ValueError, "dtype"):
        genify_matmul([seg((1, 64, 64)), seg((1, 64, 64), dtype=ttnn.float32)], b)
    with expect_error(ValueError, "placement"):
        genify_matmul([seg((1, 64, 64)), seg((1, 64, 64), mc=ttnn.L1_MEMORY_CONFIG)], b)
    out = genify_matmul([seg((1, 64, 64)), seg((1, 64, 64))], b)
    assert isinstance(out, ttnn.Tensor) and list(out.shape) == [1, 64, 192]
