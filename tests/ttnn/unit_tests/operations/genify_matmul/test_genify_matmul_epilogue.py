# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 2: fused epilogues (bias, bias_relu, swiglu, addcmul) on regime R1.

Shapes pin the epilogue against the blocking paths: a single K-block (cb_partials only for the
epilogue), a multi-K-block walk with a ragged last K-block, a 1-D grid (M = 1 / 32), folded leading
dims, an odd per-core pair count for swiglu, and L1-interleaved A / output. Accuracy gate:
helpers.TOLERANCES via run_genify_matmul (the golden-suite runner, same reference and operands).
"""

import pytest
import ttnn

from eval.golden_tests.genify_matmul.helpers import run_genify_matmul

EPILOGUES = ["bias", "bias_relu", "swiglu", "addcmul"]

SHAPES = [
    ((1, 1, 32, 32), (32, 64)),  # one tile pair: single K-block, q_n forced to 2 for swiglu
    ((1, 384, 1536), (1536, 1024)),  # 2-D grid
    ((2, 3, 64, 128), (128, 192)),  # folded leading dims, 3 swiglu pairs
    ((1, 2048), (2048, 1536)),  # M = 1 (m_non_aligned), 1-D grid
    ((1, 1, 128, 8032), (8032, 320)),  # deep K: multi K-block walk, ragged last block; 5 swiglu pairs
]
IDS = [f"A{'x'.join(map(str, a))}_B{'x'.join(map(str, b))}" for a, b in SHAPES]


def _run(inputs, epilogue, device, *, dtype=ttnn.bfloat16, weight_dtype=ttnn.bfloat16, acc=True, extras=None):
    run_genify_matmul(
        inputs,
        dtype=dtype,
        weight_dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=acc,
        placement="interleaved",
        epilogue=epilogue,
        device=device,
        extras=extras,
    )


@pytest.mark.parametrize("epilogue", EPILOGUES)
@pytest.mark.parametrize("inputs", SHAPES, ids=IDS)
def test_epilogue_bf16(inputs, epilogue, device):
    _run(inputs, epilogue, device)


@pytest.mark.parametrize("epilogue", EPILOGUES)
@pytest.mark.parametrize(
    "dtype,weight_dtype,acc",
    [
        (ttnn.bfloat8_b, ttnn.bfloat8_b, False),  # 16-bit DEST, Float16_b partials
        (ttnn.float32, ttnn.bfloat16, True),  # fp32 out: partials reloaded through SrcA
        (ttnn.bfloat16, ttnn.bfloat4_b, True),  # bf4b weight -> bf8b bias
    ],
    ids=["bf8b_acc16", "fp32", "bf16xbf4b"],
)
def test_epilogue_formats(dtype, weight_dtype, acc, epilogue, device):
    _run(((1, 384, 1536), (1536, 1024)), epilogue, device, dtype=dtype, weight_dtype=weight_dtype, acc=acc)


@pytest.mark.parametrize("epilogue", EPILOGUES)
def test_epilogue_l1_act_out(epilogue, device):
    _run(((1, 1, 128, 1024), (1024, 512)), epilogue, device, extras={"act_buffer": "l1", "out_buffer": "l1"})


@pytest.mark.parametrize("epilogue", EPILOGUES)
def test_epilogue_multi_block(epilogue, device):
    """A per-core region too big for one L1 block (S3 shrink): several blocks per core, so the bias /
    gate row is refreshed per block and the residual stream crosses block boundaries."""
    from ttnn.operations.genify_matmul.genify_matmul_program_descriptor import plan

    a_shape, b_shape = (1, 8192, 512), (512, 8192)
    n_out = b_shape[-1] // 2 if epilogue == "swiglu" else b_shape[-1]
    a = ttnn.allocate_tensor_on_device(ttnn.Shape(list(a_shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    b = ttnn.allocate_tensor_on_device(ttnn.Shape(list(b_shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 8192, n_out]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    _, grid, blk, _ = plan(a, b, o, True, device, epilogue=epilogue)
    blocks = -(-grid.core_m_tiles // blk.block_m_tiles) * -(-grid.core_n_tiles // blk.block_n_tiles)
    assert blocks > 1, f"expected a multi-block walk, got {grid} {blk}"
    for t in (a, b, o):
        ttnn.deallocate(t)
    _run((a_shape, b_shape), epilogue, device)
