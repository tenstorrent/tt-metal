# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.to_layout`` (tilize / tilize_with_val_padding / untilize /
untilize_with_unpadding) and ``ttnn.untilize`` — every layout conversion the model ran.

Shape source (kernel cache), factory line refs at revision ``2508e6e197a``:
  * WH-block tilize (``reader_unary_pad_multicore_both_dims`` ct = [logical_H, N*C, 32, elem,
    row_bytes]) and untilize (``writer_unary_stick_layout_wh_multicore`` ct = [out_H, N*C, 32,
    row_bytes]) pin the full shape [H]: the BEV-attention tensor [1, 8, 6, 10000]
    (heads x cams x queries) and its siblings.
  * Row-split tilize / untilize (``reader_unary_pad_dims_split_rows_multicore``,
    ``writer_unary_stick_layout_split_rows_multicore``) pin the row width; the per-core block
    counts on the 20-core grid pin the row count (e.g. 313 = ceil(10000/32)) [M].
  * Small-width conversions only bound the row count [L]; representative shapes are used.
Layout conversion is value-preserving: checked bit-exact (``assert_lossless``); bfloat8_b
sources are compared against their quantized values.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

Q, E = U.NUM_QUERY, U.EMBED

# fmt: off
# (shape, dtype, in_mem, out_mem)
TILIZE_CASES = [
    pytest.param((1, U.NUM_HEADS, U.NUM_CAMS, Q), "bf16", "dram", "dram", id="T1-bev-attn-1x8x6x10000"),
    pytest.param((1, Q, E), "bf16", "l1", "l1", id="T2-bev-query-l1"),
    pytest.param((1, Q, E), "bf16", "dram", "dram", id="T3-bev-query-dram"),
    pytest.param((1, Q, 128), "bf16", "dram", "dram", id="T4-1x10000x128"),
    pytest.param((1, 18), "fp32", "dram", "dram", id="T5-canbus-fp32"),
    pytest.param((1, 1), "fp32", "dram", "dram", id="T6-fp32-w1"),
    pytest.param((150, 32), "bf16", "l1", "l1", id="T7-w32-padded"),
    pytest.param((100, 8), "bf16", "l1", "l1", id="T8-w8-padded"),
    pytest.param((100, 4), "bf16", "dram", "dram", id="T9-w4-padded"),
    pytest.param((100, 2), "bf16", "dram", "dram", id="T10-w2-padded"),
    pytest.param((1, 1, 160, 32), "bf16", "dram", "dram", id="T11-w32-aligned"),
    pytest.param((1, 1, 160, 128), "bf16", "dram", "dram", id="T12-w128-aligned"),
    pytest.param((1, 1, 160, E), "bf16", "l1", "l1", id="T13-w256-aligned"),
]

# (shape, src dtype, in_mem, out_mem)
UNTILIZE_CASES = [
    pytest.param((1, U.NUM_HEADS, U.NUM_CAMS, Q), "bf16", "dram", "dram", id="U1-bev-attn-1x8x6x10000"),
    pytest.param((1, U.TSA_POINTS, U.NUM_CAMS, Q), "bf16", "dram", "dram", id="U2-1x4x6x10000"),
    pytest.param((1, 1, 1, Q), "bf16", "l1", "l1", id="U3-1x1x1x10000"),
    pytest.param((U.BEV_QUEUE, Q, E), "bfp8", "dram", "l1", id="U4-bev-queue-bfp8"),
    pytest.param((1, Q, E), "bf16", "dram", "l1", id="U5-bev-query"),
    pytest.param((1, 1, 300, 3 * E), "bfp8", "dram", "dram", id="U6-fused-qkv-w768-bfp8"),
    pytest.param((1, Q, 64, 2), "bf16", "dram", "dram", id="U7-sampling-locs-w2"),
    pytest.param((100, 8), "bf16", "dram", "l1", id="U8-w8"),
    pytest.param((100, 32), "bf16", "dram", "l1", id="U9-w32"),
    pytest.param((100, 1), "fp32", "l1", "l1", id="U10-fp32-w1"),
]
# fmt: on


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, in_mem, out_mem", TILIZE_CASES)
def test_to_layout_tile(ttnn_mesh_device, reset_seeds, shape, dt, in_mem, out_mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape, dtype=torch.float32 if dt == "fp32" else torch.bfloat16)
    x = U.to_tt_mem(x_torch, mesh, in_mem, dtype=U.DTYPES[dt], layout=ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=U.resolve_memcfg(out_mem, mesh))
    assert out.layout == ttnn.TILE_LAYOUT
    U.assert_lossless(x_torch, out, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape, dt, in_mem, out_mem", UNTILIZE_CASES)
def test_to_layout_row_major(ttnn_mesh_device, reset_seeds, shape, dt, in_mem, out_mem):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape, dtype=torch.float32 if dt == "fp32" else torch.bfloat16)
    x = U.to_tt_mem(x_torch, mesh, in_mem, dtype=U.DTYPES[dt])
    ref = U.seen(x, x_torch, mesh)  # bfp8 sources: the quantized values are what must survive
    # bfloat8_b can't be ROW_MAJOR: the model untilized those into bf16.
    kwargs = {"dtype": ttnn.bfloat16} if dt == "bfp8" else {}
    out = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=U.resolve_memcfg(out_mem, mesh), **kwargs)
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    U.assert_lossless(ref, out, mesh_device=mesh)


@U.with_default_mesh()
# ttnn.untilize on an L1 TILE tensor, W=256 (untilize_variable_num_blocks [8, 0, 16]) [L rows].
@pytest.mark.parametrize("shape", [pytest.param((1, 1, 320, E), id="U11-untilize-w256-l1")])
def test_untilize(ttnn_mesh_device, reset_seeds, shape):
    mesh = ttnn_mesh_device
    x_torch = U.torch_rand(shape)
    x = U.to_tt_mem(x_torch, mesh, "l1")
    out = ttnn.untilize(x, memory_config=ttnn.L1_MEMORY_CONFIG)
    U.assert_lossless(x_torch, out, mesh_device=mesh)
