# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
L1 BLOCK_SHARDED input, out of place (Refinement 1) and in place (Refinement 2): even shards, ROW_MAJOR + TILE,
bf16, gamma/beta bf16 RM.

Covers the sharded geometry classes the kernels branch on:
  * TILE: the shard is the input ring (K = B), the output tiles are the output shard;
  * RM tile-wide slices (k = 1) and narrow slices (k = lcm(sw, 32) / sw = 2 / 4 sticks per virtual row);
  * N > 1 (gy / N shards per image, each image combines separately);
  * RM with K < B (tail resident, head re-tilized from the shard; the output shard is written tail-first);
  * the three model grids 8x8, 4x8, 4x4.
The output contract (BLOCK_SHARDED L1, identical shard spec, input layout, values; in place: the output is the
input's buffer) is checked by the golden helper run_groupnorm_sc_N_1_HW_C.
"""

import gc

import pytest
import torch
import ttnn

from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import run_groupnorm_sc_N_1_HW_C

AXES = dict(
    dtype=ttnn.bfloat16,
    affine="gamma_beta",
    affine_dtype=ttnn.bfloat16,
    affine_layout=ttnn.ROW_MAJOR_LAYOUT,
    in_place=False,
)

CASES = [
    # (inputs, layout)
    pytest.param(((1, 1, 1024, 1280), 32, (8, 8)), ttnn.TILE_LAYOUT, id="tile_1024x1280_8x8"),
    pytest.param(((1, 1, 4096, 640), 32, (4, 8)), ttnn.TILE_LAYOUT, id="tile_4096x640_4x8"),
    pytest.param(((1, 1, 256, 640), 32, (4, 4)), ttnn.TILE_LAYOUT, id="tile_256x640_4x4"),
    pytest.param(((2, 1, 256, 1280), 32, (8, 8)), ttnn.TILE_LAYOUT, id="tile_2x256x1280_8x8"),
    pytest.param(((1, 1, 1024, 768), 32, (8, 8)), ttnn.TILE_LAYOUT, id="tile_1024x768_8x8"),
    pytest.param(((1, 1, 1024, 1280), 32, (8, 8)), ttnn.ROW_MAJOR_LAYOUT, id="rm_1024x1280_8x8"),
    pytest.param(((1, 1, 4096, 320), 32, (8, 8)), ttnn.ROW_MAJOR_LAYOUT, id="rm_4096x320_8x8_k4"),
    pytest.param(((1, 1, 4096, 384), 32, (8, 8)), ttnn.ROW_MAJOR_LAYOUT, id="rm_4096x384_8x8_k2"),
    pytest.param(((1, 1, 4096, 960), 32, (8, 8)), ttnn.ROW_MAJOR_LAYOUT, id="rm_4096x960_8x8_w15"),
    pytest.param(((2, 1, 1024, 640), 32, (4, 8)), ttnn.ROW_MAJOR_LAYOUT, id="rm_2x1024x640_4x8"),
    pytest.param(((1, 1, 256, 640), 32, (4, 4)), ttnn.ROW_MAJOR_LAYOUT, id="rm_256x640_4x4"),
]


@pytest.mark.parametrize("inputs, layout", CASES)
def test_sharded_oop(device, inputs, layout):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, layout=layout, **AXES)


@pytest.mark.parametrize("inputs, layout", CASES)
def test_sharded_inplace(device, inputs, layout):
    """in_place=True: the result overwrites the input shard (TILE: tile i packed over tile i; RM: each chunk
    untilized over the rows it was tilized from)."""
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, layout=layout, **{**AXES, "in_place": True})


# 8x8 RM shards too large for a resident ring with default tuning: 20480x1280 keeps a partial tail
# (K = 9 of B = 16 chunks), 30720x1280 streams every chunk (K = 0). The streaming shard only fits
# L1 in place: out of place, the input and output shards together overflow it.
RM_PARTIAL = pytest.param(((1, 1, 20480, 1280), 32, (8, 8)), {"partial"}, id="partial")
RM_STREAM = pytest.param(((1, 1, 30720, 1280), 32, (8, 8)), {"stream"}, id="stream_only")


def _run_rm_partial_residency(device, assert_inputs_regime, inputs, regime, in_place):
    # These shards take over half of each core's L1, so the previous case's output shard must be
    # freed before this case allocates its input.
    gc.collect()
    axes = {**AXES, "in_place": in_place}
    assert_inputs_regime(
        regime,
        inputs,
        dtype=axes["dtype"],
        layout=ttnn.ROW_MAJOR_LAYOUT,
        affine_dtype=axes["affine_dtype"],
        affine_layout=axes["affine_layout"],
        in_place=in_place,
    )
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, **axes)


@pytest.mark.parametrize("inputs, regime", [RM_PARTIAL])
def test_sharded_rm_partial_residency(device, assert_inputs_regime, inputs, regime):
    """RM with K < B: the head chunks are re-tilized from the shard in pass 2 and the output shard is
    written tail-first behind head-skip credits."""
    _run_rm_partial_residency(device, assert_inputs_regime, inputs, regime, in_place=False)


@pytest.mark.parametrize("inputs, regime", [RM_PARTIAL, RM_STREAM])
def test_sharded_rm_partial_residency_inplace(device, assert_inputs_regime, inputs, regime):
    """In place with K < B: the tail is written over rows tilized in pass 1, and each head chunk is re-tilized
    from the shard right before its rows are overwritten."""
    _run_rm_partial_residency(device, assert_inputs_regime, inputs, regime, in_place=True)
