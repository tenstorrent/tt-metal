#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""conv3d blocking sweep for the MiniMax-H3 visual VAE encoder.

The encoder is 83 % of VAE wall time after data-parallelism (STATE.md amendment 38) and
runs at ~2.3 TFLOP/s against the decoder's 14.0. The cause is that every one of its conv
shapes misses the tuned table, so ``_H3_ENCODER_BLOCKINGS`` in ``conv_minimax_h3.py`` is a
set of hand-shaped **stubs**. This replaces them with measured values.

The shapes below are the **padded** dims actually handed to ``ttnn.experimental.conv3d``,
after H3's reflect spatial pad and causal temporal pad, at ``h_factor = w_factor = 1``. That
is the shipping configuration: data-parallelism keeps a whole 256x256 tile on each device,
so the per-device shape is identical to the single-device one and the sweep is valid for
the deployed model. (H/W sharding, gated in ``test_vae_hw_parallel_minimax_h3.py``, would
change these dims and need its own sweep -- which is why the shape decision came first.)

Run one layer, or all of them sequentially:

    pytest models/tt_dit/tests/models/minimax_h3/sweep_conv3d_minimax_h3.py \\
        -k "b0_res" -s --timeout=0
    pytest models/tt_dit/tests/models/minimax_h3/sweep_conv3d_minimax_h3.py -s --timeout=0
"""

import pytest

import ttnn

from ..wan2_2.bruteforce_conv3d_sweep import TRACE_REGION_SIZE, run_sweep

# (name, C_in, C_out, kernel, stride, padding, T, H, W, h, w)
#
# T is the causal-padded frame count (T_level + 2 for a 3-tap conv); H/W include the reflect
# pad (+2 for a symmetric pad of 1, +1 for a downsampler's asymmetric bottom/right pad).
# Level geometry, from the encoder's own shape walk. Channels come from
# `block_in_channels = (block_out[0],) + block_out[:-1]` against
# `block_out = (128, 256, 256, 512, 512, 1024)`, so the levels are 128->128, 128->256,
# **256->256**, 256->512, 512->512, 512->1024 -- getting this wrong is easy and silent, and
# a first pass swept 512-channel convs at b2's size when b2 is 256->256.
# Spatial: b0 T=17 @256^2, b1 T=17 @128^2, b2 T=9 @64^2, b3 T=5 @32^2, b4/b5 T=5 @16^2
# (downsamplers on b0..b3, temporal stride on b1/b2 only).
#
# Ordered most-to-least compute (T x H x W x C_in x C_out), so a partial run still covers
# what dominates. b0's four resnet convs alone are the bulk of the encoder.
_SWEEP_LAYERS = [
    ("b1_res0_128_256", 128, 256, (3, 3, 3), (1, 1, 1), (0, 0, 0), 19, 130, 130, 1, 1),
    ("b1_res1_256_256", 256, 256, (3, 3, 3), (1, 1, 1), (0, 0, 0), 19, 130, 130, 1, 1),
    ("b0_res_128_128", 128, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 19, 258, 258, 1, 1),
    ("conv_in_32_128", 32, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 19, 258, 258, 1, 1),
    ("b0_down_128_128", 128, 128, (3, 3, 3), (1, 2, 2), (0, 0, 0), 19, 257, 257, 1, 1),
    ("b1_down_256_256", 256, 256, (3, 3, 3), (2, 2, 2), (0, 0, 0), 19, 129, 129, 1, 1),
    ("b2_res_256_256", 256, 256, (3, 3, 3), (1, 1, 1), (0, 0, 0), 11, 66, 66, 1, 1),
    ("b2_down_256_256", 256, 256, (3, 3, 3), (2, 2, 2), (0, 0, 0), 11, 65, 65, 1, 1),
    ("b3_res0_256_512", 256, 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), 7, 34, 34, 1, 1),
    ("b3_res1_512_512", 512, 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), 7, 34, 34, 1, 1),
    ("b3_down_512_512", 512, 512, (3, 3, 3), (1, 2, 2), (0, 0, 0), 7, 33, 33, 1, 1),
    ("b4_res_512_512", 512, 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), 7, 18, 18, 1, 1),
    ("b5_res0_512_1024", 512, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 7, 18, 18, 1, 1),
    ("b5_res1_1024_1024", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 7, 18, 18, 1, 1),
    ("conv_out_1024_48", 1024, 48, (3, 3, 3), (1, 1, 1), (0, 0, 0), 7, 18, 18, 1, 1),
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS,
    ids=[layer[0] for layer in _SWEEP_LAYERS],
)
def test_sweep_minimax_h3_encoder(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    device = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        f"sweep_results_minimax_h3_encoder/{layer_name}.json",
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=500,
        max_t_block=8,
        # hw_product=32 mirrors the wan sweep: a non-32 hw product hung Blackhole there, and
        # nothing about H3 makes that safer.
        hw_product=32,
    )
