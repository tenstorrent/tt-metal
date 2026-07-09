# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 720p (704x1280) BH Galaxy 4x8 conv3d blocking sweep.

The 720p warm gen logs `conv3d blocking [fallback] ... H=1 W=1` for the deep
1024-ch upsampler / decoder-s0 convs: they miss the tuned _BLOCKINGS table and
fall to the degenerate (Cin,32,1,1,1) one-pixel-per-work-unit path (~9x slower).
This sweep benchmarks hang-safe (hw_product=32) blockings for exactly those
runtime keys so they can be transplanted into _BLOCKINGS.

Key mapping (log H/W are H_out = H_in-(kH-1); run_sweep is fed H_in = H_out+kH-1):
  (3,3,3): H_in = H_out + 2 ;  (1,3,3): same, kT=1.

Run:  pytest models/tt_dit/tests/models/ltx/test_ltx_conv3d_sweep_720p.py -s --timeout=0
"""

import pytest

import ttnn
from models.tt_dit.tests.models.wan2_2.bruteforce_conv3d_sweep import TRACE_REGION_SIZE, run_sweep

# (name, C_in, C_out, kernel, stride, padding, T, H_in, W_in, h_factor, w_factor)
# Distinct degenerate-fallback keys from the 720p warm gen (non-5x5: hw=32 reachable).
_SWEEP_LAYERS_LTX_720P = [
    ("k1024_1024_t21_88", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 21, 8, 8, 4, 8),  # x8/gen
    ("k1024_1024_t21_87", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 21, 8, 7, 4, 8),  # x4
    ("k1024_1024_t3_87", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 8, 7, 4, 8),  # x4
    ("k1024_1024_t3_1312", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 13, 12, 4, 8),  # x4
    ("k1024_4096_t21_87", 1024, 4096, (3, 3, 3), (1, 1, 1), (0, 0, 0), 21, 8, 7, 4, 8),  # s0_up
    ("k1024_128_t4_1312", 1024, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 4, 13, 12, 4, 8),
    ("k1024_128_t3_87", 1024, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 3, 8, 7, 4, 8),
    ("k1024_128_t21_88", 1024, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 21, 8, 8, 4, 8),
    ("k512_128_t4_2422", 512, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 4, 24, 22, 4, 8),
    ("k128_1024_t21_87", 128, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 21, 8, 7, 4, 8),
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_1x1_ltx_720p"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_LTX_720P,
    ids=[l[0] for l in _SWEEP_LAYERS_LTX_720P],
)
def test_ltx_720p_conv3d_sweep(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    device = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_ltx_720p/{layer_name}_{C_in}x{C_out}.json"
    run_sweep(
        device,
        C_in,
        C_out,
        kernel,
        T,
        H,
        W,
        output,
        stride=stride,
        padding=padding,
        h_factor=h_factor,
        w_factor=w_factor,
        max_combos=500,
        max_t_block=8,
        hw_product=32,  # BH hang-safe (matches the 720p full-T WAN sweep)
    )
