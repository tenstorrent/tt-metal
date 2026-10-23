#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Conv3d blocking sweep for LTX-2.3 1080p / 153 frames (25 fps): LTX layer list for the wan2_2 harness.

153f shifts decoder latent T 19->20, missing ``_BLOCKINGS`` at these 15 sites; the channel-only
fallbacks cost 6.7s -> 12.1s total generate (keys harvested from get_conv3d_config fallback warnings).
Run on Galaxy (1x1 submesh, needs exclusive access): pytest <this file> -s --timeout=0 [-k <name>].
Add each winner JSON to ``_BLOCKINGS`` keyed ``(4, 8, C_in, C_out, kernel, T, H-(kH-1), W-(kW-1))``
(output dims!), then re-run ~/ltx-25fps-validate.sh to confirm ~7 s.
"""

import pytest

import ttnn

from ..wan2_2.bruteforce_conv3d_sweep import TRACE_REGION_SIZE, run_sweep

# LTX-2.3 1080p 153f, BH Galaxy 4x8 (h=tensor_parallel factor, w=sequence_parallel factor).
# T is post-temporal-pad; H/W are per-device conv INPUT dims = _BLOCKINGS output-dim key + (kH-1, kW-1).
# Names carry the table-key (output) H, so e.g. _h9 sweeps input H=11. Ordered by compute volume.
_SWEEP_LAYERS_LTX_1080P_153F = [
    # (name,                C_in, C_out, kernel,    stride,    padding,    T,   H,  W, h, w)
    # --- 94% of total volume: the T=155 / T=79 decoder convs ---
    ("t155_c128x128", 128, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 155, 70, 62, 4, 8),  # key 68x60
    ("t155_c128x48", 128, 48, (3, 3, 3), (1, 1, 1), (0, 0, 0), 155, 70, 62, 4, 8),  # key 68x60
    ("t79_c512x512", 512, 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), 79, 36, 32, 4, 8),  # key 34x30
    ("t155_c256x512", 256, 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), 155, 36, 32, 4, 8),  # key 34x30
    ("t155_c256x256", 256, 256, (3, 3, 3), (1, 1, 1), (0, 0, 0), 155, 36, 32, 4, 8),  # key 34x30
    # --- mid volume ---
    ("t41_c512x4096", 512, 4096, (3, 3, 3), (1, 1, 1), (0, 0, 0), 41, 19, 17, 4, 8),  # key 17x15
    ("t41_c512x512", 512, 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), 41, 19, 17, 4, 8),  # key 17x15
    # --- the latent upsampler (separate 0.20s -> 0.62s regression) ---
    ("t20_ups_c1024x4096", 1024, 4096, (1, 3, 3), (1, 1, 1), (0, 0, 0), 20, 7, 6, 4, 8),  # key 5x4
    # --- small: T=22 sites ---
    ("t22_c1024x1024_h10", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 12, 10, 4, 8),  # key 10x8
    ("t22_c1024x128_h10", 1024, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 12, 10, 4, 8),  # key 10x8
    ("t22_c1024x4096_h9", 1024, 4096, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 11, 10, 4, 8),  # key 9x8
    ("t22_c1024x1024_h9", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 11, 10, 4, 8),  # key 9x8
    ("t22_c1024x1024_h5", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 7, 6, 4, 8),  # key 5x4
    ("t22_c128x1024_h9", 128, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 11, 10, 4, 8),  # key 9x8
    ("t22_c128x1024_h5", 128, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 7, 6, 4, 8),  # key 5x4
]


def _hw_product(kernel, h_in: int, w_in: int) -> int:
    """Largest sweep-supported H_blk*W_blk (32, matching wan2_2 h4w8; else 16) achievable within the OUTPUT dims.

    run_sweep's filter is an exact match, so an unreachable product sweeps zero combos; 5x4-output sites cap at 16.
    """
    h_out, w_out = h_in - (kernel[1] - 1), w_in - (kernel[2] - 1)
    for product in (32, 16):
        if any(product % h == 0 and h <= h_out and product // h <= w_out for h in range(1, product + 1)):
            return product
    msg = f"no achievable hw_product for output {h_out}x{w_out}"
    raise ValueError(msg)


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(1, 1), (1, 1), {"trace_region_size": TRACE_REGION_SIZE}]],
    ids=["bh_ltx_1080p_153f_1x1"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor",
    _SWEEP_LAYERS_LTX_1080P_153F,
    ids=[l[0] for l in _SWEEP_LAYERS_LTX_1080P_153F],
)
def test_bruteforce_sweep_ltx_1080p_153f(
    mesh_device, mesh_shape, layer_name, C_in, C_out, kernel, stride, padding, T, H, W, h_factor, w_factor
):
    parent_mesh = mesh_device
    device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    output = f"sweep_results_ltx_1080p_153f/{layer_name}_{C_in}x{C_out}.json"
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
        # Never None: unconstrained, the search balloons past 500 combos with minutes-long compiles.
        hw_product=_hw_product(kernel, H, W),
    )
