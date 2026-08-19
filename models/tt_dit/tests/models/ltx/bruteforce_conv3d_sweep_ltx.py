#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Conv3d blocking sweep for the LTX-2.3 1080p / 153-frame (25 fps) shape.

Reuses the generic brute-force harness in ``..wan2_2.bruteforce_conv3d_sweep``; only the
layer list is LTX-specific.

WHY THIS EXISTS
---------------
``_BLOCKINGS`` in ``models/tt_dit/utils/conv3d.py`` is keyed on
``(h_factor, w_factor, C_in, C_out, kernel, T, H, W)``. It is tuned for the 145-frame
production shape. Moving to 153 frames (6.12 s @ 25 fps, the Console target) shifts the
decoder's latent T from 19 to 20, which walks a different T-chain (T -> 2T-1 per
depth-to-space stage) and misses the table at 15 sites. Those convs fall back to
channel-only blocking, which costs:

    VAE decode      1s   -> 5s
    Latent upsample 0.20s -> 0.62s
    total generate  6.7s -> 12.1s     (measured 2026-08-19)

The traced denoise is unaffected and 25 fps itself is free (145f@25 measured 6.6s vs
145f@24's 6.7s), so this table is the *entire* gap to serving 6.12 s at full speed.

LAYERS ARE ORDERED BY COMPUTE VOLUME (T*H*W*C_in). The first five are 94% of the total —
sweep those first if time is short.

The exact keys were harvested from the ``conv3d blocking [fallback|NONE]`` warnings that
``get_conv3d_config`` (conv3d.py:649/655) emits on a real 153-frame run, diffed against a
145-frame run so only genuinely NEW misses are listed.

RUN
---
Needs the Galaxy (stop the media server first). The sweep itself runs on a (1,1) submesh --
one conv layer on one chip -- so it does not need the full mesh, just exclusive access.

    pytest models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py -k t155_c128x128 -s --timeout=0
    pytest models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py -s --timeout=0   # all 15

Each layer writes ``sweep_results_ltx_1080p_153f/<name>_<Cin>x<Cout>.json`` containing
``best_blocking`` (the winner) and ``table_blocking``/``table_us`` (what the table would
have used), so the speedup is visible per layer.

THEN
----
Add each winner to ``_BLOCKINGS`` as
``(4, 8, C_in, C_out, kernel, T, H, W): tuple(best_blocking)``
and re-run the 153f validation (``~/ltx-25fps-validate.sh``) to confirm ~7 s.
"""

import pytest

import ttnn

from ..wan2_2.bruteforce_conv3d_sweep import TRACE_REGION_SIZE, run_sweep

# ---------------------------------------------------------------------------
# LTX-2.3, 1920x1088, 153 frames (25 fps), BH Galaxy 4x8 (h_factor=4, w_factor=8).
#
# h_factor comes from tensor_parallel (mesh axis 0, factor 4) and w_factor from
# sequence_parallel (mesh axis 1, factor 8) -- see LTXPipeline's VaeHWParallelConfig.
#
# T here is the value the conv3d kernel sees (post temporal pad). The decoder chain for
# 153 frames is latent T=20 -> 22, 41, 79, 155 at the four (3,3,3) conv depths; the
# latent upsampler asks for T=20 with a (1,3,3) kernel.
#
# Names are descriptive of (T, channels) rather than guessing the module attribute names.
# ---------------------------------------------------------------------------
_SWEEP_LAYERS_LTX_1080P_153F = [
    # (name,                 C_in, C_out, kernel,    stride,    padding,    T,   H,  W, h, w)
    # --- 94% of total volume: the T=155 / T=79 decoder convs ---
    ("t155_c128x128", 128, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 155, 68, 60, 4, 8),
    ("t155_c128x48", 128, 48, (3, 3, 3), (1, 1, 1), (0, 0, 0), 155, 68, 60, 4, 8),
    ("t79_c512x512", 512, 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), 79, 34, 30, 4, 8),
    ("t155_c256x512", 256, 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), 155, 34, 30, 4, 8),
    ("t155_c256x256", 256, 256, (3, 3, 3), (1, 1, 1), (0, 0, 0), 155, 34, 30, 4, 8),
    # --- mid volume ---
    ("t41_c512x4096", 512, 4096, (3, 3, 3), (1, 1, 1), (0, 0, 0), 41, 17, 15, 4, 8),
    ("t41_c512x512", 512, 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), 41, 17, 15, 4, 8),
    # --- the latent upsampler (separate 0.20s -> 0.62s regression) ---
    ("t20_ups_c1024x4096", 1024, 4096, (1, 3, 3), (1, 1, 1), (0, 0, 0), 20, 5, 4, 4, 8),
    # --- small: T=22 sites ---
    ("t22_c1024x1024_h10", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 10, 8, 4, 8),
    ("t22_c1024x128_h10", 1024, 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 10, 8, 4, 8),
    ("t22_c1024x4096_h9", 1024, 4096, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 9, 8, 4, 8),
    ("t22_c1024x1024_h9", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 9, 8, 4, 8),
    ("t22_c1024x1024_h5", 1024, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 5, 4, 4, 8),
    ("t22_c128x1024_h9", 128, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 9, 8, 4, 8),
    ("t22_c128x1024_h5", 128, 1024, (3, 3, 3), (1, 1, 1), (0, 0, 0), 22, 5, 4, 4, 8),
]


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
        # hw_product=32 mirrors the wan2_2 h4w8 sweeps (non-32 hw products hung BH there), but
        # the filter is `h * w != hw_product` -- an EXACT match. Layers whose per-device spatial
        # dims cannot reach 32 (e.g. the T=20 upsampler and the T=22 H=5/W=4 sites: 5*4 = 20)
        # yield zero valid combos and sweep to nothing. Only constrain where it is satisfiable.
        # Exact-match filter on (H_out_block * W_out_block): 32 where reachable; where it is not
        # (H=5,W=4 -> max 20) use 16, satisfiable as 4x4. Do NOT pass None: unconstrained, the
        # search balloons past 500 candidates and some blockings take MINUTES to compile
        # (measured: t22_c1024x1024_h5 advanced <10 combos in 27 min at 90% host CPU).
        hw_product=32 if H * W >= 32 else 16,
    )
