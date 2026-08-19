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
    # H and W here are the PADDED CONV3D INPUT dims, not the _BLOCKINGS key dims.
    # run_sweep derives the key as H_out_key = H - (kH - 1) (see wan2_2/
    # bruteforce_conv3d_sweep.py, "Seed best_us from the production blocking table"), so a
    # production key of (155, 68, 60) with a (3,3,3) kernel must be swept with input
    # (155, 70, 62). Passing the key dims directly benchmarks a smaller tensor AND looks up a
    # key that does not exist, silently falling back to the channel-level table.
    #
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
    """Largest supported (H_out_block * W_out_block) product that is actually achievable.

    ``run_sweep`` filters candidates with ``h * w != hw_product`` -- an exact match -- so a
    value no divisor pair can satisfy yields zero valid combos and the layer sweeps to
    nothing. Blocks are bounded by the OUTPUT dims, so reachability is tested against
    ``h_in - (kH - 1)``.

    32 mirrors the wan2_2 h4w8 sweeps (non-32 products hung BH on their shapes). Where 32 is
    unreachable -- the T=20 upsampler and the T=22 H=5/W=4 sites, whose 5x4 outputs cap the
    product at 20 -- fall back to 16, satisfiable as 4x4.
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
        # Never pass None here: unconstrained, the search balloons past 500 candidates and
        # some blockings take minutes each to compile (measured: t22_c1024x1024_h5 advanced
        # fewer than 10 combos in 27 min at 90% host CPU before being abandoned).
        hw_product=_hw_product(kernel, H, W),
    )
