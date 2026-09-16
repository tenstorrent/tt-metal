# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Equivalence + timing for ``WanDupUp3D``'s fast path.

A host op profile of the production 5B decode (1280x704, 81 f, ``t_chunk=7``) attributed
3.85 s of the 4.92 s device total to the eleven ``ttnn`` calls inside
``WanDupUp3D.forward`` -- 78% of the VAE decode. The op is a pure index permutation, so
``_forward_fast`` must agree with ``_forward_generic`` *exactly*, not merely to a PCC.

    pytest models/tt_dit/tests/models/wan2_2/test_dup_up3d_ti2v_5b.py -sv --timeout=0
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import WanDupUp3D

# (in_dim, out_dim, factor_t, T, H, W) at per-device production shapes.
# The 5B decoder is channel_dims=[1024,1024,1024,512,256], temperal_upsample=[T,T,F], so
# the three shortcut instances are up0/up1 (1024->1024, factor_t=2, exact NN because
# repeats == factor) and up2 (1024->512, factor_t=1, repeats == factor_s). Per-device
# latent is 11x10 (44/4, 80/8 on the 4x8 mesh), doubling up the stages.
_CASES = [
    ("up0_1024_1024_ft2", 1024, 1024, 2, 7, 11, 10),
    ("up1_1024_1024_ft2", 1024, 1024, 2, 7, 22, 20),
    ("up2_1024_512_ft1", 1024, 512, 1, 7, 44, 40),
    # First chunk of the production chunk_bounds is T=1, and the last is T=6.
    ("up0_t1", 1024, 1024, 2, 1, 11, 10),
    ("up2_t6", 1024, 512, 1, 6, 44, 40),
    # Not covered by the fast path (repeats=1 < factor_s): must fall back and still pass.
    ("fallback_repeats1", 1024, 128, 2, 2, 6, 6),
]


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["single"],
    indirect=True,
)
@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
@pytest.mark.parametrize("first_chunk", [False, True], ids=["mid_chunk", "first_chunk"])
def test_dup_up3d_fast_matches_generic(mesh_device, case, first_chunk):
    name, in_dim, out_dim, ft, T, H, W = case

    module = WanDupUp3D(in_dim, out_dim, factor_t=ft, factor_s=2)
    covered = module._fast_slice_stride is not None
    logger.info(
        f"{name}: in={in_dim} out={out_dim} ft={ft} fs=2 factor={module.factor} "
        f"repeats={module.repeats} fast_path={covered} (stride={module._fast_slice_stride})"
    )
    if name == "fallback_repeats1":
        assert not covered, "this case is meant to exercise the generic fallback"
    else:
        assert covered, f"{name} should take the fast path but did not"

    torch.manual_seed(0)
    x_torch = torch.randn(1, T, H, W, in_dim, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )

    def _run(fn, reps=3):
        out = fn(ttnn.clone(x_tt))
        ttnn.synchronize_device(mesh_device)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn(ttnn.clone(x_tt))
            ttnn.synchronize_device(mesh_device)
            times.append(time.perf_counter() - t0)
        return out, min(times)

    def _generic(t):
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        out = module._forward_generic(t)
        if first_chunk and ft > 1:
            out = out[:, ft - 1 :, :, :, :]
        return out

    out_gen, t_gen = _run(_generic)
    out_fast, t_fast = _run(lambda t: module.forward(t, first_chunk=first_chunk))

    g = ttnn.to_torch(out_gen).float()
    f = ttnn.to_torch(out_fast).float()
    speedup = t_gen / t_fast if t_fast else float("inf")
    logger.info(
        f"DUPUP3D {name} first_chunk={first_chunk}: shape {tuple(g.shape)} "
        f"generic={1e3 * t_gen:.2f}ms fast={1e3 * t_fast:.2f}ms speedup={speedup:.1f}x"
    )

    assert g.shape == f.shape, f"shape mismatch: generic {tuple(g.shape)} vs fast {tuple(f.shape)}"
    max_abs = (g - f).abs().max().item()
    assert max_abs == 0.0, f"{name}: fast path is not bit-exact vs generic, max_abs_diff={max_abs:.6e}"
