# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""`unpatchify_tiled` (one page-remap program on the tiled fp32 tokens) against today's `to_layout` +
`unpatchify_device` chain, one chip, at the decoder's tile shape: bit-identical, and both paths timed."""

import time

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.minimax_h3.decoder_minimax_h3 import unpatchify
from ....models.vae.minimax_h3.stitch_device_minimax_h3 import unpatchify_device
from ....models.vae.minimax_h3.unpatchify_minimax_h3 import unpatchify_tiled

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
FRAMES, HEIGHT, WIDTH, SEQ = 7, 16, 16, 1824  # 1792 patches + 32 suffix rows, D = 3*4*16*16


def _best(fn, mesh_device, n=10):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - t0)
    return best, out


@pytest.mark.timeout(900)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_unpatchify_tiled_matches_permute(mesh_device):
    torch.manual_seed(0)
    dims = dict(num_frames=FRAMES, height=HEIGHT, width=WIDTH, out_channels=3, patch_size=16, patch_size_t=4)
    for tag in ("a", "b"):
        x = torch.randn(1, SEQ, 3072).bfloat16()
        # The decoder emits bf16 tiles; the pipeline casts to fp32 while still TILE.
        x_dev = ttnn.typecast(ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device), ttnn.float32)
        t_chain, ref = _best(lambda: unpatchify_device(ttnn.to_layout(x_dev, ttnn.ROW_MAJOR_LAYOUT), **dims), mesh_device)
        t_gather, got = _best(lambda: unpatchify_tiled(x_dev, **dims), mesh_device)
        ref_t = ttnn.to_torch(ref).float()
        got_t = ttnn.to_torch(got).float()
        cpu = unpatchify(x.float(), **dims)
        n_diff = int((got_t != ref_t).sum())
        logger.info(
            f"UNPATCHIFY input={tag}: chain {t_chain * 1e3:.3f} ms, gather {t_gather * 1e3:.3f} ms "
            f"({t_chain / max(t_gather, 1e-9):.1f}x); differing values {n_diff} / {ref_t.numel()}; "
            f"gather==cpu {torch.equal(got_t, cpu)}"
        )
        assert tuple(got_t.shape) == (1, 3, 28, 256, 256)
        assert torch.equal(got_t, ref_t), f"{n_diff} values differ from the permute path"
        assert torch.equal(got_t, cpu)
