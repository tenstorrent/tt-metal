# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Multi-resolution end-to-end coverage.

`test_real_image_e2e.py` only validates 224x224. MoonViT's signature
feature is native-resolution (NaViT-style) input — variable (H, W)
grids in one packed sequence. This test exercises:

  - Multiple resolutions (224, 336, 448, asymmetric 336x224).
  - The variable-length attention path at each grid size.
  - The 2D RoPE precompute at each grid size.
  - The bicubic posemb interpolation at each grid size.
  - The (H, W)-keyed cache: first call to a shape is a miss, the
    second is a hit. Verified via the on-instance counters.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_multi_resolution.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit._references import _vision_tower
from models.demos.deepseek_v3.tt.moonvit.model import MoonViT


def _make_synthetic_image(height: int, width: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_h, image_w, expected_grid",
    [
        (224, 224, (16, 16)),  # baseline (square, smallest typical)
        (336, 336, (24, 24)),  # 1.5x baseline
        (448, 448, (32, 32)),  # 2x baseline — largest realistic single-image case
        (336, 224, (24, 16)),  # asymmetric — exercises non-square 2D RoPE / posemb
    ],
)
def test_moonvit_at_resolution(mesh_device, model_args, image_h, image_w, expected_grid):
    """PCC + cache integration at the given resolution."""
    # Uniform random-noise images (a high-frequency stress input). With the
    # accurate MLP GELU (see mlp.py — a per-op bisection found the fast-approx
    # GELU was the dominant bf16 error), TT-vs-fp32 PCC is ~0.977–0.990 across
    # these resolutions; natural images (test_real_image_e2e) sit even higher.
    # 0.95 keeps margin above that floor while a real regression drops PCC
    # well below 0.9.
    pcc_threshold = 0.95

    # 1. Synthesize image, run HF processor.
    img = _make_synthetic_image(image_h, image_w, seed=image_h * 31 + image_w)
    proc_out = model_args.hf_processor.image_processor(images=[img], return_tensors="pt")
    pixel_values = proc_out["pixel_values"]
    grid_hws = proc_out["image_grid_hws"]
    grid_tuple = tuple(grid_hws[0].tolist())
    assert grid_tuple == expected_grid, (
        f"processor produced grid {grid_tuple} but expected {expected_grid} " f"for input {image_h}x{image_w}"
    )

    L_new = sum((int(h) // 2) * (int(w) // 2) for h, w in grid_hws.tolist())

    # 2. HF reference (fp32) — vision_tower + projector.
    vt = _vision_tower(model_args)
    for blk in vt.encoder.blocks:
        blk.attn_implementation = "sdpa"
    vt_fp32 = vt.float()
    proj_fp32 = model_args.reference_projector().float()
    ref_merged_list = vt_fp32(pixel_values.float(), grid_hws)
    ref_out = proj_fp32(ref_merged_list)
    text_hidden = proj_fp32.linear_2.out_features
    assert ref_out.shape == (L_new, text_hidden)

    # 3. Build our MoonViT (the posemb cache lives on this instance).
    tt_tower = MoonViT.from_torch(
        model_args=model_args,
        mesh_device=mesh_device,
        with_projector=True,
        dtype=ttnn.bfloat16,
    )
    pos_emb_cache = tt_tower.patch_embed.pos_emb
    # Sanity: cache starts empty.
    assert pos_emb_cache.cache_hits == 0
    assert pos_emb_cache.cache_misses == 0

    # 4. First forward — should be exactly 1 cache miss.
    out_tt = tt_tower(pixel_values, grid_hws)
    assert (
        pos_emb_cache.cache_misses == 1
    ), f"expected 1 cache miss after first forward, got {pos_emb_cache.cache_misses}"
    assert pos_emb_cache.cache_hits == 0, f"expected 0 cache hits after first forward, got {pos_emb_cache.cache_hits}"

    # 5. PCC check on the first output.
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    out_pt = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and out_pt.shape[0] != 1:
        out_pt = out_pt[:1]
    out_pt = out_pt.view(L_new, text_hidden)
    passing, pcc_msg = comp_pcc(ref_out, out_pt, pcc_threshold)
    logger.info(
        f"[{image_h}x{image_w} -> grid {grid_tuple} L_new={L_new}] " f"{comp_allclose(ref_out, out_pt)} {pcc_msg}"
    )
    assert passing, f"PCC below threshold for {image_h}x{image_w}: {pcc_msg}"

    # 6. Second forward on the SAME image — must be a cache hit.
    _ = tt_tower(pixel_values, grid_hws)
    assert pos_emb_cache.cache_misses == 1, (
        f"second forward should not have added a miss; got total misses " f"{pos_emb_cache.cache_misses}"
    )
    assert pos_emb_cache.cache_hits == 1, f"second forward should be a cache hit; got hits {pos_emb_cache.cache_hits}"
