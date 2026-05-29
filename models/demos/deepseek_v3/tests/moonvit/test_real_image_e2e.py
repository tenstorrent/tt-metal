# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
End-to-end test against HF processor output.

Every other MoonViT test feeds synthetic `(L, 3, 14, 14)` random
patches directly. That doesn't exercise the HF `KimiVLImageProcessor`
path — PIL image → resize/normalize → patchify → `pixel_values` +
`image_grid_hws`. This test closes that gap:

  1. Build an image (synthetic programmatic, or a real skimage photo).
  2. Run the HF image processor on it to get `(L, 3, 14, 14)` patches
     plus `(N, 2)` grid metadata.
  3. Feed those into the HF MoonViT + projector reference.
  4. Feed the same into our tt-metal `MoonViT`.
  5. PCC compare at ≥ 0.95 (plus a logged relative-L2 error).

Two test entrypoints:
  - `test_real_image_through_hf_processor`: synthetic input at 224×224.
  - `test_real_photo_high_res`: a genuine photograph (skimage `cat`) at
    448×448 and 896×896 (up to a 64×64 / 4096-patch grid), exercising the
    pipeline on natural image statistics at the largest single-image sizes.

Catches bugs the synthetic-patch tests can't:
  - RGB ordering mismatches (HF uses ImageNet-style mean/std normalization
    on RGB-ordered channels).
  - Image-grid metadata convention drift between processor and model.
  - Any unexpected dtype/layout change in the processor output across
    HF revisions.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_real_image_e2e.py -v
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


def _make_synthetic_image(width: int, height: int, seed: int = 0) -> Image.Image:
    """Programmatic PIL image — no external download.

    Uses a deterministic pseudo-random RGB array. Random rather than a
    smooth gradient because (a) we want non-degenerate inputs across the
    full dynamic range, (b) it's more representative of real photo
    content than a flat color, and (c) it's still reproducible via seed.
    """
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _run_processor_e2e(mesh_device, model_args, img, pcc_threshold, tag):
    """Shared body: image → HF processor → (HF reference vs our MoonViT) PCC.

    Compares our tt-metal vision tower + projector against the HF torch
    reference (fp32) on the SAME processor output, and asserts PCC. Also
    logs a tensor-level relative-L2 error (= ‖tt−ref‖/‖ref‖), which is
    scale-aware and complements PCC's scale-invariance.
    """
    # HF processor — produces `(L, 3, 14, 14)` and `(N, 2)` grid metadata.
    proc_out = model_args.hf_processor.image_processor(images=[img], return_tensors="pt")
    pixel_values = proc_out["pixel_values"]  # (L, 3, 14, 14)
    grid_hws = proc_out["image_grid_hws"]  # (1, 2)
    L_patches = pixel_values.shape[0]
    L_new = sum((int(h) // 2) * (int(w) // 2) for h, w in grid_hws.tolist())
    logger.info(f"[{tag}] processor: pixel_values={tuple(pixel_values.shape)} grid={grid_hws.tolist()} L_new={L_new}")

    # HF reference: vision_tower + projector in fp32 for clean comparison.
    vt = _vision_tower(model_args)
    for blk in vt.encoder.blocks:
        blk.attn_implementation = "sdpa"
    vt_fp32 = vt.float()
    proj_fp32 = model_args.reference_projector().float()
    ref_out = proj_fp32(vt_fp32(pixel_values.float(), grid_hws))
    text_hidden = proj_fp32.linear_2.out_features
    assert ref_out.shape == (L_new, text_hidden), f"HF ref shape {tuple(ref_out.shape)} != ({L_new}, {text_hidden})"

    # tt-metal MoonViT with projector — fed the SAME processor output.
    tt_tower = MoonViT.from_torch(
        model_args=model_args,
        mesh_device=mesh_device,
        with_projector=True,
        dtype=ttnn.bfloat16,
    )
    out_tt = tt_tower(pixel_values, grid_hws)

    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    out_pt = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None)
    if is_mesh and out_pt.shape[0] != 1:
        out_pt = out_pt[:1]
    out_pt = out_pt.view(L_new, text_hidden)

    passing, pcc_msg = comp_pcc(ref_out, out_pt, pcc_threshold)
    rel_l2 = (torch.linalg.norm(out_pt.float() - ref_out.float()) / torch.linalg.norm(ref_out.float())).item()
    logger.info(
        f"[{tag} L_patches={L_patches} L_new={L_new}] {comp_allclose(ref_out, out_pt)} {pcc_msg} relL2={rel_l2:.4f}"
    )
    assert passing, f"PCC below threshold for {tag}: {pcc_msg}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("image_size", [224])
def test_real_image_through_hf_processor(mesh_device, model_args, image_size):
    """HF processor → our MoonViT matches HF MoonViT + projector at PCC ≥ 0.95 (synthetic input)."""
    img = _make_synthetic_image(width=image_size, height=image_size, seed=42)
    _run_processor_e2e(mesh_device, model_args, img, pcc_threshold=0.95, tag=f"synthetic-{image_size}")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("image_size", [448, 896])
def test_real_photo_high_res(mesh_device, model_args, image_size):
    """Genuine photograph through the processor at higher grids (up to 64×64 / 4096 patches).

    Uses a real photo (skimage's `cat`) rather than random noise, exercising the
    full pipeline on natural image statistics at the largest single-image grids.
    Measured PCC at 896×896 ≈ 0.96 (real) — the fp32 reference forward over 4096
    tokens runs in ~20 s, well within the test budget.
    """
    skd = pytest.importorskip("skimage.data", reason="scikit-image needed for real-photo E2E")
    photo = Image.fromarray(skd.cat()).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    _run_processor_e2e(mesh_device, model_args, photo, pcc_threshold=0.95, tag=f"real-cat-{image_size}")
