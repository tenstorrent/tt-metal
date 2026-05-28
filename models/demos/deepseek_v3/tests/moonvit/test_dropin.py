# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Tests for `DropInMoonViT` and `DropInKimiVLMultiModalProjector`.

These wrappers expose the tt-metal MoonViT vision tower / projector
with the EXACT same Python signatures as HF
`MoonVitPretrainedModel.forward` and `KimiVLMultiModalProjector.forward`.
That means a downstream HF pipeline (e.g.,
KimiVLForConditionalGeneration) can substitute our impl with a single
attribute swap — no code changes elsewhere.

We verify two things:
  (1) DropInMoonViT(image, grid_hws) returns a list[Tensor] matching
      HF MoonVitPretrainedModel(image, grid_hws) at PCC ≥ 0.95.
  (2) DropInKimiVLMultiModalProjector(image_features) matches HF
      KimiVLMultiModalProjector(image_features) at PCC ≥ 0.95.

Together (1)+(2) compose into the full vision pipeline a multimodal LLM
expects. Plus we explicitly test the composition by chaining the two
wrappers and comparing against HF's chain.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_dropin.py -v
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
from models.demos.deepseek_v3.tt.moonvit.model import DropInKimiVLMultiModalProjector, DropInMoonViT, MoonViT
from models.demos.deepseek_v3.tt.moonvit.projector import MoonViTProjector


def _make_image(size: int, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
def test_dropin_moonvit_matches_hf(mesh_device, model_args):
    """DropInMoonViT(...) returns same list format as HF MoonVitPretrainedModel."""
    pcc_threshold = 0.95

    # Two-image batch — exercises the per-image splitting in DropInMoonViT.
    img1 = _make_image(224, seed=1)
    img2 = _make_image(336, seed=2)
    proc_out = model_args.hf_processor.image_processor(images=[img1, img2], return_tensors="pt")
    pixel_values = proc_out["pixel_values"]
    grid_hws = proc_out["image_grid_hws"]
    logger.info(f"two-image batch: pixel_values={tuple(pixel_values.shape)} grid_hws={grid_hws.tolist()}")

    # HF reference list output.
    vt = _vision_tower(model_args)
    for blk in vt.encoder.blocks:
        blk.attn_implementation = "sdpa"
    vt_fp32 = vt.float()
    ref_list = vt_fp32(pixel_values.float(), grid_hws)
    assert isinstance(ref_list, list) and len(ref_list) == 2

    # Build a tt MoonViT WITHOUT projector and wrap it.
    tt_tower = MoonViT.from_torch(
        model_args=model_args,
        mesh_device=mesh_device,
        with_projector=False,
        dtype=ttnn.bfloat16,
    )
    dropin = DropInMoonViT(tt_tower)
    our_list = dropin(pixel_values, grid_hws)

    # Structural match.
    assert isinstance(our_list, list)
    assert len(our_list) == len(ref_list)
    for i, (r, o) in enumerate(zip(ref_list, our_list)):
        assert r.shape == o.shape, f"image {i}: shape mismatch {r.shape} vs {o.shape}"

    # Element-wise PCC.
    for i, (r, o) in enumerate(zip(ref_list, our_list)):
        passing, pcc_msg = comp_pcc(r, o, pcc_threshold)
        logger.info(f"  image {i}: shape {r.shape}: {comp_allclose(r, o)} {pcc_msg}")
        assert passing, f"image {i} DropInMoonViT PCC mismatch: {pcc_msg}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
def test_dropin_projector_matches_hf(mesh_device, model_args):
    """DropInKimiVLMultiModalProjector(features) matches HF projector."""
    pcc_threshold = 0.95

    # Synthesize a 2-image list of merged-format tensors (what patch_merger emits).
    merge_kh, merge_kw = model_args.merge_kernel_size
    merge_groups = merge_kh * merge_kw
    vision_hidden = model_args.hidden_size

    torch.manual_seed(0)
    image_features = [
        torch.randn(64, merge_groups, vision_hidden, dtype=torch.float32),  # 1st image
        torch.randn(144, merge_groups, vision_hidden, dtype=torch.float32),  # 2nd image
    ]

    # HF reference.
    ref_proj = model_args.reference_projector().float()
    ref_out = ref_proj(image_features)
    text_hidden = ref_proj.linear_2.out_features
    assert ref_out.shape == (64 + 144, text_hidden)

    # tt projector wrapped as drop-in.
    tt_proj = MoonViTProjector.from_torch(
        mesh_device=mesh_device, ref=model_args.reference_projector(), dtype=ttnn.bfloat16
    )
    dropin_proj = DropInKimiVLMultiModalProjector(tt_proj, mesh_device, dtype=ttnn.bfloat16)
    our_out = dropin_proj(image_features)

    assert our_out.shape == ref_out.shape
    passing, pcc_msg = comp_pcc(ref_out, our_out, pcc_threshold)
    logger.info(f"projector dropin: {comp_allclose(ref_out, our_out)} {pcc_msg}")
    assert passing, f"DropInKimiVLMultiModalProjector PCC: {pcc_msg}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
def test_dropin_chain_matches_hf(mesh_device, model_args):
    """Chained vision_tower + projector via drop-ins matches HF chain."""
    pcc_threshold = 0.95

    img = _make_image(224, seed=7)
    proc_out = model_args.hf_processor.image_processor(images=[img], return_tensors="pt")
    pixel_values = proc_out["pixel_values"]
    grid_hws = proc_out["image_grid_hws"]

    # HF reference chain.
    vt = _vision_tower(model_args)
    for blk in vt.encoder.blocks:
        blk.attn_implementation = "sdpa"
    vt_fp32 = vt.float()
    proj_fp32 = model_args.reference_projector().float()
    ref_merged = vt_fp32(pixel_values.float(), grid_hws)
    ref_out = proj_fp32(ref_merged)

    # Our drop-in chain.
    tt_tower = MoonViT.from_torch(
        model_args=model_args,
        mesh_device=mesh_device,
        with_projector=False,
        dtype=ttnn.bfloat16,
    )
    dropin_tower = DropInMoonViT(tt_tower)
    tt_proj = MoonViTProjector.from_torch(
        mesh_device=mesh_device, ref=model_args.reference_projector(), dtype=ttnn.bfloat16
    )
    dropin_proj = DropInKimiVLMultiModalProjector(tt_proj, mesh_device, dtype=ttnn.bfloat16)

    our_merged = dropin_tower(pixel_values, grid_hws)
    our_out = dropin_proj(our_merged)

    assert our_out.shape == ref_out.shape
    passing, pcc_msg = comp_pcc(ref_out, our_out, pcc_threshold)
    logger.info(f"dropin chain: {comp_allclose(ref_out, our_out)} {pcc_msg}")
    assert passing, f"chained DropIn vs HF: {pcc_msg}"


@torch.no_grad()
def test_dropin_moonvit_rejects_projector_model():
    """Constructing DropInMoonViT with a projector-bearing MoonViT must raise."""

    # Use a stub object with the minimal attributes the check needs.
    class _FakeMoonViT:
        def __init__(self):
            self.projector = object()  # non-None — should trigger the error
            self.device = None
            self.merge_kernel_size = (2, 2)

    with pytest.raises(ValueError, match="with_projector=False"):
        DropInMoonViT(_FakeMoonViT())
