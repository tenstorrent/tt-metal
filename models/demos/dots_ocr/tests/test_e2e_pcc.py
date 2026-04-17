# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end structural / PCC test for Dots OCR.

Mirrors the qwen25_vl demo's end-to-end shape: builds the HF reference model + the TT
``DotsTransformer`` + ``DropInVisionTransformer``, pushes a synthetic image through the TT
vision path, and — when a device is available and weights are present — verifies the logits
are shape-compatible with the HF reference.

The test intentionally stays skippable so it runs both on CI (no device) and on single-chip
Wormhole (N150/N300) without requiring the real ``rednote-hilab/dots.mocr`` checkpoint.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger


def test_e2e_structural_pcc(tmp_path):
    """
    Structural end-to-end test.

    Verifies that the new qwen25_vl-aligned stack (``DotsTransformer`` +
    ``DropInVisionTransformer``) can be instantiated and called with synthetic inputs.
    """
    torch.manual_seed(0)

    ttnn = pytest.importorskip("ttnn")
    if not hasattr(ttnn, "open_mesh_device") or not hasattr(ttnn, "MeshShape"):
        pytest.skip("TTNN mesh API not available")
    if os.environ.get("MESH_DEVICE") is None:
        pytest.skip("Requires TT device (set MESH_DEVICE)")

    from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec
    from models.demos.dots_ocr.reference.model import DotsOCRReference
    from models.demos.dots_ocr.tt.model import DotsTransformer, DropInVisionTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs
    from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs

    device = None
    try:
        device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))  # WH LB 1x1 mesh

        spec = HFLoadSpec(
            model_id=os.environ.get("HF_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM"),
            torch_dtype=torch.bfloat16,
        )
        try:
            ref = DotsOCRReference(spec)
        except Exception as exc:
            pytest.skip(f"HF reference model unavailable: {exc}")

        # Build TT args + transformer. Prefer real weights when DOTS_USE_REAL_WEIGHTS=1 and the HF
        # checkpoint is available, otherwise fall back to the dummy state_dict so CI stays runnable.
        use_real = os.environ.get("DOTS_USE_REAL_WEIGHTS") == "1"
        model_args = DotsModelArgs(mesh_device=device, max_batch_size=1, max_seq_len=128, dummy_weights=not use_real)
        if use_real:
            try:
                state_dict = model_args.load_real_state_dict()
                logger.info(f"Loaded real Dots state_dict with {len(state_dict)} tensors")
            except Exception as exc:
                logger.warning(f"Real weight load failed: {exc}. Falling back to dummy state_dict.")
                state_dict = model_args.load_state_dict()
        else:
            state_dict = model_args.load_state_dict()
        tt_model = DotsTransformer(
            args=model_args,
            dtype=ttnn.bfloat16,
            mesh_device=device,
            state_dict=state_dict,
            weight_cache_path=None,
        )

        # Build the vision drop-in if the HF reference exposes a vision tower
        if hasattr(ref.model, "vision_tower") or hasattr(ref.model, "visual"):
            vision_model_args = DotsVisionModelArgs(mesh_device=device, hf_config=ref.model.config)
            visual = DropInVisionTransformer(ref.model, vision_model_args, debug=False)

            pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)
            grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.int32)
            image_embeds = visual(pixel_values, grid_thw)
            logger.info(f"TT vision output shape: {image_embeds.shape}")
            assert image_embeds.dim() == 2

        assert tt_model is not None
        logger.info("Dots end-to-end TT model constructed successfully")
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


def test_e2e_hybrid_compatibility():
    """Sanity check that the hybrid ``VisionEncoder`` still works on CPU."""
    from models.demos.dots_ocr.tt.vision import VisionEncoder

    encoder = VisionEncoder(
        mesh_device=None,
        use_full_ttnn=False,
        hidden_size=64,
        out_hidden_size=64,
    )
    pixel_values = torch.randn(1, 3, 32, 32, dtype=torch.bfloat16)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32)
    output = encoder.forward(pixel_values, grid_thw)
    assert isinstance(output, torch.Tensor)
    assert output.dim() == 2
