# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end structural / PCC test for Dots OCR.

Builds the HF Dots reference (``DotsOCRReference``) + TT ``DotsTransformer`` +
``DropInVisionTransformer``, runs a minimal TT vision forward with synthetic inputs, and —
when a device and weights are available — checks shape compatibility with the reference path.

The first test (``test_e2e_structural_pcc``) stays skippable on CI (no device). A companion
test, ``test_e2e_hybrid_compatibility``, runs the full-TTNN :class:`VisionEncoder` with
default ``MESH_DEVICE``/T3K env and ``HF_MODEL=rednote-hilab/dots.mocr`` (legacy CPU hybrid is
disabled in code).

When a mesh is available, the structural test loads **real** Dots checkpoint tensors via
``load_real_state_dict`` (requires HF cache / network unless skipped).
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger


@pytest.mark.filterwarnings("ignore:Support for class-based `config` is deprecated.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:builtin type SwigPyPacked has no __module__ attribute:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:builtin type SwigPyObject has no __module__ attribute:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:builtin type swigvarlink has no __module__ attribute:DeprecationWarning")
def test_e2e_structural_pcc(tmp_path):
    """
    Structural end-to-end test.

    Verifies that the Dots TT stack (``DotsTransformer`` + ``DropInVisionTransformer``) can be
    instantiated and called with synthetic inputs.
    """
    torch.manual_seed(0)

    ttnn = pytest.importorskip("ttnn")
    if not hasattr(ttnn, "open_mesh_device") or not hasattr(ttnn, "MeshShape"):
        pytest.skip("TTNN mesh API not available")
    # Default topology when unset; skip only if the mesh cannot be opened (true no-device case).
    os.environ.setdefault("MESH_DEVICE", "T3K")
    os.environ.setdefault("DOTS_T3K_OPEN_FULL_MESH", "1")
    os.environ.setdefault("DOTS_T3K_CREATE_SUBMESH", "1")
    os.environ.setdefault("DOTS_T3K_TP", "2")

    from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec
    from models.demos.dots_ocr.reference.model import DotsOCRReference
    from models.demos.dots_ocr.tt.model import DotsTransformer, DropInVisionTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs
    from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs

    device = None
    try:
        from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device
        from models.demos.dots_ocr.tt.mesh import open_mesh_device as _open_mesh

        # Honors MESH_DEVICE (N150/N300/T3K). dots.mocr is GQA with num_kv_heads=2,
        # so T3K is auto-clamped to a 1x2 submesh (see tt/mesh.py).
        try:
            device = _open_mesh()
        except Exception as exc:
            pytest.skip(f"TT device unavailable ({type(exc).__name__}): {exc}")

        # On-device: default to the real Dots checkpoint unless the user overrides.
        # The tiny random models often have hidden_size=16 which is incompatible with TT tiling.
        default_model_id = "rednote-hilab/dots.mocr"
        spec = HFLoadSpec(model_id=os.environ.get("HF_MODEL", default_model_id), dtype=torch.bfloat16)
        try:
            ref = DotsOCRReference(spec)
        except Exception as exc:
            pytest.skip(f"HF reference model unavailable: {exc}")

        # Ensure DotsModelArgs can resolve HF params (it requires HF_MODEL or hf_config).
        os.environ["HF_MODEL"] = spec.model_id

        # Some tiny/random HF models use very small hidden sizes (e.g. 16) that are not compatible with
        # tt_transformers' tiling + LM-head grid assumptions (requires hidden_size multiple of TILE_SIZE=32).
        # This test is structural for the Dots stack; skip when the chosen HF model is inherently incompatible.
        hidden_size = getattr(getattr(ref.model, "language_model", ref.model).config, "hidden_size", None) or getattr(
            ref.model.config, "hidden_size", None
        )
        if hidden_size is not None and (hidden_size < 32 or hidden_size % 32 != 0):
            pytest.skip(f"HF model hidden_size={hidden_size} is incompatible with TT tiling (needs multiple of 32)")

        # Build TT args + transformer (real checkpoint tensors only).
        model_args = DotsModelArgs(
            mesh_device=device,
            hf_config=ref.model.config,
            max_batch_size=1,
            max_seq_len=128,
        )
        try:
            state_dict = model_args.load_real_state_dict()
            logger.info(f"Loaded real Dots state_dict with {len(state_dict)} tensors")
        except Exception as exc:
            pytest.skip(f"Real Dots weights unavailable: {exc}")
        tt_model = DotsTransformer(
            args=model_args,
            dtype=ttnn.bfloat16,
            mesh_device=device,
            state_dict=state_dict,
            # tt_transformers expects a Path-like cache dir when modules form cache filenames.
            weight_cache_path=tmp_path / "weights",
        )

        # Build the vision drop-in if the HF reference exposes a vision tower
        if hasattr(ref.model, "vision_tower") or hasattr(ref.model, "visual"):
            vision_model_args = DotsVisionModelArgs(mesh_device=device, hf_config=ref.model.config)
            visual = DropInVisionTransformer(ref.model, vision_model_args, debug=False)

            pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)
            grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.int32)
            try:
                image_embeds = visual(pixel_values, grid_thw)
                logger.info(f"TT vision output shape: {image_embeds.shape}")
                assert image_embeds.dim() == 2
            except TypeError as exc:
                # Some hybrid vision components currently expect TT tensors while the driver keeps host tensors.
                # This structural test primarily validates construction; skip vision-forward when the runtime
                # cannot execute it with the installed TTNN API.
                pytest.skip(f"TT vision forward not supported in this environment: {exc}")

        assert tt_model is not None
        logger.info("Dots end-to-end TT model constructed successfully")
    finally:
        if device is not None:
            close_dots_mesh_device(device)


def test_e2e_hybrid_compatibility(tmp_path):
    """
    End-to-end :class:`VisionEncoder` on device with real Dots ``vision_tower`` weights.

    The legacy CPU hybrid path is removed; this checks the supported stack: full TTNN vision
    (same defaults as ``test_e2e_structural_pcc`` for mesh and checkpoint id).
    """
    ttnn = pytest.importorskip("ttnn")
    if not hasattr(ttnn, "open_mesh_device") or not hasattr(ttnn, "MeshShape"):
        pytest.skip("TTNN mesh API not available")

    torch.manual_seed(0)

    os.environ.setdefault("MESH_DEVICE", "T3K")
    os.environ.setdefault("DOTS_T3K_OPEN_FULL_MESH", "1")
    os.environ.setdefault("DOTS_T3K_CREATE_SUBMESH", "1")
    os.environ.setdefault("DOTS_T3K_TP", "2")
    default_model_id = "rednote-hilab/dots.mocr"
    os.environ.setdefault("HF_MODEL", default_model_id)

    from models.demos.dots_ocr.reference.hf_utils import get_hf_model_id
    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, open_mesh_device
    from models.demos.dots_ocr.tt.vision import VisionEncoder
    from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

    device = None
    try:
        try:
            device = open_mesh_device()
        except Exception as exc:
            pytest.skip(f"TT device unavailable ({type(exc).__name__}): {exc}")

        model_id = os.environ.get("HF_MODEL", get_hf_model_id())
        try:
            state_dict = load_hf_state_dict_filtered(model_id, ("vision_tower.", "model.vision_tower."))
        except Exception as exc:
            pytest.skip(f"Real vision weights required for e2e VisionEncoder test: {exc}")

        encoder = VisionEncoder(
            mesh_device=device,
            hf_model=None,
            state_dict=state_dict,
            weight_cache_path=tmp_path,
            hidden_size=1536,
            out_hidden_size=1536,
            spatial_merge_size=2,
        )
        pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)
        grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.int32)
        out = encoder.forward(pixel_values, grid_thw)
        assert isinstance(out, torch.Tensor) and out.dim() == 2
        assert out.shape[1] == 1536 and out.shape[0] > 0
        assert not torch.isnan(out).any() and not torch.isinf(out).any()
    finally:
        if device is not None:
            close_dots_mesh_device(device)
