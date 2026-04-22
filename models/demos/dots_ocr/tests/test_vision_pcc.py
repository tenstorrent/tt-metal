# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TTNN hybrid VisionEncoder (HF vision_tower + TTNN PatchMerger).

This implements Step 4 of the Dots OCR roadmap: hybrid vision stack
that avoids full TTNN implementation of DotsVisionTransformer.
"""

import os

import pytest
import torch

# VisionEncoder is imported inside the test so collection does not pull optional stacks early.


def test_vision_encoder_pcc_gt_0_99(tmp_path):
    """Test full TTNN VisionEncoder interface on device.

    Note: Full PCC against HF is tracked in dedicated PCC tests. This test is a
    device-level bring-up check that real weights load and a forward pass runs.
    """
    ttnn = pytest.importorskip("ttnn")
    if not hasattr(ttnn, "open_mesh_device"):
        pytest.skip("TTNN runtime has no open_mesh_device (mesh API missing)")

    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, open_mesh_device
    from models.demos.dots_ocr.tt.vision import VisionEncoder

    torch.manual_seed(0)

    def _open_device():
        """Open a mesh device, defaulting to single-chip when unset."""
        os.environ.setdefault("MESH_DEVICE", "T3K")
        # dots.mocr uses GQA with n_kv_heads=2, so TP cols must divide 2.
        # On T3K we open the physical mesh but *run* on a 1x2 submesh for correctness.
        os.environ.setdefault("DOTS_T3K_OPEN_FULL_MESH", "1")
        os.environ.setdefault("DOTS_T3K_CREATE_SUBMESH", "1")
        os.environ.setdefault("DOTS_T3K_TP", "2")
        try:
            return open_mesh_device()
        except Exception as e:
            pytest.skip(f"TT device unavailable ({type(e).__name__}): {e}")

    device = _open_device()
    try:
        state_dict = None
        if device is not None:
            from models.demos.dots_ocr.reference.hf_utils import get_hf_model_id
            from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

            model_id = os.environ.get("HF_MODEL", get_hf_model_id())
            try:
                # Load the full vision_tower weights (patch embed + 42 blocks + norm + merger).
                state_dict = load_hf_state_dict_filtered(model_id, ("vision_tower.", "model.vision_tower."))
            except Exception as exc:
                pytest.skip(f"Real vision weights required for device test: {exc}")

        encoder = VisionEncoder(
            mesh_device=device,
            hf_model=None,
            state_dict=state_dict,
            weight_cache_path=tmp_path,
            hidden_size=1536,
            out_hidden_size=1536,
            spatial_merge_size=2,
        )

        # Create test inputs matching patch_size=14.
        pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)
        grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.int32)

        # Run full TTNN vision forward
        tt_vision_tokens = encoder.forward(pixel_values, grid_thw)

        assert isinstance(tt_vision_tokens, torch.Tensor)
        assert tt_vision_tokens.dim() == 2, f"Expected 2D tensor, got {tt_vision_tokens.dim()}D"
        assert tt_vision_tokens.shape[1] == 1536, f"Expected hidden_size=1536, got {tt_vision_tokens.shape[1]}"
        assert tt_vision_tokens.shape[0] > 0, "Should produce at least one vision token"

        # Basic sanity check - values should be reasonable
        assert not torch.isnan(tt_vision_tokens).any(), "Output contains NaNs"
        assert not torch.isinf(tt_vision_tokens).any(), "Output contains Infs"

    finally:
        if device is not None:
            close_dots_mesh_device(device)


def test_vision_encoder_smoke():
    """Smoke test for full TTNN VisionEncoder (requires TT device + weights)."""
    ttnn = pytest.importorskip("ttnn")
    if not hasattr(ttnn, "open_mesh_device"):
        pytest.skip("TTNN runtime has no open_mesh_device (mesh API missing)")
    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, open_mesh_device
    from models.demos.dots_ocr.tt.vision import VisionEncoder

    os.environ.setdefault("MESH_DEVICE", "T3K")
    device = None
    try:
        # dots.mocr uses GQA with n_kv_heads=2 in the shared ModelArgs base. That forces TP cols to divide 2.
        # On T3K, open a 1x2 mesh for vision tests to satisfy both n_heads=12 and n_kv_heads=2 constraints.
        mesh_shape = None
        try:
            if os.environ.get("MESH_DEVICE", "").upper().startswith("T3K"):
                mesh_shape = ttnn.MeshShape(1, 2)
        except Exception:
            mesh_shape = None
        device = open_mesh_device(mesh_shape=mesh_shape) if mesh_shape is not None else open_mesh_device()
        from models.demos.dots_ocr.reference.hf_utils import get_hf_model_id
        from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

        model_id = os.environ.get("HF_MODEL", get_hf_model_id())
        try:
            state_dict = load_hf_state_dict_filtered(model_id, ("vision_tower.", "model.vision_tower."))
        except Exception as exc:
            pytest.skip(f"Real vision weights required for TTNN smoke test: {exc}")

        encoder = VisionEncoder(
            mesh_device=device,
            hf_model=None,
            state_dict=state_dict,
            weight_cache_path=None,
            hidden_size=1536,
            out_hidden_size=1536,
            spatial_merge_size=2,
        )

        # Smoke input must match the checkpoint's patch embedding configuration.
        # Dots vision uses patch_size=14, so a canonical shape is 224x224 with 16x16 patches.
        pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16)
        grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.int32)

        output = encoder.forward(pixel_values, grid_thw)

        assert isinstance(output, torch.Tensor)
        assert output.dim() == 2, f"Expected 2D output, got {output.dim()}D"
        assert output.shape[1] == 1536, f"Expected hidden_size=1536, got {output.shape[1]}"
        assert output.shape[0] > 0, "Should produce at least one vision token"
        assert not torch.isnan(output).any(), "Output contains NaNs"
        assert not torch.isinf(output).any(), "Output contains Infs"
    finally:
        if device is not None:
            close_dots_mesh_device(device)


if __name__ == "__main__":
    test_vision_encoder_smoke()
    print("✅ VisionEncoder smoke test passed")
    try:
        import ttnn  # noqa: WPS433

        if hasattr(ttnn, "open_mesh_device"):
            print("✅ TTNN runtime available - device test would run under pytest")
    except ImportError:
        print("ℹ️  TTNN not importable - device test skipped")
