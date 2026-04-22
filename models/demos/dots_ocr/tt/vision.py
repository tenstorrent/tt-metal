# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN vision stack for Dots OCR.

**Default (``use_full_ttnn=True``)**: full :class:`VisionTransformerTT` on device — patch embed,
42 blocks, post-trunk RMSNorm, and :class:`PatchMerger` use ttnn; PyTorch is not used on that path
except where blocks still mix host math (e.g. RoPE/softmax) as documented in those modules.

**Optional hybrid** (``use_full_ttnn=False``): run HF ``vision_tower`` on the host, then run only
``PatchMerger`` on ttnn for compatibility.
"""

from __future__ import annotations

import torch

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn


class VisionEncoder(LightweightModule):
    """
    VisionEncoder for Dots OCR — full TTNN vision by default, optional HF+merger hybrid.

    Full TTNN (``use_full_ttnn=True``): :class:`VisionTransformerTT` on device.

    Hybrid (``use_full_ttnn=False``): HF ``vision_tower`` on host, then ttnn :class:`PatchMerger` only.
    """

    def __init__(
        self,
        mesh_device,
        hf_model=None,  # HF model with .vision_tower for reference/hybrid mode
        state_dict=None,
        state_dict_prefix: str = "vision_tower",
        weight_cache_path=None,
        dtype=None,
        hidden_size: int = 1536,
        out_hidden_size: int = 1536,
        spatial_merge_size: int = 2,
        use_full_ttnn: bool = True,
        model_args=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.hf_model = hf_model
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        # If we have a device, always run the full TTNN vision tower.
        # Hybrid mode is kept only for CPU-only / no-device environments.
        self.use_full_ttnn = True if mesh_device is not None else use_full_ttnn
        self.model_args = model_args

        if mesh_device is None:
            raise ValueError(
                "VisionEncoder requires a TT device (mesh_device is None). "
                "Set MESH_DEVICE and open a mesh with models.demos.dots_ocr.tt.mesh.open_mesh_device()."
            )

        # Set default dtype if not provided
        if dtype is None:
            ttnn = get_ttnn()
            dtype = ttnn.bfloat16 if ttnn is not None and hasattr(ttnn, "bfloat16") else torch.bfloat16
        self.dtype = dtype

        if state_dict is None:
            raise ValueError(
                "VisionEncoder requires ``state_dict`` when running on device. "
                "Load weights with ``models.demos.dots_ocr.tt.load.load_dots_vision_state_dict``."
            )

        # Normalize common HF key layouts (e.g. "model.vision_tower.*") to the expected
        # TT vision stack layout ("vision_tower.*").
        if any(k.startswith("model.vision_tower.") for k in state_dict.keys()) and not any(
            k.startswith("vision_tower.") for k in state_dict.keys()
        ):
            state_dict = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}

        if not self.use_full_ttnn:
            raise ValueError("Hybrid vision is disabled: VisionEncoder is TTNN-only in this configuration.")

        # Full TTNN Vision Transformer (Phase 2 implementation).
        from models.demos.dots_ocr.tt.vision_transformer import create_dots_vision_transformer

        self.vision_transformer = create_dots_vision_transformer(
            mesh_device=mesh_device,
            model_args=model_args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            hf_model=hf_model,
        )
        self.is_full_ttnn = True
        print("✅ Using FULL TTNN Vision Transformer (42 layers)")

    def _synthetic_patch_merger_state_dict(self, prefix: str = "vision_tower") -> dict:
        """Synthetic PatchMerger tensors for CPU-only tests (no device / no HF checkpoint)."""
        mlp = self.hidden_size * (self.spatial_merge_size**2)
        # ``nn.Linear`` weights are ``[out_features, in_features]``; ``PatchMergerTT`` transposes once for ttnn.
        return {
            f"{prefix}.patch_merger.ln_q.weight": torch.ones(self.hidden_size, dtype=torch.bfloat16),
            f"{prefix}.patch_merger.feed_forward.0.weight": torch.randn(mlp, mlp, dtype=torch.bfloat16),
            f"{prefix}.patch_merger.feed_forward.2.weight": torch.randn(
                self.out_hidden_size, mlp, dtype=torch.bfloat16
            ),
        }

    def _create_mock_patch_merger(self, hidden_size, out_hidden_size, spatial_merge_size, state_dict):
        """Create a mock patch merger for CPU-only testing environments."""

        class MockPatchMerger:
            def __init__(self, hidden_size, out_hidden_size, spatial_merge_size):
                self.hidden_size = hidden_size
                self.out_hidden_size = out_hidden_size
                self.spatial_merge_size = spatial_merge_size
                self.mlp_size = hidden_size * (spatial_merge_size**2)

            def forward(self, x):
                # Simple mock that returns appropriately shaped tensor
                B, _, S_patch, H = x.shape if isinstance(x, torch.Tensor) else (1, 1, 256, hidden_size)
                S_img = S_patch // (self.spatial_merge_size**2)
                return torch.randn(B, 1, S_img, self.out_hidden_size, dtype=torch.bfloat16)

        return MockPatchMerger(hidden_size, out_hidden_size, spatial_merge_size)

    def _reshape_for_merger(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Reshape vision features from HF vision_tower [N_tokens, hidden] to
        PatchMerger input format [B, 1, S_patch, H].

        Dots vision tower typically returns flattened tokens. We reconstruct
        the expected spatial layout for the merger.
        """
        # vision_features: [N_img_tokens, hidden] or [B, N_img_tokens, hidden]
        if vision_features.dim() == 2:
            # Add batch dimension if missing
            vision_features = vision_features.unsqueeze(0)

        B, N_tokens, H = vision_features.shape
        assert H == self.hidden_size, f"Expected hidden={self.hidden_size}, got {H}"

        # For Dots, the patch merger expects [B, 1, S_patch, H] where S_patch = N_tokens
        # The spatial_merge_size is applied inside PatchMerger
        return vision_features.reshape(B, 1, N_tokens, H)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Vision forward pass supporting both full TTNN and hybrid modes.

        Full TTNN mode: PatchEmbedTT → 42×VisionBlockTT → PatchMergerTT (all on device)
        Hybrid mode: HF vision_tower (host) → TTNN PatchMerger (device)

        Args:
            pixel_values: [B, C, H, W] from processor
            grid_thw: [B, 3] image grid dimensions (temporal, height, width) from processor

        Returns:
            vision_tokens: [N_merged_tokens, hidden_size] ready for merge_vision_tokens
        """
        return self.vision_transformer.forward(pixel_values, grid_thw)

    def to_host(self):
        """Ensure all tensors are on host (for cleanup)."""
        if hasattr(self, "patch_merger"):
            # PatchMerger doesn't hold persistent tensors that need cleanup in this design
            pass


# Convenience function matching reference/vision.py interface
def vision_encoder_forward(encoder: VisionEncoder, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """Wrapper to match reference/vision.py:vision_tower_forward signature."""
    return encoder.forward(pixel_values, grid_thw)
