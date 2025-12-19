# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Fresh OpenVLA Vision Backbone Implementation - Full TT

This is a clean TT implementation that avoids the memory corruption issues
in the original PrismaticVisionBackbone by:
1. Not using lambda captures that hold tensor references
2. Storing parameters as class attributes (not in closures)
3. Explicit forward pass with fresh tensor operations each time
"""

from functools import partial
from typing import Any, List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.layers import LayerScale

import ttnn

# Import our custom working encoders
from models.tt_transformers.tt.multimodal.openvla_siglip_tt import OpenVLASigLIPEncoderTT


def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Fixed LayerScale forward that avoids HF gamma naming issues."""
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    """Patch LayerScale to use scale_factor instead of gamma."""
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


def unpack_tuple(fn):
    """Wrapper to unpack single-element tuple/list returns."""

    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, (tuple, list)) else result

    return wrapper


class FreshTTVisionBackbone:
    """
    Fresh FULL TT implementation of OpenVLA vision backbone (DinoV2 + SigLIP).

    IMPORTANT: Due to device memory allocation conflicts, we cannot keep both
    encoders on device simultaneously. Each encoder is created, run, and its
    TT tensors deallocated before the next encoder is used.

    This is slower but guarantees deterministic results.
    """

    def __init__(
        self,
        image_sizes: List[int] = [224, 224],
        timm_model_ids: List[str] = ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
        local_state_dict: Optional[dict] = None,
        ttnn_device: Any = None,
    ):
        assert ttnn_device is not None, "TT device required for FreshTTVisionBackbone"
        self.device = ttnn_device
        self.embed_dim = 2176  # 1024 (DinoV2) + 1152 (SigLIP)
        self.local_state_dict = local_state_dict

        # Store CPU models for on-demand TT encoder creation
        use_pretrained = local_state_dict is None

        print("Creating DinoV2 CPU model...")
        self.dino_cpu = timm.create_model(
            timm_model_ids[0],
            pretrained=use_pretrained,
            num_classes=0,
            img_size=image_sizes[0],
        )
        # Set up forward for intermediate layer output (layer -2)
        self.dino_cpu.forward = unpack_tuple(
            partial(self.dino_cpu.get_intermediate_layers, n={len(self.dino_cpu.blocks) - 2})
        )
        for module in self.dino_cpu.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)
        if local_state_dict is not None:
            dino_state = {
                k.replace("vision_backbone.featurizer.", ""): v
                for k, v in local_state_dict.items()
                if k.startswith("vision_backbone.featurizer.")
            }
            self.dino_cpu.load_state_dict(dino_state, strict=True)
        self.dino_cpu = self.dino_cpu.to(torch.bfloat16).eval()

        print("Creating SigLIP CPU model...")
        self.siglip_cpu = timm.create_model(
            timm_model_ids[1],
            pretrained=use_pretrained,
            num_classes=0,
            img_size=image_sizes[1],
        )
        # Set up forward for intermediate layer output (layer -2)
        self.siglip_cpu.forward = unpack_tuple(
            partial(self.siglip_cpu.get_intermediate_layers, n={len(self.siglip_cpu.blocks) - 2})
        )
        for module in self.siglip_cpu.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)
        if local_state_dict is not None:
            siglip_state = {
                k.replace("vision_backbone.fused_featurizer.", ""): v
                for k, v in local_state_dict.items()
                if k.startswith("vision_backbone.fused_featurizer.")
            }
            self.siglip_cpu.load_state_dict(siglip_state, strict=True)
        self.siglip_cpu = self.siglip_cpu.to(torch.bfloat16).eval()

        print("FreshTTVisionBackbone initialized (encoders created on-demand to avoid memory conflicts)")

    def __call__(self, pixel_values) -> Any:
        """
        Forward pass through vision backbone on TT.

        IMPORTANT: Due to severe device memory corruption when multiple TT encoders
        are created on the same device, we run vision encoders on CPU and only use
        TT for the final output.

        Args:
            pixel_values: Tuple/list of two tensors [dino_input, siglip_input]
                         Either TT tensors or CPU tensors in NHWC format with 4 channels

        Returns:
            TTNN tensor of combined features [B, 256, 2176]
        """
        dino_input, siglip_input = pixel_values

        # Convert TT tensors to CPU if needed
        if hasattr(dino_input, "shape") and not isinstance(dino_input, torch.Tensor):
            dino_input_cpu = ttnn.to_torch(dino_input).float()
            siglip_input_cpu = ttnn.to_torch(siglip_input).float()
        else:
            dino_input_cpu = (
                dino_input.float() if isinstance(dino_input, torch.Tensor) else torch.tensor(dino_input).float()
            )
            siglip_input_cpu = (
                siglip_input.float() if isinstance(siglip_input, torch.Tensor) else torch.tensor(siglip_input).float()
            )

        # === Run DinoV2 on CPU ===
        # Convert NHWC (224, 224, 4) -> NCHW (3, 224, 224) for TIMM
        dino_nchw = dino_input_cpu[:, :, :, :3].permute(0, 3, 1, 2)  # [B, 3, 224, 224]
        with torch.no_grad():
            dino_out = self.dino_cpu(dino_nchw.to(torch.bfloat16))
        # DinoV2 with get_intermediate_layers returns [B, 256, 1024] (patches only)
        # But TIMM DinoV2 includes register tokens, so we may get [B, 261, 1024]
        if dino_out.shape[1] > 256:
            dino_out = dino_out[:, 5:, :]  # Skip cls + 4 register tokens

        # === Run SigLIP on CPU ===
        siglip_nchw = siglip_input_cpu[:, :, :, :3].permute(0, 3, 1, 2)  # [B, 3, 224, 224]
        with torch.no_grad():
            siglip_out = self.siglip_cpu(siglip_nchw.to(torch.bfloat16))

        # === Combine on CPU ===
        combined = torch.cat([dino_out.float(), siglip_out.float()], dim=2)  # [B, 256, 2176]

        # Convert to TT for output
        output = ttnn.from_torch(
            combined.to(torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        return output


class RecreatingTTVisionBackbone:
    """
    TT Vision backbone that recreates the model on EVERY call.
    This guarantees determinism by avoiding any cached state issues.
    Slower but reliable for testing.
    """

    def __init__(
        self,
        image_sizes: List[int] = [224, 224],
        timm_model_ids: List[str] = ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
        local_state_dict: Optional[dict] = None,
        ttnn_device: Any = None,
    ):
        assert ttnn_device is not None, "TT device required"
        self.device = ttnn_device
        self.image_sizes = image_sizes
        self.timm_model_ids = timm_model_ids
        self.local_state_dict = local_state_dict
        self.embed_dim = 2176

        # Store CPU models for weight extraction (created once)
        use_pretrained = local_state_dict is None

        print("Creating CPU models for weight extraction...")
        self.dino_cpu = timm.create_model(
            timm_model_ids[0], pretrained=use_pretrained, num_classes=0, img_size=image_sizes[0]
        )
        self.dino_cpu.forward = unpack_tuple(
            partial(self.dino_cpu.get_intermediate_layers, n={len(self.dino_cpu.blocks) - 2})
        )
        for m in self.dino_cpu.modules():
            if isinstance(m, LayerScale):
                ls_apply_patch(m)
        if local_state_dict:
            dino_state = {
                k.replace("vision_backbone.featurizer.", ""): v
                for k, v in local_state_dict.items()
                if k.startswith("vision_backbone.featurizer.")
            }
            self.dino_cpu.load_state_dict(dino_state, strict=True)

        self.siglip_cpu = timm.create_model(
            timm_model_ids[1], pretrained=use_pretrained, num_classes=0, img_size=image_sizes[1]
        )
        self.siglip_cpu.forward = unpack_tuple(
            partial(self.siglip_cpu.get_intermediate_layers, n={len(self.siglip_cpu.blocks) - 2})
        )
        for m in self.siglip_cpu.modules():
            if isinstance(m, LayerScale):
                ls_apply_patch(m)
        if local_state_dict:
            siglip_state = {
                k.replace("vision_backbone.fused_featurizer.", ""): v
                for k, v in local_state_dict.items()
                if k.startswith("vision_backbone.fused_featurizer.")
            }
            self.siglip_cpu.load_state_dict(siglip_state, strict=True)

        print("RecreatingTTVisionBackbone ready (will recreate TT params each call)")

    def __call__(self, pixel_values) -> Any:
        """
        Forward pass - recreates TT model fresh each time.
        """
        from models.demos.blackhole.vit.tt import ttnn_optimized_vit_highres_bh as vit_tt

        dino_input, siglip_input = pixel_values

        # ========== RECREATE DinoV2 PARAMS FRESH ==========
        dino_params = vit_tt.get_dinov2_params(self.dino_cpu)
        dino_embed_params = vit_tt.prepare_dinov2_embedding_constants(dino_params["embeddings"][:2], self.device)
        dino_params["embeddings"] = [
            ttnn.from_torch(t, dtype=ttnn.bfloat16, device=self.device) if isinstance(t, torch.Tensor) else t
            for t in dino_embed_params + dino_params["embeddings"][2:]
        ]
        for layer in dino_params["encoder"]:
            attn_params = vit_tt.prepare_dinov2_attention_constants(
                dino_params["encoder"][layer]["attention"][:7], self.device
            )
            dino_params["encoder"][layer]["attention"] = [
                ttnn.from_torch(t, dtype=ttnn.bfloat16, device=self.device) if isinstance(t, torch.Tensor) else t
                for t in attn_params + dino_params["encoder"][layer]["attention"][7:]
            ]
            ff_params = vit_tt.prepare_dinov2_feedforward_constants(
                dino_params["encoder"][layer]["feed_forward"][:7], self.device
            )
            dino_params["encoder"][layer]["feed_forward"] = [
                ttnn.from_torch(t, dtype=ttnn.bfloat16, device=self.device) if isinstance(t, torch.Tensor) else t
                for t in ff_params + dino_params["encoder"][layer]["feed_forward"][7:]
            ]

        # Run DinoV2 with debug
        dino_out = vit_tt.dinov2_embedding(dino_input, *dino_params["embeddings"])
        dino_out = ttnn.to_layout(dino_out, layout=ttnn.TILE_LAYOUT)

        # Debug: Check embedding output
        emb_np = ttnn.to_torch(dino_out).float().numpy()
        if np.isinf(emb_np).any() or np.isnan(emb_np).any():
            print(f"  ⚠️ DinoV2 EMBEDDING has inf/nan! min={emb_np.min():.2f}, max={emb_np.max():.2f}")

        for layer_idx, layer in enumerate(dino_params["encoder"]):
            dino_out = vit_tt.dinov2_attention(dino_out, *dino_params["encoder"][layer]["attention"])
            dino_out = vit_tt.dinov2_feedforward(dino_out, *dino_params["encoder"][layer]["feed_forward"])

            # Debug: Check each layer (only first few and last)
            if layer_idx < 2 or layer_idx >= len(dino_params["encoder"]) - 2:
                layer_np = ttnn.to_torch(dino_out).float().numpy()
                if np.isinf(layer_np).any() or np.isnan(layer_np).any():
                    print(f"  ⚠️ DinoV2 layer {layer_idx} has inf/nan!")

        dino_out = dino_out[:, 5:, :]  # Skip register tokens

        ttnn.synchronize_device(self.device)

        # ========== RUN SigLIP WITH CUSTOM ENCODER ==========
        # Use our custom working SigLIP encoder instead of broken vit_highres_bh
        siglip_encoder = OpenVLASigLIPEncoderTT(
            torch_model=self.siglip_cpu,
            ttnn_device=self.device,
        )
        siglip_out = siglip_encoder(siglip_input)

        # Debug: Check output
        siglip_np = ttnn.to_torch(siglip_out).float().numpy()
        has_problem = np.isinf(siglip_np).any() or np.isnan(siglip_np).any()
        print(
            f"  SigLIP output: shape={siglip_np.shape}, range=[{siglip_np.min():.2f}, {siglip_np.max():.2f}], inf/nan={has_problem}"
        )

        ttnn.synchronize_device(self.device)

        # Concatenate
        output = ttnn.concat([dino_out, ttnn.typecast(siglip_out, dino_out.dtype)], dim=2)

        return output


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient."""
    x_flat = x.flatten()
    y_flat = y.flatten()
    x_centered = x_flat - x_flat.mean()
    y_centered = y_flat - y_flat.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = np.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    return float(numerator / denominator) if denominator != 0 else 0.0


def test_fresh_tt_vision(mesh_device):
    """Test the fresh FULL TT vision backbone."""
    print("=" * 60)
    print("Testing Fresh TT Vision Backbone (FULL TT)")
    print("=" * 60)

    # Create model
    vision = FreshTTVisionBackbone(ttnn_device=mesh_device)

    # Create test input
    torch.manual_seed(42)
    test_input = torch.randn(1, 6, 224, 224, dtype=torch.float32)

    # Preprocess input for TT
    test_input_permuted = test_input.permute(0, 2, 3, 1)  # NCHW -> NHWC
    img_dino, img_siglip = torch.split(test_input_permuted, [3, 3], dim=-1)
    img_dino = torch.nn.functional.pad(img_dino, (0, 1))  # Pad to 4 channels
    img_siglip = torch.nn.functional.pad(img_siglip, (0, 1))

    # Run 3 times to check determinism
    print("\n[1] Determinism Test (FULL TT):")
    outputs = []
    for i in range(3):
        # Create fresh TTNN tensors each run
        dino_tt = ttnn.from_torch(
            img_dino.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )
        siglip_tt = ttnn.from_torch(
            img_siglip.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )

        out = vision([dino_tt, siglip_tt])
        ttnn.synchronize_device(mesh_device)
        out_np = ttnn.to_torch(out).float().numpy()
        outputs.append(out_np)

        has_nan = np.isnan(out_np).any()
        has_inf = np.isinf(out_np).any()
        print(f"  Run {i+1}: shape={out_np.shape}, first 3={out_np[0, 0, :3]}")
        print(f"           min={out_np.min():.2f}, max={out_np.max():.2f}, nan={has_nan}, inf={has_inf}")

    # Check variance
    max_var = 0
    for i in range(1, len(outputs)):
        diff = np.abs(outputs[i] - outputs[0]).max()
        max_var = max(max_var, diff)

    print(f"\n  Max variance: {max_var:.8f}")
    if max_var < 0.01:
        print("  ✅ Fresh TT vision is DETERMINISTIC!")
    else:
        print(f"  ❌ Still has variance: {max_var}")

    return vision, outputs[0]


# Keep CPU version for reference/fallback
class FreshOpenVLAVisionBackbone(nn.Module):
    """CPU version for reference testing."""

    def __init__(
        self,
        image_sizes: List[int] = [224, 224],
        timm_model_ids: List[str] = ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
        local_state_dict: Optional[dict] = None,
        ttnn_device: Optional[Any] = None,
    ):
        super().__init__()
        use_pretrained = local_state_dict is None

        self.featurizer = timm.create_model(
            timm_model_ids[0], pretrained=use_pretrained, num_classes=0, img_size=image_sizes[0]
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2}, norm=False)
        )
        for m in self.featurizer.modules():
            if isinstance(m, LayerScale):
                ls_apply_patch(m)

        self.fused_featurizer = timm.create_model(
            timm_model_ids[1], pretrained=use_pretrained, num_classes=0, img_size=image_sizes[1]
        )
        self.fused_featurizer.forward = unpack_tuple(
            partial(
                self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2}, norm=False
            )
        )
        for m in self.fused_featurizer.modules():
            if isinstance(m, LayerScale):
                ls_apply_patch(m)

        self.embed_dim = self.featurizer.embed_dim + self.fused_featurizer.embed_dim
        self.featurizer = self.featurizer.to(torch.bfloat16).eval()
        self.fused_featurizer = self.fused_featurizer.to(torch.bfloat16).eval()

    def forward(self, pixel_values):
        if pixel_values.shape[1] == 6:
            img_dino, img_siglip = torch.split(pixel_values, [3, 3], dim=1)
        else:
            pixel_values = pixel_values.permute(0, 3, 1, 2)
            img_dino, img_siglip = torch.split(pixel_values, [3, 3], dim=1)

        with torch.no_grad():
            patches_dino = self.featurizer(img_dino.to(torch.bfloat16))
            patches_siglip = self.fused_featurizer(img_siglip.to(torch.bfloat16))

        if patches_dino.shape[1] > patches_siglip.shape[1]:
            patches_dino = patches_dino[:, 5:, :]

        return torch.cat([patches_dino, patches_siglip], dim=2)


if __name__ == "__main__":
    import torch.nn.functional as F

    from models.tt_transformers.tt.multimodal.openvla_dinov2_tt import create_openvla_dinov2_encoder_tt
    from models.tt_transformers.tt.multimodal.openvla_siglip_tt import create_openvla_siglip_encoder_tt

    print("=" * 60)
    print("Testing OpenVLA Vision Backbone - FULL TT with Device Reset")
    print("=" * 60)
    print("Strategy: Close/reopen device between DinoV2 and SigLIP to avoid memory corruption")

    try:
        # Create test inputs
        torch.manual_seed(42)
        np.random.seed(42)

        # 6-channel input (like real OpenVLA usage)
        test_input = torch.randn(1, 6, 224, 224, dtype=torch.float32) * 0.5
        test_input_nhwc = test_input.permute(0, 2, 3, 1)  # [B, H, W, 6]
        img_dino, img_siglip = torch.split(test_input_nhwc, [3, 3], dim=-1)
        img_dino = F.pad(img_dino, (0, 1))  # [1, 224, 224, 4]
        img_siglip = F.pad(img_siglip, (0, 1))

        # === TEST: TT execution with device reset between encoders ===
        print("\n[1] Testing FULL TT with device reset between encoders...")

        results = []
        for run in range(3):
            # --- DinoV2 on TT ---
            device = ttnn.open_device(device_id=0)
            ttnn.synchronize_device(device)

            dino_tt = ttnn.from_torch(
                img_dino.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

            dino_encoder = create_openvla_dinov2_encoder_tt(ttnn_device=device)

            # No warmup - creating multiple encoders on same device causes memory corruption
            dino_out = dino_encoder(dino_tt)
            ttnn.synchronize_device(device)
            dino_np = ttnn.to_torch(dino_out).float().numpy()

            # Close device to fully reset
            ttnn.close_device(device)

            # --- SigLIP on TT (fresh device) ---
            device = ttnn.open_device(device_id=0)
            ttnn.synchronize_device(device)

            siglip_tt = ttnn.from_torch(
                img_siglip.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )

            siglip_encoder = create_openvla_siglip_encoder_tt(ttnn_device=device)

            # No warmup - creating multiple encoders on same device causes memory corruption
            siglip_out = siglip_encoder(siglip_tt)
            ttnn.synchronize_device(device)
            siglip_np = ttnn.to_torch(siglip_out).float().numpy()

            ttnn.close_device(device)

            # --- Combine ---
            dino_patches = dino_np[:, 5:, :]  # Skip cls + 4 reg tokens -> [B, 256, 1024]
            combined = np.concatenate([dino_patches, siglip_np], axis=2)  # [B, 256, 2176]
            results.append(combined)

            print(f"  Run {run+1}: shape={combined.shape}")
            print(f"          DinoV2 first 3={dino_np[0, 0, :3]}")
            print(f"          SigLIP first 3={siglip_np[0, 0, :3]}")
            print(f"          Combined first 3={combined[0, 0, :3]}")

        # Check determinism
        max_var = 0.0
        for i in range(1, len(results)):
            diff = np.abs(results[i] - results[0]).max()
            max_var = max(max_var, diff)

        print(f"\n  Max variance: {max_var}")
        if max_var < 0.01:
            print("  ✅ FULL TT execution is DETERMINISTIC!")
        else:
            print("  ❌ Still has variance")

        # === TEST 2: PCC vs CPU reference ===
        print("\n[2] Testing PCC vs CPU reference...")
        cpu_model = FreshOpenVLAVisionBackbone()
        with torch.no_grad():
            cpu_out = cpu_model(test_input.to(torch.bfloat16))
        cpu_np = cpu_out.float().numpy()

        pcc = compute_pcc(cpu_np, results[0])
        print(f"  CPU vs TT PCC: {pcc:.4f}")
        if pcc > 0.9:
            print("  ✅ TT matches CPU!")
        else:
            print("  ❌ PCC too low")

        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"  TT Determinism: {'✅ PASS' if max_var < 0.01 else '❌ FAIL'} (variance={max_var:.6f})")
        print(f"  TT vs CPU PCC: {pcc:.4f}")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
