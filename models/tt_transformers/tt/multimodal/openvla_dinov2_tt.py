# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Custom TT DinoV2 encoder for OpenVLA.

This follows the same working pattern as our custom SigLIP encoder,
avoiding the broken ttnn_optimized_vit_highres_bh.py implementation.

OpenVLA DinoV2 specs (vit_large_patch14_reg4_dinov2.lvd142m):
- Hidden dim: 1024
- Layers: 24 (we use layer -2 = 23 layers)
- Heads: 16
- Head dim: 64 (1024 / 16) - tile-aligned!
- MLP intermediate: 4096
- Image size: 224x224
- Patch size: 14
- Num patches: 256 + 5 register tokens = 261
"""

import math
from typing import Any, Dict, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerScale

import ttnn


def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Fixed LayerScale forward that avoids HF gamma naming issues."""
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    """Patch LayerScale to use scale_factor instead of gamma."""
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


class OpenVLADinoV2EncoderTT:
    """
    Custom TT DinoV2 encoder following the same working pattern as SigLIP.

    Key differences from ttnn_optimized_vit_highres_bh.py:
    1. Manual QKV split/reshape (no closures that corrupt)
    2. Explicit attention computation
    3. Fresh tensor operations each forward pass
    """

    def __init__(
        self,
        torch_model,
        ttnn_device,
        hidden_size: int = 1024,
        num_layers: int = 24,  # Full model has 24, we use up to layer -2
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
        patch_size: int = 14,
        num_register_tokens: int = 4,  # DinoV2 has register tokens
        use_layer_index: int = -2,  # Use second-to-last layer output
    ):
        self.device = ttnn_device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 64 - tile aligned!
        self.mlp_dim = int(hidden_size * mlp_ratio)  # 4096
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 256
        self.num_register_tokens = num_register_tokens
        self.use_layer_index = use_layer_index

        # Calculate actual layers to run
        # get_intermediate_layers(n={22}) returns output AFTER block 22
        # So we need to run blocks 0-22 = 23 layers total
        self.layers_to_run = num_layers + use_layer_index + 1  # 24 + (-2) + 1 = 23

        # Initialize TT parameters from torch model
        self._init_tt_params(torch_model)

    def _init_tt_params(self, torch_model):
        """Initialize TT tensor parameters from the torch model."""
        print(
            f"Initializing TT DinoV2 params: {self.layers_to_run} layers (of {self.num_layers}), {self.hidden_size} hidden"
        )

        # Patch embedding
        patch_embed = torch_model.patch_embed

        # Preprocess conv weights for TT linear: [out, in, kH, kW] -> [kH*kW*in, out]
        weight = patch_embed.proj.weight  # [1024, 3, 14, 14]
        out_c, in_c, kh, kw = weight.shape

        # Pad input channels from 3 to 4 for TT
        weight_padded = F.pad(weight, (0, 0, 0, 0, 0, 1))  # [1024, 4, 14, 14]
        weight_reshaped = weight_padded.permute(2, 3, 1, 0).reshape(-1, out_c)  # [784, 1024]

        self.tt_patch_embed_weight = ttnn.from_torch(
            weight_reshaped.to(torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        if patch_embed.proj.bias is not None:
            self.tt_patch_embed_bias = ttnn.from_torch(
                patch_embed.proj.bias.unsqueeze(0).to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            self.tt_patch_embed_bias = None

        # CLS token
        self.tt_cls_token = ttnn.from_torch(
            torch_model.cls_token.to(torch.bfloat16),  # [1, 1, 1024]
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Register tokens (DinoV2 specific)
        if hasattr(torch_model, "reg_token") and torch_model.reg_token is not None:
            self.tt_reg_token = ttnn.from_torch(
                torch_model.reg_token.to(torch.bfloat16),  # [1, num_reg, 1024]
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            self.tt_reg_token = None

        # Position embeddings - DinoV2 pos_embed is for patches only [1, 256, 1024]
        print(f"  pos_embed shape from TIMM: {torch_model.pos_embed.shape}")
        self.tt_pos_embed = ttnn.from_torch(
            torch_model.pos_embed.to(torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Layer parameters
        self.tt_layer_params = []

        for i in range(self.layers_to_run):
            block = torch_model.blocks[i]

            # DinoV2 uses separate Q, K, V projections (not combined QKV)
            # Actually TIMM DinoV2 uses combined qkv
            qkv_weight = block.attn.qkv.weight  # [3072, 1024]
            qkv_bias = block.attn.qkv.bias if block.attn.qkv.bias is not None else None

            # Get LayerScale parameters if present
            ls1_scale = None
            ls2_scale = None
            if hasattr(block, "ls1") and block.ls1 is not None:
                if hasattr(block.ls1, "scale_factor"):
                    ls1_scale = block.ls1.scale_factor
                elif hasattr(block.ls1, "gamma"):
                    ls1_scale = block.ls1.gamma
            if hasattr(block, "ls2") and block.ls2 is not None:
                if hasattr(block.ls2, "scale_factor"):
                    ls2_scale = block.ls2.scale_factor
                elif hasattr(block.ls2, "gamma"):
                    ls2_scale = block.ls2.gamma

            layer_params = {
                # Pre-attention LayerNorm
                "norm1_weight": ttnn.from_torch(
                    block.norm1.weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "norm1_bias": ttnn.from_torch(
                    block.norm1.bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                if block.norm1.bias is not None
                else None,
                # QKV projection (combined)
                "qkv_weight": ttnn.from_torch(
                    qkv_weight.T.contiguous().to(torch.bfloat16),  # [1024, 3072]
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "qkv_bias": ttnn.from_torch(
                    qkv_bias.unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                if qkv_bias is not None
                else None,
                # Output projection
                "proj_weight": ttnn.from_torch(
                    block.attn.proj.weight.T.contiguous().to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "proj_bias": ttnn.from_torch(
                    block.attn.proj.bias.unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                if block.attn.proj.bias is not None
                else None,
                # LayerScale for attention
                "ls1_scale": ttnn.from_torch(
                    ls1_scale.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                if ls1_scale is not None
                else None,
                # Pre-MLP LayerNorm
                "norm2_weight": ttnn.from_torch(
                    block.norm2.weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "norm2_bias": ttnn.from_torch(
                    block.norm2.bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                if block.norm2.bias is not None
                else None,
                # MLP FC1
                "fc1_weight": ttnn.from_torch(
                    block.mlp.fc1.weight.T.contiguous().to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "fc1_bias": ttnn.from_torch(
                    block.mlp.fc1.bias.unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                # MLP FC2
                "fc2_weight": ttnn.from_torch(
                    block.mlp.fc2.weight.T.contiguous().to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "fc2_bias": ttnn.from_torch(
                    block.mlp.fc2.bias.unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                # LayerScale for MLP
                "ls2_scale": ttnn.from_torch(
                    ls2_scale.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                if ls2_scale is not None
                else None,
            }
            self.tt_layer_params.append(layer_params)

        print(f"Initialized {len(self.tt_layer_params)} TT DinoV2 layers")

    def _attention_tt(
        self,
        hidden_states: ttnn.Tensor,
        layer_params: dict,
    ) -> ttnn.Tensor:
        """
        TT attention with manual split/reshape.

        DinoV2 has head_dim=64 which IS tile-aligned, but we still use
        manual split to avoid any potential issues with the helper functions.
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # QKV projection: [B, S, 1024] -> [B, S, 3072]
        qkv = ttnn.linear(
            hidden_states,
            layer_params["qkv_weight"],
            bias=layer_params["qkv_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        # Manual split QKV: [B, S, 3*H*D] -> Q, K, V each [B, S, H*D]
        qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT)

        q = ttnn.slice(qkv, (0, 0, 0), (batch_size, seq_len, self.hidden_size))
        k = ttnn.slice(qkv, (0, 0, self.hidden_size), (batch_size, seq_len, 2 * self.hidden_size))
        v = ttnn.slice(qkv, (0, 0, 2 * self.hidden_size), (batch_size, seq_len, 3 * self.hidden_size))
        ttnn.deallocate(qkv)

        # Reshape for multi-head attention: [B, S, H*D] -> [B, H, S, D]
        q = ttnn.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        q = ttnn.to_layout(q, layout=ttnn.TILE_LAYOUT)

        k = ttnn.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        k = ttnn.to_layout(k, layout=ttnn.TILE_LAYOUT)

        v = ttnn.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))
        v = ttnn.to_layout(v, layout=ttnn.TILE_LAYOUT)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)

        # Q @ K^T
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        ttnn.deallocate(k)

        attn_weights = ttnn.matmul(
            q,
            k_t,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)

        # Scale
        attn_weights = ttnn.mul(attn_weights, scale)

        # Softmax
        attn_probs = ttnn.softmax(attn_weights, dim=-1)
        ttnn.deallocate(attn_weights)

        # Attention output
        attn_output = ttnn.matmul(
            attn_probs,
            v,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(attn_probs)
        ttnn.deallocate(v)

        # Reshape back: [B, H, S, D] -> [B, S, H*D]
        attn_output = ttnn.to_layout(attn_output, layout=ttnn.ROW_MAJOR_LAYOUT)
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, self.hidden_size))
        attn_output = ttnn.to_layout(attn_output, layout=ttnn.TILE_LAYOUT)

        # Output projection
        output = ttnn.linear(
            attn_output,
            layer_params["proj_weight"],
            bias=layer_params["proj_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(attn_output)

        return output

    def _mlp_tt(
        self,
        hidden_states: ttnn.Tensor,
        layer_params: dict,
    ) -> ttnn.Tensor:
        """TT MLP with GELU activation (DinoV2 uses GELU)."""
        # FC1 with GELU
        hidden = ttnn.linear(
            hidden_states,
            layer_params["fc1_weight"],
            bias=layer_params["fc1_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            activation="gelu",
        )

        # FC2
        output = ttnn.linear(
            hidden,
            layer_params["fc2_weight"],
            bias=layer_params["fc2_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(hidden)

        return output

    def _encoder_layer_tt(
        self,
        hidden_states: ttnn.Tensor,
        layer_params: dict,
    ) -> ttnn.Tensor:
        """
        Single TT encoder layer with LayerScale.

        DinoV2 architecture:
        - LayerNorm -> Attn -> LayerScale -> Residual
        - LayerNorm -> MLP -> LayerScale -> Residual
        """
        residual = hidden_states

        # Pre-attention LayerNorm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=layer_params["norm1_weight"],
            bias=layer_params["norm1_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Attention
        attn_output = self._attention_tt(hidden_states, layer_params)

        # LayerScale for attention (if present)
        if layer_params["ls1_scale"] is not None:
            attn_output = ttnn.mul(attn_output, layer_params["ls1_scale"])

        # Residual
        hidden_states = ttnn.add(
            residual,
            attn_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(attn_output)

        residual = hidden_states

        # Pre-MLP LayerNorm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=layer_params["norm2_weight"],
            bias=layer_params["norm2_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # MLP
        mlp_output = self._mlp_tt(hidden_states, layer_params)

        # LayerScale for MLP (if present)
        if layer_params["ls2_scale"] is not None:
            mlp_output = ttnn.mul(mlp_output, layer_params["ls2_scale"])

        # Residual
        hidden_states = ttnn.add(
            residual,
            mlp_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(mlp_output)

        return hidden_states

    def __call__(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass through TT DinoV2 encoder.

        Args:
            pixel_values: [batch, H, W, 4] TT tensor in NHWC format with padded channels

        Returns:
            [batch, num_patches + 1 + num_reg, hidden_size] TT tensor
        """
        batch_size = pixel_values.shape[0]

        # Patch embedding
        hidden_states = ttnn.reshape(
            pixel_values, (batch_size, self.image_size, self.image_size // self.patch_size, 4 * self.patch_size)
        )
        hidden_states = ttnn.fold(hidden_states, stride_h=self.patch_size, stride_w=1)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_patch_embed_weight,
            bias=self.tt_patch_embed_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        # Reshape to sequence: [B, 256, 1024]
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, self.num_patches, self.hidden_size))

        # DinoV2 order (from ttnn_optimized_vit_highres_bh.py):
        # 1. Add pos_embed to patch embeddings ONLY (not to cls/reg)
        # 2. Concat: [cls, reg_tokens, patches_with_pos]
        hidden_torch = ttnn.to_torch(hidden_states).float()  # [B, 256, 1024]
        pos_torch = ttnn.to_torch(self.tt_pos_embed).float()  # [1, 256, 1024]

        # Add position embeddings to patches
        hidden_torch = hidden_torch + pos_torch

        # Get CLS and reg tokens
        cls_torch = ttnn.to_torch(self.tt_cls_token).float()  # [1, 1, 1024]

        # Concat: [cls, reg_tokens, patches_with_pos]
        if self.tt_reg_token is not None:
            reg_torch = ttnn.to_torch(self.tt_reg_token).float()  # [1, 4, 1024]
            hidden_torch = torch.cat(
                [cls_torch.expand(batch_size, -1, -1), reg_torch.expand(batch_size, -1, -1), hidden_torch], dim=1
            )  # [B, 1+4+256=261, 1024]
        else:
            hidden_torch = torch.cat([cls_torch.expand(batch_size, -1, -1), hidden_torch], dim=1)  # [B, 257, 1024]

        # Convert back to TT
        hidden_states = ttnn.from_torch(
            hidden_torch.to(torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        # Encoder layers
        for i in range(len(self.tt_layer_params)):
            hidden_states = self._encoder_layer_tt(hidden_states, self.tt_layer_params[i])

        return hidden_states

    def deallocate(self):
        """Deallocate all TT tensors to free device memory."""
        # Deallocate patch embedding
        if hasattr(self, "tt_patch_embed_weight") and self.tt_patch_embed_weight is not None:
            ttnn.deallocate(self.tt_patch_embed_weight)
        if hasattr(self, "tt_patch_embed_bias") and self.tt_patch_embed_bias is not None:
            ttnn.deallocate(self.tt_patch_embed_bias)

        # Deallocate special tokens
        if hasattr(self, "tt_cls_token") and self.tt_cls_token is not None:
            ttnn.deallocate(self.tt_cls_token)
        if hasattr(self, "tt_reg_token") and self.tt_reg_token is not None:
            ttnn.deallocate(self.tt_reg_token)
        if hasattr(self, "tt_pos_embed") and self.tt_pos_embed is not None:
            ttnn.deallocate(self.tt_pos_embed)

        # Deallocate layer params
        for layer_params in self.tt_layer_params:
            for key, tensor in layer_params.items():
                if tensor is not None:
                    try:
                        ttnn.deallocate(tensor)
                    except Exception:
                        pass

        self.tt_layer_params = []


def create_openvla_dinov2_encoder_tt(
    local_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ttnn_device: Any = None,
) -> OpenVLADinoV2EncoderTT:
    """
    Create TT DinoV2 encoder for OpenVLA.
    """
    # Create TIMM model
    torch_model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m",
        pretrained=True if local_state_dict is None else False,
        num_classes=0,
        img_size=224,
    )

    # Apply LayerScale patch
    for module in torch_model.modules():
        if isinstance(module, LayerScale):
            ls_apply_patch(module)

    # Load weights from state dict if provided
    if local_state_dict is not None:
        featurizer_state_dict = {
            k.replace("vision_backbone.featurizer.", ""): v
            for k, v in local_state_dict.items()
            if k.startswith("vision_backbone.featurizer.")
        }
        if featurizer_state_dict:
            torch_model.load_state_dict(featurizer_state_dict, strict=True)
            print("Loaded OpenVLA DinoV2 weights from state dict")

    return OpenVLADinoV2EncoderTT(torch_model, ttnn_device)


# ============================================================================
# Test Functions
# ============================================================================
def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson Correlation Coefficient."""
    x_flat = torch.as_tensor(x).flatten().float()
    y_flat = torch.as_tensor(y).flatten().float()
    x_centered = x_flat - x_flat.mean()
    y_centered = y_flat - y_flat.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    return (numerator / denominator).item() if denominator != 0 else 0.0


if __name__ == "__main__":
    from functools import partial

    def unpack_tuple(fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            return result[0] if isinstance(result, (tuple, list)) else result

        return wrapper

    print("=" * 60)
    print("Testing Custom TT DinoV2 Encoder for OpenVLA")
    print("=" * 60)

    # Setup device
    device = ttnn.open_device(device_id=0)
    ttnn.synchronize_device(device)

    try:
        # Create test input
        torch.manual_seed(42)
        np.random.seed(42)
        test_input = torch.randn(1, 3, 224, 224, dtype=torch.float32) * 0.5

        # Create CPU reference model
        print("\n[1] Creating CPU reference model...")
        cpu_model = timm.create_model(
            "vit_large_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
        )
        # Use intermediate layer output (layer -2)
        cpu_model.forward = unpack_tuple(partial(cpu_model.get_intermediate_layers, n={len(cpu_model.blocks) - 2}))
        for m in cpu_model.modules():
            if isinstance(m, LayerScale):
                ls_apply_patch(m)
        cpu_model.eval()
        cpu_model = cpu_model.to(torch.bfloat16)

        with torch.no_grad():
            cpu_output = cpu_model(test_input.to(torch.bfloat16))
        cpu_output_np = cpu_output.float().numpy()
        print(f"  CPU output: shape={cpu_output_np.shape}, first 3={cpu_output_np[0, 0, :3]}")

        # Create TT encoder
        print("\n[2] Creating TT DinoV2 encoder...")
        tt_encoder = create_openvla_dinov2_encoder_tt(
            local_state_dict=None,
            ttnn_device=device,
        )

        # Prepare TT input: [B, C, H, W] -> [B, H, W, C] -> pad to 4 channels
        test_input_nhwc = test_input.permute(0, 2, 3, 1)
        test_input_padded = F.pad(test_input_nhwc, (0, 1))

        pixel_values_tt = ttnn.from_torch(
            test_input_padded.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Test determinism (skip first run as warmup)
        print("\n[3] Testing TT determinism (1 warmup + 3 runs)...")

        # Warmup run
        print("  Warmup run...")
        warmup_out = tt_encoder(pixel_values_tt)
        ttnn.synchronize_device(device)

        tt_outputs = []
        for run in range(3):
            output_tt = tt_encoder(pixel_values_tt)
            ttnn.synchronize_device(device)
            out_np = ttnn.to_torch(output_tt).float().numpy()
            tt_outputs.append(out_np)

            has_nan = np.isnan(out_np).any()
            has_inf = np.isinf(out_np).any()
            print(f"  Run {run+1}: shape={out_np.shape}, first 3={out_np[0, 0, :3]}")
            print(f"           min={out_np.min():.2f}, max={out_np.max():.2f}, nan={has_nan}, inf={has_inf}")

        # Check determinism (after warmup, runs should be identical)
        max_variance = 0.0
        for i in range(1, len(tt_outputs)):
            diff = np.abs(tt_outputs[i] - tt_outputs[0]).max()
            max_variance = max(max_variance, diff)

        print(f"\n  Max variance: {max_variance}")
        if max_variance < 0.01:
            print("  ✅ TT DinoV2 is DETERMINISTIC!")
        else:
            print("  ❌ TT DinoV2 has VARIANCE")

        # Test PCC vs CPU (skip register tokens from TT output)
        print("\n[4] Testing PCC vs CPU...")
        # CPU output: intermediate layer output [B, 256, 1024] (patches only, no cls/reg)
        # TT output: [cls, reg1, reg2, reg3, reg4, patch1, ...] = [B, 261, 1024]

        print(f"  CPU output shape: {cpu_output_np.shape}")
        print(f"  TT output shape: {tt_outputs[0].shape}")

        # Skip cls + 4 reg tokens from TT (first 5 tokens)
        tt_patches = tt_outputs[0][:, 5:, :]
        print(f"  TT patches shape (after skip 5): {tt_patches.shape}")

        # If shapes still don't match, truncate to min
        min_seq = min(cpu_output_np.shape[1], tt_patches.shape[1])
        cpu_compare = cpu_output_np[:, :min_seq, :]
        tt_compare = tt_patches[:, :min_seq, :]
        print(f"  Comparing: CPU {cpu_compare.shape} vs TT {tt_compare.shape}")

        pcc = compute_pcc(cpu_compare, tt_compare)
        print(f"  CPU vs TT PCC (patches only): {pcc:.4f}")

        if pcc > 0.9:
            print("  ✅ PCC > 0.9 - TT implementation matches CPU!")
        elif pcc > 0.5:
            print("  ⚠️ PCC moderate - some differences")
        else:
            print("  ❌ PCC too low - check implementation")

    finally:
        ttnn.close_device(device)
        print("\nDevice closed.")
