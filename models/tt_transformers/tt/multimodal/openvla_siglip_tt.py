# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Custom TT SigLIP implementation for OpenVLA.

This follows SmolVLA's working pattern using ttnn.transformer.* helpers
instead of the broken ttnn_optimized_vit_highres_bh.py implementation.

OpenVLA SigLIP specs (vit_so400m_patch14_siglip_224):
- Hidden dim: 1152
- Layers: 27
- Heads: 16
- Head dim: 72 (1152 / 16)
- MLP intermediate: 4304
- Image size: 224x224
- Patch size: 14
- Num patches: 256 (16x16)
"""

import math
from typing import Any, Dict, Optional

import numpy as np
import timm
import torch
import torch.nn.functional as F

import ttnn


class OpenVLASigLIPEncoderTT:
    """
    Custom TT SigLIP encoder following SmolVLA's working pattern.

    Key differences from ttnn_optimized_vit_highres_bh.py:
    1. Uses ttnn.transformer.split_query_key_value_and_split_heads
    2. Uses ttnn.transformer.concatenate_heads
    3. Explicit scaling before softmax
    4. Consistent core_grid usage
    """

    def __init__(
        self,
        torch_model,
        ttnn_device,
        hidden_size: int = 1152,
        num_layers: int = 27,
        num_heads: int = 16,
        mlp_ratio: float = 4304 / 1152,  # ~3.736
        image_size: int = 224,
        patch_size: int = 14,
        use_layer_index: int = -2,  # Use second-to-last layer output (like CPU get_intermediate_layers)
    ):
        self.device = ttnn_device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 72
        self.mlp_dim = int(hidden_size * mlp_ratio)  # 4304
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 256

        # Calculate actual layers to run
        # get_intermediate_layers(n={25}) returns output AFTER block 25
        # So we need to run blocks 0-25 = 26 layers total
        self.layers_to_run = num_layers + use_layer_index + 1  # 27 + (-2) + 1 = 26

        # Skip final norm when getting intermediate layers (matches get_intermediate_layers(norm=False))
        self.skip_final_norm = use_layer_index != 0  # Skip norm for intermediate output

        # Initialize TT parameters from torch model
        self._init_tt_params(torch_model)

    def _init_tt_params(self, torch_model):
        """Initialize TT tensor parameters from the torch model."""
        print(
            f"Initializing TT SigLIP params: {self.layers_to_run} layers (of {self.num_layers}), {self.hidden_size} hidden"
        )

        # Patch embedding
        patch_embed = torch_model.patch_embed

        # Preprocess conv weights for TT linear: [out, in, kH, kW] -> [kH*kW*in, out]
        weight = patch_embed.proj.weight  # [1152, 3, 14, 14]
        out_c, in_c, kh, kw = weight.shape

        # Pad input channels from 3 to 4 for TT
        weight_padded = F.pad(weight, (0, 0, 0, 0, 0, 1))  # [1152, 4, 14, 14]
        weight_reshaped = weight_padded.permute(2, 3, 1, 0).reshape(-1, out_c)  # [784, 1152]

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

        # Position embeddings (if exists) - SigLIP doesn't use cls token
        if hasattr(torch_model, "pos_embed") and torch_model.pos_embed is not None:
            self.tt_pos_embed = ttnn.from_torch(
                torch_model.pos_embed.to(torch.bfloat16),  # [1, 256, 1152]
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            self.tt_pos_embed = None

        # Layer parameters
        self.tt_layer_params = []

        for i in range(self.layers_to_run):
            block = torch_model.blocks[i]
            # Combine Q, K, V weights into single QKV weight for efficiency
            q_weight = block.attn.qkv.weight[: self.hidden_size, :]
            k_weight = block.attn.qkv.weight[self.hidden_size : 2 * self.hidden_size, :]
            v_weight = block.attn.qkv.weight[2 * self.hidden_size :, :]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

            q_bias = block.attn.qkv.bias[: self.hidden_size] if block.attn.qkv.bias is not None else None
            k_bias = (
                block.attn.qkv.bias[self.hidden_size : 2 * self.hidden_size]
                if block.attn.qkv.bias is not None
                else None
            )
            v_bias = block.attn.qkv.bias[2 * self.hidden_size :] if block.attn.qkv.bias is not None else None
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0) if q_bias is not None else None

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
                    qkv_weight.T.contiguous().to(torch.bfloat16),  # [1152, 3456]
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
            }
            self.tt_layer_params.append(layer_params)

        # Final LayerNorm (if exists)
        if hasattr(torch_model, "norm") and torch_model.norm is not None:
            self.tt_final_norm_weight = ttnn.from_torch(
                torch_model.norm.weight.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            self.tt_final_norm_bias = (
                ttnn.from_torch(
                    torch_model.norm.bias.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                if torch_model.norm.bias is not None
                else None
            )
        else:
            self.tt_final_norm_weight = None
            self.tt_final_norm_bias = None

        print(f"Initialized {self.layers_to_run} TT SigLIP layers")

    def _attention_tt(
        self,
        hidden_states: ttnn.Tensor,
        layer_params: dict,
    ) -> ttnn.Tensor:
        """
        TT attention with MANUAL split/reshape for non-tile-aligned head_dim.

        OpenVLA SigLIP has head_dim=72 which is NOT a multiple of 32,
        so we can't use ttnn.transformer.split_query_key_value_and_split_heads.

        We manually:
        1. Project to QKV
        2. Reshape and split manually
        3. Do attention
        4. Reshape back
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # QKV projection: [B, S, 1152] -> [B, S, 3456]
        qkv = ttnn.linear(
            hidden_states,
            layer_params["qkv_weight"],
            bias=layer_params["qkv_bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        # Manual split QKV: [B, S, 3*H*D] -> Q, K, V each [B, S, H*D]
        # Need to go to ROW_MAJOR for reshape/slice
        qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Split into Q, K, V
        q = ttnn.slice(qkv, (0, 0, 0), (batch_size, seq_len, self.hidden_size))
        k = ttnn.slice(qkv, (0, 0, self.hidden_size), (batch_size, seq_len, 2 * self.hidden_size))
        v = ttnn.slice(qkv, (0, 0, 2 * self.hidden_size), (batch_size, seq_len, 3 * self.hidden_size))
        ttnn.deallocate(qkv)

        # Reshape for multi-head attention: [B, S, H*D] -> [B, H, S, D]
        # Q: [B, S, 1152] -> [B, S, 16, 72] -> [B, 16, S, 72]
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

        # Q @ K^T: [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
        k_t = ttnn.permute(k, (0, 1, 3, 2))  # Transpose last two dims
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

        # Attention output: [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
        attn_output = ttnn.matmul(
            attn_probs,
            v,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(attn_probs)
        ttnn.deallocate(v)

        # Reshape back: [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
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
        """TT MLP with GELU activation (SigLIP uses GELU)."""
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
        """Single TT encoder layer: LayerNorm -> Attn -> Residual -> LayerNorm -> MLP -> Residual."""
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

        # Residual
        hidden_states = ttnn.add(
            residual,
            mlp_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(mlp_output)

        return hidden_states

    def __call__(self, pixel_values: ttnn.Tensor, layer_end_index: Optional[int] = None) -> ttnn.Tensor:
        """
        Forward pass through TT SigLIP encoder.

        Args:
            pixel_values: [batch, H, W, 4] TT tensor in NHWC format with padded channels
            layer_end_index: Optional, run only up to this layer (for debugging)

        Returns:
            [batch, num_patches, hidden_size] TT tensor
        """
        batch_size = pixel_values.shape[0]

        # Patch embedding
        # Input: [B, 224, 224, 4] -> fold -> [B*256, 784] -> linear -> [B*256, 1152]
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

        # Reshape to sequence: [B, 256, 1152]
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, self.num_patches, self.hidden_size))
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

        # Add position embeddings if exists
        if self.tt_pos_embed is not None:
            hidden_states = ttnn.add(
                hidden_states,
                self.tt_pos_embed,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        # Encoder layers
        num_layers = layer_end_index if layer_end_index is not None else self.layers_to_run
        for i in range(num_layers):
            hidden_states = self._encoder_layer_tt(hidden_states, self.tt_layer_params[i])

        # Final LayerNorm (skip if getting intermediate layers to match get_intermediate_layers(norm=False))
        if self.tt_final_norm_weight is not None and not self.skip_final_norm:
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=self.tt_final_norm_weight,
                bias=self.tt_final_norm_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        return hidden_states

    def deallocate(self):
        """Deallocate all TT tensors to free device memory."""
        # Deallocate patch embedding
        if hasattr(self, "tt_patch_embed_weight") and self.tt_patch_embed_weight is not None:
            ttnn.deallocate(self.tt_patch_embed_weight)
        if hasattr(self, "tt_patch_embed_bias") and self.tt_patch_embed_bias is not None:
            ttnn.deallocate(self.tt_patch_embed_bias)

        # Deallocate position embedding
        if hasattr(self, "tt_pos_embed") and self.tt_pos_embed is not None:
            ttnn.deallocate(self.tt_pos_embed)

        # Deallocate final norm
        if hasattr(self, "tt_final_norm_weight") and self.tt_final_norm_weight is not None:
            ttnn.deallocate(self.tt_final_norm_weight)
        if hasattr(self, "tt_final_norm_bias") and self.tt_final_norm_bias is not None:
            ttnn.deallocate(self.tt_final_norm_bias)

        # Deallocate layer params
        for layer_params in self.tt_layer_params:
            for key, tensor in layer_params.items():
                if tensor is not None:
                    try:
                        ttnn.deallocate(tensor)
                    except Exception:
                        pass

        self.tt_layer_params = []


def create_openvla_siglip_encoder_tt(
    local_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ttnn_device: Any = None,
) -> OpenVLASigLIPEncoderTT:
    """
    Create TT SigLIP encoder for OpenVLA.

    Args:
        local_state_dict: Optional state dict with OpenVLA weights
        ttnn_device: TT device

    Returns:
        OpenVLASigLIPEncoderTT instance
    """
    # Create TIMM model
    torch_model = timm.create_model(
        "vit_so400m_patch14_siglip_224",
        pretrained=True if local_state_dict is None else False,
        num_classes=0,
        img_size=224,
    )

    # Load weights from state dict if provided
    if local_state_dict is not None:
        fused_featurizer_state_dict = {
            k.replace("vision_backbone.fused_featurizer.", ""): v
            for k, v in local_state_dict.items()
            if k.startswith("vision_backbone.fused_featurizer.")
        }
        if fused_featurizer_state_dict:
            torch_model.load_state_dict(fused_featurizer_state_dict, strict=True)
            print("Loaded OpenVLA SigLIP weights from state dict")

    return OpenVLASigLIPEncoderTT(torch_model, ttnn_device)


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
    print("=" * 60)
    print("Testing Custom TT SigLIP Encoder for OpenVLA")
    print("=" * 60)

    # Setup device
    device = ttnn.open_device(device_id=0)
    ttnn.synchronize_device(device)

    try:
        # Create test input
        torch.manual_seed(42)
        np.random.seed(42)
        test_input = torch.randn(1, 3, 224, 224, dtype=torch.float32) * 0.5  # Normalized

        # Create CPU reference model
        print("\n[1] Creating CPU reference model...")
        cpu_model = timm.create_model(
            "vit_so400m_patch14_siglip_224",
            pretrained=True,
            num_classes=0,
            img_size=224,
        )
        cpu_model.eval()
        cpu_model = cpu_model.to(torch.bfloat16)  # Convert model to bfloat16

        with torch.no_grad():
            cpu_output = cpu_model.forward_features(test_input.to(torch.bfloat16))
        cpu_output_np = cpu_output.float().numpy()
        print(f"  CPU output: shape={cpu_output_np.shape}, first 3={cpu_output_np[0, 0, :3]}")

        # Create TT encoder
        print("\n[2] Creating TT SigLIP encoder...")
        tt_encoder = create_openvla_siglip_encoder_tt(
            local_state_dict=None,  # Use pretrained weights
            ttnn_device=device,
        )

        # Prepare TT input: [B, C, H, W] -> [B, H, W, C] -> pad to 4 channels
        test_input_nhwc = test_input.permute(0, 2, 3, 1)  # [1, 224, 224, 3]
        test_input_padded = F.pad(test_input_nhwc, (0, 1))  # [1, 224, 224, 4]

        pixel_values_tt = ttnn.from_torch(
            test_input_padded.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        # Test determinism
        print("\n[3] Testing TT determinism (3 runs)...")
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

        # Check determinism
        max_variance = 0.0
        for i in range(1, len(tt_outputs)):
            diff = np.abs(tt_outputs[i] - tt_outputs[0]).max()
            max_variance = max(max_variance, diff)

        print(f"\n  Max variance: {max_variance}")
        if max_variance < 0.01:
            print("  ✅ TT SigLIP is DETERMINISTIC!")
        else:
            print("  ❌ TT SigLIP has VARIANCE")

        # Test PCC vs CPU
        print("\n[4] Testing PCC vs CPU...")
        pcc = compute_pcc(cpu_output_np, tt_outputs[0])
        print(f"  CPU vs TT PCC: {pcc:.4f}")

        if pcc > 0.9:
            print("  ✅ PCC > 0.9 - TT implementation matches CPU!")
        elif pcc > 0.5:
            print("  ⚠️ PCC moderate - some differences")
        else:
            print("  ❌ PCC too low - check implementation")

    finally:
        ttnn.close_device(device)
        print("\nDevice closed.")
