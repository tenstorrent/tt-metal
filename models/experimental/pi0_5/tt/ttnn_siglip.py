# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
SigLIP Vision Tower - TTNN Implementation

This module implements the SigLIP vision encoder using TTNN operations.

SigLIP Architecture:
    - Patch embedding (Unfold + TTNN linear - optimized)
    - Positional embedding (learned)
    - Transformer encoder blocks (fused QKV, native head operations)
    - Multi-modal projector (linear to match language model dimension)

Optimizations over baseline:
    1. Unfold + TTNN linear for patch embedding (from Gemma3)
    2. Fused QKV projection (single linear instead of 3)
    3. Native ttnn.experimental.nlp_create_qkv_heads
    4. Native ttnn.experimental.nlp_concat_heads for output
"""

import math
import os
from typing import Dict, Optional, Tuple

import torch
import ttnn
import tt_lib.fallback_ops as fallback_ops  # For position embedding interpolation (native TTNN interpolate not available)

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.tt.ttnn_common import (
    get_sdpa_compute_kernel_config,
    get_sdpa_exp_approx_mode,
    sdpa_prefill_chunk_sizes,
    get_ln_weight_memory_config,
    tensor_1d_to_2d_ttnn,
)
from models.experimental.pi0_5.tt.ttnn_gemma import build_matmul_pcfg, build_sharded_norm_pcfg, _RMS_NORM_COMPUTE_CONFIG


# Tier 2 — ViT-BH-style block-sharded encoder data path for SigLIP.
# Single env-controllable switch so we can revert at runtime without rebuild.
#
# When enabled, hidden states stay block-sharded across the entire 27-layer
# encoder, only re-tiling for SDPA (which wants its own height-sharded layout
# for the attention heads). Eliminates the 108 LN-pair reshards we measured
# in the microbench (~7 ms) and lets the QKV/O-proj/FC1/FC2 matmuls run on
# block-sharded inputs (avoids DRAM re-read; +15-20% matmul throughput).
#
# Common grid: 12x8 = 96 cores. Picked because:
#   x=12 divides 36 hidden tiles (1152/32) cleanly
#   y=8  divides 16 M tiles (2*256/32 for batch=2, seq=256) cleanly
#   x=12 divides 144 QKV-out tiles (3*16*96/32) cleanly
#   x=12 divides 144 padded-intermediate tiles (4608/32) — see PADDING below.
#
# PADDING: SigLIP intermediate=4304 → 135 tiles, doesn't divide grid_x=12.
# We pad weights to 144 tiles (4608) at load time (6.7% extra weight memory).
# GELU(0)=0 so the padding columns produce zero downstream, harmless.
_SIGLIP_BS_GRID = (12, 8)  # (x, y) — 96 cores
_SIGLIP_INTERMEDIATE_PADDED_TILES = 144  # 144*32 = 4608 (padded from 4304/135)
_SIGLIP_INTERMEDIATE_PADDED = _SIGLIP_INTERMEDIATE_PADDED_TILES * 32


def _siglip_bs_enabled() -> bool:
    """Master switch for SigLIP block-sharded encoder path. Default ON.

    History: was flipped OFF after a 40-task LIBERO sweep on pi05_libero
    weights showed -42.5 pp at N=5 / -60 pp at N=10. Root cause was NOT
    matmul precision but a structural bug — the BS path flattened the
    batch dim into the M dim of SDPA, causing cross-image attention when
    the SigLIP batch is 2 (production: wrist + base camera stacked).
    Fixed by un-flattening batch around SDPA inside attention.forward_bs.

    Set PI0_SIGLIP_BS=0 to disable (e.g. to A/B against the baseline).
    """
    v = os.environ.get("PI0_SIGLIP_BS")
    if v is None:
        return True
    return v.strip().lower() in ("1", "true", "yes", "on")


def _make_bs_memcfg(b: int, m: int, hidden: int, grid_x: int, grid_y: int) -> "ttnn.MemoryConfig":
    """Build a block-sharded L1 memcfg for an encoder-data-path tensor."""
    return ttnn.create_sharded_memory_config(
        (b, 1, m, hidden),
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def _build_bs_matmul_pcfg(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    grid_x: int,
    grid_y: int,
    *,
    in0_block_w: Optional[int] = None,
    activation=None,
    dst_budget: int = 4,
) -> "ttnn.MatmulMultiCoreReuseMultiCastProgramConfig":
    """Build a 2D block-sharded matmul program config on a FIXED grid.

    Stricter variant of build_matmul_pcfg: the grid is pinned (so the BS output
    uses the same shard spec across all matmuls in the encoder) and we require
    exact divisibility.

    `in0_block_w` MUST divide per-core K-tiles (= k_tiles / grid_x), since the
    matmul kernel reads activations in blocks of size in0_block_w along the K
    dim *within each core's shard*. ViT-BH-hiRes uses `in0_block_w = per_core_K`
    (i.e. process the full per-core K slice in one inner iteration). We default
    to the largest divisor of per_core_K that's ≤ 4 to keep CBs small.
    """
    assert m_tiles % grid_y == 0, f"m_tiles {m_tiles} must divide grid_y {grid_y}"
    assert n_tiles % grid_x == 0, f"n_tiles {n_tiles} must divide grid_x {grid_x}"
    assert k_tiles % grid_x == 0, f"k_tiles {k_tiles} must divide grid_x {grid_x}"

    per_core_M = m_tiles // grid_y
    per_core_N = n_tiles // grid_x
    per_core_K = k_tiles // grid_x

    if in0_block_w is None:
        in0_block_w = min(per_core_K, 4)
    while in0_block_w > 1 and per_core_K % in0_block_w != 0:
        in0_block_w -= 1
    in0_block_w = max(1, in0_block_w)

    out_subblock_w = min(per_core_N, dst_budget)
    while out_subblock_w > 1 and per_core_N % out_subblock_w != 0:
        out_subblock_w -= 1
    out_subblock_h = max(1, dst_budget // out_subblock_w)
    out_subblock_h = min(per_core_M, out_subblock_h)
    while out_subblock_h > 1 and per_core_M % out_subblock_h != 0:
        out_subblock_h -= 1

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=activation,
    )


# ============================================================================
# Helper Functions
# ============================================================================


def nearest_32(x: int) -> int:
    """Round up to nearest multiple of 32 for TTNN tile alignment."""
    return ((x + 31) // 32) * 32


# ============================================================================
# Patch Embedding (TTNN - Optimized)
# ============================================================================


class PatchEmbeddingTTNN:
    """
    Convert image patches to embeddings using TTNN 6D permute + linear.

    OPTIMIZED: Uses TTNN's MultiCoreTileInvariant 6D permute for patch extraction,
    staying in TILE layout throughout to minimize layout conversions.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize patch embedding with TTNN weights.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights for conv2d (will be converted to linear format)
            device: TTNN device
        """
        self.config = config
        self.device = device
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size

        # Handle both formats: vision_model.embeddings.patch_embedding (checkpoint) and patch_embedding (legacy)
        conv_weight = weights.get("patch_embedding.weight") or weights.get(
            "vision_model.embeddings.patch_embedding.weight"
        )
        conv_bias = weights.get("patch_embedding.bias") or weights.get("vision_model.embeddings.patch_embedding.bias")

        # Convert conv2d weight to linear format
        # Conv weight: (out_channels, in_channels, kernel_h, kernel_w) = (hidden_size, 3, patch_size, patch_size)
        # Linear weight: (in_features, out_features) where in_features = 3 * patch_size * patch_size
        out_channels = conv_weight.shape[0]  # hidden_size
        in_channels = conv_weight.shape[1]  # 3
        in_features = in_channels * conv_weight.shape[2] * conv_weight.shape[3]  # 3 * 14 * 14 = 588

        # Store raw in_features for unfold output
        self.in_features = in_features
        self.in_channels = in_channels

        # Reorder weight to match our unfold's channel-last output order (h, w, c)
        # Conv weight: (out, c, h, w) -> permute to (out, h, w, c) -> flatten to (out, h*w*c)
        # This matches our _unfold_conv2d which produces (B, num_patches, h*w*c) order
        linear_weight = conv_weight.permute(0, 2, 3, 1).contiguous()  # (hidden_size, 14, 14, 3)
        linear_weight = linear_weight.view(out_channels, -1)  # (hidden_size, 588)

        # Pad input dimension to tile-aligned (588 -> 608)
        self.in_features_padded = nearest_32(in_features)
        pad_len = self.in_features_padded - in_features

        # Transpose for TTNN linear: (hidden_size, in_features) -> (in_features, hidden_size)
        linear_weight = linear_weight.T.contiguous()

        # Transfer to device first, then pad on device
        linear_weight_ttnn = ttnn.from_torch(
            linear_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Pad on device using ttnn.pad
        if pad_len > 0:
            linear_weight_ttnn = ttnn.pad(
                linear_weight_ttnn,
                padding=((0, pad_len), (0, 0)),  # Pad first dim (in_features)
                value=0.0,
            )

        self._linear_weight = linear_weight_ttnn

        # Bias (if present)
        if conv_bias is not None:
            self._linear_bias = tensor_1d_to_2d_ttnn(conv_bias, device, dtype=ttnn.bfloat16)
        else:
            self._linear_bias = None

        # === Opt-in: ttnn.fold-based patch extraction (PI0_SIGLIP_USE_FOLD=1) ===
        # Mirrors the SmolVLA/ViT pattern: replaces permute+reshape+permute+reshape
        # with a single ttnn.fold op. Requires NHWC input (channel-last) padded to
        # C=4 (power-of-2 align). Validated by test_patch_fold_pcc.py (PCC=0.99998).
        import os as _os

        self._use_fold = _os.environ.get("PI0_SIGLIP_USE_FOLD", "").lower() in ("1", "true", "yes", "on")
        if self._use_fold:
            # ttnn.fold supports C=3 natively (verified by test_fold_c3_smoke.py) — no
            # padding-to-4 needed. Keep C=3 so the downstream matmul stays at the
            # original (588 padded to 608) × 1152 size; skip the C-pad inflation.
            self._fold_in_channels = in_channels  # 3
            self._fold_in_features = self.patch_size * self.patch_size * in_channels  # 14*14*3=588
            # Build fold weight: (out, in, kh, kw)=(1152, 3, 14, 14) → linear (kH*kW*C, out)=(588, 1152)
            # Reuse the same shape as the linear path: (588, 1152), padded to (608, 1152) on device.
            w_fold = conv_weight.permute(2, 3, 1, 0).contiguous()  # (14, 14, 3, 1152)
            w_fold = w_fold.reshape(-1, out_channels)  # (588, 1152)
            w_ttnn = ttnn.from_torch(
                w_fold,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Pad in_features 588 → 608 to tile-align
            pad_f = self.in_features_padded - self._fold_in_features
            if pad_f > 0:
                w_ttnn = ttnn.pad(w_ttnn, padding=((0, pad_f), (0, 0)), value=0.0)
            self._fold_weight = w_ttnn
            if conv_bias is not None:
                self._fold_bias = ttnn.from_torch(
                    conv_bias.reshape(1, -1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            else:
                self._fold_bias = None

        # Query device grid to use all available cores
        device_grid = device.compute_with_storage_grid_size()
        self.grid_size = (device_grid.x, device_grid.y)
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

        # Compute kernel config
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _unfold_conv2d(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Unfold using TTNN 6D permute with MultiCoreTileInvariant optimization.
        Stays in TILE layout throughout - no layout conversions.

        The permute pattern (0, 1, 3, 2, 4, 5) keeps the last 2 dimensions (4, 5)
        in place, enabling the optimized MultiCoreTileInvariant kernel.

        Args:
            x: TTNN tensor (batch_size, height, width, channels) - channel-last, TILE layout

        Returns:
            TTNN tensor (batch_size, num_patches, patch_size * patch_size * channels) - TILE layout
        """
        batch_size = x.shape[0]
        img_h = x.shape[1]
        img_w = x.shape[2]
        img_c = x.shape[3]

        patches_h = img_h // self.patch_size
        patches_w = img_w // self.patch_size

        # Reshape to 6D: (B, H, W, C) -> (B, patches_h, patch_size, patches_w, patch_size, C)
        x = ttnn.reshape(x, (batch_size, patches_h, self.patch_size, patches_w, self.patch_size, img_c))

        # Optimized 6D permute - last 2 dims (4, 5) stay in place
        # Uses MultiCoreTileInvariant kernel for TILE layout
        # (B, patches_h, patch_size, patches_w, patch_size, C) -> (B, patches_h, patches_w, patch_size, patch_size, C)
        x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))

        # Flatten to 3D: (B, patches_h, patches_w, patch_size, patch_size, C) -> (B, num_patches, patch_features)
        x = ttnn.reshape(x, (batch_size, patches_h * patches_w, self.patch_size * self.patch_size * img_c))

        return x

    def _forward_fold(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """ttnn.fold-based patch extraction (PI0_SIGLIP_USE_FOLD=1).

        Replaces the 4-op permute+unfold+pad+linear chain with: pre-reshape (metadata)
        + ttnn.fold + linear. ttnn.fold supports C=3 natively (no C-padding needed),
        so the downstream matmul stays at the same (588 padded to 608) × 1152 size.

        Two input formats supported (auto-detected by shape):
          (a) (B, H, W, 3) ROW_MAJOR — already-permuted host upload (the fast path)
          (b) (B, 3, H, W) TILE BCHW — inline-converts via permute (slower)

        Output: (B, num_patches, hidden_size).
        """
        last_dim = int(x.shape[-1])
        if last_dim == self._fold_in_channels:
            # NHWC ROW_MAJOR (host pre-permute, no C-pad) — need to do reshape on device.
            B = int(x.shape[0])
            H = int(x.shape[1])
            W = int(x.shape[2])
            # Pre-reshape (B, H, W, C) → (B, H, W/patch, C*patch). Costs ~0.29 ms in TTNN.
            x = ttnn.reshape(x, (B, H, W // self.patch_size, self._fold_in_channels * self.patch_size))
        elif last_dim == self._fold_in_channels * self.patch_size:
            # Pre-reshaped on host: (B, H, W/patch, C*patch) — skip the device reshape.
            # This is the FAST PATH (saves ~0.29 ms vs the device reshape).
            B = int(x.shape[0])
            H = int(x.shape[1])
            W = int(x.shape[2]) * self.patch_size
        else:
            # BCHW TILE — inline convert (slow path)
            B = int(x.shape[0])
            H = int(x.shape[2])
            W = int(x.shape[3])
            x = ttnn.permute(x, (0, 2, 3, 1))  # BCHW → BHWC
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(x, (B, H, W // self.patch_size, self._fold_in_channels * self.patch_size))

        # fold(stride_h=patch_size, stride_w=1): (B, P_h, P_w, kH*kW*C) = (B, 16, 16, 588)
        x = ttnn.fold(x, self.patch_size, 1)

        P_h = H // self.patch_size
        P_w = W // self.patch_size
        x = ttnn.reshape(x, (B, P_h * P_w, self._fold_in_features))  # (B, 256, 588)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Pad in_features 588 → 608 (tile align) for the matmul
        pad_amount = self.in_features_padded - self._fold_in_features
        if pad_amount > 0:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_amount)], value=0.0)

        out = ttnn.linear(
            x,
            self._fold_weight,
            bias=self._fold_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=self.core_grid,
        )
        ttnn.deallocate(x)
        return out

    def forward(self, pixel_values) -> ttnn.Tensor:
        """
        OPTIMIZED: Extract patch embeddings entirely on device using TILE layout.
        Minimizes layout conversions by staying in TILE throughout.

        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)

        Returns:
            TTNN tensor (batch_size, num_patches, hidden_size)
        """

        x = pixel_values

        if self._use_fold:
            return self._forward_fold(x)

        # Step 2: Permute to channel-last: (B, C, H, W) -> (B, H, W, C)
        # Note: This uses generic kernel since last 2 dims move, but unavoidable
        x = ttnn.permute(x, (0, 2, 3, 1))

        # Step 3: Unfold using optimized 6D permute (MultiCoreTileInvariant)
        x = self._unfold_conv2d(x)

        # Step 4: Pad to tile-aligned if needed (588 -> 608)
        current_features = x.shape[-1]
        if current_features < self.in_features_padded:
            pad_amount = self.in_features_padded - current_features
            # Use ttnn.pad: pad last dimension
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_amount)], value=0.0)

        # Step 5: TTNN linear (already in TILE - no conversion needed!)
        out = ttnn.linear(
            x,
            self._linear_weight,
            bias=self._linear_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=self.core_grid,
        )

        ttnn.deallocate(x)

        return out


# ============================================================================
# Vision Transformer Block (TTNN - Optimized)
# ============================================================================


class SigLIPAttentionTTNN:
    """
    SigLIP self-attention using TTNN operations.

    OPTIMIZED: Uses fused QKV projection (single linear) and native TTNN head operations.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize attention with TTNN weights.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights to convert
            device: TTNN device
        """
        self.config = config
        self.device = device
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Query device grid to use all available cores (P150: up to 13x10)
        device_grid = device.compute_with_storage_grid_size()
        self.grid_size = (device_grid.x, device_grid.y)
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

        # Pad head_dim to multiple of 32 for TTNN tile alignment
        self.padded_head_dim = ((self.head_dim + 31) // 32) * 32  # 72 -> 96
        padding_size = self.padded_head_dim - self.head_dim

        # Pad weights on host using pure PyTorch (avoids expensive device round-trips)
        def pad_head_dim_weight(weight, heads_out=True):
            """Pad weight tensor's head dimension using PyTorch on host."""
            dim = weight.shape[0]

            if padding_size > 0:
                if heads_out:
                    weight = weight.T
                weight = weight.reshape(dim, self.num_heads, self.head_dim)
                weight = torch.nn.functional.pad(weight, (0, padding_size))
                weight = weight.reshape(dim, self.num_heads * self.padded_head_dim)
                if heads_out:
                    weight = weight.T
            return weight

        def pad_head_dim_bias(bias):
            """Pad 1D bias using PyTorch on host."""
            if padding_size > 0:
                bias = bias.view(self.num_heads, self.head_dim)
                bias = torch.nn.functional.pad(bias, (0, padding_size))
                bias = bias.reshape(self.num_heads * self.padded_head_dim)
            return bias

        # OPTIMIZATION: Fused QKV weights - single linear instead of 3
        # Pad each weight on host, then transfer to device
        wq_padded = pad_head_dim_weight(weights["self_attn.q_proj.weight"])
        wk_padded = pad_head_dim_weight(weights["self_attn.k_proj.weight"])
        wv_padded = pad_head_dim_weight(weights["self_attn.v_proj.weight"])

        # Concatenate Q, K, V weights on device
        wq_ttnn = ttnn.from_torch(
            wq_padded.T.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        wk_ttnn = ttnn.from_torch(
            wk_padded.T.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        wv_ttnn = ttnn.from_torch(
            wv_padded.T.contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.wqkv = ttnn.concat([wq_ttnn, wk_ttnn, wv_ttnn], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Fused QKV biases
        if "self_attn.q_proj.bias" in weights:
            bq_padded = pad_head_dim_bias(weights["self_attn.q_proj.bias"])
            bk_padded = pad_head_dim_bias(weights["self_attn.k_proj.bias"])
            bv_padded = pad_head_dim_bias(weights["self_attn.v_proj.bias"])

            # Concatenate biases on device (using tensor_1d_to_2d_ttnn to avoid torch.unsqueeze)
            bq_ttnn = tensor_1d_to_2d_ttnn(bq_padded, device, dtype=ttnn.bfloat8_b)
            bk_ttnn = tensor_1d_to_2d_ttnn(bk_padded, device, dtype=ttnn.bfloat8_b)
            bv_ttnn = tensor_1d_to_2d_ttnn(bv_padded, device, dtype=ttnn.bfloat8_b)
            self.bqkv = ttnn.concat([bq_ttnn, bk_ttnn, bv_ttnn], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            self.bqkv = None

        # Output projection - pad input head dim, output is hidden_size
        wo_padded = pad_head_dim_weight(weights["self_attn.out_proj.weight"], heads_out=False)
        self.wo = ttnn.from_torch(
            wo_padded.T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "self_attn.out_proj.bias" in weights:
            self.bo = tensor_1d_to_2d_ttnn(weights["self_attn.out_proj.bias"], device, dtype=ttnn.bfloat8_b)
        else:
            self.bo = None

        # Compute kernel configs — HiFi2 for QKV/O linears matches tt_transformers
        # inference defaults and saves cycles on the SigLIP attention path.
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # SDPA compute kernel — env-controllable for A/B testing. See ttnn_common.py.
        # Default (PI0_SDPA_HIFI=2, PI0_SDPA_EXP_APPROX=1) matches ViT-BH-hiRes §5.4
        # apart from math_fidelity (we run HiFi2 by default, ViT uses HiFi4).
        self.compute_kernel_config_sdpa = get_sdpa_compute_kernel_config()

        # Tier 2 — precompute QKV/O-proj tile dims for BS path.
        # QKV out = 3 * num_heads * padded_head_dim. For SigLIP base:
        # 3 * 16 * 96 = 4608 = 144 tiles. O-proj input = 16 * 96 = 1536 = 48 tiles.
        self._qkv_n_tiles = int(self.wqkv.shape[-1]) // 32  # expected 144
        self._oproj_k_tiles = int(self.wo.shape[-2]) // 32  # expected 48
        self._oproj_n_tiles = int(self.wo.shape[-1]) // 32  # expected 36

    def forward_bs(
        self,
        hidden_states: ttnn.Tensor,
        bs_memcfg_hidden: "ttnn.MemoryConfig",
        bs_memcfg_qkv: "ttnn.MemoryConfig",
        bs_memcfg_attn: "ttnn.MemoryConfig",
        *,
        n_batch: int,
        n_seq: int,
    ) -> ttnn.Tensor:
        """ViT-BH-style attention with block-sharded data path.

        Round-trip: BS in → QKV(BS) → L1 → nlp_create_qkv_heads → DRAM →
        SDPA → L1 → nlp_concat_heads → BS → O-proj(BS) → BS out (4D).

        The BS residual stream is (1, 1, n_batch*n_seq, hidden) — batch is
        flattened into the M dim so matmuls run on the full token block.
        SDPA, however, MUST see the un-flattened (n_batch, num_heads, n_seq, head_dim)
        or it will attend across image boundaries (cross-image contamination
        when n_batch>1 — confirmed against pi05_libero_finetuned at bs=2).
        We therefore reshape around the nlp_create_qkv_heads → SDPA → concat
        section, leaving the BS layout entry/exit shape intact.
        """
        gx, gy = _SIGLIP_BS_GRID
        b = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[-2])
        hidden_t = int(hidden_states.shape[-1]) // 32  # 36
        m_tiles = (b * seq_len) // 32  # 16

        # 1) QKV linear — BS in → BS out
        qkv_pcfg = _build_bs_matmul_pcfg(
            m_tiles,
            hidden_t,
            self._qkv_n_tiles,
            gx,
            gy,
            dst_budget=4,
        )
        xqkv = ttnn.linear(
            hidden_states,
            self.wqkv,
            bias=self.bqkv,
            dtype=ttnn.bfloat8_b,
            memory_config=bs_memcfg_qkv,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=qkv_pcfg,
        )

        # 2) BS → L1 interleaved for nlp_create_qkv_heads. Then un-flatten
        # the batch dim so SDPA computes attention per-image (not across).
        xqkv = ttnn.sharded_to_interleaved(xqkv, memory_config=ttnn.L1_MEMORY_CONFIG)
        qkv_dim = int(xqkv.shape[-1])
        xqkv = ttnn.reshape(xqkv, (n_batch, 1, n_seq, qkv_dim))

        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # 3) SDPA — runs on its own grid, accepts L1 inputs.
        # Chunk sizes use the per-batch seq_len (n_seq) — production has
        # n_seq=256 per image; the previous code used b*n_seq here, which was
        # both wrong semantically and not consistent with the chunked
        # attention's intended granularity.
        q_chunk, k_chunk = sdpa_prefill_chunk_sizes(n_seq, n_seq)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid_size,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=get_sdpa_exp_approx_mode(n_seq),
        )
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config_sdpa,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # 4) Concat heads back to L1, re-flatten batch into M, then reshard
        # to BS for the O-proj matmul.
        attn_concat = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)
        concat_dim = int(attn_concat.shape[-1])
        attn_concat = ttnn.reshape(attn_concat, (1, 1, n_batch * n_seq, concat_dim))
        attn_concat = ttnn.to_memory_config(attn_concat, bs_memcfg_attn)

        # 5) O-proj — BS in → BS out.
        oproj_pcfg = _build_bs_matmul_pcfg(
            m_tiles,
            self._oproj_k_tiles,
            self._oproj_n_tiles,
            gx,
            gy,
            dst_budget=4,
        )
        output = ttnn.linear(
            attn_concat,
            self.wo,
            bias=self.bo,
            dtype=ttnn.bfloat8_b,
            memory_config=bs_memcfg_hidden,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=oproj_pcfg,
        )
        ttnn.deallocate(attn_concat)
        return output  # 4D BS

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        OPTIMIZED forward pass using fused QKV and native TTNN head operations.

        Key optimizations:
        1. Single fused QKV linear (3x fewer linear ops)
        2. Native ttnn.experimental.nlp_create_qkv_heads
        3. Native ttnn.experimental.nlp_concat_heads

        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Reshape to 4D for nlp_create_qkv_heads: [batch, 1, seq, hidden]
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, -1))

        # OPTIMIZATION 1: Single fused QKV linear (instead of 3 separate)
        # Output: [batch, 1, seq, 3 * num_heads * padded_head_dim]
        ndim = len(hidden_states.shape)
        m_eff_attn = 1
        for i in range(ndim - 1):
            m_eff_attn *= int(hidden_states.shape[i])
        attn_k = (int(hidden_states.shape[-1]) + 31) // 32
        attn_n_qkv = (int(self.wqkv.shape[-1]) + 31) // 32
        m_t_attn = (m_eff_attn + 31) // 32

        qkv_pcfg = build_matmul_pcfg(
            m_t_attn,
            attn_k,
            attn_n_qkv,
            self.grid_size[0],
            self.grid_size[1],
            in0_block_w=4,
            dst_budget=4,
        )

        if qkv_pcfg is not None:
            xqkv_fused = ttnn.linear(
                hidden_states,
                self.wqkv,
                bias=self.bqkv,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                program_config=qkv_pcfg,
            )
        else:
            xqkv_fused = ttnn.linear(
                hidden_states,
                self.wqkv,
                bias=self.bqkv,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                core_grid=self.core_grid,
            )

        # OPTIMIZATION 2: Native TTNN head splitting
        # This splits the fused QKV into separate Q, K, V with proper head layout
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,  # SigLIP uses MHA, not MQA
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # Chunk sizes aligned with tt_transformers prefill SDPA (64 vs 2048-boundary heuristic)
        q_chunk, k_chunk = sdpa_prefill_chunk_sizes(seq_len, seq_len)

        # SDPA configuration - use full device grid for maximum parallelism.
        # exp_approx_mode is per-shape (04 §3c): seq_len<=256 → True (single
        # K-chunk, no accumulation depth); longer seqs → False (exact).
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid_size,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=get_sdpa_exp_approx_mode(seq_len),
        )

        # SDPA - stays entirely on device, L1 output feeds the o-proj matmul.
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config_sdpa,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # OPTIMIZATION 3: Native TTNN head concatenation
        # This concatenates heads back to [batch, 1, seq, num_heads * padded_head_dim]
        attn_concat = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output)

        # Output projection — sharded program_config + fused bias.
        wo_k = (int(self.wo.shape[-2]) + 31) // 32
        wo_n = (int(self.wo.shape[-1]) + 31) // 32
        wo_pcfg = build_matmul_pcfg(
            m_t_attn,
            wo_k,
            wo_n,
            self.grid_size[0],
            self.grid_size[1],
            in0_block_w=4,
            dst_budget=4,
        )
        if wo_pcfg is not None:
            output = ttnn.linear(
                attn_concat,
                self.wo,
                bias=self.bo,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                program_config=wo_pcfg,
            )
        else:
            output = ttnn.linear(
                attn_concat,
                self.wo,
                bias=self.bo,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                core_grid=self.core_grid,
            )
        ttnn.deallocate(attn_concat)

        # Reshape back to 3D: [batch, 1, seq, hidden] -> [batch, seq, hidden]
        output = ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))

        return output


class SigLIPMLPTTNN:
    """
    SigLIP MLP with GELU activation using TTNN.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize MLP with TTNN weights.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights to convert
            device: TTNN device
        """
        self.config = config
        self.device = device

        # Tier 2 BS path: pad intermediate (4304) → 4608 (144 tiles) so FC1/FC2
        # align on grid_x=12. Padding columns become zeros (GELU(0)=0 → safe).
        # Falls back to unpadded if BS path disabled.
        self.bs_enabled = _siglip_bs_enabled()
        orig_intermediate = weights["mlp.fc1.weight"].shape[0]  # (intermediate, hidden)
        self._intermediate_padded = _SIGLIP_INTERMEDIATE_PADDED if self.bs_enabled else orig_intermediate
        pad_n = self._intermediate_padded - orig_intermediate

        # FC1 weight: torch shape (intermediate, hidden) → T → (hidden, intermediate)
        # When BS is on, right-pad the OUT dim (intermediate) to 4608.
        fc1_w_torch = weights["mlp.fc1.weight"]
        if pad_n > 0:
            fc1_w_torch = torch.nn.functional.pad(fc1_w_torch, (0, 0, 0, pad_n))  # pad rows
        fc1_weight = fc1_w_torch.T.contiguous()
        self.fc1_weight = ttnn.from_torch(
            fc1_weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "mlp.fc1.bias" in weights:
            fc1_b_torch = weights["mlp.fc1.bias"]
            if pad_n > 0:
                fc1_b_torch = torch.nn.functional.pad(fc1_b_torch, (0, pad_n))
            self.fc1_bias = tensor_1d_to_2d_ttnn(fc1_b_torch, device, dtype=ttnn.bfloat8_b)
        else:
            self.fc1_bias = None

        # FC2 weight: torch shape (hidden, intermediate) → T → (intermediate, hidden)
        # When BS is on, top-pad the IN dim (intermediate) to 4608 — zero-padding
        # rows means the padded inputs (from FC1's zero outputs) contribute zero.
        fc2_w_torch = weights["mlp.fc2.weight"]
        if pad_n > 0:
            fc2_w_torch = torch.nn.functional.pad(fc2_w_torch, (0, pad_n, 0, 0))  # pad cols
        fc2_weight = fc2_w_torch.T.contiguous()
        self.fc2_weight = ttnn.from_torch(
            fc2_weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "mlp.fc2.bias" in weights:
            self.fc2_bias = tensor_1d_to_2d_ttnn(weights["mlp.fc2.bias"], device, dtype=ttnn.bfloat8_b)
        else:
            self.fc2_bias = None

        # Query device grid to use all available cores
        device_grid = device.compute_with_storage_grid_size()
        self.grid_size = (device_grid.x, device_grid.y)
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

        # Compute kernel config — HiFi2 sufficient for SigLIP MLP (bf8_b weights anyway).
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward_bs(
        self,
        hidden_states: ttnn.Tensor,
        bs_memcfg_hidden: "ttnn.MemoryConfig",
        bs_memcfg_intermediate: "ttnn.MemoryConfig",
    ) -> ttnn.Tensor:
        """Block-sharded BS forward path: BS in → FC1 BS → FC2 BS out.

        hidden_states is 4D (b, 1, m, hidden) in block-sharded L1 on grid (12,8).
        Returns same shape/layout for the next residual add.
        """
        b = int(hidden_states.shape[0])
        m_padded = int(hidden_states.shape[-2])
        hidden_t = int(hidden_states.shape[-1]) // 32  # 36
        m_tiles = (b * m_padded) // 32  # 16 for b=2, m=256
        gx, gy = _SIGLIP_BS_GRID

        fc1_pcfg = _build_bs_matmul_pcfg(
            m_tiles,
            hidden_t,
            _SIGLIP_INTERMEDIATE_PADDED_TILES,
            gx,
            gy,
            activation=(ttnn.UnaryOpType.GELU, True),
            dst_budget=4,
        )
        x = ttnn.linear(
            hidden_states,
            self.fc1_weight,
            bias=self.fc1_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=bs_memcfg_intermediate,
            compute_kernel_config=self.compute_kernel_config,
            program_config=fc1_pcfg,
        )

        fc2_pcfg = _build_bs_matmul_pcfg(
            m_tiles,
            _SIGLIP_INTERMEDIATE_PADDED_TILES,
            hidden_t,
            gx,
            gy,
            dst_budget=4,
        )
        output = ttnn.linear(
            x,
            self.fc2_weight,
            bias=self.fc2_bias,
            dtype=ttnn.bfloat8_b,
            memory_config=bs_memcfg_hidden,
            compute_kernel_config=self.compute_kernel_config,
            program_config=fc2_pcfg,
        )
        ttnn.deallocate(x)
        return output

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass using TTNN operations.

        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        # Compute matmul tile counts. SigLIP runs at bs=2 (2 images stacked).
        # The matmul kernel collapses leading dims into M: M_eff = product of dims
        # before the last (hidden) dim. Input shape: (2, 1, 256, 1152) → M_eff = 512.
        ndim = len(hidden_states.shape)
        m_eff = 1
        for i in range(ndim - 1):
            m_eff *= int(hidden_states.shape[i])
        hidden = int(hidden_states.shape[-1])
        intermediate = int(self.fc1_weight.shape[-1])
        m_t = (m_eff + 31) // 32
        # Ceil division — the matmul kernel pads K up to a tile-multiple internally,
        # and the in0_block_w divisor check applies to the *padded* K.
        k_t_fc1 = (hidden + 31) // 32
        n_t_fc1 = (intermediate + 31) // 32
        k_t_fc2 = (intermediate + 31) // 32
        n_t_fc2 = (hidden + 31) // 32

        # SigLIP compute config uses fp32_dest_acc_en=True → DST budget = 4 tiles.
        fc1_pcfg = build_matmul_pcfg(
            m_t,
            k_t_fc1,
            n_t_fc1,
            self.grid_size[0],
            self.grid_size[1],
            in0_block_w=4,
            activation=(ttnn.UnaryOpType.GELU, True),
            dst_budget=4,
        )
        fc2_pcfg = build_matmul_pcfg(
            m_t,
            k_t_fc2,
            n_t_fc2,
            self.grid_size[0],
            self.grid_size[1],
            in0_block_w=4,
            dst_budget=4,
        )

        # FC1 with GELU
        if fc1_pcfg is not None:
            x = ttnn.linear(
                hidden_states,
                self.fc1_weight,
                bias=self.fc1_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                program_config=fc1_pcfg,
            )
        else:
            x = ttnn.linear(
                hidden_states,
                self.fc1_weight,
                bias=self.fc1_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=self.core_grid,
                activation="gelu",
            )

        # FC2
        if fc2_pcfg is not None:
            output = ttnn.linear(
                x,
                self.fc2_weight,
                bias=self.fc2_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                program_config=fc2_pcfg,
            )
        else:
            output = ttnn.linear(
                x,
                self.fc2_weight,
                bias=self.fc2_bias,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=self.core_grid,
            )
        ttnn.deallocate(x)

        return output


class SigLIPBlockTTNN:
    """
    Complete SigLIP transformer block using TTNN.
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize block with TTNN weights.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights to convert
            device: TTNN device
        """
        self.config = config
        self.device = device

        # Layer norms — PI0_LN_WEIGHTS_L1=1 opts into L1 placement.
        _ln_mc = get_ln_weight_memory_config()
        self.ln1_weight = ttnn.from_torch(
            weights["layer_norm1.weight"].reshape(1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_ln_mc,
        )

        if "layer_norm1.bias" in weights:
            self.ln1_bias = ttnn.from_torch(
                weights["layer_norm1.bias"].reshape(1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_ln_mc,
            )
        else:
            self.ln1_bias = None

        self.ln2_weight = ttnn.from_torch(
            weights["layer_norm2.weight"].reshape(1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_ln_mc,
        )

        if "layer_norm2.bias" in weights:
            self.ln2_bias = ttnn.from_torch(
                weights["layer_norm2.bias"].reshape(1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_ln_mc,
            )
        else:
            self.ln2_bias = None

        # Attention and MLP - using native TTNN with padded head dim workaround
        self.attention = SigLIPAttentionTTNN(config, weights, device)
        self.mlp = SigLIPMLPTTNN(config, weights, device)

        # Sharded LayerNorm config (ViT-BH §5.3). SigLIP hidden=1152 → 36 tiles.
        # M is built lazily on first call since it depends on (batch, seq_len).
        self._ln_sharded_pcfg = None
        self._ln_sharded_memcfg = None
        self._ln_sharded_m_padded = 0

        # Tier 2 — BS forward path master switch + cached LN PCFG on the
        # encoder common grid (12, 8) so LN matches the matmul shard spec.
        self.bs_enabled = _siglip_bs_enabled()
        self._bs_ln_pcfg = None
        self._bs_ln_grid_cached = None

    def _get_bs_ln_pcfg(self, b: int, m_padded: int) -> "ttnn.LayerNormShardedMultiCoreProgramConfig":
        """LN program config on the SIGLIP_BS_GRID (12, 8). Cached."""
        key = (b, m_padded)
        if self._bs_ln_pcfg is not None and self._bs_ln_grid_cached == key:
            return self._bs_ln_pcfg
        gx, gy = _SIGLIP_BS_GRID
        m_tiles = (b * m_padded) // 32  # 16 for b=2 m=256
        hidden_tiles = self.config.hidden_size // 32  # 36
        assert m_tiles % gy == 0, f"BS LN: m_tiles {m_tiles} must divide gy {gy}"
        assert hidden_tiles % gx == 0, f"BS LN: hidden_tiles {hidden_tiles} must divide gx {gx}"
        block_h = m_tiles // gy  # 2
        block_w = hidden_tiles // gx  # 3
        self._bs_ln_pcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            subblock_w=block_w,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )
        self._bs_ln_grid_cached = key
        return self._bs_ln_pcfg

    def _get_sharded_ln(self, b: int, m_padded: int):
        key = (b, m_padded)
        if self._ln_sharded_pcfg is None or self._ln_sharded_m_padded != key:
            total_m_tiles = (b * m_padded) // 32
            hidden_tiles = self.config.hidden_size // 32  # 1152/32 = 36
            cfg = build_sharded_norm_pcfg(
                total_m_tiles, hidden_tiles, max_grid_x=12, max_grid_y=min(8, max(1, total_m_tiles))
            )
            if cfg is not None:
                pc, memcfg_factory, _grid = cfg
                self._ln_sharded_pcfg = pc
                self._ln_sharded_memcfg = memcfg_factory(b, m_padded, m_padded, self.config.hidden_size)
                self._ln_sharded_m_padded = key
            else:
                self._ln_sharded_pcfg = None
                self._ln_sharded_memcfg = None
        return self._ln_sharded_pcfg, self._ln_sharded_memcfg

    def _sharded_layer_norm(self, x, weight, bias):
        b = x.shape[0]
        m_padded = x.shape[1]
        sh_pc, sh_mc = self._get_sharded_ln(b, m_padded)
        if sh_pc is None:
            return ttnn.layer_norm(
                x,
                weight=weight,
                bias=bias,
                epsilon=self.config.layer_norm_eps,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        x_sh = ttnn.to_memory_config(x, sh_mc)
        out = ttnn.layer_norm(
            x_sh,
            weight=weight,
            bias=bias,
            epsilon=self.config.layer_norm_eps,
            program_config=sh_pc,
            memory_config=sh_mc,
            compute_kernel_config=_RMS_NORM_COMPUTE_CONFIG,
        )
        if x_sh is not x:
            ttnn.deallocate(x_sh)
        out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.L1_MEMORY_CONFIG)
        return out

    def _sharded_layer_norm_bs(self, x_bs, weight, bias, bs_memcfg):
        """LN with BS in → BS out (NO s2i at exit). Input must already be BS
        on _SIGLIP_BS_GRID."""
        b = int(x_bs.shape[0])
        m_padded = int(x_bs.shape[-2])
        pc = self._get_bs_ln_pcfg(b, m_padded)
        return ttnn.layer_norm(
            x_bs,
            weight=weight,
            bias=bias,
            epsilon=self.config.layer_norm_eps,
            program_config=pc,
            memory_config=bs_memcfg,
            compute_kernel_config=_RMS_NORM_COMPUTE_CONFIG,
        )

    def forward_bs(
        self,
        hidden_states: ttnn.Tensor,
        bs_memcfg_hidden: "ttnn.MemoryConfig",
        bs_memcfg_qkv: "ttnn.MemoryConfig",
        bs_memcfg_attn: "ttnn.MemoryConfig",
        bs_memcfg_intermediate: "ttnn.MemoryConfig",
        *,
        n_batch: int,
        n_seq: int,
    ) -> ttnn.Tensor:
        """Full block in BS: input 4D BS → output 4D BS. No internal reshards
        except for the SDPA round-trip inside attention.

        n_batch / n_seq plumbed through so attention.forward_bs can un-flatten
        the batch dim around SDPA (otherwise SDPA computes attention across
        the b stacked images instead of per-image — see attention.forward_bs).
        """
        normed = self._sharded_layer_norm_bs(hidden_states, self.ln1_weight, self.ln1_bias, bs_memcfg_hidden)
        attn = self.attention.forward_bs(
            normed,
            bs_memcfg_hidden,
            bs_memcfg_qkv,
            bs_memcfg_attn,
            n_batch=n_batch,
            n_seq=n_seq,
        )
        ttnn.deallocate(normed)
        # Residual add on BS — both operands share bs_memcfg_hidden.
        hidden_states = ttnn.add(hidden_states, attn, memory_config=bs_memcfg_hidden)
        ttnn.deallocate(attn)

        normed = self._sharded_layer_norm_bs(hidden_states, self.ln2_weight, self.ln2_bias, bs_memcfg_hidden)
        mlp_out = self.mlp.forward_bs(normed, bs_memcfg_hidden, bs_memcfg_intermediate)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_out, memory_config=bs_memcfg_hidden)
        ttnn.deallocate(mlp_out)
        return hidden_states

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass using native TTNN operations.

        Args:
            hidden_states: TTNN tensor (batch_size, seq_len, hidden_size)

        Returns:
            TTNN tensor (batch_size, seq_len, hidden_size)
        """
        # Pre-attention LayerNorm (sharded if available)
        normed = self._sharded_layer_norm(hidden_states, self.ln1_weight, self.ln1_bias)

        # Native TTNN attention with padded head dim workaround
        attn_output = self.attention.forward(normed)
        ttnn.deallocate(normed)

        # Residual connection - use L1 for intermediate computation
        hidden_states = ttnn.add(hidden_states, attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)

        # Pre-MLP LayerNorm (sharded if available)
        normed = self._sharded_layer_norm(hidden_states, self.ln2_weight, self.ln2_bias)

        # MLP with residual - use L1 for intermediate computation
        mlp_output = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(mlp_output)

        return hidden_states


# ============================================================================
# Full Vision Tower (TTNN)
# ============================================================================


class SigLIPVisionTowerTTNN:
    """
    Complete SigLIP vision tower using TTNN operations.

    Fully implemented in TTNN:
        - Patch embedding on host (Unfold) + device (TTNN linear)
        - Position embedding addition on device
        - All transformer blocks on device (TTNN)
        - Final layer norm on device
    """

    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize vision tower.

        Args:
            config: SigLIP configuration
            weights: PyTorch weights (will be converted)
            device: TTNN device
        """
        self.config = config
        self.device = device

        # Patch embedding (Unfold + TTNN linear)
        self.patch_embed = PatchEmbeddingTTNN(config, weights, device)

        # Position embedding on device (handle both formats)
        pos_emb = weights.get("position_embedding.weight") or weights.get(
            "vision_model.embeddings.position_embedding.weight"
        )

        if pos_emb is not None:
            # Calculate target number of patches based on config
            num_patches = (config.image_size // config.patch_size) ** 2

            # Check if we need to interpolate position embeddings
            if pos_emb.shape[0] != num_patches:
                original_num_patches = pos_emb.shape[0]
                original_size = int(math.sqrt(original_num_patches))
                target_size = int(math.sqrt(num_patches))

                # Reshape to 2D grid: (num_patches, hidden_size) -> (1, hidden_size, H, W)
                pos_emb_2d = pos_emb.view(1, original_size, original_size, -1)
                pos_emb_2d = pos_emb_2d.permute(0, 3, 1, 2)  # (1, hidden_size, H, W)

                # Interpolate using bicubic (via TTNN fallback_ops)
                # Note: This is still rare - only when checkpoint resolution differs
                # fallback_ops.interpolate to replace torch.nn.functional.interpolate
                pos_emb_interpolated = fallback_ops.interpolate(
                    pos_emb_2d,
                    size=(target_size, target_size),
                    mode="bicubic",
                    align_corners=False,
                )

                # Reshape back: (1, hidden_size, H, W) -> (num_patches, hidden_size)
                pos_emb = pos_emb_interpolated.permute(0, 2, 3, 1).flatten(0, 2)

            # Create position IDs
            self.position_ids = ttnn.arange(0, num_patches, 1, dtype=ttnn.uint32, device=device)
            self.position_ids = ttnn.reshape(self.position_ids, (1, -1))

            # Load position embedding weights
            self.pos_emb_weights = ttnn.as_tensor(
                pos_emb,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.position_ids = None
            self.pos_emb_weights = None

        # Initialize TTNN transformer blocks
        self.blocks = []
        for i in range(config.num_hidden_layers):
            block_weights = self._get_layer_weights(weights, i)
            self.blocks.append(SigLIPBlockTTNN(config, block_weights, device))

        # Tier 2 — BS encoder data path. Memcfgs built lazily on first forward
        # since they depend on the actual (batch, seq_len) at runtime.
        self.bs_enabled = _siglip_bs_enabled()
        self._bs_memcfgs_cache: Dict[Tuple[int, int], Tuple] = {}

        # Final layer norm weights (handle both formats)
        post_ln_weight = weights.get("post_layernorm.weight") or weights.get("vision_model.post_layernorm.weight")
        post_ln_bias = weights.get("post_layernorm.bias") or weights.get("vision_model.post_layernorm.bias")

        if post_ln_weight is not None:
            _post_ln_mc = get_ln_weight_memory_config()
            self.post_ln_weight = tensor_1d_to_2d_ttnn(
                post_ln_weight, device, dtype=ttnn.bfloat16, memory_config=_post_ln_mc
            )
            self.post_ln_bias = (
                tensor_1d_to_2d_ttnn(post_ln_bias, device, dtype=ttnn.bfloat16, memory_config=_post_ln_mc)
                if post_ln_bias is not None
                else None
            )
        else:
            self.post_ln_weight = None
            self.post_ln_bias = None

    def _get_bs_memcfgs(self, b: int, m_padded: int):
        key = (b, m_padded)
        if key not in self._bs_memcfgs_cache:
            gx, gy = _SIGLIP_BS_GRID
            total_m = b * m_padded  # collapse leading dims into M
            # Hidden-shape BS memcfg (LN + O-proj + FC2 out + residual stream).
            mc_hidden = _make_bs_memcfg(1, total_m, self.config.hidden_size, gx, gy)
            # QKV-shape BS memcfg (after fused QKV linear). N = 144*32 = 4608.
            mc_qkv = _make_bs_memcfg(1, total_m, 144 * 32, gx, gy)
            # O-proj input shape after concat_heads = num_heads*padded_head_dim = 1536.
            mc_attn = _make_bs_memcfg(1, total_m, 48 * 32, gx, gy)
            # Intermediate BS memcfg (after FC1; padded to 4608).
            mc_intermediate = _make_bs_memcfg(1, total_m, _SIGLIP_INTERMEDIATE_PADDED, gx, gy)
            self._bs_memcfgs_cache[key] = (mc_hidden, mc_qkv, mc_attn, mc_intermediate)
        return self._bs_memcfgs_cache[key]

    def _get_layer_weights(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Extract weights for a specific layer."""
        # Handle both formats: vision_model.encoder.layers.X (checkpoint) and encoder.layers.X (legacy)
        prefixes = [f"vision_model.encoder.layers.{layer_idx}.", f"encoder.layers.{layer_idx}."]
        layer_weights = {}
        for prefix in prefixes:
            for key, value in weights.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix) :]
                    layer_weights[new_key] = value
        return layer_weights

    def forward(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        Process images to embeddings (TTNN).

        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)

        Returns:
            TTNN tensor (batch_size, num_patches, hidden_size)
        """
        # Patch embedding (hybrid - Unfold on host, linear on device)
        hidden_states = self.patch_embed.forward(pixel_values)

        # Add position embeddings (on device)
        if self.pos_emb_weights is not None:
            num_patches_actual = hidden_states.shape[1]
            num_patches_expected = self.position_ids.shape[1]

            # Check if we need to interpolate position embeddings dynamically
            if num_patches_actual != num_patches_expected:
                # Dynamic position embedding interpolation (rare - only when image size differs)
                original_size = int(math.sqrt(num_patches_expected))
                target_size = int(math.sqrt(num_patches_actual))

                # Reshape position embeddings for interpolation
                # pos_emb_weights: [num_patches, hidden_size] -> [1, hidden_size, H, W]
                pos_emb_2d = ttnn.reshape(self.pos_emb_weights, (1, original_size, original_size, -1))
                pos_emb_2d = ttnn.permute(pos_emb_2d, (0, 3, 1, 2))

                # Interpolate using bicubic (via TTNN fallback_ops - handles TTNN tensors)
                # fallback_ops.interpolate to replace torch.nn.functional.interpolate
                pos_emb_interpolated = fallback_ops.interpolate(
                    pos_emb_2d,
                    size=(target_size, target_size),
                    mode="bicubic",
                    align_corners=False,
                )

                # Reshape back: [1, hidden_size, H, W] -> [num_patches, hidden_size]
                pos_emb_resized = ttnn.permute(pos_emb_interpolated, (0, 2, 3, 1))
                pos_emb_resized = ttnn.reshape(pos_emb_resized, (target_size * target_size, -1))

                # Create new position IDs for actual number of patches
                position_ids_new = ttnn.arange(0, num_patches_actual, 1, dtype=ttnn.uint32, device=self.device)
                position_ids_new = ttnn.reshape(position_ids_new, (1, -1))

                # Convert resized embeddings to TTNN
                pos_emb_weights_new = ttnn.as_tensor(
                    pos_emb_resized,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                # Use ttnn.embedding with resized weights
                positional_embeddings = ttnn.embedding(
                    position_ids_new,
                    pos_emb_weights_new,
                    layout=ttnn.TILE_LAYOUT,
                )
            else:
                # Use pre-loaded position embeddings
                positional_embeddings = ttnn.embedding(
                    self.position_ids,
                    self.pos_emb_weights,
                    layout=ttnn.TILE_LAYOUT,
                )

            hidden_states = ttnn.add(hidden_states, positional_embeddings, memory_config=ttnn.L1_MEMORY_CONFIG)

        if self.bs_enabled:
            # Tier 2 — enter BS once before the encoder loop, exit once after.
            # Saves 27*2*2=108 LN reshards + lets matmuls consume BS directly.
            b, num_patches, hidden = hidden_states.shape
            # Reshape to 4D (1, 1, b*num_patches, hidden) for block-sharding.
            hidden_states = ttnn.reshape(hidden_states, (1, 1, int(b) * int(num_patches), int(hidden)))
            mc_hidden, mc_qkv, mc_attn, mc_intermediate = self._get_bs_memcfgs(int(b), int(num_patches))
            hidden_states = ttnn.to_memory_config(hidden_states, mc_hidden, dtype=ttnn.bfloat16)

            # Plumb (n_batch, n_seq) through so attention.forward_bs can un-flatten
            # the batch dim around SDPA — otherwise SDPA attends across the b stacked
            # images. See attention.forward_bs for the full reasoning.
            for block in self.blocks:
                hidden_states = block.forward_bs(
                    hidden_states,
                    mc_hidden,
                    mc_qkv,
                    mc_attn,
                    mc_intermediate,
                    n_batch=int(b),
                    n_seq=int(num_patches),
                )

            # Exit BS once: BS → L1 interleaved → 3D shape for post_ln + projector.
            hidden_states = ttnn.sharded_to_interleaved(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
            hidden_states = ttnn.reshape(hidden_states, (int(b), int(num_patches), int(hidden)))
        else:
            for block in self.blocks:
                hidden_states = block.forward(hidden_states)

        # Final layer norm (on device)
        if self.post_ln_weight is not None:
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=self.post_ln_weight,
                bias=self.post_ln_bias,
                epsilon=self.config.layer_norm_eps,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        return hidden_states


# ============================================================================
# Multi-modal Projector (TTNN)
# ============================================================================


class MultiModalProjectorTTNN:
    """
    Projects vision features to language model dimension using TTNN.
    """

    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
    ):
        """
        Initialize projector with TTNN weights.

        Args:
            weights: PyTorch weights to convert
            device: TTNN device
        """
        self.device = device

        # Query device grid to use all available cores
        device_grid = device.compute_with_storage_grid_size()
        self.grid_size = (device_grid.x, device_grid.y)
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

        # Convert weight to TTNN format (transposed)
        self.weight = ttnn.from_torch(
            weights["linear.weight"].T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if "linear.bias" in weights:
            self.bias = tensor_1d_to_2d_ttnn(weights["linear.bias"], device, dtype=ttnn.bfloat16)
        else:
            self.bias = None

    def forward(self, vision_features: ttnn.Tensor) -> ttnn.Tensor:
        """
        Project vision features using TTNN linear.
        """
        ndim = len(vision_features.shape)
        m_eff = 1
        for i in range(ndim - 1):
            m_eff *= int(vision_features.shape[i])
        m_t = (m_eff + 31) // 32
        k_t = (int(vision_features.shape[-1]) + 31) // 32
        n_t = (int(self.weight.shape[-1]) + 31) // 32
        pcfg = build_matmul_pcfg(m_t, k_t, n_t, self.grid_size[0], self.grid_size[1])
        if pcfg is not None:
            return ttnn.linear(
                vision_features,
                self.weight,
                bias=self.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pcfg,
            )
        return ttnn.linear(
            vision_features,
            self.weight,
            bias=self.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=self.core_grid,
        )


# Default exports
PatchEmbedding = PatchEmbeddingTTNN
SigLIPAttention = SigLIPAttentionTTNN
SigLIPMLP = SigLIPMLPTTNN
SigLIPBlock = SigLIPBlockTTNN
SigLIPVisionTower = SigLIPVisionTowerTTNN
MultiModalProjector = MultiModalProjectorTTNN
