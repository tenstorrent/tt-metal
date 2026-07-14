# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN native Sam2HieraImageEncoder matching HuggingFace modeling_sam2.py.
Implements 12 Sam2MultiScaleBlock with windowed/global attention, query pooling.
Architecture matches exact HF reference.

CURRENT LIMITATIONS:
- Patch embedding (ttnn.conv2d) port pending: requires conv_config/compute_config
  setup similar to models/demos/stable_diffusion_xl_base/tt/sdxl_utility.py
- LayerNorm on CPU (needs ttnn.layer_norm tested on hardware)
- GELU on CPU (needs ttnn.gelu tested on hardware)
- MaxPool2d for query pooling on CPU (needs ttnn.max_pool2d tested on hardware)
- Attention math on CPU via torch (ttnn SDPA needs hardware to validate)
"""

from typing import Dict, List, Optional, Tuple
import torch
import ttnn
import math
import numpy as np


def do_pool(x: torch.Tensor, query_stride: Optional[Tuple[int, int]]) -> torch.Tensor:
    """Max pool for query pooling at stage transitions. (B,H,W,C) -> (B,H',W',C).
    TODO: Replace with ttnn.max_pool2d once hardware CI available."""
    if query_stride is None:
        return x
    x = x.permute(0, 3, 1, 2)                    # NHWC -> NCHW
    x = torch.nn.functional.max_pool2d(x, kernel_size=query_stride, stride=query_stride, ceil_mode=False)
    return x.permute(0, 2, 3, 1)                  # NCHW -> NHWC


def window_partition(hidden_state: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition into non-overlapping windows with padding. (B,H,W,C) -> (B*nW,ws,ws,C)."""
    B, H, W, C = hidden_state.shape
    pad_h = (-H) % window_size
    pad_w = (-W) % window_size
    if pad_h > 0 or pad_w > 0:
        hidden_state = torch.nn.functional.pad(hidden_state, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = H + pad_h, W + pad_w
    n_h = padded_h // window_size
    n_w = padded_w // window_size
    hidden_state = hidden_state.view(B, n_h, window_size, n_w, window_size, C)
    windows = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (padded_h, padded_w)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    """Reverse window partition. (B*nW,ws,ws,C) -> (B,H,W,C)."""
    padded_h, padded_w = pad_hw
    H, W = hw
    n_h = padded_h // window_size
    n_w = padded_w // window_size
    B = windows.shape[0] // (n_h * n_w)
    hidden_state = windows.view(B, n_h, n_w, window_size, window_size, -1)
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous()
    hidden_state = hidden_state.view(B, padded_h, padded_w, -1)
    return hidden_state[:, :H, :W, :].contiguous()


class TtnnMultiScaleAttention:
    """Matches HF Sam2MultiScaleAttention — qkv linear + SDPA + optional query pooling.
    
    NOTE: Attention math runs on CPU for now.
    TODO: Port to ttnn.transformer.scaled_dot_product_attention once hardware CI available.
    The ttnn SDPA API (from qwen3_vl) is: 
      ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
    """

    def __init__(self, dim: int, dim_out: int, num_heads: int,
                 query_stride: Optional[Tuple[int, int]], device: ttnn.Device,
                 state_dict: Optional[dict] = None, prefix: str = ""):
        self.device = device
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.query_stride = query_stride

        def _load(name, shape):
            key = f"{prefix}.{name}"
            if state_dict and key in state_dict:
                return state_dict[key]
            return torch.randn(shape)

        self.qkv_w = _load("qkv.weight", (dim_out * 3, dim)).T.contiguous()
        self.qkv_b = _load("qkv.bias", (dim_out * 3,))
        self.proj_w = _load("proj.weight", (dim_out, dim_out)).T.contiguous()
        self.proj_b = _load("proj.bias", (dim_out,))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward. Input/Output: [B, N, C] flattened.
        Currently on CPU — TODO: port to device with ttnn.linear + ttnn.SDPA."""
        B, N, C = hidden_states.shape
        H = W = int(math.sqrt(N))

        # QKV projection (torch for now)
        qkv = hidden_states @ self.qkv_w.to(hidden_states.dtype) + self.qkv_b.to(hidden_states.dtype)
        qkv = qkv.view(B, H, W, 3, self.num_heads, -1)
        q, k, v = qkv[:, :, :, 0, :, :], qkv[:, :, :, 1, :, :], qkv[:, :, :, 2, :, :]

        # Attention
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = torch.nn.functional.softmax(attn, dtype=torch.float32, dim=-1)
        out = (attn.to(q.dtype) @ v).transpose(2, 3)  # [B, H, W, nH, hd]

        # Query pooling
        out_2d = out.reshape(B, H, W, -1)
        if self.query_stride is not None:
            out_2d = do_pool(out_2d, self.query_stride)
            H_new, W_new = out_2d.shape[1], out_2d.shape[2]
        else:
            H_new, W_new = H, W

        # Output projection
        out = out_2d.reshape(B, H_new * W_new, -1)
        out = out @ self.proj_w.to(out.dtype) + self.proj_b.to(out.dtype)
        return out


class TtnnFeedForward:
    """Matches HF Sam2FeedForward — proj_in + GELU + proj_out (2 layer MLP).
    TODO: Port to ttnn.linear + ttnn.gelu once tested on hardware."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = "gelu",
                 device: ttnn.Device = None, state_dict: Optional[dict] = None, prefix: str = ""):
        self.device = device
        self.activation_name = activation

        def _load(name, shape):
            key = f"{prefix}.{name}"
            if state_dict and key in state_dict:
                return state_dict[key]
            return torch.randn(shape)

        self.in_w = _load("mlp.proj_in.weight", (hidden_dim, input_dim)).T.contiguous()
        self.in_b = _load("mlp.proj_in.bias", (hidden_dim,))
        self.out_w = _load("mlp.proj_out.weight", (output_dim, hidden_dim)).T.contiguous()
        self.out_b = _load("mlp.proj_out.bias", (output_dim,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through MLP on CPU."""
        x = x @ self.in_w.to(x.dtype) + self.in_b.to(x.dtype)
        x = torch.nn.functional.gelu(x)
        x = x @ self.out_w.to(x.dtype) + self.out_b.to(x.dtype)
        return x


class TtnnMultiScaleBlock:
    """Matches HF Sam2MultiScaleBlock — layernorm → window_attn → residual → layernorm → mlp → residual."""

    def __init__(self, dim: int, dim_out: int, num_heads: int, window_size: int,
                 query_stride: Optional[Tuple[int, int]], mlp_ratio: float,
                 activation: str, device: ttnn.Device, state_dict: Optional[dict] = None, prefix: str = ""):
        self.device = device
        self.dim = dim
        self.dim_out = dim_out
        self.window_size = window_size
        self.query_stride = query_stride

        def _load(name, shape):
            key = f"{prefix}.{name}"
            if state_dict and key in state_dict:
                return state_dict[key]
            return torch.randn(shape)

        self.ln1_w = _load("layer_norm1.weight", (dim,))
        self.ln1_b = _load("layer_norm1.bias", (dim,))
        self.ln2_w = _load("layer_norm2.weight", (dim_out,))
        self.ln2_b = _load("layer_norm2.bias", (dim_out,))

        self.attn = TtnnMultiScaleAttention(dim, dim_out, num_heads, query_stride, device, state_dict, prefix)
        mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = TtnnFeedForward(dim_out, mlp_hidden, dim_out, activation, device, state_dict, prefix)

        if dim != dim_out:
            self.proj_w = _load("proj.weight", (dim_out, dim)).T.contiguous()
            self.proj_b = _load("proj.bias", (dim_out,))

    def forward(self, hidden_states: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Forward with (B, N, C) flattened input, returns (B, N', C) flattened output.
        All on CPU — TODO: port to device operations."""
        # LayerNorm1
        hidden_states = torch.nn.functional.layer_norm(
            hidden_states, (self.dim,), self.ln1_w, self.ln1_b)
        hidden_states_2d = hidden_states.view(-1, H, W, self.dim)

        # Residual projection
        residual = hidden_states_2d
        if self.dim != self.dim_out:
            r = hidden_states_2d @ self.proj_w.to(hidden_states_2d.dtype) + self.proj_b.to(hidden_states_2d.dtype)
            residual = do_pool(r, self.query_stride)

        # Window partition
        ws = self.window_size
        pad_hw = None
        if ws > 0:
            hidden_states_2d, pad_hw = window_partition(hidden_states_2d, ws)
            sh = hidden_states_2d.shape
            hidden_states_pt = hidden_states_2d.view(sh[0], sh[1] * sh[2], sh[3])
        else:
            hidden_states_pt = hidden_states_2d.view(-1, H * W, self.dim)

        # Attention
        attn_out = self.attn.forward(hidden_states_pt)

        # Window unpartition
        B_total = attn_out.shape[0]
        if ws > 0:
            attn_out_windows = attn_out.view(B_total, ws, ws, -1)
            hidden_states = window_unpartition(attn_out_windows, ws, pad_hw, (H, W))
        else:
            new_dim = attn_out.shape[-1]
            new_H, new_W = H, W
            hidden_states = attn_out.view(-1, new_H, new_W, new_dim)

        new_H, new_W = hidden_states.shape[1], hidden_states.shape[2]
        hidden_states = residual + hidden_states

        # LayerNorm2 + MLP + Residual2
        ln_out = torch.nn.functional.layer_norm(
            hidden_states, (self.dim_out,), self.ln2_w, self.ln2_b)
        mlp_out = self.mlp.forward(ln_out.reshape(-1, new_H * new_W, self.dim_out))
        hidden_states = hidden_states + mlp_out.reshape(-1, new_H, new_W, self.dim_out)

        return hidden_states.reshape(-1, new_H * new_W, self.dim_out)


class Sam2HieraImageEncoderTT:
    """TTNN native Hiera image encoder matching Sam2HieraDetModel.
    12 blocks, windowed/global attention, 4 stages, query pooling at stage boundaries.
    
    NOTE: Currently runs on CPU via torch for correctness validation.
    TODO: Port each block to TTNN ops (ttnn.linear, ttnn.SDPA, ttnn.layer_norm, ttnn.gelu)
    once hardware CI is available for validation.
    """

    def __init__(
        self,
        device: ttnn.Device,
        config: dict,
        state_dict: Optional[dict] = None,
    ):
        self.device = device
        self.config = config

        embed_dim_per_stage = config.get("embed_dim_per_stage", [96, 192, 384, 768])
        blocks_per_stage = config.get("blocks_per_stage", [1, 2, 7, 2])
        window_size_per_stage = config.get("window_size_per_stage", [8, 4, 14, 7])
        num_heads_per_stage = config.get("num_attention_heads_per_stage", [1, 2, 4, 8])
        global_attention_blocks = config.get("global_attention_blocks", [5, 7, 9])
        mlp_ratio = config.get("mlp_ratio", 4.0)
        hidden_act = config.get("hidden_act", "gelu")
        query_stride = config.get("query_stride", [2, 2])
        num_query_pool_stages = config.get("num_query_pool_stages", 3)
        num_channels = config.get("num_channels", 3)
        hidden_size = config.get("hidden_size", 96)

        patch_kernel = config.get("patch_kernel_size", [7, 7])
        patch_stride = config.get("patch_stride", [4, 4])
        patch_padding = config.get("patch_padding", [3, 3])

        def _load(prefix, shape):
            if state_dict and prefix in state_dict:
                return state_dict[prefix]
            return torch.randn(shape)

        # Patch embedding weights (CPU ops for now)
        # TODO: Port to ttnn.conv2d with prepare_conv_params pattern
        # See models/demos/stable_diffusion_xl_base/tt/tt_downsample2d.py for reference.
        # ttnn.conv2d expects NHWC input ([B,H,W,C]), returns [result, [H,W], [weights,bias]],
        # and requires conv_config + compute_config from model_config.
        self.patch_w = _load("vision_encoder.backbone.patch_embed.projection.weight",
                             (hidden_size, num_channels, patch_kernel[0], patch_kernel[1]))
        self.patch_b = _load("vision_encoder.backbone.patch_embed.projection.bias", (hidden_size,))
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.patch_kernel = patch_kernel
        self.hidden_size = hidden_size

        # Positional embeddings (CPU torch tensors — only used at init, uploaded to device)
        pos_embed = _load("vision_encoder.backbone.pos_embed", (1, hidden_size, 7, 7))
        pos_embed_window = _load("vision_encoder.backbone.pos_embed_window", (1, hidden_size, 8, 8))
        self.pos_embed = pos_embed
        self.pos_embed_window = pos_embed_window

        # Build 12 blocks
        self.blocks = []
        self.stage_ends = (np.cumsum(blocks_per_stage) - 1).tolist()
        total_block_idx = 0

        for stage_idx, bps in enumerate(blocks_per_stage):
            for block_idx in range(bps):
                if stage_idx > 0 and block_idx == 0:
                    dim = embed_dim_per_stage[stage_idx - 1]
                else:
                    dim = embed_dim_per_stage[stage_idx]
                dim_out = embed_dim_per_stage[stage_idx]

                if stage_idx > 0 and block_idx == 0:
                    ws = window_size_per_stage[stage_idx - 1]
                else:
                    ws = window_size_per_stage[stage_idx]
                ws = 0 if total_block_idx in global_attention_blocks else ws

                qs = (query_stride if 0 < stage_idx <= num_query_pool_stages and block_idx == 0 else None)

                prefix = f"vision_encoder.backbone.blocks.{total_block_idx}"
                block = TtnnMultiScaleBlock(
                    dim=dim, dim_out=dim_out,
                    num_heads=num_heads_per_stage[stage_idx],
                    window_size=ws, query_stride=tuple(qs) if qs else None,
                    mlp_ratio=mlp_ratio, activation=hidden_act,
                    device=device, state_dict=state_dict, prefix=prefix,
                )
                self.blocks.append(block)
                total_block_idx += 1

    def _get_pos_embed(self, H: int, W: int) -> torch.Tensor:
        """Interpolate positional embedding to target spatial size (on CPU)."""
        pos_interp = torch.nn.functional.interpolate(self.pos_embed, size=(H, W), mode="bicubic")
        tile_h = H // self.pos_embed_window.shape[2]
        tile_w = W // self.pos_embed_window.shape[3]
        tiled = self.pos_embed_window.repeat(1, 1, tile_h, tile_w)
        pos_interp = pos_interp + tiled[:, :, :H, :W]
        return pos_interp  # [B, C, H, W]

    def forward(self, pixel_values: torch.Tensor) -> Dict:
        """Forward through backbone. Returns dict with 'last_hidden_state' and 'intermediate_hidden_states'.
        CPU-only — TODO: port to TTNN ops.
        Matches HF Sam2HieraDetModel.forward()."""
        # Patch embedding via torch conv2d
        # TODO: Replace with ttnn.conv2d with NHWC layout and prepare_conv_params
        hs = torch.nn.functional.conv2d(
            pixel_values, self.patch_w, bias=self.patch_b,
            stride=self.patch_stride, padding=self.patch_padding,
        )  # [B, C, H, W]
        hs = hs.permute(0, 2, 3, 1)  # NCHW -> NHWC: [B, H, W, C]

        # Add position embedding
        B, H, W, C = hs.shape
        pos_emb = self._get_pos_embed(H, W).permute(0, 2, 3, 1)
        hs = hs + pos_emb.to(hs.device, hs.dtype)

        intermediate_hidden_states = []
        for i, block in enumerate(self.blocks):
            hs = block.forward(hs, H, W)
            new_n = hs.shape[1]
            new_h = int(math.sqrt(new_n))
            H, W = new_h, new_h

            if i in self.stage_ends:
                intermediate_hidden_states.append(hs.view(-1, H, W, hs.shape[-1]))

        return {
            "last_hidden_state": hs.view(-1, H, W, hs.shape[-1]),
            "intermediate_hidden_states": tuple(intermediate_hidden_states),
        }
