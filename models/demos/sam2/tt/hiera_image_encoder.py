# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN native Sam2HieraImageEncoder matching HuggingFace modeling_sam2.py.
Implements 12 Sam2MultiScaleBlock with windowed/global attention, multi-scale attention,
and query pooling for stage transitions. Architecture matches exact HF reference.
"""

from typing import Dict, List, Optional, Tuple
import torch
import ttnn
import math
import numpy as np


def do_pool(x: torch.Tensor, query_stride: Optional[Tuple[int, int]]) -> torch.Tensor:
    """Max pool for query pooling at stage transitions. (B,H,W,C) -> (B,H',W',C) on CPU.
    NOTE: For production use ttnn.max_pool2d; here on CPU since shapes vary."""
    if query_stride is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = torch.nn.functional.max_pool2d(x, kernel_size=query_stride, stride=query_stride, ceil_mode=False)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    return x


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
    """Matches HF Sam2MultiScaleAttention — qkv linear + SDPA + optional query pooling."""

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

        self.qkv_w = ttnn.from_torch(
            _load("qkv.weight", (dim_out * 3, dim)).T.contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.qkv_b = ttnn.from_torch(
            _load("qkv.bias", (dim_out * 3,)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.proj_w = ttnn.from_torch(
            _load("proj.weight", (dim_out, dim_out)).T.contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.proj_b = ttnn.from_torch(
            _load("proj.bias", (dim_out,)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward on device. Returns attn_output [B,H,W,C]."""
        B, H, W, C = hidden_states.shape
        tt_x = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        # qkv projection
        qkv = ttnn.linear(tt_x, self.qkv_w, bias=self.qkv_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tt_x)

        # Reshape qkv on CPU for now (ttnn reshape limitations)
        qkv_pt = ttnn.to_torch(qkv)
        ttnn.deallocate(qkv)
        qkv_pt = qkv_pt.view(B, H * W, 3, self.num_heads, -1)
        q, k, v = qkv_pt[:, :, 0, :, :], qkv_pt[:, :, 1, :, :], qkv_pt[:, :, 2, :, :]

        # SDPA on CPU for now (matching HF eager path exactly)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = torch.nn.functional.softmax(attn, dtype=torch.float32, dim=-1)
        out = attn.to(q.dtype) @ v  # [B, H*W, nH, head_dim]
        out = out.transpose(1, 2).reshape(B, H, W, -1)

        # Query pooling
        if self.query_stride is not None:
            out = do_pool(out.reshape(B, H, W, -1), self.query_stride)
        else:
            out = out.reshape(B, H, W, -1)

        # Output projection
        tt_out = ttnn.from_torch(out.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        result = ttnn.linear(tt_out, self.proj_w, bias=self.proj_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tt_out)
        result_pt = ttnn.to_torch(result)
        ttnn.deallocate(result)
        return result_pt


class TtnnFeedForward:
    """Matches HF Sam2FeedForward — proj_in + GELU + proj_out (2 layer MLP)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: str = "gelu",
                 device: ttnn.Device = None, state_dict: Optional[dict] = None, prefix: str = ""):
        self.device = device
        self.activation_name = activation

        def _load(name, shape):
            key = f"{prefix}.{name}"
            if state_dict and key in state_dict:
                return state_dict[key]
            return torch.randn(shape)

        self.in_w = ttnn.from_torch(
            _load("mlp.proj_in.weight", (hidden_dim, input_dim)).T.contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.in_b = ttnn.from_torch(
            _load("mlp.proj_in.bias", (hidden_dim,)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.out_w = ttnn.from_torch(
            _load("mlp.proj_out.weight", (output_dim, hidden_dim)).T.contiguous(),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.out_b = ttnn.from_torch(
            _load("mlp.proj_out.bias", (output_dim,)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through MLP. Input/Output on CPU."""
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        x = ttnn.linear(tt_x, self.in_w, bias=self.in_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tt_x)

        # GELU on CPU (ttnn.gelu exists but needs testing)
        x_pt = ttnn.to_torch(x)
        ttnn.deallocate(x)
        x_pt = torch.nn.functional.gelu(x_pt)

        tt_x2 = ttnn.from_torch(x_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        out = ttnn.linear(tt_x2, self.out_w, bias=self.out_b, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(tt_x2)

        result = ttnn.to_torch(out)
        ttnn.deallocate(out)
        return result


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

        # LayerNorms
        ln1_w = _load("layer_norm1.weight", (dim,))
        ln1_b = _load("layer_norm1.bias", (dim,))
        ln2_w = _load("layer_norm2.weight", (dim_out,))
        ln2_b = _load("layer_norm2.bias", (dim_out,))
        self.ln1_w = ttnn.from_torch(ln1_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.ln1_b = ttnn.from_torch(ln1_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.ln2_w = ttnn.from_torch(ln2_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.ln2_b = ttnn.from_torch(ln2_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Attention
        self.attn = TtnnMultiScaleAttention(dim, dim_out, num_heads, query_stride, device, state_dict, prefix)

        # MLP
        mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = TtnnFeedForward(dim_out, mlp_hidden, dim_out, activation, device, state_dict, prefix)

        # Proj (skip connection dim mismatch)
        if dim != dim_out:
            self.proj_w = ttnn.from_torch(
                _load("proj.weight", (dim_out, dim)).T.contiguous(),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            )
            self.proj_b = ttnn.from_torch(
                _load("proj.bias", (dim_out,)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            )

    def forward(self, hidden_states: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Forward with (B, N, C) flattened input, returns (B, N', C) flattened output."""
        # LayerNorm1 on CPU
        hidden_states = torch.nn.functional.layer_norm(
            hidden_states, (self.dim,),
            self.ln1_w.to(torch.float32) if hasattr(self.ln1_w, 'to') else torch.ones(self.dim),
            self.ln1_b.to(torch.float32) if hasattr(self.ln1_b, 'to') else torch.zeros(self.dim),
        )
        hidden_states_2d = hidden_states.view(-1, H, W, self.dim)

        # Residual projection
        residual = hidden_states_2d
        if self.dim != self.dim_out:
            # proj + pool on CPU
            tt_r = ttnn.from_torch(hidden_states_2d.contiguous(), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=self.device)
            r = ttnn.linear(tt_r, self.proj_w, bias=self.proj_b, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(tt_r)
            r_pt = ttnn.to_torch(r)
            ttnn.deallocate(r)
            residual = do_pool(r_pt, self.query_stride)

        # Window partition
        ws = self.window_size
        pad_hw = None
        if ws > 0:
            hidden_states_2d, pad_hw = window_partition(hidden_states_2d, ws)
            sh = hidden_states_2d.shape
            hidden_states_pt = hidden_states_2d.view(sh[0], -1, sh[3])
        else:
            hidden_states_pt = hidden_states_2d.view(-1, H * W, self.dim)

        # Attention (will reshape internally)
        # Flatten windows for attn: [B*nW, ws, ws, C] -> [B*nW, ws*ws, C]
        if ws > 0:
            sh = hidden_states_2d.shape
            attn_in = hidden_states_2d.view(sh[0], sh[1] * sh[2], sh[3])
        else:
            attn_in = hidden_states_pt

        attn_out_flat = self.attn.forward(attn_in)
        B_total = attn_out_flat.shape[0]

        # Window unpartition
        if ws > 0:
            # Reshape attn output back to windows
            attn_out_windows = attn_out_flat.view(B_total, ws, ws, -1)
            hidden_states = window_unpartition(attn_out_windows, ws, pad_hw, (H, W))
        else:
            hidden_states = attn_out_flat.view(-1, H, W, attn_out_flat.shape[-1])

        # Residual 1
        new_H, new_W = hidden_states.shape[1], hidden_states.shape[2]
        hidden_states = residual + hidden_states

        # LayerNorm2 + MLP + Residual2
        ln_out = torch.nn.functional.layer_norm(
            hidden_states, (self.dim_out,),
            self.ln2_w.to(torch.float32) if hasattr(self.ln2_w, 'to') else torch.ones(self.dim_out),
            self.ln2_b.to(torch.float32) if hasattr(self.ln2_b, 'to') else torch.zeros(self.dim_out),
        )
        mlp_out = self.mlp.forward(ln_out.reshape(-1, new_H * new_W, self.dim_out))
        hidden_states = hidden_states + mlp_out.reshape(-1, new_H, new_W, self.dim_out)

        return hidden_states.reshape(-1, new_H * new_W, self.dim_out)


class Sam2HieraImageEncoderTT:
    """TTNN native Hiera image encoder matching Sam2HieraDetModel.
    12 blocks, windowed/global attention, 4 stages, query pooling at stage boundaries."""

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

        # Patch embedding: Conv2d(num_channels→hidden_size, k=7,s=4,p=3)
        patch_kernel = config.get("patch_kernel_size", [7, 7])
        patch_stride = config.get("patch_stride", [4, 4])
        patch_padding = config.get("patch_padding", [3, 3])

        def _load(prefix, shape):
            if state_dict and prefix in state_dict:
                return state_dict[prefix]
            return torch.randn(shape)

        self.patch_conv_w = ttnn.from_torch(
            _load("vision_encoder.backbone.patch_embed.projection.weight", (hidden_size, num_channels, patch_kernel[0], patch_kernel[1])),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.patch_conv_b = ttnn.from_torch(
            _load("vision_encoder.backbone.patch_embed.projection.bias", (hidden_size,)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
        self.patch_kernel = patch_kernel
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.hidden_size = hidden_size

        # Positional embeddings
        pos_embed = _load("vision_encoder.backbone.pos_embed", (1, hidden_size, 7, 7))
        pos_embed_window = _load("vision_encoder.backbone.pos_embed_window", (1, hidden_size, 8, 8))
        self.pos_embed = ttnn.from_torch(pos_embed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.pos_embed_window = ttnn.from_torch(pos_embed_window, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Build 12 blocks
        self.blocks = []
        self.stage_ends = (np.cumsum(blocks_per_stage) - 1).tolist()
        total_block_idx = 0

        for stage_idx, bps in enumerate(blocks_per_stage):
            for block_idx in range(bps):
                # dim logic: first block of stage takes dim from prev stage
                if stage_idx > 0 and block_idx == 0:
                    dim = embed_dim_per_stage[stage_idx - 1]
                else:
                    dim = embed_dim_per_stage[stage_idx]
                dim_out = embed_dim_per_stage[stage_idx]

                # window size: 0 for global attention blocks
                if stage_idx > 0 and block_idx == 0:
                    ws = window_size_per_stage[stage_idx - 1]
                else:
                    ws = window_size_per_stage[stage_idx]
                ws = 0 if total_block_idx in global_attention_blocks else ws

                # query stride: first block of stage 1,2,3 (not stage 0 which has no pool)
                qs = (query_stride if 0 < stage_idx <= num_query_pool_stages and block_idx == 0 else None)

                prefix = f"vision_encoder.backbone.blocks.{total_block_idx}"
                block = TtnnMultiScaleBlock(
                    dim=dim, dim_out=dim_out,
                    num_heads=num_heads_per_stage[stage_idx],
                    window_size=ws, query_stride=qs,
                    mlp_ratio=mlp_ratio, activation=hidden_act,
                    device=device, state_dict=state_dict, prefix=prefix,
                )
                self.blocks.append(block)
                total_block_idx += 1

    def _get_pos_embed(self, H: int, W: int) -> torch.Tensor:
        """Interpolate positional embedding to target spatial size (on CPU)."""
        pos = self.pos_embed
        pos_win = self.pos_embed_window
        # ttnn.interpolate doesn't exist, do on CPU
        pos_pt = ttnn.to_torch(pos)
        pos_win_pt = ttnn.to_torch(pos_win)
        pos_interp = torch.nn.functional.interpolate(pos_pt, size=(H, W), mode="bicubic")
        # tile window embedding
        tile_h = H // pos_win_pt.shape[2]
        tile_w = W // pos_win_pt.shape[3]
        tiled = pos_win_pt.repeat(1, 1, tile_h, tile_w)
        pos_interp = pos_interp + tiled[:, :, :H, :W]
        return pos_interp  # [B, C, H, W]

    def forward(self, pixel_values: torch.Tensor) -> Dict:
        """Forward through backbone. Returns dict with 'last_hidden_state' and 'intermediate_hidden_states'.
        Matches HF Sam2HieraDetModel.forward()."""
        # Patch embedding via ttnn.conv2d
        tt_pv = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        hidden_states = ttnn.conv2d(
            input_tensor=tt_pv,
            weight_tensor=self.patch_conv_w,
            bias_tensor=self.patch_conv_b,
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=tuple(self.patch_kernel),
            stride=tuple(self.patch_stride),
            padding=tuple(self.patch_padding),
            device=self.device,
        )
        ttnn.deallocate(tt_pv)
        # Conv2d output is [B, C, H, W]; HF expects [B, H, W, C]
        hs_pt = ttnn.to_torch(hidden_states)
        ttnn.deallocate(hidden_states)
        hs_pt = hs_pt.permute(0, 2, 3, 1)  # -> [B, H, W, C]

        # Add position embedding
        B, H, W, C = hs_pt.shape
        pos_emb = self._get_pos_embed(H, W).permute(0, 2, 3, 1)  # [B, H, W, C]
        hs_pt = hs_pt + pos_emb.to(hs_pt.device, hs_pt.dtype)

        intermediate_hidden_states = []
        for i, block in enumerate(self.blocks):
            hs_pt = block.forward(hs_pt, H, W)
            # After block, shape may have changed due to query pooling
            new_n = hs_pt.shape[1]
            new_h = int(math.sqrt(new_n))
            H, W = new_h, new_h

            if i in self.stage_ends:
                intermediate_hidden_states.append(hs_pt.view(-1, H, W, hs_pt.shape[-1]))

        return {
            "last_hidden_state": hs_pt.view(-1, H, W, hs_pt.shape[-1]),
            "intermediate_hidden_states": tuple(intermediate_hidden_states),
        }
