# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN DiT (Diffusion Transformer) estimator for CosyVoice3 flow decoder.

The DiT estimates the velocity field for the flow matching ODE solver.
It is called 10 times per inference (one per ODE step), with batch=2
for classifier-free guidance.

Architecture:
    TimestepEmbedding → InputEmbedding → 22× DiTBlock → AdaLN_Final → proj_out

For initial bring-up, input composition and timestep embedding run on host,
while the 22 DiT blocks run on device (the compute-hot path).
"""

import math

import torch
import torch.nn.functional as F
from einops import repeat
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.wormhole.cosy_voice.tt.flow.dit_modules import TtDiTBlock


class TtDiT(LightweightModule):
    """
    Full DiT estimator for CosyVoice3 flow decoder.

    For initial bring-up:
    - TimestepEmbedding runs on host (small MLP, called 10× total)
    - InputEmbedding runs on host (concat + linear + conv_pos, called 10×)
    - 22 DiT blocks run on DEVICE (bulk compute: 10 × 22 = 220 evaluations)
    - Final norm + projection runs on host (small, called 10×)
    """

    def __init__(
        self,
        device,
        state_dict,
        dtype=ttnn.bfloat16,
        dim=1024,
        depth=22,
        heads=16,
        dim_head=64,
        ff_mult=2,
        mel_dim=80,
        spk_dim=80,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mel_dim = mel_dim
        self.device = device

        # --- Host-side modules (small, run once per ODE step) ---

        # TimestepEmbedding: SinusoidalPosEmb(256) → Linear(256,1024) → SiLU → Linear(1024,1024)
        self.time_mlp_w1 = state_dict["decoder.estimator.time_embed.time_mlp.0.weight"]
        self.time_mlp_b1 = state_dict["decoder.estimator.time_embed.time_mlp.0.bias"]
        self.time_mlp_w2 = state_dict["decoder.estimator.time_embed.time_mlp.2.weight"]
        self.time_mlp_b2 = state_dict["decoder.estimator.time_embed.time_mlp.2.bias"]

        # InputEmbedding: proj Linear(320, 1024) + CausalConvPositionEmbedding
        self.input_proj_w = state_dict["decoder.estimator.input_embed.proj.weight"]
        self.input_proj_b = state_dict["decoder.estimator.input_embed.proj.bias"]

        # Conv position embedding (2 grouped causal convs)
        self.conv_pos_conv1_w = state_dict["decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight"]
        self.conv_pos_conv1_b = state_dict["decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias"]
        self.conv_pos_conv2_w = state_dict["decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight"]
        self.conv_pos_conv2_b = state_dict["decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias"]

        # Final norm + projection (host)
        self.norm_out_w = state_dict["decoder.estimator.norm_out.linear.weight"]
        self.norm_out_b = state_dict["decoder.estimator.norm_out.linear.bias"]
        self.proj_out_w = state_dict["decoder.estimator.proj_out.weight"]
        self.proj_out_b = state_dict["decoder.estimator.proj_out.bias"]

        # --- Device-side modules (22 DiT blocks — bulk compute) ---
        logger.info(f"Loading {depth} DiT blocks to device...")
        self.blocks = []
        for i in range(depth):
            prefix = f"decoder.estimator.transformer_blocks.{i}"
            block = TtDiTBlock(dim, heads, dim_head, ff_mult, device, state_dict, prefix, dtype)
            self.blocks.append(block)
        logger.info(f"DiT: {depth} blocks loaded")

    def _sinusoidal_pos_emb(self, x, dim=256):
        """Host-side sinusoidal position embedding."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = 1000 * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

    def _timestep_embedding(self, t):
        """Host-side timestep MLP: sin_pos_emb → Linear → SiLU → Linear."""
        h = self._sinusoidal_pos_emb(t).to(t.dtype)
        h = F.linear(h, self.time_mlp_w1, self.time_mlp_b1)
        h = F.silu(h)
        h = F.linear(h, self.time_mlp_w2, self.time_mlp_b2)
        return h  # (batch, dim)

    def _input_embedding(self, x, cond, mu, spks):
        """Host-side input embedding: concat → proj → causal_conv_pos + residual."""
        # x, cond, mu: (batch, seq, mel_dim=80), spks: (batch, spk_dim=80)
        spks_expanded = repeat(spks, "b c -> b t c", t=x.shape[1])
        inp = torch.cat([x, cond, mu, spks_expanded], dim=-1)  # (batch, seq, 320)
        x = F.linear(inp, self.input_proj_w, self.input_proj_b)  # (batch, seq, 1024)

        # Causal conv position embedding
        residual = x
        h = x.permute(0, 2, 1)  # (batch, 1024, seq)
        k = self.conv_pos_conv1_w.shape[2]  # kernel_size=31
        h = F.pad(h, (k - 1, 0))
        h = F.conv1d(h, self.conv_pos_conv1_w, self.conv_pos_conv1_b, groups=16)
        h = F.mish(h)
        h = F.pad(h, (k - 1, 0))
        h = F.conv1d(h, self.conv_pos_conv2_w, self.conv_pos_conv2_b, groups=16)
        h = F.mish(h)
        x = h.permute(0, 2, 1) + residual  # (batch, seq, 1024)
        return x

    def _final_norm_and_proj(self, x, t_emb):
        """Host-side final AdaLN + projection."""
        # AdaLayerNormZero_Final: SiLU → Linear(dim, dim*2) → split → modulate
        emb = F.silu(t_emb)
        emb = F.linear(emb, self.norm_out_w, self.norm_out_b)
        scale, shift = torch.chunk(emb, 2, dim=1)

        # LayerNorm (no affine)
        x = F.layer_norm(x, [self.dim])
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]

        # Project to mel dim
        x = F.linear(x, self.proj_out_w, self.proj_out_b)
        return x  # (batch, seq, mel_dim)

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        """
        Full DiT forward pass.

        Args:
            x: (batch, mel_dim, seq) - noised input
            mask: (batch, 1, seq) - mask
            mu: (batch, mel_dim, seq) - encoder output
            t: (batch,) - timestep
            spks: (batch, spk_dim) - speaker embedding
            cond: (batch, mel_dim, seq) - conditioning

        Returns:
            (batch, mel_dim, seq) - estimated velocity
        """
        # Transpose to (batch, seq, mel_dim) for host processing
        x_host = x.transpose(1, 2)
        mu_host = mu.transpose(1, 2)
        cond_host = cond.transpose(1, 2)
        spks_host = spks.unsqueeze(1) if spks.dim() == 2 else spks

        batch, seq_len = x_host.shape[0], x_host.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        # 1. Host: Timestep embedding
        t_emb = self._timestep_embedding(t)  # (batch, dim)

        # 2. Host: Input embedding
        x_host = self._input_embedding(x_host, cond_host, mu_host, spks_host.squeeze(1))  # (batch, seq, dim)

        # 3. Transfer to device for DiT blocks
        x_tt = ttnn.from_torch(
            x_host.unsqueeze(0),  # (1, batch, seq, dim)
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        t_tt = ttnn.from_torch(
            t_emb.unsqueeze(0).unsqueeze(0),  # (1, 1, batch, dim)
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        # 4. Device: Run 22 DiT blocks
        for block in self.blocks:
            x_tt = block(x_tt, t_tt)

        # 5. Transfer back to host for final norm + projection
        x_host = ttnn.to_torch(x_tt).squeeze(0).float()  # (batch, seq, dim)

        # 6. Host: Final AdaLN + projection
        output = self._final_norm_and_proj(x_host, t_emb)

        # Transpose back to (batch, mel_dim, seq)
        return output.transpose(1, 2)
