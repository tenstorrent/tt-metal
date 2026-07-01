# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import ttnn

from models.experimental.audiox.tt.common import linear_weight, to_tt
from models.experimental.audiox.tt.continuous_transformer import TtContinuousTransformer


def _strip_prefix(sd: dict, prefix: str) -> dict:
    plen = len(prefix)
    return {k[plen:]: v for k, v in sd.items() if k.startswith(prefix)}


class TtDiffusionTransformer:
    """TTNN port of the AudioX DiT outer wrapper. Produces the same forward path
    as audiox/models/dit.py:DiffusionTransformer for the inference config used
    by AudioX (continuous transformer, prepend conditioning, no CFG/patching).

    Conv1d 1x1 layers are folded into ttnn.linear over the channel dim while
    the activations are in [B, T, C] form (equivalent to a kernel-1 conv on
    [B, C, T]), avoiding extra transposes around the residual."""

    def __init__(
        self,
        mesh_device,
        state_dict: dict,
        depth: int,
        num_heads: int,
        io_channels: int = 64,
        embed_dim: int = 1536,
        lazy_layers: bool | None = None,
    ):
        sd = state_dict
        self.mesh_device = mesh_device
        self.io_channels = io_channels
        self.embed_dim = embed_dim
        dim_heads = embed_dim // num_heads
        if lazy_layers is None:
            lazy_layers = os.getenv("AUDIOX_TT_LAZY_LAYERS", "0") == "1"

        # FourierFeatures stores weight as [fourier_dim/2, 1]; forward does x @ weight.T,
        # so transpose once at construction and use ttnn.linear directly.
        self.fourier_w = to_tt(sd["timestep_features.weight"].transpose(0, 1).contiguous(), mesh_device)

        self.te0_w = to_tt(linear_weight(sd["to_timestep_embed.0.weight"]), mesh_device)
        self.te0_b = to_tt(sd["to_timestep_embed.0.bias"], mesh_device)
        self.te2_w = to_tt(linear_weight(sd["to_timestep_embed.2.weight"]), mesh_device)
        self.te2_b = to_tt(sd["to_timestep_embed.2.bias"], mesh_device)

        self.tc0_w = to_tt(linear_weight(sd["to_cond_embed.0.weight"]), mesh_device)
        self.tc2_w = to_tt(linear_weight(sd["to_cond_embed.2.weight"]), mesh_device)

        # Conv1d 1x1 weight is [out, in, 1]; squeeze the kernel dim to get a [out, in] linear weight.
        self.pre_w = to_tt(linear_weight(sd["preprocess_conv.weight"].squeeze(-1)), mesh_device)
        self.post_w = to_tt(linear_weight(sd["postprocess_conv.weight"].squeeze(-1)), mesh_device)

        self.transformer = TtContinuousTransformer(
            mesh_device=mesh_device,
            state_dict=_strip_prefix(sd, "transformer."),
            depth=depth,
            dim_heads=dim_heads,
            cross_attend=True,
            dim_in=io_channels,
            dim_out=io_channels,
            lazy_layers=lazy_layers,
        )

    def deallocate(self) -> None:
        for tensor in (
            self.fourier_w,
            self.te0_w,
            self.te0_b,
            self.te2_w,
            self.te2_b,
            self.tc0_w,
            self.tc2_w,
            self.pre_w,
            self.post_w,
        ):
            ttnn.deallocate(tensor, force=True)
        self.transformer.deallocate()

    def __call__(self, x: ttnn.Tensor, t: ttnn.Tensor, cross_attn_cond: ttnn.Tensor) -> ttnn.Tensor:
        # x:    [B, C, T] (NCT, matching upstream)
        # t:    [B] or [B, 1]
        # cond: [B, S_cond, cond_token_dim]
        if len(t.shape) == 1:
            t = ttnn.unsqueeze(t, -1)

        f = ttnn.linear(t, self.fourier_w)
        f = ttnn.multiply(f, 2.0 * math.pi)
        fourier = ttnn.concat([ttnn.cos(f), ttnn.sin(f)], dim=-1)

        timestep = ttnn.linear(fourier, self.te0_w, bias=self.te0_b)
        timestep = ttnn.silu(timestep)
        timestep = ttnn.linear(timestep, self.te2_w, bias=self.te2_b)
        prepend = ttnn.unsqueeze(timestep, 1)

        cond = ttnn.linear(cross_attn_cond, self.tc0_w)
        cond = ttnn.silu(cond)
        cond = ttnn.linear(cond, self.tc2_w)

        x = ttnn.transpose(x, 1, 2)  # [B, T, C]
        x = ttnn.add(x, ttnn.linear(x, self.pre_w))

        out = self.transformer(x, prepend_embeds=prepend, context=cond)

        # Strip the prepended timestep token along the sequence dim.
        batch, seq, ch = out.shape
        out = ttnn.slice(out, [0, 1, 0], [batch, seq, ch])

        out = ttnn.add(out, ttnn.linear(out, self.post_w))
        out = ttnn.transpose(out, 1, 2)  # [B, C, T]
        return out
