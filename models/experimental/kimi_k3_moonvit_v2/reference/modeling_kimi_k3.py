# SPDX-FileCopyrightText: © 2025-2026 The Moonshot AI Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
#
# Copied from huggingface.co/moonshotai/Kimi-K3. LICENSE-Kimi-K3 in this folder is "the LICENSE file
# in this repository" that the notice below refers to.
# Omitted from upstream (the language model): the LlavaCausalLMOutputWithPast and KimiLinearForCausalLM
# imports, KimiK3PreTrainedModel and KimiK3ForConditionalGeneration. Everything else is unchanged.

# coding=utf-8
# Copyright 2025-2026 The Moonshot AI Team and HuggingFace Inc. team. All rights reserved.
#
# The code is based on llava (llava/modeling_llava.py), but modified for Kimi-K3.
#
# Licensing Information:
# - Code derived from llava (llava/modeling_llava.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the Kimi K3 License (see the LICENSE file in this repository).
#
# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# NOTE: Reference implementation for model architecture; see the model card for production deployment.
import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import activations

try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh

    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh
from transformers.activations import PytorchGELUTanh
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_flash_attn_2_available

from .configuration_kimi_k3 import KimiK3Config

# Flash attention imports
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None


def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    deterministic: bool = False,
):
    """Multi-head attention using flash attention 2.

    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.
            The first element should be 0 and the last element should be q.shape[0].
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.
            The first element should be 0 and the last element should be k.shape[0].

    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
    """
    attn_out = flash_attn_varlen_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
        deterministic=deterministic,
    )
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]

    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight = attn_weight.masked_fill(~attention_mask, torch.finfo(attn_weight.dtype).min)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "eager": eager_attention,
}


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@get_rope_shape_decorate
@torch.compile(dynamic=True)
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From:
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    def __init__(self, height: int, width: int, num_frames: int, dim: int, interpolation_mode: str = "bicubic") -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames)).float().unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


class MoonVision3dPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        patch_embed_proj_bias: bool = True,
        pos_emb_interpolation_mode: str = "bicubic",
    ):
        super().__init__()
        assert isinstance(patch_size, int | Sequence), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size, bias=patch_embed_proj_bias)

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
                interpolation_mode=pos_emb_interpolation_mode,
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 3): temporal, height and width

        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws)
        return x


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, "freqs_cis"):
            self.register_buffer("freqs_cis", self._precompute_freqs_cis(device), persistent=False)

        shapes = grid_thws.tolist()
        assert all(1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1) for t, h, w in shapes],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonViTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        qkv_hidden_size: int | None = None,
        norm_type: str = "layernorm",
        mlp_type: str = "mlp2",
        *,
        attn_implementation: str = "flash_attention_2",
        activation=F.gelu,
        attn_bias: bool = False,
        linear_bias: bool = True,
        use_deterministic_attn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = hidden_dim if qkv_hidden_size is None else qkv_hidden_size
        self.hidden_size_per_attention_head = self.qkv_hidden_size // self.num_heads
        self.attn_implementation = attn_implementation
        self.use_deterministic_attn = use_deterministic_attn

        if norm_type == "layernorm":
            self.norm0 = nn.LayerNorm(hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            self.norm0 = nn.RMSNorm(hidden_dim)
            self.norm1 = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Not support norm_type: {norm_type}")

        if mlp_type == "mlp2":
            self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation, bias=linear_bias)
        else:
            raise NotImplementedError(f"Not support mlp_type: {mlp_type}")

        self.wqkv = nn.Linear(hidden_dim, self.qkv_hidden_size * 3, bias=attn_bias)
        self.wo = nn.Linear(self.qkv_hidden_size, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(
            xq,
            xk,
            xv,
            q_cu_seqlens=cu_seqlens,
            k_cu_seqlens=cu_seqlens,
            max_seqlen_k=max_seqlen,
            max_seqlen_q=max_seqlen,
            deterministic=self.use_deterministic_attn,
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attention_qkvpacked(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonViT3dEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, block_cfg: dict, use_deterministic_attn: bool = False) -> None:
        super().__init__()
        self.use_deterministic_attn = use_deterministic_attn

        qkv_hidden_size = (
            block_cfg["hidden_dim"] if block_cfg.get("qkv_hidden_size") is None else block_cfg["qkv_hidden_size"]
        )
        self.rope_2d = Rope2DPosEmbRepeated(qkv_hidden_size // block_cfg["num_heads"], 512, 512)
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(**block_cfg, use_deterministic_attn=self.use_deterministic_attn)
                for _ in range(num_layers)
            ]
        )
        norm_type = block_cfg.get("norm_type", "layernorm")
        if norm_type == "layernorm":
            self.final_layernorm = nn.LayerNorm(hidden_dim)
        elif norm_type == "rmsnorm":
            self.final_layernorm = nn.RMSNorm(hidden_dim)
        else:
            raise NotImplementedError(f"Not support norm_type: {norm_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws=grid_thws, device=hidden_states.device)

        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )

        max_seqlen = lengths.max()
        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0, dtype=torch.int32)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        # Get the current sequence
        seq = x[pre_sum : pre_sum + t * h * w]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)  # temporal pooling
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs


class MoonViT3dPretrainedModel(PreTrainedModel):
    config_class = None
    model_type = "moonvit3d"
    _no_split_modules = ["MoonViTEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
            patch_embed_proj_bias=getattr(config, "patch_embed_proj_bias", True),
            pos_emb_interpolation_mode=getattr(config, "pos_emb_interpolation_mode", "bicubic"),
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "qkv_hidden_size": getattr(config, "qkv_hidden_size", None),
                "mlp_dim": config.intermediate_size,
                "norm_type": getattr(config, "norm_type", "layernorm"),
                "mlp_type": getattr(config, "mlp_type", "mlp2"),
                "activation": PytorchGELUTanh(),
                "attn_bias": getattr(config, "attn_bias", True),
                "linear_bias": getattr(config, "linear_bias", True),
                "attn_implementation": config._attn_implementation,
            },
            use_deterministic_attn=getattr(self, "use_deterministic_attn", False),
        )

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): Temporal, height and width.

        Returns:
            torch.Tensor: The output tokens.
        """
        # grid_thws = grid_thws.to('cpu')
        assert grid_thws.ndim == 2, f"grid_thws should be 2D, got {grid_thws.ndim}"
        assert grid_thws.size(1) == 3, f"No support for thw: {grid_thws}"
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type == "sd2_tpool":  # spatial downsampling 2x with temporal pooling all
            hidden_states = tpool_patch_merger(hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size)
        else:
            raise NotImplementedError(f"Not support {self.merge_type}")

        return hidden_states


# ============================================================================
# MM Projector Helper Classes (from mm_projector/modeling_mm_projectors.py)
# ============================================================================


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO, use faster LayerNorm
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        assert isinstance(x, list | tuple), f"x is not a list or tuple: {type(x)}"
        lengths = [item.shape[0] for item in x]
        x = torch.cat(x, dim=0)
        x = self.pre_norm(x)
        x = self.proj(x)
        x = torch.split(x, lengths, dim=0)

        return x


class PatchMergerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        eps = config.projector_ln_eps
        self.hidden_size = config.mm_hidden_size * (config.merge_kernel_size[0] * config.merge_kernel_size[1])
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size, eps=eps)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list) or isinstance(x, tuple):
            x = [self.proj(self.pre_norm(item).view(item.shape[0], -1)) for item in x]
        else:
            # B, N, N_k, C = x.shape
            B = x.shape[0]
            x = self.proj(self.pre_norm(x).view(B, -1, self.hidden_size))
        return x


class PatchMergerMLPV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        eps = config.projector_ln_eps
        self.hidden_size = config.mm_hidden_size * (config.merge_kernel_size[0] * config.merge_kernel_size[1])
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(config.hidden_size, eps=eps)
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list) or isinstance(x, tuple):
            lengths = [item.shape[0] for item in x]
            x = torch.concat([item.view(item.shape[0], -1) for item in x], dim=0)
            x = self.post_norm(self.proj(x))
            x = torch.split(x, lengths, dim=0)
        else:
            # B, N, N_k, C = x.shape
            B = x.shape[0]
            x = self.proj(x.view(B, -1, self.hidden_size))
            x = self.post_norm(x)
        return x


class VisionTowerConfig(PretrainedConfig):
    model_type = "moonvit3d"

    def __init__(self, config: KimiK3Config, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = config.patch_size
        self.init_pos_emb_height = config.init_pos_emb_height
        self.init_pos_emb_width = config.init_pos_emb_width
        self.init_pos_emb_time = config.init_pos_emb_time
        self.pos_emb_type = config.pos_emb_type
        self.num_attention_heads = config.vt_num_attention_heads
        self.num_hidden_layers = config.vt_num_hidden_layers
        self.hidden_size = config.vt_hidden_size
        self.intermediate_size = config.vt_intermediate_size
        self.merge_kernel_size = config.merge_kernel_size
        self.merge_type = config.merge_type
        self._attn_implementation = config._attn_implementation
        self.qkv_hidden_size = getattr(config, "qkv_hidden_size", None)
        self.norm_type = getattr(config, "norm_type", "layernorm")
        self.attn_bias = getattr(config, "attn_bias", True)
        self.patch_embed_proj_bias = getattr(config, "patch_embed_proj_bias", True)
        self.mlp_type = getattr(config, "mlp_type", "mlp2")
        self.linear_bias = getattr(config, "linear_bias", True)
        self.pos_emb_interpolation_mode = getattr(config, "pos_emb_interpolation_mode", "bilinear")


class ProjectorConfig:
    def __init__(self, config: KimiK3Config):
        self.mm_projector_type = config.mm_projector_type
        self.mm_hidden_size = config.mm_hidden_size
        self.hidden_size = config.text_hidden_size
        self.merge_kernel_size = config.merge_kernel_size
        self.projector_hidden_act = config.projector_hidden_act
        self.projector_ln_eps = config.projector_ln_eps
