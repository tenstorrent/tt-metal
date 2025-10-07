# SPDX-FileCopyrightText: Copyright (c) 2025 Motif Technologies

# SPDX-License-Identifier: MIT License

"""Diffusion Transformer (MMDiT) backbone and attention modules for Motif Image 6B Preview.

This module defines the transformer blocks for multi-modal text–image conditioning, including
patch projection, text/time conditioning, SD3-style joint attention, and the final head that
predicts latent-space velocity for rectified flow sampling.
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from loguru import logger

try:
    motif_ops = torch.ops.motif
    MotifRMSNorm = motif_ops.T5LayerNorm
    ScaledDotProductAttention = None
    MotifFlashAttention = motif_ops.flash_attention
except Exception:  # if motif_ops is not available
    MotifRMSNorm = None
    ScaledDotProductAttention = None
    MotifFlashAttention = None

NUM_MODULATIONS = 6
SD3_LATENT_CHANNEL = 16
LOW_RES_POSEMB_BASE_SIZE = 16
HIGH_RES_POSEMB_BASE_SIZE = 64


class IdentityConv2d(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self._initialize_identity()

    def _initialize_identity(self):
        k = self.conv.kernel_size[0]

        nn.init.zeros_(self.conv.weight)

        center = k // 2
        for i in range(self.conv.in_channels):
            self.conv.weight.data[i, i, center, center] = 1.0

        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.mask = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float)
        if self.mask is not None:
            hidden_states = self.mask.to(hidden_states.device).to(hidden_states.dtype) * hidden_states
        variance = hidden_states.pow(2).sum(-1, keepdim=True)
        if self.mask is not None:
            variance /= torch.count_nonzero(self.mask)
        else:
            variance /= hidden_states.shape[-1]
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            self.input_size, self.hidden_size = input_size, input_size * 4
        else:
            self.input_size, self.hidden_size = input_size, hidden_size

        self.gate_proj = nn.Linear(self.input_size, self.hidden_size)
        self.down_proj = nn.Linear(self.hidden_size, self.input_size)

        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.act_fn(self.gate_proj(x))
        down_proj = self.down_proj(down_proj)

        return down_proj


class TextTimeEmbToGlobalParams(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(emb_dim, hidden_dim * NUM_MODULATIONS)

    def forward(self, emb):
        emb = F.silu(emb)  # emb: B x D
        params = self.projection(emb)  # emb: B x C
        params = params.reshape(params.shape[0], NUM_MODULATIONS, params.shape[-1] // NUM_MODULATIONS)  # emb: B x 6 x C
        return params.chunk(6, dim=1)  # [B x 1 x C] x 6


class TextTimeEmbedding(nn.Module):
    """
    Input:
        pooled_text_emb (B x C_l)
        time_steps (B)

    Output:
        ()
    """

    def __init__(self, time_channel, text_channel, embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0):
        super().__init__()
        self.time_proj = Timesteps(
            time_channel, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=downscale_freq_shift
        )
        self.time_emb = TimestepEmbedding(time_channel, time_channel * 4, out_dim=embed_dim)  # Encode time emb with MLP
        self.pooled_text_emb = TimestepEmbedding(
            text_channel, text_channel * 4, out_dim=embed_dim
        )  # Encode pooled text with MLP

    def forward(self, pooled_text_emb, time_steps):
        time_steps = self.time_proj(time_steps)
        time_emb = self.time_emb(time_steps)
        pooled_text_emb = self.pooled_text_emb(pooled_text_emb)

        return time_emb + pooled_text_emb


class LatentPatchModule(nn.Module):
    def __init__(self, patch_size, embedding_dim, latent_channels, vae_type):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.projection_SD3 = nn.Conv2d(SD3_LATENT_CHANNEL, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.latent_channels = latent_channels

    def forward(self, x):
        assert x.shape[1] == SD3_LATENT_CHANNEL, (
            f"VAE-Latent channel is not matched with '{SD3_LATENT_CHANNEL}'. current shape: {x.shape}"
        )
        patches = self.projection_SD3(x)  # Shape: (B, embedding_dim, num_patches_h, num_patches_w)
        patches = patches.contiguous()
        patches = patches.flatten(2)  # Shape: (B, embedding_dim, num_patches)

        patches = patches.transpose(1, 2)  # Shape: (B, num_patches, embedding_dim)
        patches = patches.contiguous()
        return patches

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        n = x.shape[0]
        c = self.latent_channels
        p = self.patch_size

        # check the valid patching
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.contiguous()
        # (N x T x [C * patch_size**2]) -> (N x H x W x P_1 x P_2 x C)
        x = x.reshape(shape=(n, h, w, p, p, c))
        # x = torch.einsum('nhwpqc->nchpwq', x)  # Note that einsum possibly be the problem.

        # (N x H x W x P_1 x P_2 x C) -> (N x C x H x P_1 x W x P_2)
        # (0 . 1 . 2 .  3 .   4 .  5) -> (0 . 5 . 1 .  3    2 .  4 )
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(shape=(n, c, h * p, h * p)).contiguous()


class TextConditionModule(nn.Module):
    def __init__(self, text_dim, latent_dim):
        super().__init__()
        self.projection = nn.Linear(text_dim, latent_dim)

    def forward(self, t5_xxl, clip_a, clip_b):
        clip_emb = torch.cat([clip_a, clip_b], dim=-1)
        clip_emb = torch.nn.functional.pad(clip_emb, (0, t5_xxl.shape[-1] - clip_emb.shape[-1]))
        text_emb = torch.cat([clip_emb, t5_xxl], dim=-2)
        text_emb = self.projection(text_emb)
        return text_emb


class MotifDiTBlock(nn.Module):
    def __init__(self, emb_dim, t_emb_dim, attn_emb_dim, mlp_dim, attn_config, text_dim=4096):
        super().__init__()
        self.affine_params_c = TextTimeEmbToGlobalParams(t_emb_dim, emb_dim)
        self.affine_params_x = TextTimeEmbToGlobalParams(t_emb_dim, emb_dim)

        self.norm_1_c = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm_1_x = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.linear_1_c = nn.Linear(emb_dim, attn_emb_dim)
        self.linear_1_x = nn.Linear(emb_dim, attn_emb_dim)

        self.attn = JointAttn(attn_config)
        self.norm_2_c = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm_2_x = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.mlp_3_c = MLP(emb_dim, mlp_dim)
        self.mlp_3_x = MLP(emb_dim, mlp_dim)

    def forward(self, x_emb, c_emb, t_emb, perturbed=False):
        """
        x_emb (N, TOKEN_LENGTH x 2, C)
        c_emb (N, T + REGISTER_TOKENS, C)
        t_emb (N, modulation_dim)
        """

        device = x_emb.device

        # get global affine transformation parameters
        alpha_x, beta_x, gamma_x, delta_x, epsilon_x, zeta_x = self.affine_params_x(t_emb)  # scale and shift for image
        alpha_c, beta_c, gamma_c, delta_c, epsilon_c, zeta_c = self.affine_params_c(t_emb)  # scale and shift for text

        # projection and affine transform before attention
        x_emb_pre_attn = self.linear_1_x((1 + alpha_x) * self.norm_1_x(x_emb) + beta_x)
        c_emb_pre_attn = self.linear_1_c((1 + alpha_c) * self.norm_1_c(c_emb) + beta_c)

        # attn_output, attn_weight (None), past_key_value (None)
        x_emb_post_attn, c_emb_post_attn = self.attn(
            x_emb_pre_attn, c_emb_pre_attn, perturbed
        )  # mixed feature for both text and image (N, [T_x + T_c], C)

        # scale with gamma and residual with the original inputs
        x_emb_post_attn = x_emb_post_attn.to(gamma_x.device)
        x_emb_post_attn = (1 + gamma_x) * x_emb_post_attn + x_emb  # NOTE: nan loss for self.linear_2_x.bias
        c_emb_post_attn = c_emb_post_attn.to(gamma_c.device)
        c_emb_post_attn = (1 + gamma_c) * c_emb_post_attn + c_emb

        # norm the features -> affine transform with modulation -> MLP
        normalized_x_emb = self.norm_2_x(x_emb_post_attn).to(delta_x.device)
        normalized_c_emb = self.norm_2_c(c_emb_post_attn).to(delta_c.device)
        x_emb_final = self.mlp_3_x(delta_x * normalized_x_emb + epsilon_x)
        c_emb_final = self.mlp_3_c(delta_c * normalized_c_emb + epsilon_c)

        # final scaling with zeta and residual with the original inputs
        x_emb_final = zeta_x.to(device) * x_emb_final.to(device) + x_emb.to(device)
        c_emb_final = zeta_c.to(device) * c_emb_final.to(device) + c_emb.to(device)

        return x_emb_final, c_emb_final


class MotifDiT(nn.Module):
    """Multi-modal Diffusion Transformer backbone.

    Takes latent patches, text embeddings (sequence + pooled), and discrete timesteps, then
    produces the predicted velocity in the latent space used by the rectified-flow sampler.

    Args:
        config: Model configuration with architectural hyperparameters and options (e.g. attention mode).

    Forward inputs:
        latent: Latent tensor [B, C=16, H/vae, W/vae].
        t: Discrete timestep indices [B].
        text_embs: List of encoder sequence embeddings [T5, CLIP-L, CLIP-G].
        pooled_text_embs: Concatenated pooled CLIP embeddings [B, 2048].

    Returns:
        Tensor: Predicted velocity d(x)/dt in latent space with shape [B, 16, H/vae, W/vae].
    """

    ENCODED_TEXT_DIM = 4096

    def __init__(self, config):
        super(MotifDiT, self).__init__()
        self.patch_size = config.patch_size
        self.h, self.w = config.height // config.vae_compression, config.width // config.vae_compression

        self.latent_chennels = 16

        # Embedding for (1) text; (2) input image; (3) time
        self.text_cond = TextConditionModule(self.ENCODED_TEXT_DIM, config.hidden_dim)
        self.patching = LatentPatchModule(config.patch_size, config.hidden_dim, self.latent_chennels, config.vae_type)
        self.time_emb = TextTimeEmbedding(config.time_embed_dim, config.pooled_text_dim, config.modulation_dim)

        # main multi-modal DiT blocks
        self.mmdit_blocks = nn.ModuleList(
            [
                MotifDiTBlock(
                    config.hidden_dim, config.modulation_dim, config.hidden_dim, config.mlp_hidden_dim, config
                )
                for layer_idx in range(config.num_layers)
            ]
        )

        self.final_modulation = nn.Linear(config.modulation_dim, config.hidden_dim * 2)
        self.final_linear_SD3 = nn.Linear(config.hidden_dim, SD3_LATENT_CHANNEL * config.patch_size**2)
        self.skip_register_token_num = config.skip_register_token_num

        if getattr(config, "pos_emb_size", None):
            pos_emb_size = config.pos_emb_size
        else:
            pos_emb_size = HIGH_RES_POSEMB_BASE_SIZE if config.height > 512 else LOW_RES_POSEMB_BASE_SIZE
        logger.info(f"Positional embedding of Motif-DiT is set to {pos_emb_size}")

        self.pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(
                config.hidden_dim, (self.h // self.patch_size, self.w // self.patch_size), base_size=pos_emb_size
            )
        )

        # set register tokens (https://arxiv.org/abs/2309.16588)
        if config.register_token_num > 0:
            self.register_token_num = config.register_token_num
            self.register_tokens = nn.Parameter(torch.randn(1, self.register_token_num, config.hidden_dim))
            self.register_parameter("register_tokens", self.register_tokens)

            # if needed, add additional register tokens for higher resolution training
            self.additional_register_token_num = config.additional_register_token_num
            if config.additional_register_token_num > 0:
                self.register_tokens_highres = nn.Parameter(
                    torch.randn(1, self.additional_register_token_num, config.hidden_dim)
                )
                self.register_parameter("register_tokens_highres", self.register_tokens_highres)

        if config.use_final_layer_norm:
            self.final_norm = nn.LayerNorm(config.hidden_dim)

        if config.conv_header:
            logger.info("use convolution header after de-patching")
            self.depatching_conv_header = IdentityConv2d(SD3_LATENT_CHANNEL)

        if config.use_time_token_in_attn:
            self.t_token_proj = nn.Linear(config.modulation_dim, config.hidden_dim)

    def forward(self, latent, t, text_embs: List[torch.Tensor], pooled_text_embs, guiding_feature=None):
        """
        latent (torch.Tensor)
        t (torch.Tensor)
        text_embs (List[torch.Tensor])
        pooled_text_embs (torch.Tensor)
        """
        # 1. get inputs for the MMDiT blocks
        emb_c = self.text_cond(*text_embs)  # (N, L, D), text conditions
        emb_t = self.time_emb(pooled_text_embs, t).to(emb_c.device)  # (N, D), time and pooled text conditions

        emb_x = (self.patching(latent) + self.pos_embed).to(
            emb_c.device
        )  # (N, T, D), where T = H*W / (patch_size ** 2), input latent patches

        # additional "register" tokens, to convey the global information and prevent high-norm abnormal patch
        # see https://openreview.net/forum?id=2dnO3LLiJ1
        if hasattr(self, "register_tokens"):
            if hasattr(self, "register_tokens_highres"):
                emb_x = torch.cat(
                    (
                        self.register_tokens_highres.expand(emb_x.shape[0], -1, -1),
                        self.register_tokens.expand(emb_x.shape[0], -1, -1),
                        emb_x,
                    ),
                    dim=1,
                )
            else:
                emb_x = torch.cat((self.register_tokens.expand(emb_x.shape[0], -1, -1), emb_x), dim=1)

        # time embedding into text embedding
        if hasattr(self, "t_token_proj"):
            t_token = self.t_token_proj(emb_t).unsqueeze(1)
            emb_c = torch.cat([emb_c, t_token], dim=1)  # (N, [T_c + 1], C)

        # 2. MMDiT Blocks
        for block_idx, block in enumerate(self.mmdit_blocks):
            emb_x, emb_c = block(emb_x, emb_c, emb_t)

            # accumulating the feature_similarity loss
            # TODO: add modeling_dit related test
            if hasattr(self, "num_feature_align_layers") and block_idx == self.num_feature_align_layers:
                self.feature_alignment_loss = self.feature_align_mlp(emb_x, guiding_feature)  # exclude register tokens

            # Remove the register tokens at the certain layer (the last layer as default).
            if block_idx == len(self.mmdit_blocks) - (1 + self.skip_register_token_num):
                if hasattr(self, "register_tokens_highres"):
                    emb_x = emb_x[
                        :, self.register_token_num + self.additional_register_token_num :
                    ]  # remove the register tokens for the output layer
                elif hasattr(self, "register_tokens"):
                    emb_x = emb_x[:, self.register_token_num :]  # remove the register tokens for the output layer

        # 3. final modulation (shift-and-scale)
        scale, shift = self.final_modulation(emb_t).chunk(2, -1)  # (N, D) x 2
        scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)  # (N, 1, D) x 2

        if hasattr(self, "final_norm"):
            emb_x = self.final_norm(emb_x)

        final_emb = (scale + 1) * emb_x + shift

        # 4. final linear layer to reduce channel and un-patching
        emb_x = self.final_linear_SD3(final_emb)  # (N, T, D) to (N, T, out_channels * patch_size**2)
        emb_x = self.patching.unpatchify(emb_x)  # (N, out_channels, H, W)

        if hasattr(self, "depatching_conv_header"):
            emb_x = self.depatching_conv_header(emb_x)
        return emb_x


class JointAttn(nn.Module):
    """SD3-style joint attention that mixes image and text tokens.

    This layer computes attention over the concatenation of image queries/keys/values and
    text queries/keys/values, then splits the outputs back into image and text streams.

    Inputs:
        hidden_states: Image token representations [B, Lx, D] or a 4D feature map [B, C, H, W].
        encoder_hidden_states: Text token representations [B, Lc, D] or [B, C, H, W].

    Returns:
        Tuple[Tensor, Tensor]: Updated (hidden_states, encoder_hidden_states) with the same rank as inputs.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        self.add_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.add_k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.add_v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.add_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm_x = MotifRMSNorm(self.head_dim) if MotifRMSNorm else RMSNorm(self.head_dim)
        self.k_norm_x = MotifRMSNorm(self.head_dim) if MotifRMSNorm else RMSNorm(self.head_dim)

        self.q_norm_c = MotifRMSNorm(self.head_dim) if MotifRMSNorm else RMSNorm(self.head_dim)
        self.k_norm_c = MotifRMSNorm(self.head_dim) if MotifRMSNorm else RMSNorm(self.head_dim)
        self.q_scale = nn.Parameter(torch.ones(self.num_heads))

        # Attention mode : {'sdpa', 'flash', None}
        self.attn_mode = config.attn_mode

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # `context` projections.
        query_c = self.add_q_proj(encoder_hidden_states)
        key_c = self.add_k_proj(encoder_hidden_states)
        value_c = self.add_v_proj(encoder_hidden_states)

        # head first
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.num_heads

        def norm_qk(x, f_norm):
            x = x.view(batch_size, -1, self.num_heads, head_dim)
            b, l, h, d_h = x.shape
            x = x.reshape(b * l, h, d_h)
            x = f_norm(x)
            return x.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)  # [b, h, l, d_h]

        query = norm_qk(query, self.q_norm_x)  # [b, h, l, d_h]
        key = norm_qk(key, self.k_norm_x)  # [b, h, l, d_h]
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)  # [b, h, l, d_h]

        query_c = norm_qk(query_c, self.q_norm_c) * self.q_scale.reshape(1, self.num_heads, 1, 1)  # [b, h, l_c, d]
        key_c = norm_qk(key_c, self.k_norm_c)  # [b, h, l_c, d]
        value_c = value_c.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)  # [b, h, l_c, d]

        # attention
        query = torch.cat([query, query_c], dim=2).contiguous()  # [b, h, l + l_c, d]
        key = torch.cat([key, key_c], dim=2).contiguous()  # [b, h, l + l_c, d]
        value = torch.cat([value, value_c], dim=2).contiguous()  # [b, h, l + l_c, d]

        # deprecated.
        hidden_states = self.joint_attention(batch_size, query, key, value, head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = self.o_proj(hidden_states)
        encoder_hidden_states = self.add_o_proj(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states

    def joint_attention(self, batch_size, query, key, value, head_dim):
        if self.attn_mode == "sdpa" and ScaledDotProductAttention is not None:
            # NOTE: SDPA does not support high-resolution (long-context).
            q_len = query.size(-2)
            masked_bias = torch.zeros((batch_size, self.num_heads, query.size(-2), key.size(-2)))

            query = query.transpose(1, 2).reshape(batch_size, q_len, self.hidden_size).contiguous()
            key = key.transpose(1, 2).reshape(batch_size, q_len, self.hidden_size).contiguous()
            value = value.transpose(1, 2).reshape(batch_size, q_len, self.hidden_size).contiguous()

            scale_factor = 1.0
            scale_factor /= float(self.head_dim) ** 0.5

            hidden_states = ScaledDotProductAttention(
                query,
                key,
                value,
                masked_bias,
                dropout_rate=0.0,
                training=self.training,
                attn_weight_scale_factor=scale_factor,
                num_kv_groups=1,
            )
        elif self.attn_mode == "flash" and MotifFlashAttention is not None:
            query = query.permute(0, 2, 1, 3).contiguous()  # [b, l + l_c, h, d]
            key = key.permute(0, 2, 1, 3).contiguous()  # [b, l + l_c, h, d]
            value = value.permute(0, 2, 1, 3).contiguous()  # [b, l + l_c, h, d]
            scale_factor = 1.0 / math.sqrt(self.head_dim)

            # NOTE (1): masking of motif flash-attention uses (`1`: un-mask, `0`: mask) and has [Batch, Seq] shape
            # NOTE (2): Q,K,V must be [Batch, Seq, Heads, Dim] and contiguous.
            mask = torch.ones((batch_size, query.size(-3)))
            hidden_states = MotifFlashAttention(
                query,
                key,
                value,
                padding_mask=mask,
                softmax_scale=scale_factor,
                causal=False,
            )
            hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * head_dim).contiguous()
        elif self.attn_mode == "flash" and MotifFlashAttention is None:
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
        else:
            raise ValueError(f"Invalid attention mode: {self.attn_mode}")

        return hidden_states


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb.astype(np.float32)
