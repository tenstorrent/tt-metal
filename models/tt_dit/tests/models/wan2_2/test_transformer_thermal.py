# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Thermal and reliability stress test.

Usage:
    pytest test_transformer_thermal.py -s

    # 500 iterations:
    NUM_ITERATIONS=500 pytest test_transformer_thermal.py -s
"""

import math
import os
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn
from models.tt_dit.layers.embeddings import WanPatchEmbed, WanTimeTextImageEmbedding
from models.tt_dit.layers.linear import Linear, gelu_tanh
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import DistributedLayerNorm
from models.tt_dit.models.transformers.wan2_2.transformer_wan import WanTransformerBlock
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import register_conv3d_configs
from models.tt_dit.utils.matmul import FusedMMRSConfig, register_fused_mmrs_configs, register_matmul_configs
from models.tt_dit.utils.mochi import get_rot_transformation_mat, stack_cos_sin
from models.tt_dit.utils.padding import pad_vision_seq_parallel
from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor, from_torch, unflatten
from models.tt_dit.utils.test import ring_params

_MATMUL_11x10 = {
    (32, 32, 32): (1, 1, 1, (1, 1)),
    (32, 96, 192): (1, 2, 4, (1, 4)),
    (32, 192, 384): (1, 6, 12, (1, 4)),
    (32, 256, 5120): (1, 6, 10, (1, 1)),
    (32, 1280, 30720): (1, 20, 5, (1, 1)),
    (32, 3072, 10240): (1, 24, 2, (1, 2)),
    (32, 5120, 1280): (1, 32, 1, (1, 1)),
    (32, 10240, 10240): (1, 64, 3, (1, 3)),
    (64, 96, 192): (1, 2, 4, (1, 1)),
    (64, 192, 384): (1, 5, 5, (1, 1)),
    (96, 96, 192): (1, 3, 3, (1, 1)),
    (128, 5120, 2560): (1, 12, 20, (1, 2)),
    (512, 4096, 5120): (2, 32, 6, (2, 1)),
    (512, 5120, 5120): (2, 32, 5, (2, 1)),
    (6144, 384, 384): (10, 6, 4, (2, 2)),
    (6144, 384, 1152): (6, 12, 3, (1, 3)),
    (6144, 3456, 5120): (6, 12, 10, (1, 2)),
    (6144, 5120, 64): (12, 2, 2, (3, 1)),
    (6144, 5120, 3456): (6, 12, 8, (1, 1)),
    (6240, 384, 384): (6, 2, 2, (2, 2)),
    (6240, 384, 1152): (20, 3, 4, (1, 4)),
    (6240, 3456, 5120): (20, 6, 4, (1, 4)),
    (6240, 5120, 64): (3, 20, 2, (3, 1)),
    (6240, 5120, 3456): (10, 8, 4, (1, 4)),
    (14400, 384, 384): (9, 12, 3, (3, 1)),
    (14400, 384, 1152): (6, 12, 2, (1, 1)),
    (14400, 3456, 5120): (15, 12, 4, (1, 2)),
    (14400, 5120, 3456): (15, 20, 1, (3, 1)),
}

_MATMUL_12x9 = {
    (6144, 5120, 1280): (10, 8, 6, (2, 2)),
    (6144, 5120, 3456): (8, 5, 12, (4, 1)),
    (6144, 5120, 3840): (6, 5, 16, (1, 1)),
    (6240, 5120, 1280): (10, 8, 6, (2, 2)),
    (6240, 5120, 3456): (10, 4, 12, (1, 4)),
    (6240, 5120, 3840): (6, 5, 16, (1, 1)),
    (14400, 5120, 1280): (10, 8, 6, (2, 1)),
    (14400, 5120, 3456): (8, 5, 12, (1, 4)),
    (14400, 5120, 3840): (6, 5, 16, (3, 1)),
}

_FUSED_MMRS = {
    ttnn.CoreCoord(12, 10): {
        (14400, 3456, 5120): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 8, 4, 8, 2, 1, None, 1),
        (6144, 3456, 5120): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 8, 4, 8, 2, 1, None, 1),
        (4800, 3456, 5120): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 8, 4, 8, 2, 1, None, 1),
    },
}

_CONV3D = {
    (32, 96, (3, 3, 3)): (32, 96, 1, 8, 16),
    (96, 192, (3, 3, 3)): (96, 96, 1, 4, 8),
    (384, 32, (3, 3, 3)): (384, 32, 1, 1, 4),
    (192, 192, (3, 3, 3)): (96, 96, 1, 8, 8),
    (32, 384, (3, 3, 3)): (32, 384, 1, 8, 8),
    (192, 384, (3, 3, 3)): (96, 128, 1, 8, 4),
    (384, 384, (3, 3, 3)): (128, 96, 1, 8, 4),
    (384, 768, (3, 1, 1)): (384, 384, 1, 16, 4),
}

SDPA_CHUNK_SIZE_OVERRIDES = {
    (True, 8, 4): (224, 512),
}


def _register_configs():
    register_matmul_configs({"11x10": _MATMUL_11x10, "12x9": _MATMUL_12x9})
    register_fused_mmrs_configs(_FUSED_MMRS)
    register_conv3d_configs(_CONV3D)


_register_configs()


class _RotaryEmbeddingND:
    def __init__(self, head_dims, base, rotate_mode=1, dtype=torch.float32):
        self.head_dims = head_dims
        self.base = base
        self.dtype = dtype
        self.rotate_mode = rotate_mode

    def _get_freqs(self, head_dim, device, base_override=None):
        base = base_override if base_override is not None else self.base
        freqs = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)[: (head_dim // 2)] / head_dim)
        )
        return freqs

    def _get_rope_with_pos(self, pos, head_dim, device, base_override=None, no_repeat=False):
        freqs = self._get_freqs(head_dim, device=device, base_override=base_override)
        if not no_repeat:
            if self.rotate_mode == 0:
                freqs = torch.concatenate([freqs, freqs], dim=-1)
            else:
                freqs = torch.concatenate([freqs[..., None], freqs[..., None]], dim=-1).view(-1)
        freqs_pos = pos[:, None] * freqs[None, :]
        emb = torch.stack((torch.cos(freqs_pos), torch.sin(freqs_pos)), dim=-1)
        return emb.to(dtype=self.dtype)

    def __call__(self, pids, base_override=None, no_repeat=False):
        ndim = pids.shape[-1]
        assert ndim == len(self.head_dims)
        if base_override is None:
            base_override = [None] * len(self.head_dims)
        all_freqs = []
        for i in range(len(self.head_dims)):
            freqs = self._get_rope_with_pos(
                pids[..., i],
                self.head_dims[i],
                device=pids.device,
                base_override=base_override[i],
                no_repeat=no_repeat,
            )
            all_freqs.append(freqs)
        return torch.concatenate(all_freqs, dim=-2)


class _FourierFeatures(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.proj = nn.Linear(in_feats, out_feats // 2, bias=True)

    def forward(self, x):
        x = self.proj(x)
        x = torch.stack([torch.sin(x), torch.cos(x)], dim=-1)
        return x.reshape(*x.shape[:-2], -1)


class _FourierMLPPosEmb(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.d_out = d_out
        self.fourier = _FourierFeatures(in_feats=2, out_feats=d_out)
        self.linear1 = nn.Linear(d_out, d_out, bias=True)
        self.linear2 = nn.Linear(d_out, d_out, bias=True)
        self.norm_out_weight = nn.Parameter(torch.ones(d_out))
        self.proj_out = nn.Linear(d_out, d_out, bias=True)

    def forward(self, grid_size, coords):
        h, w = grid_size
        start_h, end_h, start_w, end_w = coords
        hs = torch.linspace(start_h, end_h, steps=h)
        ws = torch.linspace(start_w, end_w, steps=w)
        hs, ws = torch.broadcast_tensors(hs[:, None], ws[None, :])
        hws = torch.stack([hs, ws], dim=-1)

        y = self.fourier(hws)
        y = self.linear1(y)
        y = torch.nn.functional.silu(y)
        y = self.linear2(y)
        variance = y.pow(2).mean(-1, keepdim=True)
        y = y * torch.rsqrt(variance + 1e-6) * self.norm_out_weight
        y = self.proj_out(y)
        return y

    @staticmethod
    def prepare_torch_state(state_dict, prefix="fourier_pos_emb."):
        mapped = {}
        key_map = {
            f"{prefix}base.fourier.proj.weight": "fourier.proj.weight",
            f"{prefix}base.fourier.proj.bias": "fourier.proj.bias",
            f"{prefix}linear1.weight": "linear1.weight",
            f"{prefix}linear1.bias": "linear1.bias",
            f"{prefix}linear2.weight": "linear2.weight",
            f"{prefix}linear2.bias": "linear2.bias",
            f"{prefix}norm_out.weight": "norm_out_weight",
            f"{prefix}proj_out.weight": "proj_out.weight",
            f"{prefix}proj_out.bias": "proj_out.bias",
        }
        for src_key, dst_key in key_map.items():
            if src_key in state_dict:
                mapped[dst_key] = state_dict[src_key]
        return mapped


class _ConditioningEmbedding(Module):
    def __init__(self, d_in, dim, kl=True, mesh_device=None):
        super().__init__()
        self.dim = dim
        self.kl = kl
        output_dim = dim * 2 if kl else dim
        self.linear1 = Linear(d_in, output_dim, bias=True, mesh_device=mesh_device)
        self.linear2 = Linear(output_dim, output_dim, bias=True, mesh_device=mesh_device)

    def forward(self, x_embed, x_eps=None):
        x = self.linear1(x_embed)
        x = gelu_tanh(x)
        x = self.linear2(x)
        if self.kl:
            mu, logvar = ttnn.chunk(x, 2, -1)
            if x_eps is not None:
                sigma = ttnn.exp(ttnn.multiply(logvar, 0.5))
                x = ttnn.add(ttnn.multiply(x_eps, sigma), mu)
            else:
                x = mu
        return x


class _TransformerBlock(WanTransformerBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ffn.ff1.fused_activation_fn = (ttnn.UnaryOpType.GELU, True)
        self.ffn.ff1.activation_fn = None
        self._sst_chunks = None

    def forward(
        self,
        spatial_1BND,
        prompt_1BLP,
        temb_1BTD,
        N,
        rope_cos,
        rope_sin,
        trans_mat,
        temb_chunks=None,
    ):
        if temb_chunks is None:
            assert temb_1BTD.shape[2] == 6
            return super().forward(
                spatial_1BND,
                prompt_1BLP,
                temb_1BTD,
                N,
                rope_cos,
                rope_sin,
                trans_mat,
            )

        if self._sst_chunks is None:
            sst_flat = ttnn.reshape(self.scale_shift_table.data, [1, 1, 1, -1])
            sst_bf16 = ttnn.typecast(sst_flat, dtype=ttnn.bfloat16)
            raw_chunks = ttnn.chunk(sst_bf16, 6, dim=3)
            D_f = sst_bf16.shape[3] // 6
            ones = ttnn.from_torch(
                torch.ones(1, 1, 1, D_f, dtype=torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.mesh_device),
            )
            self._sst_chunks = (
                raw_chunks[0],
                raw_chunks[1] + ones,
                raw_chunks[2],
                raw_chunks[3],
                raw_chunks[4] + ones,
                raw_chunks[5],
            )
            ttnn.deallocate(ones)

        shift_msa_1BND = self._sst_chunks[0] + temb_chunks[0]
        scale_msa_1BND = self._sst_chunks[1] + temb_chunks[1]
        gate_msa_1BND = self._sst_chunks[2] + temb_chunks[2]
        c_shift_msa_1BND = self._sst_chunks[3] + temb_chunks[3]
        c_scale_msa_1BND = self._sst_chunks[4] + temb_chunks[4]
        c_gate_msa_1BND = self._sst_chunks[5] + temb_chunks[5]

        N_padded = spatial_1BND.shape[2]
        D_local = spatial_1BND.shape[3]
        batched_shape = [1, N_padded // 32, 32, D_local]

        spatial_normed_1BND = self.norm1(
            ttnn.reshape(spatial_1BND, batched_shape),
            dynamic_weight=ttnn.reshape(scale_msa_1BND, batched_shape),
            dynamic_bias=ttnn.reshape(shift_msa_1BND, batched_shape),
        )
        spatial_normed_1BND = ttnn.reshape(spatial_normed_1BND, [1, 1, N_padded, D_local])

        spatial_1BND = self.attn1(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=spatial_1BND,
            addcmul_gate=gate_msa_1BND,
        )

        spatial_normed_1BND = self.norm2(spatial_1BND)
        attn_output_1BND = self.attn2(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            prompt_1BLP=prompt_1BLP,
        )
        spatial_1BND = spatial_1BND + attn_output_1BND

        spatial_normed_1BND = self.norm3(
            ttnn.reshape(spatial_1BND, batched_shape),
            dynamic_weight=ttnn.reshape(c_scale_msa_1BND, batched_shape),
            dynamic_bias=ttnn.reshape(c_shift_msa_1BND, batched_shape),
        )
        spatial_normed_1BND = ttnn.reshape(spatial_normed_1BND, [1, 1, N_padded, D_local])

        if self.ccl_manager.topology == ttnn.Topology.Linear:
            if self.parallel_config.tensor_parallel.factor > 1:
                spatial_normed_1BND = self.ccl_manager.all_gather_persistent_buffer(
                    spatial_normed_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            spatial_ff_1BND = self.ffn(spatial_normed_1BND, compute_kernel_config=self.ff_compute_kernel_config)
            spatial_1BND = ttnn.addcmul(spatial_1BND, spatial_ff_1BND, c_gate_msa_1BND)
        else:
            spatial_1BND = self.ffn.forward_fused_addcmul(
                spatial_normed_1BND,
                spatial_1BND,
                c_gate_msa_1BND,
                scalar=1.0,
                compute_kernel_config=self.ff_compute_kernel_config,
                parallel_config=self.parallel_config,
            )

        return spatial_1BND


class _Transformer3DModel(Module):
    def __init__(
        self,
        *,
        patch_size=(1, 2, 2),
        in_channels=16,
        out_channels=16,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        num_heads=40,
        num_layers=40,
        cross_attn_norm=True,
        eps=1e-6,
        use_extra_cond=True,
        d_extra_cond=3072,
        kl_extra_cond=True,
        use_fourier_coords=True,
        mesh_device,
        ccl_manager=None,
        parallel_config,
        is_fsdp=True,
        sdpa_chunk_size_overrides=None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = dim // num_heads
        self.eps = eps
        self.use_extra_cond = use_extra_cond
        self.use_fourier_coords = use_fourier_coords

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_fsdp = is_fsdp
        self.fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None
        tp_mesh_axis = parallel_config.tensor_parallel.mesh_axis

        self.patch_embedding = WanPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            mesh_device=mesh_device,
            tp_mesh_axis=tp_mesh_axis,
        )

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            time_freq_dim=freq_dim,
            time_proj_dim=dim * 6,
            text_embed_dim=text_dim,
            mesh_device=mesh_device,
            tp_mesh_axis=tp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        if use_extra_cond:
            self.extra_cond_embed = _ConditioningEmbedding(
                d_in=d_extra_cond,
                dim=dim,
                kl=kl_extra_cond,
                mesh_device=mesh_device,
            )

        if use_fourier_coords:
            self._fourier_pos_emb = _FourierMLPPosEmb(d_out=dim)

        self._rope = None

        self.blocks = ModuleList(
            _TransformerBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                cross_attention_norm=cross_attn_norm,
                eps=eps,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                sdpa_chunk_size_overrides=sdpa_chunk_size_overrides,
            )
            for _ in range(num_layers)
        )

        self.norm_out = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        out_dim_total = math.prod(patch_size) * out_channels
        self.proj_out = Linear(dim, out_dim_total, bias=True, mesh_device=mesh_device)

        self.scale_shift_table = Parameter(
            total_shape=[1, 2, dim],
            device=mesh_device,
            mesh_axes=[None, None, tp_mesh_axis],
            dtype=ttnn.float32,
        )

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _get_rope_instance(self):
        if self._rope is None:
            self._rope = _RotaryEmbeddingND(head_dims=[44, 42, 42], base=10000)
        return self._rope

    def _prepare_rope_features(self, hidden_states):
        B, C, T, H, W = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        t_patches, h_patches, w_patches = T // p_t, H // p_h, W // p_w

        rope = self._get_rope_instance()

        zeros = torch.zeros(t_patches, h_patches, w_patches, dtype=torch.int32)
        t_ids = torch.arange(t_patches, dtype=torch.int32)[:, None, None] + zeros
        h_ids = torch.arange(h_patches, dtype=torch.int32)[None, :, None] + zeros
        w_ids = torch.arange(w_patches, dtype=torch.int32)[None, None, :] + zeros
        all_ids = torch.stack([t_ids, h_ids, w_ids], dim=-1).reshape(-1, 3)

        freqs = rope(all_ids, no_repeat=True)
        rope_cos_raw = freqs[..., 0]
        rope_sin_raw = freqs[..., 1]

        rope_cos_1N1D = rope_cos_raw.unsqueeze(0).unsqueeze(2)
        rope_sin_1N1D = rope_sin_raw.unsqueeze(0).unsqueeze(2)
        rope_cos_1N1D, rope_sin_1N1D = stack_cos_sin(rope_cos_1N1D, rope_sin_1N1D)

        rope_cos_1HND = rope_cos_1N1D.permute(0, 2, 1, 3)
        rope_sin_1HND = rope_sin_1N1D.permute(0, 2, 1, 3)
        trans_mat = get_rot_transformation_mat()

        return rope_cos_1HND, rope_sin_1HND, trans_mat

    def _get_rope_cached(self, spatial_shape):
        cache_key = spatial_shape
        if getattr(self, "_rope_cache_key", None) != cache_key:
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis
            sp_factor = self.parallel_config.sequence_parallel.factor

            rope_cos, rope_sin, trans_mat = self._prepare_rope_features(torch.empty(spatial_shape))
            rope_cos = pad_vision_seq_parallel(rope_cos, num_devices=sp_factor)
            rope_sin = pad_vision_seq_parallel(rope_sin, num_devices=sp_factor)

            self._rope_cos_cache = from_torch(
                rope_cos,
                device=self.mesh_device,
                dtype=ttnn.float32,
                mesh_axes=[..., sp_axis, None],
            )
            self._rope_sin_cache = from_torch(
                rope_sin,
                device=self.mesh_device,
                dtype=ttnn.float32,
                mesh_axes=[..., sp_axis, None],
            )
            self._rope_trans_mat_cache = bf16_tensor(trans_mat, device=self.mesh_device)
            self._rope_cache_key = cache_key
        return self._rope_cos_cache, self._rope_sin_cache, self._rope_trans_mat_cache

    def _get_fourier_cached(self, F, H, W, coords):
        pF, pH, pW = self.patch_size
        h_patches, w_patches, t_patches = H // pH, W // pW, F // pF
        cache_key = (h_patches, w_patches, t_patches, coords)
        if getattr(self, "_fourier_cache_key", None) != cache_key:
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis

            grid_pos_emb = self._fourier_pos_emb(
                grid_size=(h_patches, w_patches),
                coords=coords,
            )
            grid_pos_emb = grid_pos_emb.reshape(1, 1, h_patches * w_patches, self.dim)
            grid_pos_emb = grid_pos_emb.expand(1, 1, -1, -1).repeat(1, 1, t_patches, 1)
            grid_pos_emb = pad_vision_seq_parallel(
                grid_pos_emb,
                num_devices=self.parallel_config.sequence_parallel.factor,
            )
            self._fourier_cache = from_torch(
                grid_pos_emb.bfloat16(),
                device=self.mesh_device,
                mesh_axes=[..., sp_axis, self.parallel_config.tensor_parallel.mesh_axis],
            )
            self._fourier_cache_key = cache_key
        return self._fourier_cache

    def _get_prompt_cached(self, prompt, extra_cond, extra_cond_eps, context_lens):
        cache_key = (id(prompt), id(extra_cond), context_lens)
        if getattr(self, "_prompt_cache_key", None) != cache_key:
            prompt_1BLP = self.condition_embedder.forward_text(prompt)
            if self.use_extra_cond and extra_cond is not None:
                cond_token = self.extra_cond_embed(extra_cond, extra_cond_eps)
                cond_token = ttnn.unsqueeze(cond_token, 0)
                prompt_1BLP = ttnn.concat([cond_token, prompt_1BLP], dim=2)
            if context_lens is not None:
                effective_lens = context_lens
                if self.use_extra_cond and extra_cond is not None:
                    effective_lens += 1
                prompt_1BLP = prompt_1BLP[:, :, :effective_lens, :]
            self._prompt_cache = prompt_1BLP
            self._prompt_cache_key = cache_key
        return self._prompt_cache

    def _get_perframe_mask(self, N_seq, ratio):
        cache_key = (N_seq, ratio)
        if getattr(self, "_perframe_mask_key", None) != cache_key:
            sp_factor = self.parallel_config.sequence_parallel.factor
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis

            mask = torch.ones(1, 1, N_seq, 1, dtype=torch.float32)
            mask[:, :, :ratio, :] = 0.0
            mask = pad_vision_seq_parallel(mask, num_devices=sp_factor)
            self._perframe_mask_cache = from_torch(
                mask,
                device=self.mesh_device,
                dtype=ttnn.float32,
                mesh_axes=[..., sp_axis, None],
            )
            self._perframe_mask_key = cache_key
        return self._perframe_mask_cache

    def _preprocess_spatial(self, spatial):
        B, C, F, H, W = spatial.shape
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        spatial = spatial.reshape(B, C, patch_F, pF, patch_H, pH, patch_W, pW)
        spatial = spatial.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, B, N, pF * pH * pW * C)
        spatial = pad_vision_seq_parallel(
            spatial,
            num_devices=self.parallel_config.sequence_parallel.factor,
        )
        return spatial, N

    def _postprocess_spatial(self, spatial_1BNI, F, H, W, N):
        B = spatial_1BNI.shape[1]
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW

        spatial_BND = spatial_1BNI.squeeze(0)[:, :N]
        spatial = spatial_BND.reshape(B, patch_F, patch_H, patch_W, pF, pH, pW, self.out_channels)
        spatial = spatial.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.out_channels, F, H, W)
        return spatial

    def forward(
        self,
        spatial,
        prompt,
        timestep,
        context_mask=None,
        context_lens=None,
        extra_cond=None,
        extra_cond_eps=None,
        coords=(-1.0, 1.0, -1.0, 1.0),
    ):
        B, C, F, H, W = spatial.shape
        pF, pH, pW = self.patch_size

        t0 = time.perf_counter()
        tt_rope_cos, tt_rope_sin, tt_trans_mat = self._get_rope_cached(spatial.shape)
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        t_rope = time.perf_counter() - t0

        t0 = time.perf_counter()
        per_frame = timestep.dim() == 2

        if per_frame:
            T_frames = F // pF
            h_patches = H // pH
            w_patches = W // pW
            ratio = h_patches * w_patches
            N_seq = T_frames * ratio

            cond_ts_tt = float32_tensor(
                timestep[:, 0].unsqueeze(1).unsqueeze(1).unsqueeze(1),
                device=self.mesh_device,
            )
            gen_ts_tt = float32_tensor(
                timestep[:, 1].unsqueeze(1).unsqueeze(1).unsqueeze(1),
                device=self.mesh_device,
            )
            temb_cond, proj_cond = self.condition_embedder.forward_timestep(cond_ts_tt)
            temb_gen, proj_gen = self.condition_embedder.forward_timestep(gen_ts_tt)
            temb_11BD = temb_gen

            mask_tt = self._get_perframe_mask(N_seq, ratio)
            per_token_proj = proj_cond + mask_tt * (proj_gen - proj_cond)
        else:
            timestep_tt = float32_tensor(
                timestep.unsqueeze(1).unsqueeze(1).unsqueeze(1),
                device=self.mesh_device,
            )
            temb_11BD, timestep_proj_1BTD = self.condition_embedder.forward_timestep(timestep_tt)
            timestep_proj_1BTD = unflatten(ttnn.squeeze(timestep_proj_1BTD, -2), -1, (6, -1))
            per_token_proj = None

        prompt_1BLP = self._get_prompt_cached(prompt, extra_cond, extra_cond_eps, context_lens)
        t_temb_prompt = time.perf_counter() - t0

        t0 = time.perf_counter()
        spatial_1BNI, N = self._preprocess_spatial(spatial)
        spatial_tt = bf16_tensor(
            spatial_1BNI,
            device=self.mesh_device,
            mesh_axis=sp_axis,
            shard_dim=-2,
        )
        t_patchify = time.perf_counter() - t0

        t0 = time.perf_counter()
        spatial_tt = self.patch_embedding(spatial_tt)
        t_patch_embed = time.perf_counter() - t0

        t_fourier = 0.0
        if self.use_fourier_coords and coords is not None:
            t0 = time.perf_counter()
            grid_pos_emb_tt = self._get_fourier_cached(F, H, W, coords)
            spatial_tt = ttnn.add(spatial_tt, grid_pos_emb_tt)
            t_fourier = time.perf_counter() - t0

        t0 = time.perf_counter()
        if per_frame:
            temb_bf16 = ttnn.typecast(per_token_proj, dtype=ttnn.bfloat16)
            temb_chunks = tuple(ttnn.chunk(temb_bf16, 6, dim=3))
            temb_for_blocks = temb_bf16
        else:
            temb_for_blocks = timestep_proj_1BTD
            temb_chunks = None

        for block in self.blocks:
            spatial_tt = block(
                spatial_1BND=spatial_tt,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=temb_for_blocks,
                N=N,
                rope_cos=tt_rope_cos,
                rope_sin=tt_rope_sin,
                trans_mat=tt_trans_mat,
                temb_chunks=temb_chunks,
            )
        t_blocks = time.perf_counter() - t0

        t0 = time.perf_counter()
        if per_frame:
            temb_pertoken = temb_cond + mask_tt * (temb_gen - temb_cond)
            shift_sst, scale_sst = ttnn.chunk(self.scale_shift_table.data, 2, -2)
            shift_out = shift_sst + temb_pertoken
            scale_out = scale_sst + temb_pertoken

            N_padded = spatial_tt.shape[2]
            D_local = spatial_tt.shape[3]
            head_batched_shape = [1, N_padded // 32, 32, D_local]
            spatial_norm = self.norm_out(
                ttnn.reshape(spatial_tt, head_batched_shape),
                dynamic_weight=(1.0 + ttnn.reshape(scale_out, head_batched_shape)),
                dynamic_bias=ttnn.reshape(shift_out, head_batched_shape),
                dtype=ttnn.float32,
            )
            spatial_norm = ttnn.reshape(spatial_norm, [1, 1, N_padded, D_local])
        else:
            scale_shift_1BSD = self.scale_shift_table.data + temb_11BD
            shift_11BD, scale_11BD = ttnn.chunk(scale_shift_1BSD, 2, -2)
            spatial_norm = self.norm_out(
                spatial_tt,
                dynamic_weight=(1 + scale_11BD),
                dynamic_bias=shift_11BD,
                dtype=ttnn.float32,
            )

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm = self.ccl_manager.all_gather_persistent_buffer(
                spatial_norm,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        proj_out = self.proj_out(
            spatial_norm,
            compute_kernel_config=self.hifi4_compute_kernel_config,
            dtype=ttnn.float32,
        )
        t_output_head = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.parallel_config.sequence_parallel.factor > 1:
            proj_out = self.ccl_manager.all_gather_persistent_buffer(
                proj_out,
                dim=2,
                mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            )
        proj_out_host = ttnn.to_torch(ttnn.get_device_tensors(proj_out)[0])
        output = self._postprocess_spatial(proj_out_host, F, H, W, N)
        t_d2h = time.perf_counter() - t0

        step_idx = getattr(self, "_timing_step", 0)
        self._timing_step = step_idx + 1
        logger.info(
            f"DiT forward step {step_idx}: "
            f"rope={t_rope:.3f}s  temb+prompt={t_temb_prompt:.3f}s  "
            f"patchify+h2d={t_patchify:.3f}s  patch_embed={t_patch_embed:.3f}s  "
            f"fourier={t_fourier:.3f}s  blocks={t_blocks:.3f}s  "
            f"output_head={t_output_head:.3f}s  d2h+unpatch={t_d2h:.3f}s  "
            f"total={t_rope + t_temb_prompt + t_patchify + t_patch_embed + t_fourier + t_blocks + t_output_head + t_d2h:.3f}s"
        )

        return output

    def _prepare_torch_state(self, state):
        new_state = {}
        fourier_keys = {}

        for key, value in list(state.items()):
            new_key = self._remap_key(key, value)
            if new_key is None:
                continue
            if new_key.startswith("_fourier_"):
                fourier_keys[new_key[len("_fourier_") :]] = value
            else:
                if ".norm_q.weight" in new_key or ".norm_k.weight" in new_key:
                    if value.numel() == self.head_dim:
                        value = value.repeat(self.num_heads)
                new_state[new_key] = value

        state.clear()
        state.update(new_state)

        if fourier_keys and hasattr(self, "_fourier_pos_emb"):
            self._fourier_pos_emb.load_state_dict(fourier_keys, strict=False)

    def _remap_key(self, key, value):
        if key.startswith("patch_embedding."):
            return key

        if key.startswith("time_embedding.0."):
            return f"condition_embedder.time_embedder.linear_1.{key[len('time_embedding.0.'):]}"
        if key.startswith("time_embedding.2."):
            return f"condition_embedder.time_embedder.linear_2.{key[len('time_embedding.2.'):]}"

        if key.startswith("time_projection.1."):
            return f"condition_embedder.time_proj.{key[len('time_projection.1.'):]}"

        if key.startswith("text_embedding.0."):
            return f"condition_embedder.text_embedder.linear_1.{key[len('text_embedding.0.'):]}"
        if key.startswith("text_embedding.2."):
            return f"condition_embedder.text_embedder.linear_2.{key[len('text_embedding.2.'):]}"

        if key.startswith("extra_cond_embedding.0."):
            return f"extra_cond_embed.linear1.{key[len('extra_cond_embedding.0.'):]}"
        if key.startswith("extra_cond_embedding.2."):
            return f"extra_cond_embed.linear2.{key[len('extra_cond_embedding.2.'):]}"

        if key.startswith("fourier_pos_emb."):
            mapped = _FourierMLPPosEmb.prepare_torch_state({key: value}, prefix="fourier_pos_emb.")
            if mapped:
                target_key = next(iter(mapped.keys()))
                return f"_fourier_{target_key}"
            return None

        if key.startswith("head.norm."):
            return f"norm_out.{key[len('head.norm.'):]}"
        if key.startswith("head.head."):
            return f"proj_out.{key[len('head.head.'):]}"
        if key == "head.modulation":
            return "scale_shift_table"

        if key.startswith("blocks."):
            return self._remap_block_key(key, value)

        return None

    def _remap_block_key(self, key, value):
        parts = key.split(".", 2)
        if len(parts) < 3:
            return None
        prefix = f"blocks.{parts[1]}"
        rest = parts[2]

        attn_map = {
            "self_attn.q.": "attn1.to_q.",
            "self_attn.k.": "attn1.to_k.",
            "self_attn.v.": "attn1.to_v.",
            "self_attn.o.": "attn1.to_out.0.",
            "self_attn.norm_q.": "attn1.norm_q.",
            "self_attn.norm_k.": "attn1.norm_k.",
            "cross_attn.q.": "attn2.to_q.",
            "cross_attn.k.": "attn2.to_k.",
            "cross_attn.v.": "attn2.to_v.",
            "cross_attn.o.": "attn2.to_out.0.",
            "cross_attn.norm_q.": "attn2.norm_q.",
            "cross_attn.norm_k.": "attn2.norm_k.",
        }
        for src_pat, dst_pat in attn_map.items():
            if rest.startswith(src_pat):
                return f"{prefix}.{dst_pat}{rest[len(src_pat):]}"

        if rest.startswith("norm1."):
            return f"{prefix}.norm1.{rest[6:]}"
        if rest.startswith("norm3."):
            return f"{prefix}.norm2.{rest[6:]}"
        if rest.startswith("norm2."):
            return f"{prefix}.norm3.{rest[6:]}"

        if rest.startswith("ffn.0."):
            return f"{prefix}.ffn.net.0.proj.{rest[6:]}"
        if rest.startswith("ffn.2."):
            return f"{prefix}.ffn.net.2.{rest[6:]}"

        if rest == "modulation":
            return f"{prefix}.scale_shift_table"

        return f"{prefix}.{rest}"


DIM = 5120
FFN_DIM = 13824
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS
IN_CHANNELS = 16
OUT_CHANNELS = 16
TEXT_DIM = 4096
FREQ_DIM = 256
PATCH_SIZE = (1, 2, 2)
D_EXTRA_COND = 3072

NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "40"))
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "1000"))
LATENT_FRAMES = int(os.environ.get("LATENT_FRAMES", "32"))
LATENT_H = 90
LATENT_W = 160
CTX_LEN = int(os.environ.get("CTX_LEN", "512"))
NUM_DENOISE_STEPS = 8


def _generate_random_state_dict(num_layers):
    torch.manual_seed(42)
    sd = {}

    def w(*shape):
        return torch.randn(*shape) * 0.02

    def z(*shape):
        return torch.zeros(*shape)

    sd["patch_embedding.weight"] = w(DIM, IN_CHANNELS, *PATCH_SIZE)
    sd["patch_embedding.bias"] = z(DIM)

    sd["time_embedding.0.weight"] = w(DIM, FREQ_DIM)
    sd["time_embedding.0.bias"] = z(DIM)
    sd["time_embedding.2.weight"] = w(DIM, DIM)
    sd["time_embedding.2.bias"] = z(DIM)

    sd["time_projection.1.weight"] = w(DIM * 6, DIM)
    sd["time_projection.1.bias"] = z(DIM * 6)

    sd["text_embedding.0.weight"] = w(DIM, TEXT_DIM)
    sd["text_embedding.0.bias"] = z(DIM)
    sd["text_embedding.2.weight"] = w(DIM, DIM)
    sd["text_embedding.2.bias"] = z(DIM)

    ec_dim = DIM * 2
    sd["extra_cond_embedding.0.weight"] = w(ec_dim, D_EXTRA_COND)
    sd["extra_cond_embedding.0.bias"] = z(ec_dim)
    sd["extra_cond_embedding.2.weight"] = w(ec_dim, ec_dim)
    sd["extra_cond_embedding.2.bias"] = z(ec_dim)

    half = DIM // 2
    sd["fourier_pos_emb.base.fourier.proj.weight"] = w(half, 2)
    sd["fourier_pos_emb.base.fourier.proj.bias"] = z(half)
    sd["fourier_pos_emb.linear1.weight"] = w(DIM, DIM)
    sd["fourier_pos_emb.linear1.bias"] = z(DIM)
    sd["fourier_pos_emb.linear2.weight"] = w(DIM, DIM)
    sd["fourier_pos_emb.linear2.bias"] = z(DIM)
    sd["fourier_pos_emb.norm_out.weight"] = torch.ones(DIM)
    sd["fourier_pos_emb.proj_out.weight"] = w(DIM, DIM)
    sd["fourier_pos_emb.proj_out.bias"] = z(DIM)

    out_features = PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2] * OUT_CHANNELS
    sd["head.head.weight"] = w(out_features, DIM)
    sd["head.head.bias"] = z(out_features)
    sd["head.modulation"] = z(1, 2, DIM)

    for i in range(num_layers):
        p = f"blocks.{i}"

        for proj in ("q", "k", "v", "o"):
            sd[f"{p}.self_attn.{proj}.weight"] = w(DIM, DIM)
            sd[f"{p}.self_attn.{proj}.bias"] = z(DIM)
        sd[f"{p}.self_attn.norm_q.weight"] = torch.ones(HEAD_DIM)
        sd[f"{p}.self_attn.norm_k.weight"] = torch.ones(HEAD_DIM)

        for proj in ("q", "k", "v", "o"):
            sd[f"{p}.cross_attn.{proj}.weight"] = w(DIM, DIM)
            sd[f"{p}.cross_attn.{proj}.bias"] = z(DIM)
        sd[f"{p}.cross_attn.norm_q.weight"] = torch.ones(HEAD_DIM)
        sd[f"{p}.cross_attn.norm_k.weight"] = torch.ones(HEAD_DIM)

        sd[f"{p}.ffn.0.weight"] = w(FFN_DIM, DIM)
        sd[f"{p}.ffn.0.bias"] = z(FFN_DIM)
        sd[f"{p}.ffn.2.weight"] = w(DIM, FFN_DIM)
        sd[f"{p}.ffn.2.bias"] = z(DIM)

        sd[f"{p}.norm3.weight"] = torch.ones(DIM)
        sd[f"{p}.norm3.bias"] = z(DIM)

        sd[f"{p}.modulation"] = z(1, 6, DIM)

    return sd


def _make_sigma_schedule(num_steps, shift=5.0, latent_frames=LATENT_FRAMES):
    def _shifted(s, smax, smin, n):
        raw = np.linspace(smax, smin, n + 1)
        return torch.from_numpy((s * raw / (1 + (s - 1) * raw)).astype(np.float32))

    cond = _shifted(1.0, 0.0002, 0.00019, num_steps).unsqueeze(1)
    gen = _shifted(shift, 0.999, 0.0, num_steps).unsqueeze(1)
    return torch.cat([cond, gen.expand(-1, latent_frames - 1)], dim=1)


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        tensor_parallel=ParallelFactor(
            mesh_axis=tp_axis,
            factor=tuple(mesh_device.shape)[tp_axis],
        ),
        sequence_parallel=ParallelFactor(
            mesh_axis=sp_axis,
            factor=tuple(mesh_device.shape)[sp_axis],
        ),
        cfg_parallel=None,
    )


def _create_model(mesh_device, ccl_manager, parallel_config, is_fsdp, num_layers):
    return _Transformer3DModel(
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        dim=DIM,
        ffn_dim=FFN_DIM,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
        num_heads=NUM_HEADS,
        num_layers=num_layers,
        cross_attn_norm=True,
        use_extra_cond=True,
        d_extra_cond=D_EXTRA_COND,
        kl_extra_cond=True,
        use_fourier_coords=True,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        sdpa_chunk_size_overrides=SDPA_CHUNK_SIZE_OVERRIDES,
    )


@pytest.mark.timeout(86400)
@pytest.mark.parametrize(
    (
        "mesh_device",
        "mesh_shape",
        "sp_axis",
        "tp_axis",
        "num_links",
        "device_params",
        "topology",
        "is_fsdp",
    ),
    [
        pytest.param(
            (4, 8),
            (4, 8),
            1,
            0,
            2,
            ring_params,
            ttnn.Topology.Ring,
            False,
            id="bh_4x8sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_transformer_thermal(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
) -> None:
    """Thermal / reliability stress test."""

    num_layers = NUM_LAYERS
    num_iterations = NUM_ITERATIONS
    latent_frames = LATENT_FRAMES
    latent_h = LATENT_H
    latent_w = LATENT_W
    ctx_len = CTX_LEN

    p_t, p_h, p_w = PATCH_SIZE
    seq_len = (latent_frames // p_t) * (latent_h // p_h) * (latent_w // p_w)

    logger.info(
        f"Thermal stress test config:\n"
        f"  Mesh:       {mesh_shape}\n"
        f"  Layers:     {num_layers}\n"
        f"  Latents:    {latent_frames} x {latent_h} x {latent_w}  ->  {seq_len} tokens\n"
        f"  Context:    {ctx_len}\n"
        f"  Iterations: {num_iterations}"
    )

    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=topology,
    )

    logger.info("Creating model ...")
    model = _create_model(
        mesh_device,
        ccl_manager,
        parallel_config,
        is_fsdp,
        num_layers,
    )

    logger.info("Generating and loading random weights ...")
    t0 = time.perf_counter()
    state_dict = _generate_random_state_dict(num_layers)
    model.load_torch_state_dict(state_dict)
    del state_dict
    logger.info(f"Weight loading: {time.perf_counter() - t0:.1f}s")

    torch.manual_seed(0)
    B = 1
    spatial = torch.randn(
        B,
        IN_CHANNELS,
        latent_frames,
        latent_h,
        latent_w,
        dtype=torch.float32,
    )
    prompt_cpu = torch.randn(1, B, ctx_len, TEXT_DIM, dtype=torch.bfloat16)
    extra_cond_cpu = torch.randn(B, 1, D_EXTRA_COND, dtype=torch.bfloat16)

    prompt_tt = from_torch(prompt_cpu, device=mesh_device)
    extra_cond_tt = from_torch(extra_cond_cpu, device=mesh_device)

    sigmas = _make_sigma_schedule(NUM_DENOISE_STEPS, latent_frames=latent_frames)

    logger.info(f"Starting thermal loop: {num_iterations} iterations ...")
    times = []
    width = len(str(num_iterations))

    for i in range(num_iterations):
        step = i % NUM_DENOISE_STEPS
        sigma_t = sigmas[step]
        timestep = (sigma_t * 1000.0).unsqueeze(0)

        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(
                spatial=spatial,
                prompt=prompt_tt,
                timestep=timestep,
                extra_cond=extra_cond_tt,
                extra_cond_eps=None,
                coords=(-1.0, 1.0, -1.0, 1.0),
                context_lens=ctx_len,
            )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()

        avg = sum(times) / len(times)
        logger.info(
            f"[{i + 1:>{width}}/{num_iterations}]  "
            f"compute_time={elapsed:.3f}s  avg={avg:.3f}s  "
            f"shape={tuple(output.shape)}  "
            f"range=[{output.min().item():.4f}, {output.max().item():.4f}]"
            + ("  *** NaN ***" if has_nan else "")
            + ("  *** Inf ***" if has_inf else "")
        )

        if has_nan or has_inf:
            pytest.fail(f"Numerical fault at iteration {i + 1}: " f"NaN={has_nan}, Inf={has_inf}")

    t = torch.tensor(times)
    logger.info(
        f"\n{'=' * 60}\n"
        f"Thermal stress test complete\n"
        f"  Iterations: {num_iterations}\n"
        f"  Mean:       {t.mean():.3f}s\n"
        f"  Std:        {t.std():.3f}s\n"
        f"  Min:        {t.min():.3f}s\n"
        f"  Max:        {t.max():.3f}s\n"
        f"  Total:      {t.sum():.1f}s\n"
        f"{'=' * 60}"
    )

    del model
