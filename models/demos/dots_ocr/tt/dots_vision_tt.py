# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
TTNN Dots vision tower (Wormhole).

Reuses ttnn patterns from Qwen2.5-VL / Qwen3-VL (RMSNorm, SwiGLU MLP, LayerNorm+GELU merger)
for norms/MLP. The trunk uses TTNN tensors end-to-end; `DotsVisionTransformerTT.forward`
accepts and returns PyTorch tensors, converting only at the API boundary.

RoPE frequencies follow the same geometry as `DotsVisionTransformer.rot_pos_emb`.

Typical `state_dict` prefix: ``"vision_tower."`` (keys like ``vision_tower.blocks.0...``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm as TtRmsNorm
from models.demos.dots_ocr.reference.dots_ocr.configuration_dots import DotsVisionConfig
from models.demos.qwen3_vl.tt.vision_layernorm import LayerNorm as TtLayerNorm
from models.tt_transformers.tt.common import Mode


@dataclass
class DotsVisionTtConfig:
    """Runtime bundle for ttnn vision stack."""

    embed_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    spatial_merge_size: int
    rms_norm_eps: float
    use_bias: bool
    post_norm: bool
    hidden_size: int
    patch_size: int
    temporal_patch_size: int
    num_channels: int

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_attention_heads


def _w128(x: int) -> int:
    return ((x + 127) // 128) * 128


def _pad_seq_dim_ttnn(x: ttnn.Tensor, s: int, s_pad: int) -> ttnn.Tensor:
    """Pad sequence dim (dim=2) from s to s_pad; no-op if already padded."""
    if s_pad <= s:
        return x
    pad_rows = s_pad - s
    out = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_rows), (0, 0)), value=0.0)
    ttnn.deallocate(x)
    return out


def _get_compute_cfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _qkv_from_state(state_dict: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    w = state_dict[key]
    d = w.shape[0] // 3
    wq, wk, wv = w[:d], w[d : 2 * d], w[2 * d :]
    return torch.cat([wq.transpose(0, 1), wk.transpose(0, 1), wv.transpose(0, 1)], dim=-1).unsqueeze(0).unsqueeze(0)


def _wo_from_state(state_dict: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    return state_dict[key].transpose(-1, -2).unsqueeze(0).unsqueeze(0)


def _rotate_half_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = ttnn.slice(
        x,
        [0, 0, 0, 0],
        [x.shape[0], x.shape[1], x.shape[2], half],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x2 = ttnn.slice(
        x,
        [0, 0, 0, half],
        [x.shape[0], x.shape[1], x.shape[2], d],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    neg_x2 = ttnn.multiply(x2, -1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x2)
    out = ttnn.concat([neg_x2, x1], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(neg_x2)
    ttnn.deallocate(x1)
    return out


def apply_rotary_pos_emb_vision_ttnn(tensor_tt: ttnn.Tensor, freqs_tt: ttnn.Tensor) -> ttnn.Tensor:
    """
    TTNN equivalent of reference apply_rotary_pos_emb_vision using raw freqs.
    ``tensor_tt``: [1, S, H, D] TILE BF16 (same layout as reference ``q.unsqueeze(0)``).
    ``freqs_tt``: [S, head_dim / 2] TILE BF16 (matches reference ``rotary_pos_emb`` slice).
    Returns TILE BF16 tensor with the same rank as ``tensor_tt``. ``freqs_tt`` is not deallocated.
    """
    cos_half = ttnn.cos(freqs_tt)
    sin_half = ttnn.sin(freqs_tt)

    cos_full = ttnn.concat([cos_half, cos_half], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sin_full = ttnn.concat([sin_half, sin_half], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(cos_half)
    ttnn.deallocate(sin_half)

    cos = ttnn.reshape(cos_full, (1, cos_full.shape[0], 1, cos_full.shape[1]))
    sin = ttnn.reshape(sin_full, (1, sin_full.shape[0], 1, sin_full.shape[1]))
    ttnn.deallocate(cos_full)
    ttnn.deallocate(sin_full)

    rotated = _rotate_half_ttnn(tensor_tt)
    out_a = ttnn.multiply(tensor_tt, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_b = ttnn.multiply(rotated, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.add(out_a, out_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn.deallocate(tensor_tt)
    ttnn.deallocate(rotated)
    ttnn.deallocate(cos)
    ttnn.deallocate(sin)
    ttnn.deallocate(out_a)
    ttnn.deallocate(out_b)

    return out


class VisionRotaryEmbeddingTt(LightweightModule):
    """Path-A TTNN rotary frequencies (raw freqs, not precomputed cos/sin)."""

    def __init__(self, mesh_device: Any, dim: int, theta: float = 10000.0):
        super().__init__()
        self.mesh = mesh_device
        self.dim = dim
        self.theta = theta

        # pi0-style TTNN math:
        # inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))
        idx = ttnn.arange(0, dim, 2, dtype=ttnn.float32, device=mesh_device)
        idx = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
        exponent = ttnn.multiply(idx, 1.0 / dim)
        ttnn.deallocate(idx)
        theta_pow = ttnn.pow(theta, exponent)
        ttnn.deallocate(exponent)
        self.inv_freq = ttnn.reciprocal(theta_pow)
        ttnn.deallocate(theta_pow)

    def forward(self, seqlen: int) -> ttnn.Tensor:
        # freqs = outer(arange(seqlen), inv_freq)
        seq = ttnn.arange(0, seqlen, 1, dtype=ttnn.float32, device=self.mesh)
        seq = ttnn.to_layout(seq, ttnn.TILE_LAYOUT)
        seq_col = ttnn.reshape(seq, (seqlen, 1))
        ttnn.deallocate(seq)
        inv_row = ttnn.reshape(self.inv_freq, (1, self.inv_freq.shape[-1]))
        freqs = ttnn.multiply(seq_col, inv_row)
        ttnn.deallocate(seq_col)
        # Do not deallocate ``inv_row``: it aliases ``self.inv_freq``; deallocating it frees the
        # module buffer and breaks the next ``forward`` (e.g. staged PCC then full ``forward``).
        return freqs


class DotsPatchEmbedTt(LightweightModule):
    """TTNN counterpart of reference DotsPatchEmbed (proj + RMSNorm)."""

    def __init__(
        self,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        self.mesh = mesh_device
        self.cfg = cfg
        self.compute_cfg = _get_compute_cfg()
        in_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size

        w_conv = state_dict[f"{state_dict_prefix}proj.weight"]  # [D, C, P, P]
        w_lin = w_conv.reshape(cfg.embed_dim, in_dim).transpose(0, 1).unsqueeze(0).unsqueeze(0)
        b_key = f"{state_dict_prefix}proj.bias"
        b_proj = state_dict[b_key] if b_key in state_dict else None

        cache = (
            (lambda p: (weight_cache_path / p) if weight_cache_path else None) if weight_cache_path else lambda _: None
        )
        self.w_proj = ttnn.as_tensor(
            w_lin,
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}proj") if weight_cache_path else None,
        )
        self.b_proj = (
            ttnn.as_tensor(
                b_proj,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            if b_proj is not None
            else None
        )
        self.norm = TtRmsNorm(
            device=mesh_device,
            dim=cfg.embed_dim,
            eps=cfg.rms_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_key="norm",
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
        )

    def forward(self, x: ttnn.Tensor, grid_thw=None) -> ttnn.Tensor:
        """
        Patch projection + RMSNorm. Input ``x`` is [1, 1, N, C*P*P] TILE BF16 (same layout as
        reference patchifier after flatten); ``grid_thw`` is unused (kept for API compatibility).
        """
        del grid_thw
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x_proj = ttnn.linear(
            x,
            self.w_proj,
            bias=self.b_proj,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        return self.norm(x_proj, mode=Mode.PREFILL)


class DotsMlpTt(LightweightModule):
    """SwiGLU FFN: silu(fc1) * fc3 -> fc2 (matches Dots reference naming)."""

    def __init__(
        self,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        self.mesh = mesh_device
        self.cfg = cfg
        self.compute_cfg = _get_compute_cfg()

        def t_linear_weight(name: str) -> torch.Tensor:
            w = state_dict[f"{state_dict_prefix}{name}.weight"]
            return w.transpose(0, 1).unsqueeze(0).unsqueeze(0)

        cache = (
            (lambda p: (weight_cache_path / p) if weight_cache_path else None) if weight_cache_path else lambda _: None
        )

        self.w1 = ttnn.as_tensor(
            t_linear_weight("fc1"),
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}fc1") if weight_cache_path else None,
        )
        self.w2 = ttnn.as_tensor(
            t_linear_weight("fc2"),
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}fc2") if weight_cache_path else None,
        )
        self.w3 = ttnn.as_tensor(
            t_linear_weight("fc3"),
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}fc3") if weight_cache_path else None,
        )
        if cfg.use_bias:
            self.b1 = ttnn.as_tensor(
                state_dict[f"{state_dict_prefix}fc1.bias"],
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            self.b2 = ttnn.as_tensor(
                state_dict[f"{state_dict_prefix}fc2.bias"],
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            self.b3 = ttnn.as_tensor(
                state_dict[f"{state_dict_prefix}fc3.bias"],
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        else:
            self.b1 = self.b2 = self.b3 = None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        s = x.shape[-2]
        if s >= 1024:
            x = ttnn.reshape(x, (1, s // 1024, 1024, -1))
        w1o = ttnn.linear(
            x,
            self.w1,
            bias=self.b1,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3o = ttnn.linear(
            x,
            self.w3,
            bias=self.b3,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        mid = ttnn.mul(
            w1o,
            w3o,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1o)
        ttnn.deallocate(w3o)
        out = ttnn.linear(
            mid,
            self.w2,
            bias=self.b2,
            compute_kernel_config=self.compute_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(mid)
        shp = out.shape
        if len(shp) == 4 and shp[1] != 1:
            return ttnn.reshape(out, (1, 1, shp[0] * shp[1] * shp[2], shp[3]))
        return out


class DotsAttnQkvprojTt(LightweightModule):
    """
    TTNN QKV and output projections; RoPE apply and ``F.scaled_dot_product_attention`` run on
    host PyTorch inside this module (TTNN in/out at the submodule boundary).
    """

    def __init__(
        self,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        self.cfg = cfg
        self.compute_cfg = _get_compute_cfg()
        self.mesh = mesh_device
        wqkv = _qkv_from_state(state_dict, f"{state_dict_prefix}qkv.weight")
        cache = (
            (lambda p: (weight_cache_path / p) if weight_cache_path else None) if weight_cache_path else lambda _: None
        )
        self.wqkv = ttnn.as_tensor(
            wqkv,
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}qkv") if weight_cache_path else None,
        )
        self.wo = ttnn.as_tensor(
            _wo_from_state(state_dict, f"{state_dict_prefix}proj.weight"),
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}wo") if weight_cache_path else None,
        )
        if cfg.use_bias and f"{state_dict_prefix}qkv.bias" in state_dict:
            d = 3 * cfg.embed_dim
            b = state_dict[f"{state_dict_prefix}qkv.bias"]
            self.bqkv = ttnn.as_tensor(
                b,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        else:
            self.bqkv = None
        if cfg.use_bias and f"{state_dict_prefix}proj.bias" in state_dict:
            self.bo = ttnn.as_tensor(
                state_dict[f"{state_dict_prefix}proj.bias"],
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        else:
            self.bo = None
        self.scale = cfg.head_dim**-0.5

    def forward(
        self,
        x: ttnn.Tensor,
        rotary_pos_emb: ttnn.Tensor,
        cu_seqlens: ttnn.Tensor,
        seqlen_in: int,
    ) -> ttnn.Tensor:
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            bias=self.bqkv,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        tq = ttnn.to_torch(xqkv)
        ttnn.deallocate(xqkv)
        qkv = tq[0, 0, :seqlen_in, :]
        s = seqlen_in
        d = self.cfg.embed_dim
        h = self.cfg.num_attention_heads
        hd = self.cfg.head_dim
        q, k, v = qkv.reshape(s, 3, h, hd).permute(1, 0, 2, 3).unbind(0)

        rot_dim = int(rotary_pos_emb.shape[-1])
        rpe_s = ttnn.slice(
            rotary_pos_emb,
            [0, 0, 0, 0],
            [1, 1, s, rot_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        freqs_2d = ttnn.reshape(rpe_s, (s, rot_dim))

        q_tt = ttnn.from_torch(
            q.unsqueeze(0).float(),
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        q_tt = ttnn.to_memory_config(q_tt, ttnn.DRAM_MEMORY_CONFIG)
        q_tt = apply_rotary_pos_emb_vision_ttnn(q_tt, freqs_2d)
        q = ttnn.to_torch(q_tt).squeeze(0).to(qkv.dtype)
        ttnn.deallocate(q_tt)

        k_tt = ttnn.from_torch(
            k.unsqueeze(0).float(),
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        k_tt = ttnn.to_memory_config(k_tt, ttnn.DRAM_MEMORY_CONFIG)
        k_tt = apply_rotary_pos_emb_vision_ttnn(k_tt, freqs_2d)
        k = ttnn.to_torch(k_tt).squeeze(0).to(qkv.dtype)
        ttnn.deallocate(k_tt)

        # Do not deallocate ``rpe_s`` / ``freqs_2d``: they are views into ``rotary_pos_emb``;
        # deallocating them frees the shared backing store and breaks later vision blocks.

        cu_1d = ttnn.to_torch(cu_seqlens).reshape(-1).to(torch.int32)
        mask = torch.zeros(1, s, s, dtype=torch.bool, device=q.device)
        for b in range(1, cu_1d.numel()):
            a, z = int(cu_1d[b - 1].item()), int(cu_1d[b].item())
            mask[..., a:z, a:z] = True
        q1 = q.transpose(0, 1).unsqueeze(0)
        k1 = k.transpose(0, 1).unsqueeze(0)
        v1 = v.transpose(0, 1).unsqueeze(0)
        m3 = mask.unsqueeze(0)  # [1, S, S], True = allow (match reference VisionSdpaAttention)
        o = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=m3, dropout_p=0.0, is_causal=False, scale=self.scale)
        o = o.squeeze(0).transpose(0, 1).reshape(s, d)
        out_pad = _w128(s)
        if out_pad > s:
            o_full = o.new_zeros(out_pad, d)
            o_full[:s] = o
        else:
            o_full = o
        ttn_in = ttnn.from_torch(
            o_full.unsqueeze(0).unsqueeze(0).bfloat16(),
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttn_in = ttnn.to_memory_config(ttn_in, ttnn.DRAM_MEMORY_CONFIG)
        out1 = ttnn.linear(
            ttn_in,
            self.wo,
            bias=self.bo,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttn_in)
        return out1


class DotsPatchMergerTt(LightweightModule):
    """
    LayerNorm on token dim, then GELU-MLP (matches default Dots `PatchMerger` with
    pre_norm=layernorm and Sequential Linear-GELU-Linear).
    """

    def __init__(
        self,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        self.cfg = cfg
        self.mlp_in = (cfg.spatial_merge_size**2) * cfg.embed_dim
        self.ln = TtLayerNorm(
            device=mesh_device,
            dim=cfg.embed_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_q",
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            eps=1e-6,
        )
        self.compute_cfg = _get_compute_cfg()

        def tw_linear(name: str) -> ttnn.Tensor:
            w = torch.transpose(state_dict[f"{state_dict_prefix}mlp.{name}.weight"], -2, -1)
            return ttnn.as_tensor(
                w,
                dtype=ttnn.bfloat8_b,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        self.w0 = tw_linear("0")
        self.w2 = tw_linear("2")
        self.b0 = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}mlp.0.bias"],
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.b2 = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}mlp.2.bias"],
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def forward(self, x: ttnn.Tensor, seqlen: int) -> ttnn.Tensor:
        merge2 = self.cfg.spatial_merge_size**2
        assert seqlen % merge2 == 0, "Token count must be divisible by spatial_merge_size**2 for PatchMerger"
        xu = ttnn.slice(x, [0, 0, 0, 0], [1, 1, seqlen, x.shape[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        x1 = self.ln(xu)
        ttnn.deallocate(xu)
        n_merge = seqlen // merge2
        xrm = ttnn.to_layout(x1, ttnn.ROW_MAJOR_LAYOUT)
        xrm = ttnn.reshape(xrm, (1, 1, n_merge, self.mlp_in))
        xrm = ttnn.to_layout(xrm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x1)
        w1c = ttnn.linear(
            xrm,
            self.w0,
            bias=self.b0,
            activation="gelu",
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xrm)
        out = ttnn.linear(
            w1c,
            self.w2,
            bias=self.b2,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1c)
        return out


class DotsVisionBlockTt(LightweightModule):
    def __init__(
        self,
        layer_idx: int,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        sp = f"{prefix}blocks.{layer_idx}."
        self.layer_idx = layer_idx
        self.rms1 = TtRmsNorm(
            device=mesh_device,
            dim=cfg.embed_dim,
            eps=cfg.rms_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=sp,
            weight_key="norm1",
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
        )
        self.rms2 = TtRmsNorm(
            device=mesh_device,
            dim=cfg.embed_dim,
            eps=cfg.rms_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=sp,
            weight_key="norm2",
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
        )
        self.attn = DotsAttnQkvprojTt(mesh_device, state_dict, f"{sp}attn.", cfg, weight_cache_path)
        self.mlp = DotsMlpTt(mesh_device, state_dict, f"{sp}mlp.", cfg, weight_cache_path)

    def forward(self, x: ttnn.Tensor, rotary_pos_emb: ttnn.Tensor, cu_seqlens: ttnn.Tensor, seqlen: int) -> ttnn.Tensor:
        x0 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        n1 = self.rms1(x0, mode=Mode.PREFILL)
        ao = self.attn(n1, rotary_pos_emb, cu_seqlens, seqlen)
        t1 = ttnn.add(x0, ao, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x0)
        ttnn.deallocate(ao)
        n2 = self.rms2(t1, mode=Mode.PREFILL)
        m = self.mlp(n2)
        ttnn.deallocate(n2)
        out = ttnn.add(t1, m, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(m)
        ttnn.deallocate(t1)
        return out


class DotsVisionTransformerTT(LightweightModule):
    """
    ttnn Dots vision trunk + merger. RoPE and patch/trunk norms/attn/MLP/post/merger use ttnn.
    """

    def __init__(
        self,
        vision_config: Union[DotsVisionConfig, Dict[str, Any]],
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str = "vision_tower.",
        weight_cache_path: Optional[Any] = None,
    ):
        super().__init__()
        if isinstance(vision_config, DotsVisionConfig):
            self.dots_cfg = vision_config
        elif isinstance(vision_config, Mapping):
            self.dots_cfg = DotsVisionConfig(**vision_config)
        elif hasattr(vision_config, "to_dict"):
            # Handles dynamically loaded config classes from trust_remote_code.
            self.dots_cfg = DotsVisionConfig(**vision_config.to_dict())
        else:
            raise TypeError(
                "vision_config must be DotsVisionConfig, mapping, or to_dict()-compatible config object; "
                f"got {type(vision_config)!r}"
            )
        self.pfx = state_dict_prefix
        self.mesh = mesh_device
        self.cfg = DotsVisionTtConfig(
            embed_dim=self.dots_cfg.embed_dim,
            num_hidden_layers=self.dots_cfg.num_hidden_layers,
            num_attention_heads=self.dots_cfg.num_attention_heads,
            intermediate_size=self.dots_cfg.intermediate_size,
            spatial_merge_size=self.dots_cfg.spatial_merge_size,
            rms_norm_eps=self.dots_cfg.rms_norm_eps,
            use_bias=self.dots_cfg.use_bias,
            post_norm=bool(self.dots_cfg.post_norm),
            hidden_size=self.dots_cfg.hidden_size,
            patch_size=self.dots_cfg.patch_size,
            temporal_patch_size=self.dots_cfg.temporal_patch_size,
            num_channels=self.dots_cfg.num_channels,
        )
        self.patch_embed = DotsPatchEmbedTt(
            mesh_device,
            state_dict,
            f"{self.pfx}patch_embed.patchifier.",
            self.cfg,
            weight_cache_path,
        )
        head_dim = self.cfg.embed_dim // self.cfg.num_attention_heads
        self.rotary_dim = head_dim // 2
        self.rotary_pos_emb = VisionRotaryEmbeddingTt(mesh_device, self.rotary_dim, theta=10000.0)
        if self.dots_cfg.post_norm:
            self.post_norm = TtRmsNorm(
                device=mesh_device,
                dim=self.cfg.embed_dim,
                eps=self.cfg.rms_norm_eps,
                state_dict=state_dict,
                state_dict_prefix=self.pfx,
                weight_key="post_trunk_norm",
                weight_cache_path=weight_cache_path,
                weight_dtype=ttnn.bfloat16,
            )
        else:
            self.post_norm = None
        self.merger = DotsPatchMergerTt(mesh_device, state_dict, f"{self.pfx}merger.", self.cfg, weight_cache_path)
        self.blocks = [
            DotsVisionBlockTt(i, mesh_device, state_dict, self.pfx, self.cfg, weight_cache_path)
            for i in range(self.cfg.num_hidden_layers)
        ]

    def _pixels_flat_torch(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int]:
        hs = pixel_values.view(
            -1,
            self.cfg.num_channels,
            self.cfg.temporal_patch_size,
            self.cfg.patch_size,
            self.cfg.patch_size,
        )[:, :, 0]
        n = int(hs.shape[0])
        hs = hs.reshape(n, self.cfg.num_channels * self.cfg.patch_size * self.cfg.patch_size)
        return hs, n

    def _patch_pixels_to_ttnn(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        hs, _n = self._pixels_flat_torch(pixel_values)
        pixel_tt = ttnn.from_torch(
            hs.unsqueeze(0).unsqueeze(0).bfloat16(),
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        pixel_tt = ttnn.to_memory_config(pixel_tt, ttnn.DRAM_MEMORY_CONFIG)
        return self.patch_embed(pixel_tt, grid_thw=None)

    @torch.inference_mode()
    def _patchify(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Host patch embed for staged PCC / callers that expect torch ``[N, embed_dim]``.
        Same math as ``_patch_pixels_to_ttnn`` + ``to_torch`` slice; ``grid_thw`` is unused
        (reference patchifier ignores it).
        """
        del grid_thw
        out_tt = self._patch_pixels_to_ttnn(hidden_states)
        n = int(out_tt.shape[2])
        o = ttnn.to_torch(out_tt)[0, 0, :n, : self.cfg.embed_dim].to(hidden_states.dtype)
        ttnn.deallocate(out_tt)
        return o

    def _prepare_ttnn(self, emb: torch.Tensor, mesh_device: Any) -> Tuple[ttnn.Tensor, int, int]:
        """Pad sequence to ``_w128`` and upload for staged PCC (``[1,1,S_pad,D]`` TILE)."""
        s, d = int(emb.shape[0]), int(emb.shape[1])
        s_pad = _w128(s)
        if s_pad > s:
            p = emb.new_zeros(s_pad, d)
            p[:s] = emb
            tile = p
        else:
            tile = emb
        tt = ttnn.from_torch(
            tile.unsqueeze(0).unsqueeze(0).bfloat16(),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt = ttnn.to_memory_config(tt, ttnn.DRAM_MEMORY_CONFIG)
        return tt, s, s_pad

    def _cu_seqlens_ttnn_from_grid(self, grid_thw: torch.Tensor) -> ttnn.Tensor:
        g = grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
        cu = torch.repeat_interleave(g[:, 1] * g[:, 2], g[:, 0], dim=0).cumsum(0, dtype=torch.int32)
        cu = F.pad(cu, (1, 0), value=0)
        return ttnn.from_torch(
            cu.reshape(1, 1, 1, -1).to(torch.int32),
            dtype=ttnn.int32,
            device=self.mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _get_pos_ids_by_grid_ttnn(self, grid_thw_list: list[tuple[int, int, int]]) -> tuple[list[int], list[int]]:
        h_all: list[int] = []
        w_all: list[int] = []
        sm = self.cfg.spatial_merge_size
        for t, h, w in grid_thw_list:
            h_block: list[int] = []
            w_block: list[int] = []
            for hb in range(0, h, sm):
                for wb in range(0, w, sm):
                    for hi in range(sm):
                        for wi in range(sm):
                            h_block.append(hb + hi)
                            w_block.append(wb + wi)
            for _ in range(t):
                h_all.extend(h_block)
                w_all.extend(w_block)
        return h_all, w_all

    def _rot_pos_ttnn(self, grid_thw_list: list[tuple[int, int, int]]) -> ttnn.Tensor:
        h_ids, w_ids = self._get_pos_ids_by_grid_ttnn(grid_thw_list)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw_list)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        # ttnn.embedding currently requires BF16 weight tensors.
        rotary_pos_emb_full = ttnn.typecast(rotary_pos_emb_full, ttnn.bfloat16)
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        h_idx_tt = ttnn.as_tensor(
            np.asarray(h_ids, dtype=np.uint32).reshape(1, -1),
            dtype=ttnn.uint32,
            device=self.mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        w_idx_tt = ttnn.as_tensor(
            np.asarray(w_ids, dtype=np.uint32).reshape(1, -1),
            dtype=ttnn.uint32,
            device=self.mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        h_freqs = ttnn.embedding(h_idx_tt, rotary_pos_emb_full, layout=ttnn.TILE_LAYOUT)
        w_freqs = ttnn.embedding(w_idx_tt, rotary_pos_emb_full, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(h_idx_tt)
        ttnn.deallocate(w_idx_tt)
        ttnn.deallocate(rotary_pos_emb_full)
        token_count = len(h_ids)
        h_freqs = ttnn.reshape(h_freqs, (token_count, h_freqs.shape[-1]))
        w_freqs = ttnn.reshape(w_freqs, (token_count, w_freqs.shape[-1]))
        rotary = ttnn.concat([h_freqs, w_freqs], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h_freqs)
        ttnn.deallocate(w_freqs)
        d = rotary.shape[-1]
        return ttnn.reshape(rotary, (1, 1, token_count, d))

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, return_host_torch: bool = True
    ) -> Union[torch.Tensor, ttnn.Tensor]:
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)
        t = int(grid_thw.shape[0])
        hidden_tt = self._patch_pixels_to_ttnn(pixel_values)
        seqlen = int(hidden_tt.shape[2])
        s_pad = _w128(seqlen)
        hidden_tt = _pad_seq_dim_ttnn(hidden_tt, seqlen, s_pad)
        grid_thw_list = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        rotary_tt = self._rot_pos_ttnn(grid_thw_list)
        cu_tt = self._cu_seqlens_ttnn_from_grid(grid_thw)
        x = hidden_tt
        for blk in self.blocks:
            x = blk(x, rotary_tt, cu_tt, seqlen)
        ttnn.deallocate(rotary_tt)
        ttnn.deallocate(cu_tt)
        if self.post_norm is not None:
            x = self.post_norm(x, mode=Mode.PREFILL)
        x = self.merger(x, seqlen)
        if not return_host_torch:
            return x
        s_merge = seqlen // (self.cfg.spatial_merge_size**2)
        o_full = ttnn.to_torch(x)
        if o_full.dim() == 5:
            o = o_full[:, 0, 0, :s_merge, : self.cfg.hidden_size]
        elif o_full.dim() == 4:
            o = o_full[:, 0, :s_merge, : self.cfg.hidden_size]
        elif o_full.dim() == 3:
            o = o_full[:, :s_merge, : self.cfg.hidden_size]
        else:
            raise RuntimeError(f"Unexpected merged tensor rank {o_full.dim()} with shape {tuple(o_full.shape)}")
        ttnn.deallocate(x)
        return o.squeeze(0) if t == 1 else o.reshape(-1, self.cfg.hidden_size)
