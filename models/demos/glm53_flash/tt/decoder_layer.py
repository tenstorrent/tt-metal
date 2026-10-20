# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.3-Flash decoder layer on a single device (HF ``Glm5NextTextDecoderLayer``).

    residual = R                                  # 4 mHC streams, packed [1,1,T,4*4096] fp32
    post, comb, h = attn_hc(R)                    # fused mHC parametrization (fp32)
    y = attn(input_layernorm(h))                  # KDA or DSA, bf16
    R = post * y + comb^T R
    post, comb, h = ffn_hc(R)
    y = mlp(post_attention_layernorm(h))          # dense SwiGLU or MoE
    R = post * y + comb^T R

Prefill runs one user at a time (``[1,1,T_pad,16384]``), decode runs B users (``[1,1,B,16384]``).
Autoregressive state is device-owned: the KDA ``KdaState`` per layer (carried through the
forward calls) and the DSA paged latent cache (``TtLatentCache``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import TtDSA, TtLatentCache
from models.demos.glm53_flash.tt.kda import TtKDA
from models.demos.glm53_flash.tt.mesh import GLMMesh, single_chip
from models.demos.glm53_flash.tt.mhc import build_mhc
from models.demos.glm53_flash.tt.mlp import TtClampedSwiGLU
from models.demos.glm53_flash.tt.moe import TtMoE


def _strip(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


@dataclass
class LayerState:
    """Per-layer autoregressive state for one forward call."""

    kda: KdaState | None = None  # KDA layers: recurrent [B,H,K,V] fp32 + conv history [B,3,W] bf16
    cache: TtLatentCache | None = None  # DSA layers: paged latent cache + page table


class TtDecoderLayer(LightweightModule):
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        layer_idx: int,
        state_dict: dict[str, torch.Tensor],
        *,
        weight_cache_path: Path | None = None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        mesh: GLMMesh | None = None,
    ):
        """``state_dict``: raw checkpoint tensors of ``model.language_model.layers.{i}.`` with the
        prefix stripped and the fp8 groups dequantized (``reference/checkpoint.load_layer_raw``).
        ``weight_dtype`` covers the big non-expert matmuls (attention projections, dense MLP, shared
        expert); ``expert_dtype`` / ``expert_down_dtype`` the routed experts' gate/up and down weights.
        ``mesh`` (``tt/mesh.py``): the multi-chip placement; the hyper-connections and norms are
        replicated, the sublayers are tensor-parallel over the columns."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.is_kda = cfg.is_kda(layer_idx)
        self.is_moe = cfg.is_moe(layer_idx)
        self.hidden = cfg.hidden_size
        self.n_streams = cfg.hc_mult
        sd = state_dict
        self.attn_hc = build_mhc(device, cfg, sd["hc_attn_fn"], sd["hc_attn_base"], sd["hc_attn_scale"])
        self.ffn_hc = build_mhc(device, cfg, sd["hc_ffn_fn"], sd["hc_ffn_base"], sd["hc_ffn_scale"])
        self.input_norm_w = ttnn.from_torch(
            sd["input_layernorm.weight"].float().reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.post_norm_w = ttnn.from_torch(
            sd["post_attention_layernorm.weight"].float().reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        attn_sd = _strip(sd, "self_attn.")
        if self.is_kda:
            self.attn = TtKDA(
                device,
                cfg,
                attn_sd,
                layer_idx=layer_idx,
                weight_cache_path=weight_cache_path,
                weight_dtype=weight_dtype,
                mesh=self.mesh,
            )
        else:
            self.attn = TtDSA(device, cfg, attn_sd, layer_idx=layer_idx, weight_dtype=weight_dtype, mesh=self.mesh)
        mlp_sd = _strip(sd, "mlp.")
        if self.is_moe:
            self.mlp = TtMoE(
                device,
                cfg,
                mlp_sd,
                weight_dtype=weight_dtype,
                expert_dtype=expert_dtype,
                expert_down_dtype=expert_down_dtype,
                mesh=self.mesh,
            )
        else:
            self.mlp = TtClampedSwiGLU(
                device,
                mlp_sd["gate_proj.weight"],
                mlp_sd["up_proj.weight"],
                mlp_sd["down_proj.weight"],
                cfg.swiglu_limit,
                weight_dtype=weight_dtype,
                mesh=self.mesh,
            )
        self.norm_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ------------------------------------------------------------------ helpers
    def _norm(self, h: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(h, epsilon=self.cfg.rms_norm_eps, weight=w, compute_kernel_config=self.norm_compute)

    # ------------------------------------------------------------------ prefill
    def prefill_forward(
        self,
        streams: ttnn.Tensor,
        state: LayerState,
        *,
        user: int,
        seq_len: int,
        start: int = 0,
        actual_start: ttnn.Tensor | None = None,
        actual_end: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, LayerState]:
        """streams [1,1,T_pad,4*hidden] fp32 for one user: row i is the token at absolute position
        ``start + i`` (``start % 32 == 0``), T_pad % 32 == 0 and rows past ``seq_len`` are padding.
        KDA: ``state.kda`` is the carry after position ``start`` (zeros for a new prompt); DSA: the
        user's cache holds [0, start). Returns the new streams and the layer state after the chunk.
        ``actual_start`` / ``actual_end`` override the KDA device scalars derived from ``start`` and
        ``seq_len`` (global positions, as ttKDA defines them)."""
        T = streams.shape[-2]
        new_state = LayerState(kda=state.kda, cache=state.cache)

        def attention(h):
            h = self._norm(h, self.input_norm_w)  # [1,1,T,hidden] bf16
            if self.is_kda:
                start_t = actual_start if actual_start is not None else self.attn.scalar(start)
                end = actual_end
                # the exclusive global valid end; a chunk at start > 0 always carries it so the kernels'
                # chronology is the clipped interval [start, start + seq_len) (derive_interval)
                if end is None and (seq_len != T or start):
                    end = self.attn.scalar(start + seq_len)
                y, new_state.kda = self.attn.prefill_forward(
                    ttnn.reshape(h, [1, T, self.hidden]), state.kda, start_t, end
                )
                return ttnn.reshape(y, [1, 1, T, self.hidden])
            return self.attn.prefill_forward(h, state.cache, user, seq_len, start=start)

        streams = self.attn_hc.forward(streams, attention, sublayer_dtype=ttnn.bfloat16)

        def ffn(h):
            return self.mlp(self._norm(h, self.post_norm_w))

        streams = self.ffn_hc.forward(streams, ffn, sublayer_dtype=ttnn.bfloat16)
        return streams, new_state

    # ------------------------------------------------------------------- decode
    def decode_forward(self, streams: ttnn.Tensor, state: LayerState, *, positions) -> tuple[ttnn.Tensor, LayerState]:
        """streams [1,1,B,4*hidden] fp32 (one token per user); ``positions`` carries the persistent
        [B] int32 device tensors ``positions`` / ``pool_ids`` / ``ring_slots`` and the host values
        (``FunctionalDecoder.DecodePositions``). KDA layers ignore it."""
        new_state = LayerState(kda=state.kda, cache=state.cache)

        def attention(h):
            h = self._norm(h, self.input_norm_w)
            if self.is_kda:
                y, new_state.kda = self.attn.decode_forward(h, state.kda, inplace=True)
                return y
            return self.attn.decode_forward(
                h, state.cache, positions.positions, positions.pool_ids, positions.ring_slots, positions.host
            )

        streams = self.attn_hc.forward(streams, attention, sublayer_dtype=ttnn.bfloat16)

        def ffn(h):
            return self.mlp(self._norm(h, self.post_norm_w))

        streams = self.ffn_hc.forward(streams, ffn, sublayer_dtype=ttnn.bfloat16)
        return streams, new_state
