# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT flow-matching acoustic head (semantic argmax + Euler FM + CFG). Matches CPU reference."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import ttnn

from models.experimental.voxtraltts.reference.cpu_flow_matching_acoustic import AudioSpecialTokens
from models.experimental.voxtraltts.tt.attention import VoxtralTTAttention
from models.experimental.voxtraltts.tt.mlp import VoxtralTTMLP
from models.experimental.voxtraltts.tt.rmsnorm import VoxtralAcousticRMSNorm
from models.experimental.voxtraltts.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC,
    COMPUTE_KERNEL_CONFIG_VOXTRAL_SEMANTIC,
)
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL, load_voxtral_config
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict
from models.tt_transformers.tt.common import Mode


def extract_acoustic_state_dict(full_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "acoustic_transformer."
    return {k[len(prefix) :]: v for k, v in full_state_dict.items() if k.startswith(prefix)}


def _linear_weight_ttnn(w_out_in: torch.Tensor, device, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        w_out_in.transpose(-2, -1).contiguous(),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class VoxtralTTAcousticModel:
    """TTNN ``FlowMatchingAudioTransformer``: semantic head + FM trunk + ``forward`` orchestration."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        state_dict: dict[str, torch.Tensor],
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        n_layers: int,
        n_acoustic_out: int,
        norm_eps: float,
        semantic_codebook_size: int,
        acoustic_embeddings_levels: int,
        n_decoding_steps: int = 8,
        time_embedding_theta: float = 10000.0,
        dtype: ttnn.DataType = ttnn.bfloat16,
        weight_cache_path: Path | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.n_acoustic_out = n_acoustic_out
        self._acoustic_embeddings_levels = acoustic_embeddings_levels
        self._n_decoding_steps = n_decoding_steps

        self._empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)
        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        n_special = len(AudioSpecialTokens.all_special_tokens())
        self._tail_mask_start = n_special + semantic_codebook_size
        self._fm_scale_factor = float(acoustic_embeddings_levels - 1)

        self._euler_t_vals = tuple(i / n_decoding_steps for i in range(n_decoding_steps))
        self._euler_dt_vals = tuple(1.0 / n_decoding_steps for _ in range(n_decoding_steps))

        self._fm_dram_mem_config = ttnn.DRAM_MEMORY_CONFIG
        self._semantic_dram_mem_config = ttnn.DRAM_MEMORY_CONFIG
        self._matmul_act_mem_config = ttnn.L1_MEMORY_CONFIG
        # Euler ODE state-accumulation dtype. Measured: fp32 vs bf16 give equal acoustic-code
        # agreement (~0.94 vs ~0.94 over 4 cases) — matching the bf16 reference dtype does NOT
        # reduce round() flips, because the residual is dominated by TT-vs-torch bf16 matmul ULP
        # in the velocity head, not by state accumulation. fp32 retained (marginally best, most
        # precise). Knob kept for future round-error investigation.
        self._fm_acc_dtype = ttnn.float32

        empty_codes = torch.full(
            (1, 1, n_acoustic_out),
            self._empty_audio_token_id + n_special,
            dtype=torch.int32,
        )
        self._empty_acoustic_output_codes_tt = ttnn.from_torch(
            empty_codes,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._fm_dram_mem_config,
        )
        self._acoustic_offset_u32_tt = ttnn.from_torch(
            torch.tensor([[[n_special]]], dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._fm_dram_mem_config,
        )

        half = dim // 2
        inv_freq_cpu = torch.exp(-math.log(time_embedding_theta) * torch.arange(half).float() / half)
        self._timesteps_cpu = torch.linspace(0, 1, n_decoding_steps + 1)

        self._inv_freq_tt = ttnn.from_torch(
            inv_freq_cpu.reshape(1, 1, -1),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        sd = state_dict
        self.w_input_proj = _linear_weight_ttnn(sd["input_projection.weight"], mesh_device, dtype)
        self.w_time_proj = _linear_weight_ttnn(sd["time_projection.weight"], mesh_device, dtype)
        self.w_llm_proj = _linear_weight_ttnn(sd["llm_projection.weight"], mesh_device, dtype)
        self.w_velocity = _linear_weight_ttnn(sd["acoustic_codebook_output.weight"], mesh_device, dtype)
        self.w_semantic = _linear_weight_ttnn(sd["semantic_codebook_output.weight"], mesh_device, dtype)

        _sem_size = sd["semantic_codebook_output.weight"].shape[0]
        self._sem_vocab_size = _sem_size
        _sem_mask = torch.zeros(1, 1, _sem_size)
        _sem_mask[0, 0, self._empty_audio_token_id] = float("-inf")
        _sem_mask[0, 0, self._tail_mask_start :] = float("-inf")
        self._sem_mask_tt = ttnn.from_torch(
            _sem_mask,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self._compute_kernel_config = COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC
        self._semantic_compute_kernel_config = COMPUTE_KERNEL_CONFIG_VOXTRAL_SEMANTIC
        # FM trunk uses short sequences (3 concat tokens); keep matmul activations in L1 (weights in DRAM).
        self._matmul_act_mem_config = ttnn.L1_MEMORY_CONFIG

        def _rms(layer_num: int, key: str) -> VoxtralAcousticRMSNorm:
            return VoxtralAcousticRMSNorm(
                device=mesh_device,
                dim=dim,
                eps=norm_eps,
                state_dict=sd,
                layer_num=layer_num,
                weight_key=key,
                weight_cache_path=None if weight_cache_path is None else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                is_distributed=None,
                tt_ccl=None,
            )

        self.attn_norms = [_rms(i, "attention_norm") for i in range(n_layers)]
        self.ffn_norms = [_rms(i, "ffn_norm") for i in range(n_layers)]
        self.final_norm = VoxtralAcousticRMSNorm(
            device=mesh_device,
            dim=dim,
            eps=norm_eps,
            state_dict=sd,
            layer_num=None,
            weight_key="norm",
            weight_cache_path=None if weight_cache_path is None else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            is_distributed=None,
            tt_ccl=None,
        )

        self.attentions = [
            VoxtralTTAttention(
                mesh_device,
                hidden_size=dim,
                num_attention_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=head_dim,
                state_dict=sd,
                weight_prefix=f"layers.{i}.attention",
                weight_dtype=dtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
                activation_memory_config=self._matmul_act_mem_config,
            )
            for i in range(n_layers)
        ]
        self.mlps = [
            VoxtralTTMLP(
                mesh_device,
                sd,
                w1_key=f"layers.{i}.feed_forward.w1",
                w2_key=f"layers.{i}.feed_forward.w2",
                w3_key=f"layers.{i}.feed_forward.w3",
                weight_dtype=dtype,
                output_dtype=dtype,
                exact_silu=True,
                compute_kernel_config=self._compute_kernel_config,
                activation_memory_config=self._matmul_act_mem_config,
            )
            for i in range(n_layers)
        ]

        self._cos_identity = torch.ones(1, 3, head_dim, dtype=torch.bfloat16)
        self._sin_identity = torch.zeros(1, 3, head_dim, dtype=torch.bfloat16)

    @classmethod
    def create_from_model_name(
        cls,
        mesh_device: ttnn.MeshDevice,
        *,
        model_name_or_path: str = DEFAULT_VOXTRAL_MODEL,
        dtype: ttnn.DataType = ttnn.bfloat16,
        weight_cache_path: Path | None = None,
    ) -> "VoxtralTTAcousticModel":
        cfg = load_voxtral_config(model_name_or_path)
        at = cfg.audio_model_args.acoustic_transformer_args
        am = cfg.audio_model_args
        full = _load_safetensors_state_dict(model_name_or_path)
        sd = extract_acoustic_state_dict(full)
        n_decode = getattr(at, "n_decoding_steps", None)
        if n_decode is None:
            n_decode = 8
        return cls(
            mesh_device,
            state_dict=sd,
            dim=at.dim,
            head_dim=at.head_dim,
            n_heads=at.n_heads,
            n_kv_heads=at.n_kv_heads,
            n_layers=at.n_layers,
            n_acoustic_out=am.n_acoustic_codebook,
            norm_eps=at.sigma,
            semantic_codebook_size=am.semantic_codebook_size,
            acoustic_embeddings_levels=am.acoustic_codebook_size,
            n_decoding_steps=n_decode,
            time_embedding_theta=10000.0,
            dtype=dtype,
            weight_cache_path=weight_cache_path,
        )

    def _predict_velocity_impl(
        self,
        x_t: torch.Tensor | None,
        llm_hidden: torch.Tensor | None,
        t_emb: torch.Tensor | None,
        *,
        _tt_xt: ttnn.Tensor | None = None,
        _tt_te: ttnn.Tensor | None = None,
        _tt_llm: ttnn.Tensor | None = None,
        return_debug: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, dict[str, torch.Tensor]]:
        """Batched FM velocity ``v_t`` with shape ``[B, n_acoustic_codebook]``.

        ``t_emb`` must already be the sinusoidal time embedding ``[B, dim]`` (output of
        ``TimeEmbedding`` in the CPU reference), **before** ``time_projection``.
        ``x_t`` is ``[B, n_acoustic_codebook]`` noise/state. ``llm_hidden`` is ``[B, input_dim]``.

        If ``_tt_xt``/``_tt_te``/``_tt_llm`` are provided the torch inputs are ignored.
        ``_tt_xt`` and ``_tt_te`` ownership is transferred (freed after projection);
        ``_tt_llm`` is borrowed (caller frees after the loop).
        """
        if _tt_xt is not None:
            bsz = _tt_xt.shape[0]
            tt_xt = _tt_xt
            tt_te = _tt_te
            tt_llm = _tt_llm
        else:
            x_t = x_t.to(dtype=torch.bfloat16)
            llm_hidden = llm_hidden.to(dtype=torch.bfloat16)
            t_emb = t_emb.to(dtype=torch.bfloat16)
            bsz = x_t.shape[0]

            _act_mem = self._matmul_act_mem_config
            tt_xt = ttnn.from_torch(
                x_t.unsqueeze(1),
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_act_mem,
            )
            tt_te = ttnn.from_torch(
                t_emb.unsqueeze(1),
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_act_mem,
            )
            tt_llm = ttnn.from_torch(
                llm_hidden.unsqueeze(1),
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_act_mem,
            )

        _lin_mem = self._matmul_act_mem_config
        s0 = ttnn.linear(
            tt_xt,
            self.w_input_proj,
            dtype=self.dtype,
            memory_config=_lin_mem,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(tt_xt)
        s1 = ttnn.linear(
            tt_te,
            self.w_time_proj,
            dtype=self.dtype,
            memory_config=_lin_mem,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(tt_te)
        s2 = ttnn.linear(
            tt_llm,
            self.w_llm_proj,
            dtype=self.dtype,
            memory_config=_lin_mem,
            compute_kernel_config=self._compute_kernel_config,
        )
        if _tt_llm is None:
            ttnn.deallocate(tt_llm)
        debug_out: dict[str, torch.Tensor] | None = {} if return_debug else None
        if debug_out is not None:
            debug_out["proj_input"] = ttnn.to_torch(s0).float()
            debug_out["proj_time"] = ttnn.to_torch(s1).float()
            debug_out["proj_llm"] = ttnn.to_torch(s2).float()

        def _as_token_3d(x: ttnn.Tensor) -> ttnn.Tensor:
            shape = tuple(x.shape)
            if len(shape) == 3:
                return ttnn.slice(x, [0, 0, 0], [bsz, 1, shape[-1]])
            if len(shape) == 4:
                x3 = ttnn.reshape(x, (shape[0], shape[2], shape[3]))
                return ttnn.slice(x3, [0, 0, 0], [bsz, 1, shape[-1]])
            raise RuntimeError(f"Unexpected projection rank {len(shape)} for acoustic token assembly: {shape}")

        s0_tok = _as_token_3d(s0)
        s1_tok = _as_token_3d(s1)
        s2_tok = _as_token_3d(s2)
        h3 = ttnn.concat([s0_tok, s1_tok, s2_tok], dim=1, memory_config=self._matmul_act_mem_config)
        h = ttnn.reshape(h3, (bsz, 1, 3, self.dim))
        ttnn.deallocate(h3)
        ttnn.deallocate(s0_tok)
        ttnn.deallocate(s1_tok)
        ttnn.deallocate(s2_tok)
        ttnn.deallocate(s0)
        ttnn.deallocate(s1)
        ttnn.deallocate(s2)
        if debug_out is not None:
            debug_out["concat_input"] = ttnn.to_torch(h).float()
        # ckeck for torch and ttnn
        cos = self._cos_identity
        sin = self._sin_identity

        def _slice_like(x: ttnn.Tensor, ref: ttnn.Tensor) -> ttnn.Tensor:
            x_shape = tuple(x.shape)
            r_shape = tuple(ref.shape)
            if x_shape == r_shape:
                return x
            if len(x_shape) != len(r_shape):
                raise RuntimeError(f"Rank mismatch for residual add: x={x_shape}, ref={r_shape}")
            begin = [0] * len(r_shape)
            end = list(r_shape)
            x_sliced = ttnn.slice(x, begin, end)
            ttnn.deallocate(x)
            return x_sliced

        def _residual_add_rank3(h4: ttnn.Tensor, r4: ttnn.Tensor) -> ttnn.Tensor:
            """Residual on rank-3 view; ``h4`` must be an owned clone, not live activations."""
            h4_shape = tuple(h4.shape)
            r4_shape = tuple(r4.shape)
            if len(h4_shape) != 4 or len(r4_shape) != 4:
                raise RuntimeError(f"Expected rank-4 tensors for residual add, got h={h4_shape}, r={r4_shape}")
            h3 = ttnn.reshape(h4, (h4_shape[0], h4_shape[2], h4_shape[3]))
            r3 = ttnn.reshape(r4, (r4_shape[0], r4_shape[2], r4_shape[3]))
            out3 = ttnn.add(
                h3,
                r3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.dtype,
            )
            out4 = ttnn.reshape(out3, (h4_shape[0], 1, h4_shape[2], h4_shape[3]))
            ttnn.deallocate(h3)
            ttnn.deallocate(r3)
            ttnn.deallocate(out3)
            return out4

        _residual_mc = self._matmul_act_mem_config

        for i in range(self.n_layers):
            residual_attn = ttnn.clone(h, dtype=self.dtype, memory_config=_residual_mc)
            # check for torch or ttnn
            normed = self.attn_norms[i](h, mode=Mode.DECODE, norm_config={"output_mem_config": _residual_mc})
            if debug_out is not None:
                debug_out[f"layer{i}.attn_norm"] = ttnn.to_torch(normed).float()
            # check for torch or ttnn
            attn_out = self.attentions[i](normed, cos, sin, attention_mask=None, activation_memory_config=_residual_mc)
            ttnn.deallocate(normed)
            attn_out = _slice_like(attn_out, h)
            attn_out = ttnn.to_memory_config(attn_out, h.memory_config())
            if debug_out is not None:
                debug_out[f"layer{i}.attn_out"] = ttnn.to_torch(attn_out).float()
            ttnn.deallocate(h)
            h = _residual_add_rank3(residual_attn, attn_out)
            ttnn.deallocate(residual_attn)
            ttnn.deallocate(attn_out)
            if debug_out is not None:
                debug_out[f"layer{i}.post_attn"] = ttnn.to_torch(h).float()

            residual_ffn = ttnn.clone(h, dtype=self.dtype, memory_config=_residual_mc)
            # check for torch or ttnn
            normed_ff = self.ffn_norms[i](h, mode=Mode.DECODE, norm_config={"output_mem_config": _residual_mc})
            if debug_out is not None:
                debug_out[f"layer{i}.ffn_norm"] = ttnn.to_torch(normed_ff).float()
            # check for torch or ttnn
            ff_out = self.mlps[i](normed_ff, activation_memory_config=_residual_mc)
            ttnn.deallocate(normed_ff)
            ff_out = _slice_like(ff_out, h)
            ff_out = ttnn.to_memory_config(ff_out, h.memory_config())
            if debug_out is not None:
                debug_out[f"layer{i}.ffn_out"] = ttnn.to_torch(ff_out).float()
            ttnn.deallocate(h)
            h = _residual_add_rank3(residual_ffn, ff_out)
            ttnn.deallocate(residual_ffn)
            ttnn.deallocate(ff_out)
            if debug_out is not None:
                debug_out[f"layer{i}.post_ffn"] = ttnn.to_torch(h).float()

        h = self.final_norm(h, mode=Mode.DECODE)
        if debug_out is not None:
            debug_out["final_norm"] = ttnn.to_torch(h).float()

        # check for torch or ttnn
        h_shape = tuple(h.shape)
        h0 = ttnn.slice(h, [0, 0, 0, 0], [h_shape[0], 1, 1, h_shape[-1]])
        ttnn.deallocate(h)

        vel = ttnn.linear(
            h0,
            self.w_velocity,
            dtype=self.dtype,
            memory_config=self._matmul_act_mem_config,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(h0)
        if debug_out is not None:
            debug_out["velocity"] = ttnn.to_torch(vel).float()
            return vel, debug_out
        return vel

    def semantic_logits_tt(self, llm_hidden_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Masked fp32 semantic logits on device ``[B, 1, vocab]``."""
        llm_fp32 = llm_hidden_tt
        if llm_fp32.dtype != ttnn.float32:
            llm_fp32 = ttnn.typecast(llm_fp32, ttnn.float32, memory_config=self._semantic_dram_mem_config)
        sem_tt = ttnn.linear(
            llm_fp32,
            self.w_semantic,
            dtype=ttnn.float32,
            memory_config=self._matmul_act_mem_config,
            compute_kernel_config=self._semantic_compute_kernel_config,
        )
        if llm_fp32 is not llm_hidden_tt and llm_fp32.is_allocated():
            ttnn.deallocate(llm_fp32)
        masked = ttnn.add(
            sem_tt,
            self._sem_mask_tt,
            dtype=ttnn.float32,
            memory_config=self._semantic_dram_mem_config,
        )
        ttnn.deallocate(sem_tt)
        sem_shape = tuple(masked.shape)
        if sem_shape[-1] > self._sem_vocab_size:
            masked = ttnn.slice(
                masked,
                [0, 0, 0],
                [sem_shape[0], sem_shape[1], self._sem_vocab_size],
            )
        return masked

    def fm_noise_tt(self, bsz: int, seed: int) -> ttnn.Tensor:
        """FM initial noise ``[bsz, 1, n_acoustic]`` drawn on host to match the CPU reference RNG.

        The reference (``decode_one_frame``) draws ``x_0 = torch.randn(B, n_acoustic, dtype=bf16)``
        from the global torch RNG (``_noise_scale == 1.0``). A seeded ``torch.Generator`` reproduces
        that exact stream, so both ODEs start from identical noise. ``ttnn.randn`` is a *different*
        RNG and desyncs the FM start, dropping acoustic-code agreement to ~chance.
        """
        g = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(bsz, self.n_acoustic_out, generator=g, dtype=torch.bfloat16).reshape(
            bsz, 1, self.n_acoustic_out
        )
        return ttnn.from_torch(
            noise,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._fm_dram_mem_config,
        )

    def fm_pre_round_scaled_tt(self, sampled_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Clamp → scale FSQ values on device (fp32), pre-``round``."""
        mem = self._fm_dram_mem_config
        if sampled_tt.dtype == ttnn.float32:
            sampled_f32 = sampled_tt
        else:
            sampled_f32 = ttnn.typecast(sampled_tt, ttnn.float32, memory_config=mem)
            if sampled_f32 is not sampled_tt and sampled_tt.is_allocated():
                ttnn.deallocate(sampled_tt)
        clamped = ttnn.clip(sampled_f32, min=-1.0, max=1.0, memory_config=mem)
        if sampled_f32 is not clamped and sampled_f32.is_allocated():
            ttnn.deallocate(sampled_f32)
        plus_one = ttnn.add(clamped, 1.0, dtype=ttnn.float32, memory_config=mem)
        ttnn.deallocate(clamped)
        halved = ttnn.multiply(plus_one, 0.5, dtype=ttnn.float32, memory_config=mem)
        ttnn.deallocate(plus_one)
        return ttnn.multiply(halved, self._fm_scale_factor, dtype=ttnn.float32, memory_config=mem)

    def forward(
        self,
        llm_hidden_tt: ttnn.Tensor,
        noise_tt: ttnn.Tensor,
        cfg_scalar: float,
    ) -> ttnn.Tensor:
        """One acoustic frame on device → ``[B, 1, 1+n_acoustic]`` uint32 ROW_MAJOR discrete codes.

        Normal frames: ``ttnn.concat`` on device (requires matching ``memory_config``; skip ``ttnn.where``).
        End-audio frames: host ``torch.where`` + ``from_torch`` (``ttnn.where`` zeros uint32 acoustic readback).
        """
        llm_tile = self._llm_hidden_tile_bf16(llm_hidden_tt)
        owned_tile = llm_tile is not llm_hidden_tt
        bsz = int(llm_tile.shape[0])

        llm_sem = ttnn.typecast(llm_tile, ttnn.float32, memory_config=self._semantic_dram_mem_config)
        masked_logits = self.semantic_logits_tt(llm_sem)
        if llm_sem is not llm_tile and llm_sem.is_allocated():
            ttnn.deallocate(llm_sem)
        sem_idx = ttnn.argmax(masked_logits, dim=-1)
        ttnn.deallocate(masked_logits)
        semantic_code_tt = ttnn.reshape(sem_idx, (bsz, 1, 1))
        ttnn.deallocate(sem_idx)
        semantic_code_tt = ttnn.typecast(semantic_code_tt, ttnn.uint32)
        semantic_code_tt = ttnn.to_layout(semantic_code_tt, ttnn.TILE_LAYOUT, memory_config=self._fm_dram_mem_config)

        acoustic_tt = self._fm_decode_codes_tt(llm_tile, noise_tt, cfg_scalar)
        if owned_tile and llm_tile.is_allocated():
            ttnn.deallocate(llm_tile)

        is_end = ttnn.eq(semantic_code_tt, self._end_audio_token_id_tt)

        # End-audio only: ttnn.where on uint32 [B,1,36] zeros acoustic cols on readback (even is_end=False); use host torch.where/cat then from_torch.
        if ttnn.to_torch(is_end).reshape(-1).bool().any():
            ttnn.deallocate(is_end)
            sem_host = ttnn.to_torch(semantic_code_tt).reshape(bsz, 1).long()
            ac_host = ttnn.to_torch(acoustic_tt).reshape(bsz, self.n_acoustic_out).long()
            ttnn.deallocate(semantic_code_tt)
            ttnn.deallocate(acoustic_tt)
            empty_code = self._empty_audio_token_id + self._acoustic_special_token_offset
            end_mask = sem_host == self._end_audio_token_id
            ac_host = torch.where(
                end_mask.expand(-1, self.n_acoustic_out),
                torch.full_like(ac_host, empty_code),
                ac_host,
            )
            codes_host = torch.cat([sem_host, ac_host], dim=1).contiguous()
            return ttnn.from_torch(
                codes_host.to(torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=self._fm_dram_mem_config,
            )

        ttnn.deallocate(is_end)
        codes_tt = ttnn.concat(
            [semantic_code_tt, acoustic_tt],
            dim=2,
            memory_config=self._fm_dram_mem_config,
        )
        ttnn.deallocate(semantic_code_tt)
        ttnn.deallocate(acoustic_tt)
        return ttnn.to_layout(codes_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=self._fm_dram_mem_config)

    def _llm_hidden_tile_bf16(self, llm_hidden_tt: ttnn.Tensor) -> ttnn.Tensor:
        work = llm_hidden_tt
        shape = tuple(work.shape)
        if len(shape) == 2:
            work = ttnn.reshape(work, (int(shape[0]), 1, int(shape[1])))
        if work.dtype != self.dtype:
            work = ttnn.typecast(work, self.dtype, memory_config=self._matmul_act_mem_config)
        if work.layout != ttnn.TILE_LAYOUT:
            work = ttnn.to_layout(work, ttnn.TILE_LAYOUT, memory_config=self._matmul_act_mem_config)
        return work

    def _time_embedding_tt(self, t_val: float, bsz: int) -> ttnn.Tensor:
        """On-device sinusoidal time embedding; returns ``[bsz, 1, dim]`` ttnn tensor."""
        emb = ttnn.multiply(self._inv_freq_tt, t_val, dtype=self.dtype, memory_config=self._matmul_act_mem_config)
        cos_emb = ttnn.cos(emb, memory_config=self._matmul_act_mem_config)
        sin_emb = ttnn.sin(emb, memory_config=self._matmul_act_mem_config)
        ttnn.deallocate(emb)
        te = ttnn.concat([cos_emb, sin_emb], dim=2, memory_config=self._matmul_act_mem_config)
        ttnn.deallocate(cos_emb)
        ttnn.deallocate(sin_emb)
        if bsz > 1:
            te_expanded = ttnn.concat([te] * bsz, dim=0, memory_config=self._matmul_act_mem_config)
            ttnn.deallocate(te)
            te = te_expanded
        return te

    def _sampled_tt_for_velocity(self, sampled_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Cast FM Euler state to bf16 for the velocity trunk (weights are bf16)."""
        if sampled_tt.dtype == self.dtype:
            return sampled_tt
        return ttnn.typecast(sampled_tt, self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _euler_integrate_sampled(self, sampled_tt: ttnn.Tensor, v_t_3d: ttnn.Tensor, dt_val: float) -> ttnn.Tensor:
        mem = self._fm_dram_mem_config
        acc = self._fm_acc_dtype
        v_scaled = ttnn.multiply(v_t_3d, dt_val, dtype=acc, memory_config=mem)
        ttnn.deallocate(v_t_3d)
        if sampled_tt.dtype == acc:
            sampled_acc = sampled_tt
        else:
            sampled_acc = ttnn.typecast(sampled_tt, acc, memory_config=mem)
            ttnn.deallocate(sampled_tt)
        new_sampled = ttnn.add(sampled_acc, v_scaled, dtype=acc, memory_config=mem)
        ttnn.deallocate(sampled_acc)
        ttnn.deallocate(v_scaled)
        return new_sampled

    def _fm_pre_round_scaled_from_sampled_tt(self, sampled_tt: ttnn.Tensor, bsz: int) -> torch.Tensor:
        """Return pre-``round()`` scaled FSQ values ``[bsz, n_acoustic]`` on host (fp32 path on device)."""
        mem = ttnn.DRAM_MEMORY_CONFIG
        if sampled_tt.dtype == ttnn.float32:
            sampled_f32 = sampled_tt
        else:
            sampled_f32 = ttnn.typecast(sampled_tt, ttnn.float32, memory_config=mem)
            ttnn.deallocate(sampled_tt)
        clamped = ttnn.clip(sampled_f32, min=-1.0, max=1.0, memory_config=mem)
        ttnn.deallocate(sampled_f32)
        plus_one = ttnn.add(clamped, 1.0, dtype=ttnn.float32, memory_config=mem)
        ttnn.deallocate(clamped)
        halved = ttnn.multiply(plus_one, 0.5, dtype=ttnn.float32, memory_config=mem)
        ttnn.deallocate(plus_one)
        scaled = ttnn.multiply(
            halved,
            float(self._acoustic_embeddings_levels - 1),
            dtype=ttnn.float32,
            memory_config=mem,
        )
        ttnn.deallocate(halved)
        scaled_host = ttnn.to_torch(scaled).float().reshape(bsz, -1)
        ttnn.deallocate(scaled)
        return scaled_host

    def _fm_round_acoustic_codes_from_sampled_tt(self, sampled_tt: ttnn.Tensor, bsz: int) -> torch.Tensor:
        """Clamp → scale → ``round`` in fp32; host ``[bsz, n_acoustic]`` long (pre special-token offset)."""
        scaled_host = self._fm_pre_round_scaled_from_sampled_tt(sampled_tt, bsz)
        return scaled_host.round().long()

    def fm_pre_round_scaled_codes_tt(
        self,
        llm_hidden_tt: ttnn.Tensor,
        noise_tt: ttnn.Tensor,
        cfg_scalar: float,
    ) -> ttnn.Tensor:
        """Continuous pre-``round`` FSQ value ``[bsz, 1, n_acoustic]`` (``round`` of this -> acoustic codes).

        Numerical-accuracy probe: this is the continuous signal whose ``round`` produces the discrete
        acoustic codes, so its PCC vs the reference isolates op accuracy from FSQ-boundary code flips.
        """
        llm_tile = self._llm_hidden_tile_bf16(llm_hidden_tt)
        bsz = int(llm_hidden_tt.shape[0])
        sampled_tt = self._fm_decode_sampled_tt(llm_tile, noise_tt, cfg_scalar)
        if llm_tile is not llm_hidden_tt and llm_tile.is_allocated():
            ttnn.deallocate(llm_tile)
        scaled = self.fm_pre_round_scaled_tt(sampled_tt)
        out = ttnn.reshape(scaled, (bsz, 1, self.n_acoustic_out))
        if out is not scaled and scaled.is_allocated():
            ttnn.deallocate(scaled)
        return out

    def _fm_decode_codes_tt(
        self,
        llm_hidden_tt: ttnn.Tensor,
        noise_tt: ttnn.Tensor,
        cfg_scalar: float,
    ) -> ttnn.Tensor:
        bsz = int(llm_hidden_tt.shape[0])
        sampled_tt = self._fm_decode_sampled_tt(llm_hidden_tt, noise_tt, cfg_scalar)
        return self._fm_round_acoustic_codes_tt(sampled_tt, bsz)

    def _fm_decode_sampled_tt(
        self,
        llm_hidden_tt: ttnn.Tensor,
        noise_tt: ttnn.Tensor,
        cfg_scalar: float,
    ) -> ttnn.Tensor:
        """Run the flow-matching Euler ODE; return the continuous ``sampled`` tensor (pre-clamp/scale/round)."""
        bsz = int(llm_hidden_tt.shape[0])
        sampled_tt = ttnn.typecast(
            ttnn.clone(noise_tt),
            self._fm_acc_dtype,
            memory_config=self._fm_dram_mem_config,
        )

        # need to use ttnn.randn
        x_0 = torch.randn(bsz, self.n_acoustic_out, device=device, dtype=dtype)
        timesteps = self._timesteps_cpu  ### is timesteps statci then move to init, else justify

        sampled_tt = ttnn.from_torch(
            x_0.to(torch.bfloat16).unsqueeze(1),
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # float32 Euler state must stay in DRAM — L1 is not supported for fp32 accumulation kernels
        sampled_tt = ttnn.typecast(sampled_tt, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_llm = ttnn.from_torch(
            llm_hidden.to(torch.bfloat16).unsqueeze(1),
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._matmul_act_mem_config,
        )
        tt_llm_zero = ttnn.zeros_like(tt_llm)
        tt_llm_batched = ttnn.concat([tt_llm, tt_llm_zero], dim=0, memory_config=self._matmul_act_mem_config)
        ttnn.deallocate(tt_llm)
        ttnn.deallocate(tt_llm_zero)

        ca = cfg_alpha.to(dtype=dtype, device=device)  ##### check this for torch or ttnn
        if ca.dim() == 0:
            cfg_a = ca.reshape(1, 1).expand(bsz, 1)
        elif ca.dim() == 1:
            cfg_a = ca.unsqueeze(1)
        else:
            cfg_a = ca
        cfg_scalar = float(cfg_a.flatten()[0].item())

        fm_debug: dict[str, torch.Tensor] | None = {} if collect_fm_debug else None
        for i in range(len(timesteps) - 1):
            t_val = float(timesteps[i].item())
            dt_val = float((timesteps[i + 1] - timesteps[i]).item())

            te = self._time_embedding_tt(t_val, bsz)
            #  check the data type as  typecasted from fp32 to bf16
            x_in = self._sampled_tt_for_velocity(sampled_tt)
            x_batched = ttnn.concat([x_in, x_in], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if x_in is not sampled_tt and x_in.is_allocated():
                ttnn.deallocate(x_in)
            # te is L1 (from _time_embedding_tt); concat output must match input memory
            te_batched = ttnn.concat([te, te], dim=0, memory_config=self._matmul_act_mem_config)
            ttnn.deallocate(te)
            if collect_fm_debug and i == 0:
                v_out = self._predict_velocity_impl(
                    None,
                    None,
                    None,
                    _tt_xt=x_batched,
                    _tt_te=te_batched,
                    _tt_llm=tt_llm_batched,
                    return_debug=True,
                )
                assert isinstance(v_out, tuple)
                v_tt, fm_debug = v_out
            else:
                v_tt = self._predict_velocity_impl(
                    None,
                    None,
                    None,
                    _tt_xt=x_batched,
                    _tt_te=te_batched,
                    _tt_llm=tt_llm_batched,
                    return_debug=False,
                )

            v_shape = tuple(v_tt.shape)
            v_cond = ttnn.slice(v_tt, [0, 0, 0, 0], [bsz, v_shape[1], v_shape[2], v_shape[3]])
            v_uncond = ttnn.slice(v_tt, [bsz, 0, 0, 0], [2 * bsz, v_shape[1], v_shape[2], v_shape[3]])
            ttnn.deallocate(v_tt)
            # velocity must stay in DRAM here — feeds into _euler_integrate_sampled (float32 DRAM ops)
            v_cond_scaled = ttnn.multiply(v_cond, cfg_scalar, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(v_cond)
            v_uncond_scaled = ttnn.multiply(
                v_uncond, 1.0 - cfg_scalar, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(v_uncond)
            v_t_tt = ttnn.add(v_cond_scaled, v_uncond_scaled, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(v_cond_scaled)
            ttnn.deallocate(v_uncond_scaled)

            v_t_3d = ttnn.reshape(v_t_tt, (bsz, 1, self.n_acoustic_out))
            ttnn.deallocate(v_t_tt)
            # check for torch or ttnn
            sampled_tt = self._euler_integrate_sampled(sampled_tt, v_t_3d, dt_val)

        ttnn.deallocate(tt_llm_batched)
        return sampled_tt
