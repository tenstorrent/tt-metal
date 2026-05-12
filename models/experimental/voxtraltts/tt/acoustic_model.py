# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTNN acoustic ``FlowMatchingAudioTransformer``-equivalent module.

**Core trunk:** ``predict_velocity`` matches
``reference.cpu_flow_matching_acoustic.FlowMatchingAudioTransformerRef._predict_velocity``:
projections → 3× (RMSNorm + bidirectional GQA + residual + RMSNorm + SwiGLU) → final RMSNorm →
``acoustic_codebook_output``.

**Full ``forward``:** matches ``FlowMatchingAudioTransformerRef.forward``: TT linear for
``semantic_codebook_output``, masking + argmax on host, then Euler FM decoding with CFG (same math
as reference). Time embedding uses the reference sinusoidal formula on host (checkpoint has no
``inv_freq`` tensor); each step calls ``predict_velocity`` on device.

**Reuse:** ``VoxtralTTAttention`` with identity RoPE (``cos=1``, ``sin=0``). ``VoxtralTTMLP`` for
SwiGLU. ``VoxtralAcousticRMSNorm`` wraps common ``RMSNorm`` with ``COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC``.

FM trunk matmuls (``ttnn.linear``), attention SDPA, MLP linears, and acoustic norms share that config
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import ttnn

from models.experimental.voxtraltts.reference.cpu_flow_matching_acoustic import AudioSpecialTokens
from models.experimental.voxtraltts.tt.attention import VoxtralTTAttention
from models.experimental.voxtraltts.tt.mlp import VoxtralTTMLP
from models.experimental.voxtraltts.tt.rmsnorm import VoxtralAcousticRMSNorm
from models.experimental.voxtraltts.utils.config_helpers import COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL, load_voxtral_config
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict
from models.tt_transformers.tt.common import Mode


def extract_acoustic_state_dict(full_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "acoustic_transformer."
    return {k[len(prefix) :]: v for k, v in full_state_dict.items() if k.startswith(prefix)}


def _linear_weight_ttnn(w_out_in: torch.Tensor, device, dtype) -> ttnn.Tensor:
    # Checkpoint / nn.Linear: [out, in]. ttnn.linear expects [in, out].
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
        self._semantic_codebook_size = semantic_codebook_size
        self._acoustic_embeddings_levels = acoustic_embeddings_levels
        self._n_decoding_steps = n_decoding_steps

        self._empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)
        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        n_special = len(AudioSpecialTokens.all_special_tokens())
        self._tail_mask_start = n_special + semantic_codebook_size

        half = dim // 2
        self._inv_freq_cpu = torch.exp(-math.log(time_embedding_theta) * torch.arange(half).float() / half)
        self._timesteps_cpu = torch.linspace(0, 1, n_decoding_steps + 1)

        # Pre-upload inv_freq for on-device sinusoidal time embedding.
        self._inv_freq_tt = ttnn.from_torch(
            self._inv_freq_cpu.reshape(1, 1, -1),
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

        # Pre-compute additive semantic logit mask (-inf at empty and tail positions).
        _sem_size = sd["semantic_codebook_output.weight"].shape[0]
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
            )
            for i in range(n_layers)
        ]

        # Identity RoPE → no rotation (matches bidirectional acoustic attention).
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

            tt_xt = ttnn.from_torch(
                x_t.unsqueeze(1),
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_te = ttnn.from_torch(
                t_emb.unsqueeze(1),
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tt_llm = ttnn.from_torch(
                llm_hidden.unsqueeze(1),
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        s0 = ttnn.linear(
            tt_xt,
            self.w_input_proj,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(tt_xt)
        s1 = ttnn.linear(
            tt_te,
            self.w_time_proj,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(tt_te)
        s2 = ttnn.linear(
            tt_llm,
            self.w_llm_proj,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        if _tt_llm is None:
            ttnn.deallocate(tt_llm)
        debug_out: dict[str, torch.Tensor] | None = {} if return_debug else None
        if debug_out is not None:
            debug_out["proj_input"] = ttnn.to_torch(s0).float()
            debug_out["proj_time"] = ttnn.to_torch(s1).float()
            debug_out["proj_llm"] = ttnn.to_torch(s2).float()

        # Keep concat fully on TT.
        # Build sequence in rank-3 [B, 3, D] first, then reshape to [B, 1, 3, D].
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
        h3 = ttnn.concat([s0_tok, s1_tok, s2_tok], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
            # Match reference residual math on rank-3 [B, S, D] and then restore rank-4.
            # ``h4`` must be a tensor we own for reshape/deallocate (e.g. a clone of activations),
            # not the live activation tensor: reshaped views may alias device buffers that must not
            # be freed while the original tensor is still live elsewhere.
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
                use_legacy=True,
            )
            out4 = ttnn.reshape(out3, (h4_shape[0], 1, h4_shape[2], h4_shape[3]))
            ttnn.deallocate(h3)
            ttnn.deallocate(r3)
            ttnn.deallocate(out3)
            return out4

        _residual_mc = ttnn.DRAM_MEMORY_CONFIG

        for i in range(self.n_layers):
            # Snapshot activations before the sublayer; residual add uses this copy so reshape/dealloc
            # inside _residual_add_rank3 cannot corrupt the stream tensor (matches CPU ref PCC).
            residual_attn = ttnn.clone(h, dtype=self.dtype, memory_config=_residual_mc)
            normed = self.attn_norms[i](h, mode=Mode.DECODE)
            if debug_out is not None:
                debug_out[f"layer{i}.attn_norm"] = ttnn.to_torch(normed).float()
            attn_out = self.attentions[i](normed, cos, sin, attention_mask=None)
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
            normed_ff = self.ffn_norms[i](h, mode=Mode.DECODE)
            if debug_out is not None:
                debug_out[f"layer{i}.ffn_norm"] = ttnn.to_torch(normed_ff).float()
            ff_out = self.mlps[i](normed_ff)
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

        # First token only: [B, 1, 3, D] → [B, 1, 1, D]
        h_shape = tuple(h.shape)
        h0 = ttnn.slice(h, [0, 0, 0, 0], [h_shape[0], 1, 1, h_shape[-1]])
        ttnn.deallocate(h)

        vel = ttnn.linear(
            h0,
            self.w_velocity,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(h0)
        if debug_out is not None:
            debug_out["velocity"] = ttnn.to_torch(vel).float()
            return vel, debug_out
        return vel

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding ``[B, dim]`` (reference ``TimeEmbedding``, theta=10000). ``t``: ``[B, 1]``."""
        inv = self._inv_freq_cpu.to(device=t.device, dtype=t.dtype)
        emb = torch.einsum("bi, j -> bj", t, inv)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)

    def _time_embedding_tt(self, t_val: float, bsz: int) -> ttnn.Tensor:
        """On-device sinusoidal time embedding; returns ``[bsz, 1, dim]`` ttnn tensor."""
        emb = ttnn.multiply(self._inv_freq_tt, t_val, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos_emb = ttnn.cos(emb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin_emb = ttnn.sin(emb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(emb)
        te = ttnn.concat([cos_emb, sin_emb], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(cos_emb)
        ttnn.deallocate(sin_emb)
        if bsz > 1:
            te_expanded = ttnn.concat([te] * bsz, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(te)
            te = te_expanded
        return te

    def _decode_one_frame(
        self,
        semantic_code: torch.Tensor,
        llm_hidden: torch.Tensor,
        cfg_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Euler FM + CFG; same as ``FlowMatchingAudioTransformerRef.decode_one_frame``."""
        bsz = semantic_code.shape[0]
        device = llm_hidden.device
        dtype = llm_hidden.dtype
        should_decode = semantic_code != self._end_audio_token_id

        # RNG must stay torch for seed synchronisation with reference (tests use torch.manual_seed).
        x_0 = torch.randn(bsz, self.n_acoustic_out, device=device, dtype=dtype)
        timesteps = self._timesteps_cpu

        # Upload initial noise and LLM context to device once before the loop.
        sampled_tt = ttnn.from_torch(
            x_0.to(torch.bfloat16).unsqueeze(1),
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_llm = ttnn.from_torch(
            llm_hidden.to(torch.bfloat16).unsqueeze(1),
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_llm_zero = ttnn.zeros_like(tt_llm)
        tt_llm_batched = ttnn.concat([tt_llm, tt_llm_zero], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tt_llm)
        ttnn.deallocate(tt_llm_zero)

        ca = cfg_alpha.to(dtype=dtype, device=device)
        if ca.dim() == 0:
            cfg_a = ca.reshape(1, 1).expand(bsz, 1)
        elif ca.dim() == 1:
            cfg_a = ca.unsqueeze(1)
        else:
            cfg_a = ca
        cfg_scalar = float(cfg_a.flatten()[0].item())

        for i in range(len(timesteps) - 1):
            t_val = float(timesteps[i].item())
            dt_val = float((timesteps[i + 1] - timesteps[i]).item())

            te = self._time_embedding_tt(t_val, bsz)
            x_batched = ttnn.concat([sampled_tt, sampled_tt], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            te_batched = ttnn.concat([te, te], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(te)
            # ``x_batched``/``te_batched`` ownership transferred — freed inside _predict_velocity_impl.
            # ``tt_llm_batched`` borrowed — freed after the loop.
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
            v_cond_scaled = ttnn.multiply(v_cond, cfg_scalar, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(v_cond)
            v_uncond_scaled = ttnn.multiply(
                v_uncond, 1.0 - cfg_scalar, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(v_uncond)
            v_t_tt = ttnn.add(v_cond_scaled, v_uncond_scaled, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(v_cond_scaled)
            ttnn.deallocate(v_uncond_scaled)

            # Euler step on device: sampled = sampled + v_t * dt.
            # v_t_tt shape: [bsz, 1, 1, n_acoustic] → reshape to [bsz, 1, n_acoustic] to match sampled_tt.
            v_t_3d = ttnn.reshape(v_t_tt, (bsz, 1, self.n_acoustic_out))
            ttnn.deallocate(v_t_tt)
            v_scaled = ttnn.multiply(v_t_3d, dt_val, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(v_t_3d)
            new_sampled = ttnn.add(sampled_tt, v_scaled, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(sampled_tt)
            ttnn.deallocate(v_scaled)
            sampled_tt = new_sampled

        ttnn.deallocate(tt_llm_batched)

        # Post-processing: clamp to [-1, 1], scale to [0, levels-1].
        clamped_tt = ttnn.clip(sampled_tt, min=-1.0, max=1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sampled_tt)
        # Replicate reference formula exactly: ((clamped + 1) / 2) * (levels - 1)
        plus_one_tt = ttnn.add(clamped_tt, 1.0, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(clamped_tt)
        halved_tt = ttnn.multiply(plus_one_tt, 0.5, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(plus_one_tt)
        scaled_tt = ttnn.multiply(
            halved_tt,
            float(self._acoustic_embeddings_levels - 1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(halved_tt)
        rounded_tt = ttnn.round(scaled_tt, decimals=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scaled_tt)
        output_codes = ttnn.to_torch(rounded_tt).long().reshape(bsz, -1)
        ttnn.deallocate(rounded_tt)
        output_codes[~should_decode] = self._empty_audio_token_id
        offset = len(AudioSpecialTokens.all_special_tokens())
        return output_codes + offset

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        llm_hidden: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> ttnn.Tensor:
        return self._predict_velocity_impl(x_t, llm_hidden, t_emb, return_debug=False)

    def predict_velocity_debug(
        self,
        x_t: torch.Tensor,
        llm_hidden: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[ttnn.Tensor, dict[str, torch.Tensor]]:
        out = self._predict_velocity_impl(x_t, llm_hidden, t_emb, return_debug=True)
        assert isinstance(out, tuple)
        return out

    def forward(self, llm_hidden: torch.Tensor, cfg_alpha: torch.Tensor) -> torch.Tensor:
        """Semantic logits (TT) + on-device masked argmax + FM decode (device loop).

        Returns ``[B, 1 + n_acoustic_codebook]``: semantic token column-major compatible with reference.
        """
        bsz = llm_hidden.shape[0]

        tt_llm = ttnn.from_torch(
            llm_hidden.unsqueeze(1),
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sem_tt = ttnn.linear(
            tt_llm,
            self.w_semantic,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(tt_llm)

        # Apply mask on device: adds -inf at empty_audio and tail positions.
        sem_masked = ttnn.add(sem_tt, self._sem_mask_tt, dtype=self.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sem_tt)

        # On-device argmax over semantic logit dimension; sem_masked: [bsz, 1, sem_size].
        semantic_code_tt = ttnn.argmax(sem_masked, dim=2, keepdim=True)
        ttnn.deallocate(sem_masked)
        semantic_code = ttnn.to_torch(semantic_code_tt).long().reshape(bsz, 1)
        ttnn.deallocate(semantic_code_tt)

        acoustic_codes = self._decode_one_frame(semantic_code.squeeze(1), llm_hidden, cfg_alpha)

        return torch.cat([semantic_code, acoustic_codes], dim=1)
