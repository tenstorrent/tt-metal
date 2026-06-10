# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT flow-matching acoustic head (semantic argmax + Euler FM + CFG)."""

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


def _build_acoustic_1d_mcast_configs(dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
    """1D mcast program configs for acoustic FM transformer decode, single device.

    CFG doubles the batch to 2 for every euler step, so the effective M is 2 tiles
    (bsz=2 × seq_tiles=1). All configs use per_core_M=2, mcast_in0=True on an
    8×6=48-core grid.

    Returns (ff1_3_config, ff2_config, wqkv_config, wo_config, proj_config) where
    proj_config is reused for the w_time_proj and w_llm_proj dim→dim linears.
    """
    TILE = ttnn.TILE_SIZE  # 32

    def _largest_divisor(n: int, max_d: int = 8) -> int:
        for d in range(max_d, 0, -1):
            if n % d == 0:
                return d
        return 1

    def _best_subblock_w(per_core_n: int) -> int:
        for w in (4, 3, 2, 1):
            if per_core_n % w == 0:
                return w
        return 1

    def _cfg(
        gx: int, gy: int, kt: int, nc: int, nt: int, pm: int = 2
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        per_core_n = (nt + nc - 1) // nc
        in0_bw = _largest_divisor(kt // nc)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=in0_bw,
            out_subblock_h=1,
            out_subblock_w=_best_subblock_w(per_core_n),
            per_core_M=pm,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def _cfg_ff2_wo(
        gx: int, gy: int, kt: int, nc: int, nt: int, pm: int = 2
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        # For the contracting matmuls (w2, wo) the K is large; in0_block_w is capped at 4.
        per_core_n = (nt + nc - 1) // nc
        in0_bw = _largest_divisor(kt, max_d=4)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=in0_bw,
            out_subblock_h=1,
            out_subblock_w=_best_subblock_w(per_core_n),
            per_core_M=pm,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    # Sweep winner grid for single-device acoustic decode (matches Voxtral text model).
    gx, gy = 8, 6
    nc = gx * gy  # 48

    kt_dim = dim // TILE  # 96  (K=3072)
    kt_hid = hidden_dim // TILE  # 288 (K=9216)
    nt_dim = dim // TILE  # 96  (N=3072)
    nt_hid = hidden_dim // TILE  # 288 (N=9216)
    nt_wqkv = (n_heads + 2 * n_kv_heads) * head_dim // TILE  # 192 (N=6144)
    kt_wo = n_heads * head_dim // TILE  # 128 (K=4096)

    ff1_3_config = _cfg(gx, gy, kt_dim, nc, nt_hid)  # w1/w3: K=3072→N=9216
    ff2_config = _cfg_ff2_wo(gx, gy, kt_hid, nc, nt_dim)  # w2:   K=9216→N=3072
    wqkv_config = _cfg(gx, gy, kt_dim, nc, nt_wqkv)  # wqkv: K=3072→N=6144
    wo_config = _cfg_ff2_wo(gx, gy, kt_wo, nc, nt_dim)  # wo:   K=4096→N=3072
    proj_config = _cfg_ff2_wo(gx, gy, kt_dim, nc, nt_dim)  # proj: K=3072→N=3072

    return ff1_3_config, ff2_config, wqkv_config, wo_config, proj_config


def _linear_weight_ttnn(w_out_in: torch.Tensor, device, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        w_out_in.transpose(-2, -1).contiguous(),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class VoxtralTTAcousticModel:
    """TTNN flow-matching acoustic transformer (semantic head + FM trunk). All compute is device-resident."""

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
        self._acoustic_special_token_offset = n_special
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
        self._inv_freq_tt = ttnn.from_torch(
            inv_freq_cpu.reshape(1, 1, -1),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._fm_dram_mem_config,
        )

        sd = state_dict
        self.w_input_proj = _linear_weight_ttnn(sd["input_projection.weight"], mesh_device, dtype)
        self.w_time_proj = _linear_weight_ttnn(sd["time_projection.weight"], mesh_device, dtype)
        self.w_llm_proj = _linear_weight_ttnn(sd["llm_projection.weight"], mesh_device, dtype)
        self.w_velocity = _linear_weight_ttnn(sd["acoustic_codebook_output.weight"], mesh_device, dtype)
        self.w_semantic = _linear_weight_ttnn(sd["semantic_codebook_output.weight"], mesh_device, dtype)

        _sem_size = sd["semantic_codebook_output.weight"].shape[0]
        self._sem_vocab_size = _sem_size
        _sem_mask = torch.zeros(1, 1, _sem_size, dtype=torch.float32)
        _sem_mask[0, 0, self._empty_audio_token_id] = float("-inf")
        _sem_mask[0, 0, self._tail_mask_start :] = float("-inf")
        self._sem_mask_tt = ttnn.from_torch(
            _sem_mask,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._semantic_dram_mem_config,
        )
        self._end_audio_token_id_tt = ttnn.from_torch(
            torch.tensor([[[self._end_audio_token_id]]], dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._fm_dram_mem_config,
        )

        self._compute_kernel_config = COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC
        self._semantic_compute_kernel_config = COMPUTE_KERNEL_CONFIG_VOXTRAL_SEMANTIC

        # Infer hidden_dim from the w1 weight shape (dim→hidden_dim after transpose).
        _w1_key = "layers.0.feed_forward.w1"
        _w1_key_w = f"{_w1_key}.weight"
        _w1_raw = sd.get(_w1_key_w, sd.get(_w1_key))
        _hidden_dim = int(_w1_raw.shape[0]) if _w1_raw is not None else 9216

        # 1D-mcast program configs for all FM transformer matmuls.
        _ff1_3_cfg, _ff2_cfg, _wqkv_cfg, _wo_cfg, _proj_cfg = _build_acoustic_1d_mcast_configs(
            dim=dim,
            hidden_dim=_hidden_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        self._ff1_3_prg_config = _ff1_3_cfg
        self._ff2_prg_config = _ff2_cfg
        self._wqkv_prg_config = _wqkv_cfg
        self._wo_prg_config = _wo_cfg
        self._proj_prg_config = _proj_cfg

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

        # BFP8_B weights: 2× bandwidth reduction vs BF16 with same 1D-mcast program configs.
        # DS sharding requires fuse_batch semantics that 1D-mcast provides but DS does not;
        # the acoustic FM input (2,1,3,dim) needs per_core_M=2 via fuse_batch, which DS cannot match.
        _fm_wdtype = ttnn.bfloat8_b

        self.attentions = [
            VoxtralTTAttention(
                mesh_device,
                hidden_size=dim,
                num_attention_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=head_dim,
                state_dict=sd,
                weight_prefix=f"layers.{i}.attention",
                weight_dtype=_fm_wdtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
                activation_memory_config=self._matmul_act_mem_config,
                wqkv_program_config=self._wqkv_prg_config,
                wo_program_config=self._wo_prg_config,
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
                weight_dtype=_fm_wdtype,
                output_dtype=dtype,
                exact_silu=True,
                compute_kernel_config=self._compute_kernel_config,
                activation_memory_config=self._matmul_act_mem_config,
                ff1_3_program_config=self._ff1_3_prg_config,
                ff2_program_config=self._ff2_prg_config,
            )
            for i in range(n_layers)
        ]

    @classmethod
    def create_from_model_name(
        cls,
        mesh_device: ttnn.MeshDevice,
        *,
        model_name_or_path: str = DEFAULT_VOXTRAL_MODEL,
        dtype: ttnn.DataType = ttnn.bfloat16,
        weight_cache_path: Path | None = None,
        preloaded_state_dict: dict | None = None,
    ) -> "VoxtralTTAcousticModel":
        cfg = load_voxtral_config(model_name_or_path)
        at = cfg.audio_model_args.acoustic_transformer_args
        am = cfg.audio_model_args
        # Reuse an already-loaded checkpoint when provided to avoid loading the full
        # ~7.45GiB safetensors into host RAM a second time (host OOM risk).
        full = (
            preloaded_state_dict
            if preloaded_state_dict is not None
            else _load_safetensors_state_dict(model_name_or_path)
        )
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

    def predict_velocity_tt(
        self,
        x_t_tt: ttnn.Tensor,
        t_emb_tt: ttnn.Tensor,
        llm_hidden_tt: ttnn.Tensor,
        *,
        borrow_llm: bool = False,
    ) -> ttnn.Tensor:
        """FM velocity ``[B, …, n_acoustic]`` from device-resident inputs (projections + transformer trunk)."""
        bsz = int(x_t_tt.shape[0])
        lin_mem = self._matmul_act_mem_config

        s0 = ttnn.linear(
            x_t_tt,
            self.w_input_proj,
            dtype=self.dtype,
            memory_config=lin_mem,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(x_t_tt)
        s1 = ttnn.linear(
            t_emb_tt,
            self.w_time_proj,
            dtype=self.dtype,
            memory_config=lin_mem,
            compute_kernel_config=self._compute_kernel_config,
            program_config=self._proj_prg_config,
        )
        ttnn.deallocate(t_emb_tt)
        s2 = ttnn.linear(
            llm_hidden_tt,
            self.w_llm_proj,
            dtype=self.dtype,
            memory_config=lin_mem,
            compute_kernel_config=self._compute_kernel_config,
            program_config=self._proj_prg_config,
        )
        if not borrow_llm:
            ttnn.deallocate(llm_hidden_tt)

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
        h3 = ttnn.concat([s0_tok, s1_tok, s2_tok], dim=1, memory_config=lin_mem)
        h = ttnn.reshape(h3, (bsz, 1, 3, self.dim))
        ttnn.deallocate(h3)
        ttnn.deallocate(s0_tok)
        ttnn.deallocate(s1_tok)
        ttnn.deallocate(s2_tok)
        ttnn.deallocate(s0)
        ttnn.deallocate(s1)
        ttnn.deallocate(s2)

        def _slice_like(x: ttnn.Tensor, ref: ttnn.Tensor) -> ttnn.Tensor:
            x_shape = tuple(x.shape)
            r_shape = tuple(ref.shape)
            if x_shape == r_shape:
                return x
            if len(x_shape) != len(r_shape):
                raise RuntimeError(f"Rank mismatch for residual add: x={x_shape}, ref={r_shape}")
            x_sliced = ttnn.slice(x, [0] * len(r_shape), list(r_shape))
            ttnn.deallocate(x)
            return x_sliced

        def _residual_add_rank3(h4: ttnn.Tensor, r4: ttnn.Tensor) -> ttnn.Tensor:
            # Direct 4D add — avoids reshape to 3D which produced ROW_MAJOR in L1 and
            # triggered TilizeWithValPadding before every subsequent TILE_LAYOUT op.
            return ttnn.add(h4, r4, memory_config=self._matmul_act_mem_config, dtype=self.dtype)

        residual_mc = self._matmul_act_mem_config
        for i in range(self.n_layers):
            residual_attn = ttnn.clone(h, dtype=self.dtype, memory_config=residual_mc)
            normed = self.attn_norms[i](h, mode=Mode.DECODE, norm_config={"output_mem_config": residual_mc})
            attn_out = self.attentions[i](normed, None, None, attention_mask=None, activation_memory_config=residual_mc)
            ttnn.deallocate(normed)
            attn_out = _slice_like(attn_out, h)
            attn_out = ttnn.to_memory_config(attn_out, h.memory_config())
            ttnn.deallocate(h)
            h = _residual_add_rank3(residual_attn, attn_out)
            ttnn.deallocate(residual_attn)
            ttnn.deallocate(attn_out)

            residual_ffn = ttnn.clone(h, dtype=self.dtype, memory_config=residual_mc)
            normed_ff = self.ffn_norms[i](h, mode=Mode.DECODE, norm_config={"output_mem_config": residual_mc})
            ff_out = self.mlps[i](normed_ff, activation_memory_config=residual_mc)
            ttnn.deallocate(normed_ff)
            ff_out = _slice_like(ff_out, h)
            ff_out = ttnn.to_memory_config(ff_out, h.memory_config())
            ttnn.deallocate(h)
            h = _residual_add_rank3(residual_ffn, ff_out)
            ttnn.deallocate(residual_ffn)
            ttnn.deallocate(ff_out)

        h = self.final_norm(h, mode=Mode.DECODE)
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
        return vel

    def semantic_logits_tt(self, llm_hidden_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Masked fp32 semantic logits on device ``[B, 1, vocab]``."""
        llm_fp32 = llm_hidden_tt
        if llm_fp32.dtype != ttnn.float32:
            llm_fp32 = ttnn.typecast(llm_fp32, ttnn.float32, memory_config=self._matmul_act_mem_config)
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
            memory_config=self._matmul_act_mem_config,
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
        return ttnn.from_torch(
            self.fm_noise_host_torch(bsz, seed),
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._fm_dram_mem_config,
        )

    def fm_noise_host_torch(self, bsz: int, seed: int) -> torch.Tensor:
        """Host ``[bsz, 1, n_acoustic]`` bf16 FM noise (seeded torch RNG). For trace input staging."""
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(bsz, self.n_acoustic_out, generator=g, dtype=torch.bfloat16).reshape(
            bsz, 1, self.n_acoustic_out
        )

    def fm_noise_host_tt(self, bsz: int, seed: int) -> ttnn.Tensor:
        """Host (no device) ttnn noise tensor matching the FM noise buffer spec — for copy_host_to_device."""
        return ttnn.from_torch(self.fm_noise_host_torch(bsz, seed), dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

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

        acoustic_tt = self._fm_decode_codes_tt(llm_tile, noise_tt, cfg_scalar)
        codes = self._codes_from_fm(llm_tile, acoustic_tt, bsz)
        if owned_tile and llm_tile.is_allocated():
            ttnn.deallocate(llm_tile)
        return codes

    def fm_decode_codes_tt(self, llm_hidden_tt: ttnn.Tensor, noise_tt: ttnn.Tensor, cfg_scalar: float) -> ttnn.Tensor:
        """Public alias for the traceable Euler-FM core (semantic + end-audio handling stay outside)."""
        return self._fm_decode_codes_tt(llm_hidden_tt, noise_tt, cfg_scalar)

    def codes_from_fm(self, llm_hidden_tt: ttnn.Tensor, acoustic_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Finalize discrete codes from a (possibly trace-produced) acoustic tensor: semantic argmax,
        end-audio masking, concat. Untraced (per-frame data-dependent host ``is_end`` branch)."""
        llm_tile = self._llm_hidden_tile_bf16(llm_hidden_tt)
        codes = self._codes_from_fm(llm_tile, acoustic_tt, int(llm_tile.shape[0]))
        if llm_tile is not llm_hidden_tt and llm_tile.is_allocated():
            ttnn.deallocate(llm_tile)
        return codes

    def _codes_from_fm(self, llm_tile: ttnn.Tensor, acoustic_tt: ttnn.Tensor, bsz: int) -> ttnn.Tensor:
        """Semantic argmax + end-audio mask + concat with the FM acoustic codes -> discrete codes."""
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

        is_end = ttnn.eq(semantic_code_tt, self._end_audio_token_id_tt)

        # End-audio frame: ttnn.where on uint32 zeros false-branch (TTNN bug).
        # Fix: cast uint32→int32, mask on device, cast back — no CPU download needed.
        if ttnn.to_torch(is_end).reshape(-1).bool().any():
            empty_code = self._empty_audio_token_id + self._acoustic_special_token_offset
            # Step 1: cast uint32 → int32 so ttnn.where works correctly.
            sem_i32 = ttnn.typecast(semantic_code_tt, ttnn.int32, memory_config=self._fm_dram_mem_config)
            ac_i32 = ttnn.typecast(acoustic_tt, ttnn.int32, memory_config=self._fm_dram_mem_config)
            ttnn.deallocate(semantic_code_tt)
            ttnn.deallocate(acoustic_tt)
            # Step 2: broadcast is_end [B,1,1] → [B,1,n_acoustic_out] for element-wise mask.
            is_end_exp = ttnn.repeat(is_end, (1, 1, self.n_acoustic_out))
            ttnn.deallocate(is_end)
            # Step 3: fill tensor for masked (end-audio) positions.
            empty_tt = ttnn.from_torch(
                torch.full((bsz, 1, self.n_acoustic_out), empty_code, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self._fm_dram_mem_config,
            )
            # Step 4: mask — where is_end → empty_code, else keep original acoustic code.
            masked_i32 = ttnn.where(is_end_exp, empty_tt, ac_i32, memory_config=self._fm_dram_mem_config)
            ttnn.deallocate(is_end_exp)
            ttnn.deallocate(empty_tt)
            ttnn.deallocate(ac_i32)
            # Step 5: cast back to uint32 and concat with semantic code.
            sem_u32 = ttnn.typecast(sem_i32, ttnn.uint32, memory_config=self._fm_dram_mem_config)
            ac_masked = ttnn.typecast(masked_i32, ttnn.uint32, memory_config=self._fm_dram_mem_config)
            ttnn.deallocate(sem_i32)
            ttnn.deallocate(masked_i32)
            codes_tt = ttnn.concat([sem_u32, ac_masked], dim=2, memory_config=self._fm_dram_mem_config)
            ttnn.deallocate(sem_u32)
            ttnn.deallocate(ac_masked)
            return ttnn.to_layout(codes_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=self._fm_dram_mem_config)

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
        if sampled_tt.dtype == self.dtype:
            return sampled_tt
        return ttnn.typecast(sampled_tt, self.dtype, memory_config=self._fm_dram_mem_config)

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

    def _fm_round_acoustic_codes_tt(self, sampled_tt: ttnn.Tensor, bsz: int) -> ttnn.Tensor:
        scaled = self.fm_pre_round_scaled_tt(sampled_tt)
        rounded = ttnn.round(scaled, decimals=0, memory_config=self._fm_dram_mem_config)
        ttnn.deallocate(scaled)
        codes = ttnn.reshape(rounded, (bsz, 1, self.n_acoustic_out))
        ttnn.deallocate(rounded)
        codes_u32 = ttnn.typecast(codes, ttnn.uint32)
        ttnn.deallocate(codes)
        out = ttnn.add(
            codes_u32,
            self._acoustic_offset_u32_tt,
            dtype=ttnn.uint32,
            memory_config=self._fm_dram_mem_config,
        )
        ttnn.deallocate(codes_u32)
        return out

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

        tt_llm_zero = ttnn.zeros_like(llm_hidden_tt)
        tt_llm_batched = ttnn.concat([llm_hidden_tt, tt_llm_zero], dim=0, memory_config=self._matmul_act_mem_config)
        ttnn.deallocate(tt_llm_zero)

        for t_val, dt_val in zip(self._euler_t_vals, self._euler_dt_vals):
            te = self._time_embedding_tt(t_val, bsz)
            x_in = self._sampled_tt_for_velocity(sampled_tt)
            x_batched = ttnn.concat([x_in, x_in], dim=0, memory_config=self._matmul_act_mem_config)
            if x_in is not sampled_tt and x_in.is_allocated():
                ttnn.deallocate(x_in)
            te_batched = ttnn.concat([te, te], dim=0, memory_config=self._matmul_act_mem_config)
            ttnn.deallocate(te)
            v_tt = self.predict_velocity_tt(x_batched, te_batched, tt_llm_batched, borrow_llm=True)

            v_shape = tuple(v_tt.shape)
            v_cond = ttnn.slice(v_tt, [0, 0, 0, 0], [bsz, v_shape[1], v_shape[2], v_shape[3]])
            v_uncond = ttnn.slice(v_tt, [bsz, 0, 0, 0], [2 * bsz, v_shape[1], v_shape[2], v_shape[3]])
            ttnn.deallocate(v_tt)
            v_cond_scaled = ttnn.multiply(v_cond, cfg_scalar, dtype=self.dtype, memory_config=self._fm_dram_mem_config)
            ttnn.deallocate(v_cond)
            v_uncond_scaled = ttnn.multiply(
                v_uncond, 1.0 - cfg_scalar, dtype=self.dtype, memory_config=self._fm_dram_mem_config
            )
            ttnn.deallocate(v_uncond)
            v_t_tt = ttnn.add(v_cond_scaled, v_uncond_scaled, dtype=self.dtype, memory_config=self._fm_dram_mem_config)
            ttnn.deallocate(v_cond_scaled)
            ttnn.deallocate(v_uncond_scaled)

            v_t_3d = ttnn.reshape(v_t_tt, (bsz, 1, self.n_acoustic_out))
            ttnn.deallocate(v_t_tt)
            # check for torch or ttnn
            sampled_tt = self._euler_integrate_sampled(sampled_tt, v_t_3d, dt_val)

        ttnn.deallocate(tt_llm_batched)
        return sampled_tt
