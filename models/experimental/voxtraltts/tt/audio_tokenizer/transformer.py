# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT decoder transformer pieces for ``VoxtralTTSAudioTokenizer``."""

from __future__ import annotations

import ttnn

from models.common.rmsnorm import RMSNorm
from models.experimental.voxtraltts.reference.voxtral_config import VoxtralAudioTokenizerConfig
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import AudioTokenizerOptimizations
from models.experimental.voxtraltts.utils.config_helpers import (
    voxtral_audio_tokenizer_matmul_program_configs,
    voxtral_audio_tokenizer_sdpa_program_config,
    voxtral_audio_tokenizer_ttnn_sliding_window_size,
    voxtral_matmul_activation_mem_config,
)
from models.experimental.voxtraltts.tt.attention import VoxtralTTAttention
from models.experimental.voxtraltts.tt.mlp import VoxtralTTMLP
from models.tt_transformers.tt.common import Mode


class VoxtralTTAudioTokenizerDecoderTransformerBlock:
    """One vLLM-compatible audio-tokenizer decoder ``TransformerBlock``."""

    def __init__(
        self,
        device,
        *,
        state_dict: dict,
        tokenizer_cfg: VoxtralAudioTokenizerConfig,
        block_index: int = 1,
        layer_index: int = 0,
        weight_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        compute_kernel_config=None,
        optimizations: AudioTokenizerOptimizations | None = None,
    ) -> None:
        self.device = device
        self.block_index = block_index
        self.layer_index = layer_index
        self.tokenizer_cfg = tokenizer_cfg
        self.dim = tokenizer_cfg.dim
        self.output_dtype = output_dtype
        self.compute_kernel_config = compute_kernel_config
        self.optimizations = optimizations
        if optimizations is not None:
            weight_dtype = optimizations.weight_dtype
            output_dtype = optimizations.activation_dtype
            if compute_kernel_config is None:
                compute_kernel_config = optimizations.matmul_compute_kernel_config
            self.output_dtype = output_dtype
            self.compute_kernel_config = compute_kernel_config
        prefix = f"decoder_blocks.{block_index}.layers.{layer_index}"
        # RMSNorm weights stay BF16 (ROW_MAJOR); bfloat8_b requires TILE and is for matmul only.
        norm_weight_dtype = ttnn.bfloat16

        self.attention_norm = RMSNorm(
            device=device,
            dim=tokenizer_cfg.dim,
            eps=tokenizer_cfg.norm_eps,
            state_dict=state_dict,
            weight_key=f"{prefix}.attention_norm",
            weight_dtype=norm_weight_dtype,
            is_distributed=False,
        )
        self.ffn_norm = RMSNorm(
            device=device,
            dim=tokenizer_cfg.dim,
            eps=tokenizer_cfg.norm_eps,
            state_dict=state_dict,
            weight_key=f"{prefix}.ffn_norm",
            weight_dtype=norm_weight_dtype,
            is_distributed=False,
        )
        self.attention = VoxtralTTAttention(
            device,
            hidden_size=tokenizer_cfg.dim,
            num_attention_heads=tokenizer_cfg.n_heads,
            num_key_value_heads=tokenizer_cfg.n_kv_heads,
            head_dim=tokenizer_cfg.head_dim,
            state_dict=state_dict,
            weight_prefix=f"{prefix}.attention",
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
            sdpa_compute_kernel_config=(
                optimizations.sdpa_compute_kernel_config if optimizations is not None else compute_kernel_config
            ),
            is_causal=False,
            use_qk_norm=tokenizer_cfg.qk_norm,
            qk_norm_eps=tokenizer_cfg.qk_norm_eps,
            qk_norm_mode=Mode.DECODE,
        )
        self.mlp = VoxtralTTMLP(
            device,
            state_dict,
            w1_key=f"{prefix}.feed_forward.w1",
            w2_key=f"{prefix}.feed_forward.w2",
            w3_key=f"{prefix}.feed_forward.w3",
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            exact_silu=True,
            compute_kernel_config=compute_kernel_config,
        )

        self.layer_scale = tokenizer_cfg.layer_scale
        self.attention_scale = None
        self.ffn_scale = None
        if self.layer_scale:
            self.attention_scale = ttnn.from_torch(
                state_dict[f"{prefix}.attention_scale"].reshape(1, 1, 1, tokenizer_cfg.dim),
                device=device,
                dtype=output_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.ffn_scale = ttnn.from_torch(
                state_dict[f"{prefix}.ffn_scale"].reshape(1, 1, 1, tokenizer_cfg.dim),
                device=device,
                dtype=output_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    @staticmethod
    def _slice_like(x: ttnn.Tensor, ref: ttnn.Tensor) -> ttnn.Tensor:
        x_shape = tuple(x.shape)
        ref_shape = tuple(ref.shape)
        if x_shape == ref_shape:
            return x
        if len(x_shape) != len(ref_shape):
            raise RuntimeError(f"Rank mismatch: got {x_shape}, expected {ref_shape}")
        out = ttnn.slice(x, [0] * len(ref_shape), list(ref_shape))
        ttnn.deallocate(x)
        return out

    def __call__(
        self,
        x_b1td: ttnn.Tensor,
        *,
        attn_mask: ttnn.Tensor | None,
        sliding_window: int | None = None,
    ) -> ttnn.Tensor:
        """``[B, 1, T, D]`` tile → ``[B, 1, T, D]`` tile."""
        max_l1 = 32
        if self.optimizations is not None:
            max_l1 = self.optimizations.matmul_l1_max_seq_len
        seq_len = int(x_b1td.shape[2])
        act_mem = voxtral_matmul_activation_mem_config(seq_len, max_l1_seq_len=max_l1)

        matmul_prog = None
        sdpa_prog = None
        sliding_window_size = None
        use_native_sdpa = self.optimizations is not None and self.optimizations.sdpa_native_sliding_window
        if use_native_sdpa:
            if sliding_window is None:
                raise ValueError("sliding_window is required when sdpa_native_sliding_window is enabled.")
            sliding_window_size = voxtral_audio_tokenizer_ttnn_sliding_window_size(sliding_window)
            sdpa_prog = voxtral_audio_tokenizer_sdpa_program_config(self.device, seq_len)
        elif attn_mask is None:
            raise ValueError("attn_mask is required when sdpa_native_sliding_window is disabled.")

        if self.optimizations is not None and self.optimizations.matmul_program_config:
            cfg = self.tokenizer_cfg
            matmul_prog = voxtral_audio_tokenizer_matmul_program_configs(
                self.device,
                seq_len,
                dim=cfg.dim,
                hidden_dim=cfg.hidden_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
            )

        residual = ttnn.clone(x_b1td, dtype=self.output_dtype, memory_config=act_mem)
        normed = self.attention_norm(x_b1td, mode=Mode.DECODE)
        attn = self.attention(
            normed,
            cos=None,
            sin=None,
            attn_mask=None if use_native_sdpa else attn_mask,
            activation_memory_config=act_mem,
            wqkv_program_config=matmul_prog["wqkv"] if matmul_prog else None,
            wo_program_config=matmul_prog["wo"] if matmul_prog else None,
            sliding_window_size=sliding_window_size,
            sdpa_program_config=sdpa_prog,
        )
        ttnn.deallocate(normed)
        attn = self._slice_like(attn, x_b1td)
        if self.layer_scale:
            assert self.attention_scale is not None
            scaled = ttnn.mul(attn, self.attention_scale, dtype=self.output_dtype, memory_config=act_mem)
            ttnn.deallocate(attn)
            attn = scaled
        ttnn.deallocate(x_b1td)
        h = ttnn.add(residual, attn, dtype=self.output_dtype, memory_config=act_mem)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn)

        residual = ttnn.clone(h, dtype=self.output_dtype, memory_config=act_mem)
        normed = self.ffn_norm(h, mode=Mode.DECODE)
        ff = self.mlp(
            normed,
            activation_memory_config=act_mem,
            ff1_3_program_config=matmul_prog["ff1_3"] if matmul_prog else None,
            ff2_program_config=matmul_prog["ff2"] if matmul_prog else None,
        )
        ttnn.deallocate(normed)
        ff = self._slice_like(ff, h)
        if self.layer_scale:
            assert self.ffn_scale is not None
            scaled = ttnn.mul(ff, self.ffn_scale, dtype=self.output_dtype, memory_config=act_mem)
            ttnn.deallocate(ff)
            ff = scaled
        ttnn.deallocate(h)
        out = ttnn.add(residual, ff, dtype=self.output_dtype, memory_config=act_mem)
        ttnn.deallocate(residual)
        ttnn.deallocate(ff)
        return out
