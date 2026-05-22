# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT decoder transformer pieces for ``VoxtralTTSAudioTokenizer``."""

from __future__ import annotations

import ttnn

from models.common.rmsnorm import RMSNorm
from models.experimental.voxtraltts.reference.voxtral_config import VoxtralAudioTokenizerConfig
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import AudioTokenizerOptimizations
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

    def __call__(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """``[B, 1, T, D]`` tile → ``[B, 1, T, D]`` tile."""
        residual = ttnn.clone(x_b1td, dtype=self.output_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        normed = self.attention_norm(x_b1td, mode=Mode.DECODE)
        attn = self.attention(normed, cos=None, sin=None, attn_mask=attn_mask)
        ttnn.deallocate(normed)
        attn = self._slice_like(attn, x_b1td)
        if self.layer_scale:
            assert self.attention_scale is not None
            scaled = ttnn.mul(
                attn, self.attention_scale, dtype=self.output_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(attn)
            attn = scaled
        ttnn.deallocate(x_b1td)
        h = ttnn.add(residual, attn, dtype=self.output_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn)

        residual = ttnn.clone(h, dtype=self.output_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        normed = self.ffn_norm(h, mode=Mode.DECODE)
        ff = self.mlp(normed)
        ttnn.deallocate(normed)
        ff = self._slice_like(ff, h)
        if self.layer_scale:
            assert self.ffn_scale is not None
            scaled = ttnn.mul(ff, self.ffn_scale, dtype=self.output_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(ff)
            ff = scaled
        ttnn.deallocate(h)
        out = ttnn.add(residual, ff, dtype=self.output_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(residual)
        ttnn.deallocate(ff)
        return out
