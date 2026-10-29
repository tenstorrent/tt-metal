# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""EncoderLayer (HF `EmbeddingGemma2EncoderLayer`): sandwich norms + PLE block + layer_scalar.

    h = x + post_attn_norm( attn( input_norm(x), mask ) )
    h = h + post_ff_norm( mlp( pre_ff_norm(h) ) )
    h = h + post_pli_norm( W_proj( gelu_tanh(W_gate h) * pli_i ) )
    h = h * layer_scalar[i]           (learned scalar, 0.115 .. 0.969; scales the residual too)

Contract
    input   x [B, 1, S, hidden] bf16 TILE DRAM, per_layer_input [B, 1, S, P] bf16 TILE DRAM,
            mask [B, 1, S, S] additive bf16 TILE DRAM for this layer's kind
    output  [B, 1, S, hidden] bf16 TILE DRAM
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.attention import Attention
from models.demos.embedding_gemma2.tt.config import RESIDUAL_IN_L1, EncoderLayerConfig, RMSNormConfig, act_memory_config
from models.demos.embedding_gemma2.tt.mlp import MLP
from models.demos.embedding_gemma2.tt.per_layer_input import PerLayerInputBlock
from models.demos.embedding_gemma2.tt.precision import TextPrecision, resolve_precision
from models.demos.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.embedding_gemma2.tt.rope import RotaryTables
from models.demos.embedding_gemma2.tt.weights import substate

_NORMS = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm")


class EncoderLayer(LightweightModule):
    def __init__(
        self,
        config: EncoderLayerConfig,
        norms: dict[str, RMSNorm],
        attention: Attention,
        mlp: MLP,
        ple_block: PerLayerInputBlock,
        layer_scalar: float,
        residual_dtype: str = "bf16",
    ):
        super().__init__()
        self.residual_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[residual_dtype]
        self.config = config
        self.input_layernorm = norms["input_layernorm"]
        self.post_attention_layernorm = norms["post_attention_layernorm"]
        self.pre_feedforward_layernorm = norms["pre_feedforward_layernorm"]
        self.post_feedforward_layernorm = norms["post_feedforward_layernorm"]
        self.self_attn = attention
        self.mlp = mlp
        self.ple_block = ple_block
        self.layer_scalar = layer_scalar

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: EncoderLayerConfig,
        device,
        rope: RotaryTables,
        *,
        cache_dir: Path | None = None,
        masks=None,
        precision: TextPrecision | None = None,
    ) -> "EncoderLayer":
        """`state_dict` holds the HF `layers.<i>.` sub-state; `masks` the shared AttentionMasks;
        `precision` the text precision policy (None: the default precision file)."""
        precision = precision or resolve_precision(None).text
        i = config.layer_idx
        prefix = f"layers.{i}"
        norm_cfg = RMSNormConfig(dim=config.hidden_size, eps=config.rms_norm_eps)
        norms = {
            n: RMSNorm.from_state_dict(
                substate(state_dict, n),
                norm_cfg,
                device,
                cache_dir=cache_dir,
                name=f"{prefix}.{n}",
                compute_kernel_config=precision.norms.compute_kernel_config,
            )
            for n in _NORMS
        }
        attention = Attention.from_state_dict(
            substate(state_dict, "self_attn"),
            config.attention,
            device,
            rope,
            cache_dir=cache_dir,
            name=f"{prefix}.self_attn",
            masks=masks,
            precision=precision,
        )
        mlp = MLP.from_state_dict(
            substate(state_dict, "mlp"),
            config.mlp,
            device,
            cache_dir=cache_dir,
            name=f"{prefix}.mlp",
            optimized=True,
            precision=precision,
            layer_idx=i,
        )
        ple_block = PerLayerInputBlock.from_state_dict(
            substate(state_dict, "ple_block"),
            config,
            device,
            cache_dir=cache_dir,
            name=f"{prefix}.ple_block",
            precision=precision,
        )
        scalar = state_dict["layer_scalar"]
        if scalar.numel() != 1:
            raise ValueError(f"{prefix}.layer_scalar: expected 1 element, got {tuple(scalar.shape)}")
        return cls(config, norms, attention, mlp, ple_block, float(scalar.float().item()), precision.residual_dtype)

    def forward(
        self, x: ttnn.Tensor, per_layer_input: ttnn.Tensor, mask: ttnn.Tensor, block_map: ttnn.Tensor | None = None
    ) -> ttnn.Tensor:
        """`block_map`: a per-forward SDPA block map for this layer kind (masks.padded_block_map), or None."""
        B, S, H = x.shape[0], x.shape[2], x.shape[3]
        mem = act_memory_config(x.device(), B, S, H)  # [B,1,S,hidden] intermediates
        dram = mem if RESIDUAL_IN_L1 else ttnn.DRAM_MEMORY_CONFIG  # residual stream h1, h2, out
        h = self.input_layernorm(x, memory_config=mem)
        a = self.self_attn(h, mask, out_memory_config=mem, block_map=block_map)
        ttnn.deallocate(h)
        a_n = self.post_attention_layernorm(a, memory_config=mem)
        ttnn.deallocate(a)
        rdt = self.residual_dtype
        h1 = ttnn.add(x, a_n, dtype=rdt, memory_config=dram)
        ttnn.deallocate(a_n)

        f = self.pre_feedforward_layernorm(h1, memory_config=mem)
        m = self.mlp(f, memory_config=mem)
        ttnn.deallocate(f)
        m_n = self.post_feedforward_layernorm(m, memory_config=mem)
        ttnn.deallocate(m)
        h2 = ttnn.add(h1, m_n, dtype=rdt, memory_config=dram)
        ttnn.deallocate(h1)
        ttnn.deallocate(m_n)

        # (h2 + ple) * layer_scalar
        out = self.ple_block(h2, per_layer_input, out_scale=self.layer_scalar, out_memory_config=dram)
        ttnn.deallocate(h2)
        return out
