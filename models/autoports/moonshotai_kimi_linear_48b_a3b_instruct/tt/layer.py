# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pre-norm decoder layer: {KDA | MLA} attention + {dense SwiGLU | MoE}. Replicated residual [1,1,S,hidden] on every chip."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.ccl import KimiCCL
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.kda.layer import KDADecodeState, KimiKDA
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.mla.layer import KimiMLA
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.moe.moe import KimiDenseMLP, KimiMoE
from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.tt.rms_norm import RMSNorm
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState


@dataclass
class PrecisionPolicy:
    experts: object = ttnn.bfloat8_b
    shared: object = ttnn.bfloat8_b
    dense_mlp: object = ttnn.bfloat8_b
    mla_proj: object = ttnn.bfloat8_b
    kv_cache: object = ttnn.bfloat8_b  # the flash-MLA decode kernel is exercised upstream with a bfp8 latent cache


class KimiDecoderLayer:
    def __init__(
        self,
        mesh_device,
        cfg: KimiLinearConfig,
        sd: dict | None,
        *,
        layer_idx: int,
        ccl: KimiCCL,
        cache_path: Path | None,
        precision: PrecisionPolicy | None = None,
    ):
        precision = precision or PrecisionPolicy()
        self.cfg, self.layer_idx = cfg, layer_idx
        self.is_kda = cfg.is_kda_layer(layer_idx)
        self.is_moe = cfg.is_moe_layer(layer_idx)
        g = (lambda k: None) if sd is None else (lambda k: sd[k])
        self.input_norm = RMSNorm(
            mesh_device,
            g("input_layernorm.weight"),
            eps=cfg.rms_norm_eps,
            name=f"layer_{layer_idx}.input_norm",
            cache_path=cache_path,
        )
        self.post_norm = RMSNorm(
            mesh_device,
            g("post_attention_layernorm.weight"),
            eps=cfg.rms_norm_eps,
            name=f"layer_{layer_idx}.post_norm",
            cache_path=cache_path,
        )
        attn_sd = (
            None
            if sd is None
            else {
                k: v
                for k, v in sd.items()
                if not k.startswith(("moe.", "mlp.", "input_layernorm", "post_attention_layernorm"))
            }
        )
        if self.is_kda:
            self.attn = KimiKDA(
                mesh_device, cfg.kda_config(), attn_sd, layer_idx=layer_idx, ccl=ccl, weight_cache_path=cache_path
            )
        else:
            self.attn = KimiMLA(
                mesh_device, cfg, attn_sd, layer_idx=layer_idx, ccl=ccl, cache_path=cache_path, dtype=precision.mla_proj
            )
        if self.is_moe:
            self.mlp = KimiMoE(
                mesh_device,
                cfg,
                sd,
                layer_idx=layer_idx,
                ccl=ccl,
                cache_path=cache_path,
                expert_dtype=precision.experts,
                shared_dtype=precision.shared,
            )
        else:
            self.mlp = KimiDenseMLP(
                mesh_device, cfg, sd, layer_idx=layer_idx, ccl=ccl, cache_path=cache_path, dtype=precision.dense_mlp
            )

    # ---- prefill: x [1,1,T,hidden] -----------------------------------------------------------
    def forward_prefill(
        self,
        x: ttnn.Tensor,
        *,
        kda_state: KdaState | None = None,
        cache: ttnn.Tensor | None = None,
        page_table: ttnn.Tensor | None = None,
        user_id: int = 0,
        valid_len: int | None = None,
        chunk_start: int = 0,
    ):
        """Returns (x_out, new_kda_state or None)."""
        h = self.input_norm(x)
        new_state = None
        if self.is_kda:
            a, new_state = self.attn.forward_prefill(h, kda_state, valid_len=valid_len)
        else:
            a = self.attn.forward_prefill(
                h, cache, page_table, user_id=user_id, valid_len=valid_len, chunk_start=chunk_start
            )
        ttnn.deallocate(h)
        x = ttnn.add(x, a)
        ttnn.deallocate(a)
        h2 = self.post_norm(x)
        m = self.mlp.forward(h2, "prefill")
        ttnn.deallocate(h2)
        out = ttnn.add(x, m)
        ttnn.deallocate(m)
        return out, new_state

    # ---- decode: x [1,1,B,hidden] -----------------------------------------------------------
    def forward_decode(
        self,
        x: ttnn.Tensor,
        *,
        kda_state: KDADecodeState | None = None,
        cache: ttnn.Tensor | None = None,
        page_table: ttnn.Tensor | None = None,
        cur_pos: ttnn.Tensor | None = None,
    ):
        h = self.input_norm(x)
        a = (
            self.attn.forward_decode(h, kda_state)
            if self.is_kda
            else self.attn.forward_decode(h, cache, page_table, cur_pos)
        )
        ttnn.deallocate(h)
        x = ttnn.add(x, a)
        ttnn.deallocate(a)
        h2 = self.post_norm(x)
        m = self.mlp.forward(h2, "decode")
        ttnn.deallocate(h2)
        out = ttnn.add(x, m)
        ttnn.deallocate(m)
        return out
