# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One GLM-5.3-Flash decoder layer (modeling:1278-1348):

    attn_hc.pre -> input_layernorm -> (KDA | MLA) -> attn_hc.post -> ffn_hc.pre -> post_attention_layernorm
    -> (dense SwiGLU | MoE) -> ffn_hc.post

Builds the device modules from a `WeightSource` (tile cache or host state dict) at construction time.
"""

from __future__ import annotations

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.config import LayerSpec
from models.autoports.zai_org_glm_5_3_flash.tt.hyperconnection import HyperConnection
from models.autoports.zai_org_glm_5_3_flash.tt.kda import KDA
from models.autoports.zai_org_glm_5_3_flash.tt.mla import MLA
from models.autoports.zai_org_glm_5_3_flash.tt.mlp import SwiGLUMLP
from models.autoports.zai_org_glm_5_3_flash.tt.moe import MoE, Router
from models.autoports.zai_org_glm_5_3_flash.tt.precision import ACT_DTYPE, hifi4_fp32
from models.autoports.zai_org_glm_5_3_flash.tt.weights import WeightSource

DRAM = ttnn.DRAM_MEMORY_CONFIG


def _row(t: ttnn.Tensor) -> ttnn.Tensor:
    """1-D norm weight -> [1, 1, 1, N] for ttnn.rms_norm."""
    return ttnn.reshape(t, (1, 1, 1, t.shape[-1]))


class DecoderLayer:
    def __init__(self, cfg, layer_idx: int, mesh_device, source: WeightSource, *, block_size: int):
        self.cfg = cfg
        self.spec = LayerSpec.from_config(cfg, layer_idx)
        self.layer_idx = layer_idx
        self.device = mesh_device
        L = layer_idx
        dev = lambda name: source.device(L, name, mesh_device)
        host = lambda name: source.host(L, name)
        hc_kwargs = dict(
            hc_mult=cfg.hc_mult,
            hidden_size=cfg.hidden_size,
            sinkhorn_iters=cfg.hc_sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        self.attn_hc = HyperConnection(
            mesh_device, dev("hc_attn_fn"), host("hc_attn_base"), host("hc_attn_scale"), **hc_kwargs
        )
        self.ffn_hc = HyperConnection(
            mesh_device, dev("hc_ffn_fn"), host("hc_ffn_base"), host("hc_ffn_scale"), **hc_kwargs
        )
        self.input_norm = _row(dev("input_layernorm.weight"))
        self.post_attn_norm = _row(dev("post_attention_layernorm.weight"))
        self.ckc_norm = hifi4_fp32(mesh_device)

        if self.spec.attention == "kda":
            cat = lambda names: ttnn.concat([dev(n) for n in names], dim=-1, memory_config=DRAM)
            self.attn = KDA(
                mesh_device,
                w_qkv=cat(["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]),
                w_lowrank=cat(["self_attn.f_a_proj.weight", "self_attn.g_a_proj.weight", "self_attn.b_proj.weight"]),
                w_fb=dev("self_attn.f_b_proj.weight"),
                w_gb=dev("self_attn.g_b_proj.weight"),
                w_o=dev("self_attn.o_proj.weight"),
                taps=cat(["self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"]),
                o_norm=dev("self_attn.o_norm.weight"),
                dt_bias=host("self_attn.dt_bias"),
                a_log=host("self_attn.A_log"),
                num_heads=cfg.linear_num_heads,
                head_dim=cfg.linear_head_dim,
                hidden=cfg.hidden_size,
                lower_bound=cfg.linear_lower_bound,
                eps=cfg.rms_norm_eps,
            )
        else:
            self.attn = MLA(
                mesh_device,
                w_qa_kva=ttnn.concat(
                    [dev("self_attn.q_a_proj.weight"), dev("self_attn.kv_a_proj_with_mqa.weight")],
                    dim=-1,
                    memory_config=DRAM,
                ),
                qa_norm=_row(dev("self_attn.q_a_layernorm.weight")),
                kva_norm=_row(dev("self_attn.kv_a_layernorm.weight")),
                w_qb=dev("self_attn.q_b_proj.weight"),
                w_uk=dev("self_attn.kv_b_proj.W_uk"),
                w_uv=dev("self_attn.kv_b_proj.W_uv"),
                w_o=dev("self_attn.o_proj.weight"),
                num_heads=cfg.num_attention_heads,
                q_lora_rank=cfg.q_lora_rank,
                kv_lora_rank=cfg.kv_lora_rank,
                qk_head_dim=cfg.qk_head_dim,
                v_head_dim=cfg.v_head_dim,
                eps=cfg.rms_norm_eps,
                block_size=block_size,
            )

        if self.spec.mlp == "dense":
            self.mlp = SwiGLUMLP(
                mesh_device,
                dev("mlp.gate_proj.weight"),
                dev("mlp.up_proj.weight"),
                dev("mlp.down_proj.weight"),
                limit=cfg.swiglu_limit,
            )
        else:
            shared = SwiGLUMLP(
                mesh_device,
                dev("mlp.shared_experts.gate_proj.weight"),
                dev("mlp.shared_experts.up_proj.weight"),
                dev("mlp.shared_experts.down_proj.weight"),
                limit=cfg.swiglu_limit,
            )
            router = Router(
                mesh_device,
                dev("mlp.gate.weight"),
                host("mlp.gate.e_score_correction_bias"),
                num_experts=cfg.n_routed_experts,
                top_k=cfg.num_experts_per_tok,
                scaling=cfg.routed_scaling_factor,
            )
            self.mlp = MoE(
                mesh_device,
                router,
                gate_up=source.experts(L, "gate_up_proj", cfg.n_routed_experts, mesh_device),
                down=source.experts(L, "down_proj", cfg.n_routed_experts, mesh_device),
                shared=shared,
                intermediate=cfg.moe_intermediate_size,
                hidden=cfg.hidden_size,
                limit=cfg.swiglu_limit,
            )

    def _norm(self, h, weight):
        out = ttnn.rms_norm(h, weight=weight, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm)
        return ttnn.typecast(out, ACT_DTYPE) if out.dtype != ACT_DTYPE else out

    def _ffn(self, x, decode: bool):
        return self.mlp(x, decode=decode) if self.spec.mlp == "moe" else self.mlp(x)

    def forward(self, streams: ttnn.Tensor, attention, *, decode: bool) -> ttnn.Tensor:
        """streams [1,1,T,16384] fp32 -> [1,1,T,16384] fp32. `attention(h)` runs the (stateful) attention block."""
        h, post, comb = self.attn_hc.pre(streams)
        a = attention(self._norm(h, self.input_norm))
        ttnn.deallocate(h)
        streams = self.attn_hc.post(a, streams, post, comb)
        h, post, comb = self.ffn_hc.pre(streams)
        m = self._ffn(self._norm(h, self.post_attn_norm), decode)
        ttnn.deallocate(h)
        return self.ffn_hc.post(m, streams, post, comb)
