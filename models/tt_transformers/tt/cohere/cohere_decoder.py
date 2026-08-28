# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal scaffold — bounty tt-metal#49307 track)
# SPDX-License-Identifier: Apache-2.0
#
# CohereDecoderLayer for Command-R (c4ai-command-r-v01) — bounty tt-metal#49307 track.
#
# Modeled on models/tt_transformers/tt/decoder.py::TransformerBlock with the verified
# Command-R v01 deltas (see README.md in this directory for sources):
#
#   1. PARALLEL BLOCK (HF v4.39.3 modeling_cohere.py line 654):
#        residual = x
#        h        = input_layernorm(x)          # ONE norm feeds BOTH branches
#        out      = residual + self_attn(h) + mlp(h)
#      TransformerBlock's sequential attn_norm -> attn -> ff_norm -> MLP does NOT apply.
#   2. input_layernorm is CohereLayerNorm (mean-centering, fp32 internal) — NOT RMSNorm.
#   3. Attention is MHA 64 Q : 64 KV heads (head_dim 128), no QK norm, no attn bias;
#      scaling is plain 1/sqrt(head_dim). The stock Attention class applies (its
#      q_norm/k_norm loaders stay inert — no *.q_norm/*.k_norm tensors in this checkpoint).
#
# Scaffold status: ctor wiring mirrors TransformerBlock so P5 can slot it into
# model.py; forward() documents the parallel-block op order with TODO(PCC) markers at
# every point that must be PCC-validated against the tt-rd CPU reference harness
# (tt-rd scripts/command-r/) before Stage-1 bring-up. NOT runnable yet — ModelArgs has
# no cohere model_type (P5 survey: model_config.py, load_checkpoints.py, model.py).

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.attention import Attention as DefaultAttention
from models.tt_transformers.tt.cohere.cohere_norm import TtCohereLayerNorm
from models.tt_transformers.tt.mlp import MLP


class CohereDecoderLayer(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        prefetcher=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.args = args
        self.layer_num = layer_num
        self.dim = args.dim                      # 8192
        self.n_heads = args.n_heads              # 64 (MHA)
        self.n_kv_heads = args.n_kv_heads        # 64 — verified config + k_proj [8192, 8192]
        self.head_dim = self.dim // self.n_heads  # 128

        ActualAttentionClass = attention_class if attention_class is not None else DefaultAttention
        self.attention = ActualAttentionClass(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            prefetcher=prefetcher,
        )
        # SwiGLU MLP (hidden_act=silu), intermediate 22528; stock MLP applies.
        # TODO(PCC): gate/up [22528, 8192], down [8192, 22528] per config; no biases.
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=args.get_model_config(),
            prefetcher=prefetcher,
        )
        # ONE input_layernorm (LayerNorm, eps=config.layer_norm_eps=1e-5, weight-only).
        self.input_layernorm = TtCohereLayerNorm(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="input_layernorm",
            tt_ccl=self.tt_ccl,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ) -> ttnn.Tensor:
        # PARALLEL BLOCK — HF v4.39.3 line 654 op order.
        residual = x

        # TODO(PCC): mem-config contract mirrors TransformerBlock.forward
        # (args.get_residual_mem_config / get_norm_config) — finalize in P5 wiring.
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)

        # ONE norm feeds BOTH branches (no separate ff_norm in Command-R v01).
        # TODO(PCC): validate TtCohereLayerNorm vs HF CohereLayerNorm at PCC >= 0.99.
        h = self.input_layernorm(x, mode, norm_config=norm_config)

        if batch_size > 1:
            h_attn = ttnn.reshape(h, [batch_size, 1, h.shape[-2] // batch_size, -1])
        else:
            h_attn = h

        attn_out = self.attention.forward(
            h_attn,
            current_pos,
            rot_mats_global,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )
        # TODO(PCC): attention output must match TransformerBlock's skip_mem_cfg contract.
        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)

        # MLP consumes the SAME normed input h (parallel), not the post-attention residual.
        mlp_out = self.feed_forward.forward(h, mode)

        # residual + attn + mlp — single fused residual add.
        # TODO(PCC): HF computes (residual + attn) + mlp in fp32-accumulated bf16;
        # two-step ttnn.add vs chained add ordering to be validated at PCC >= 0.99.
        hidden = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16)
        out = ttnn.add(hidden, mlp_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16)
        return out  # fractured across devices
