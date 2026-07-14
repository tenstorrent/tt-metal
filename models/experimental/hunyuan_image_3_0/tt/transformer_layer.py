# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of a single HunyuanImage-3.0 transformer decoder layer.
# Mirrors ref/transformer_layer.py (pre-norm block):
#     residual = x
#     x = input_layernorm(x)
#     x = self_attn(x, rope, mask)
#     x = residual + x
#     residual = x
#     x = post_attention_layernorm(x)
#     x = mlp(x)                       # MoE for layer-0
#     x = residual + x
#
# Composes the verified TT sub-blocks:
#   HunyuanTtRMSNorm   (input / post-attention layernorms)
#   HunyuanTtAttention (self-attn: fused QKV, 2D RoPE, QK-norm, GQA, SDPA, o_proj)
#   HunyuanTtMoE       (gate -> experts -> combine -> shared MLP)

import ttnn
from models.common.lightweightmodule import LightweightModule

from .attention.attention import HunyuanTtAttention
from .attention.rms_norm import HunyuanTtRMSNorm
from .moe.moe import HunyuanTtMoE
from .moe.moe_parallel import HunyuanTtMoEParallel
from .parallel_utils import resid_mem_config


class HunyuanTtDecoderLayer(LightweightModule):
    def __init__(
        self,
        device,
        state_dict: dict,
        layer_num: int = 0,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        num_experts: int = 64,
        moe_topk: int = 8,
        use_qk_norm: bool = True,
        use_mixed_mlp_moe: bool = True,
        norm_topk_prob: bool = True,
        rms_norm_eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        stream_experts: bool = True,
        ccl_manager=None,
        expert_mesh_axis: int = 1,
        tp_axis: int = 1,
        tp_factor: int = 1,
        sp_axis: int = 0,
        sp_factor: int = 1,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.layer_num = layer_num

        prefix = f"model.layers.{layer_num}"
        self.input_layernorm = HunyuanTtRMSNorm(
            device,
            hidden_size,
            state_dict,
            f"{prefix}.input_layernorm",
            eps=rms_norm_eps,
            weight_cache_path=weight_cache_path,
        )
        self.self_attn = HunyuanTtAttention(
            device,
            state_dict,
            layer_num=layer_num,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
            eps=rms_norm_eps,
            weight_dtype=weight_dtype,
            weight_cache_path=weight_cache_path,
            ccl_manager=ccl_manager,
            tp_axis=tp_axis,
            tp_factor=tp_factor,
            sp_axis=sp_axis,
            sp_factor=sp_factor,
        )
        self.post_attention_layernorm = HunyuanTtRMSNorm(
            device,
            hidden_size,
            state_dict,
            f"{prefix}.post_attention_layernorm",
            eps=rms_norm_eps,
            weight_cache_path=weight_cache_path,
        )
        # MoE: expert-parallel (sharded, resident) when a CCLManager is provided
        # for a mesh device; otherwise the single-device dense/streaming MoE.
        if ccl_manager is not None:
            self.mlp = HunyuanTtMoEParallel(
                device,
                ccl_manager,
                state_dict,
                f"{prefix}.mlp",
                num_experts=num_experts,
                hidden_size=hidden_size,
                moe_topk=moe_topk,
                norm_topk_prob=norm_topk_prob,
                use_mixed_mlp_moe=use_mixed_mlp_moe,
                mesh_axis=expert_mesh_axis,
                weight_dtype=weight_dtype,
                sp_axis=sp_axis,
                sp_factor=sp_factor,
                weight_cache_path=weight_cache_path,
            )
        else:
            self.mlp = HunyuanTtMoE(
                device,
                hidden_size,
                num_experts,
                moe_topk,
                state_dict,
                f"{prefix}.mlp",
                use_mixed_mlp_moe=use_mixed_mlp_moe,
                norm_topk_prob=norm_topk_prob,
                weight_dtype=weight_dtype,
                stream_experts=stream_experts,
            )

    def forward(
        self,
        x,
        seq_len,
        image_infos=None,
        attention_mask=None,
        cos_sin=None,
        *,
        kv_cache=None,
        use_cache=False,
        decode_step=False,
    ):
        """
        Args:
            x:              TTNN tensor [B, S, H] in TILE_LAYOUT.
            seq_len:        S — used to build the 2D RoPE cos/sin tables.
            image_infos:    per-batch image span info for 2D RoPE (None => text-only).
            attention_mask: optional additive mask [B,1,S,S]; None => causal SDPA.
            cos_sin:        optional precomputed (cos_tt, sin_tt) RoPE tables. When
                            provided (e.g. hoisted to the model level so 32 layers
                            share one build), they are used as-is and NOT freed here
                            — the caller owns them. When None, the tables are built
                            and freed internally per call.
        Returns:
            [B, S, H] TTNN tensor.
        """
        # 2D RoPE tables for this sequence (shared if the caller passed them in).
        owns_cos_sin = cos_sin is None
        if owns_cos_sin:
            cos_tt, sin_tt = self.self_attn.rope.prepare_cos_sin(seq_len, image_infos=image_infos)
        else:
            cos_tt, sin_tt = cos_sin

        # --- self-attention block ---
        # Residual-stream ops run on the per-device (sp-sharded) sequence. Keep them
        # L1-resident up to the measured CB-clash bound, else DRAM together (moving
        # only some leaves a live tensor in the next op's CB path — see parallel_utils).
        rs_mc = resid_mem_config(x.shape[1])
        residual = x
        h = self.input_layernorm(x, out_memory_config=rs_mc)
        attn_out = self.self_attn(
            h,
            cos_tt,
            sin_tt,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=self.layer_num,
            use_cache=use_cache,
            decode_step=decode_step,
        )
        ttnn.deallocate(h)
        # Attention emits [B, 1, S, H] (4-D, from nlp_create_qkv_heads); collapse
        # the singleton head-group dim back to the [B, S, H] residual-stream rank
        # so the layer's output rank matches its input — required when stacking
        # layers (a 4-D hidden would break the next layer's QKV reshape).
        if len(attn_out.shape) == 4:
            attn_out = ttnn.reshape(attn_out, [residual.shape[0], residual.shape[1], residual.shape[2]])
        x = ttnn.add(residual, attn_out, memory_config=rs_mc)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_out)

        # --- MoE block ---
        residual = x
        h = self.post_attention_layernorm(x, out_memory_config=rs_mc)
        moe_out = self.mlp(h)
        ttnn.deallocate(h)
        out = ttnn.add(residual, moe_out, memory_config=rs_mc)
        ttnn.deallocate(residual)
        ttnn.deallocate(moe_out)

        if owns_cos_sin:
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)
        return out
