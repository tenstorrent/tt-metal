# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Single ``Ministral3DecoderLayer`` block on TT: pre-norm attention, residual, pre-norm SwiGLU MLP, residual.

Submodules are composed from existing Devstral TT modules only (no Torch in ``forward``).
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.tt.tt_ministralattn import TtMinistralAttention
from models.experimental.devstarl2_small.tt.tt_ministralmlp import TtMinistralMLP
from models.experimental.devstarl2_small.tt.tt_ministralrmsnorm import TtMinistralRMSNorm
from models.tt_transformers.tt.common import Mode


class TtMinistral3DecoderLayer(LightweightModule):
    """
    Mirrors HF ``Ministral3DecoderLayer`` ordering for prefill (``past_key_values=None``).

    Parameters match the constructors of :class:`TtMinistralAttention`, :class:`TtMinistralMLP`,
    and two :class:`TtMinistralRMSNorm` instances (input vs post-attention).
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        model_args,
        meta_state_dict,
        weight_cache_path,
        layer_num: int,
        dtype,
        transformation_mats,
        configuration,
        llama_4_scaling_beta=None,
        original_max_position_embeddings=None,
    ):
        super().__init__()
        self.layer_num = layer_num

        self.input_layernorm = TtMinistralRMSNorm(
            mesh_device,
            model_args,
            meta_state_dict,
            weight_cache_path,
            layer_num,
            tt_ccl,
            post_attention=False,
        )
        self.self_attn = TtMinistralAttention(
            mesh_device,
            tt_ccl,
            model_args,
            meta_state_dict,
            weight_cache_path,
            layer_num,
            dtype,
            transformation_mats,
            configuration=configuration,
            llama_4_scaling_beta=llama_4_scaling_beta,
            original_max_position_embeddings=original_max_position_embeddings,
        )
        self.post_attention_layernorm = TtMinistralRMSNorm(
            mesh_device,
            model_args,
            meta_state_dict,
            weight_cache_path,
            layer_num,
            tt_ccl,
            post_attention=True,
        )
        self.mlp = TtMinistralMLP(
            mesh_device,
            tt_ccl,
            model_args,
            meta_state_dict,
            weight_cache_path,
            layer_num,
            dtype,
            model_args.get_model_config(),
        )

    def forward_prefill(
        self,
        x_11SH: ttnn.Tensor,
        rot_mats,
        position_ids: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """``x_11SH``: ``[batch, 1, seq, hidden]`` (same layout as :meth:`TtMinistralAttention.forward_prefill`)."""
        residual = x_11SH
        h = self.input_layernorm(x_11SH, Mode.PREFILL)
        attn_out = self.self_attn.forward_prefill(h, rot_mats, position_ids=position_ids)
        # Attention may return width-fractured output on multi-device; gather to full hidden width
        # so layernorm gamma (full width) remains valid for this isolated decoder-layer PCC path.
        if self.self_attn.num_devices > 1:
            attn_out = ttnn.all_gather(
                attn_out,
                dim=3,
                num_links=1,
                topology=self.self_attn.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        # Align memory layout with residual before elementwise add to avoid binary subtile broadcast issues.
        skip_mem_cfg = residual.memory_config()
        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
        h = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
        residual_mlp = h
        h = self.post_attention_layernorm(h, Mode.PREFILL)
        ff_out = self.mlp(h, Mode.PREFILL)
        if self.self_attn.num_devices > 1:
            ff_out = ttnn.all_gather(
                ff_out,
                dim=3,
                num_links=1,
                topology=self.self_attn.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ff_out = ttnn.to_memory_config(ff_out, residual_mlp.memory_config())
        return ttnn.add(residual_mlp, ff_out, memory_config=residual_mlp.memory_config())


__all__ = ["TtMinistral3DecoderLayer"]
