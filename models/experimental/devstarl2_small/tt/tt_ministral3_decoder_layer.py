# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Single ``Ministral3DecoderLayer`` block on TT: pre-norm attention, residual, pre-norm SwiGLU MLP, residual. Submodules are composed from existing Devstral TT modules only (no Torch in ``forward``).

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.tt.tt_ministralattn import TtMinistralAttention
from models.experimental.devstarl2_small.tt.tt_ministralmlp import TtMinistralMLP
from models.experimental.devstarl2_small.tt.tt_ministralrmsnorm import TtMinistralRMSNorm
from models.tt_transformers.tt.common import Mode


class TtMinistral3DecoderLayer(LightweightModule):
    """HF ``Ministral3DecoderLayer`` order: pre-norm attention + residual, pre-norm MLP + residual.

    Submodule ctor args match :class:`TtMinistralAttention`, :class:`TtMinistralMLP`, and the two norms."""

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
        # Cached for ``forward_prefill`` / ``forward_decode`` so the two
        # per-layer ``ttnn.all_gather`` calls can size ``num_links`` from the
        # actual fabric (e.g. 2 on BH-QB P150x4, 1 on single chip / T3K 1x4
        # submesh) instead of being hard-pinned to ``num_links=1`` which under-
        # uses ethernet bandwidth.
        self.tt_ccl = tt_ccl

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
        # Multi-chip PCC path: gather fractured attn output before adding residual so RMSNorm sees full width.
        if self.self_attn.num_devices > 1:
            attn_out = ttnn.all_gather(
                attn_out,
                dim=3,
                num_links=self.tt_ccl.get_num_links(),
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
                num_links=self.tt_ccl.get_num_links(),
                topology=self.self_attn.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ff_out = ttnn.to_memory_config(ff_out, residual_mlp.memory_config())
        return ttnn.add(residual_mlp, ff_out, memory_config=residual_mlp.memory_config())

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats,
        user_id: int = 0,
        page_table=None,
    ) -> ttnn.Tensor:
        """Decode one token using KV from ``forward_prefill``.

        ``x``: residual-mem width-sharded tensor; ``current_pos`` ``[1,batch]``; ``rot_mats`` from ``get_rot_mats``."""
        args = self.self_attn.args
        num_devices = self.self_attn.num_devices
        residual_mem_cfg = args.get_residual_mem_config(Mode.DECODE, None)
        attn_input_mem_cfg = args.get_attn_input_mem_config(Mode.DECODE, None)
        mlp_input_mem_cfg = args.get_mlp_input_mem_config(Mode.DECODE, None)

        # Decode residual is width-fractured on multi-chip; all_gather post-norm before QKV/FF1 need full dim per chip.
        x = ttnn.to_memory_config(x, residual_mem_cfg)

        # Attention: DRAM norm → gather → shard into DRAM-sharded QKV matmuls.
        x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        h_normed = self.input_layernorm(x_dram, Mode.DECODE)
        ttnn.deallocate(x_dram)
        if num_devices > 1:
            h_normed = ttnn.all_gather(
                h_normed,
                dim=3,
                num_links=self.tt_ccl.get_num_links(),
                topology=args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        h = ttnn.to_memory_config(h_normed, attn_input_mem_cfg)
        ttnn.deallocate(h_normed)

        attn_out = self.self_attn.forward(
            h,
            current_pos,
            rot_mats,
            user_id=user_id,
            mode=Mode.DECODE,
            page_table=page_table,
        )
        attn_out = ttnn.to_memory_config(attn_out, residual_mem_cfg)
        skip1 = ttnn.add(x, attn_out, memory_config=residual_mem_cfg)
        ttnn.deallocate(attn_out)

        # MLP: same DRAM norm → gather → shard pattern as attention path.
        skip1_dram = ttnn.to_memory_config(skip1, ttnn.DRAM_MEMORY_CONFIG)
        h2_normed = self.post_attention_layernorm(skip1_dram, Mode.DECODE)
        ttnn.deallocate(skip1_dram)
        if num_devices > 1:
            h2_normed = ttnn.all_gather(
                h2_normed,
                dim=3,
                num_links=self.tt_ccl.get_num_links(),
                topology=args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        h2 = ttnn.to_memory_config(h2_normed, mlp_input_mem_cfg)
        ttnn.deallocate(h2_normed)

        ff_out = self.mlp(h2, Mode.DECODE)  # mlp deallocates h2 internally
        ff_out = ttnn.to_memory_config(ff_out, residual_mem_cfg)
        result = ttnn.add(skip1, ff_out, memory_config=residual_mem_cfg)
        ttnn.deallocate(ff_out)
        ttnn.deallocate(skip1)
        return result


__all__ = ["TtMinistral3DecoderLayer"]
