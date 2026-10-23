# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Hybrid LFM2 decoder layer: either a full-attention ``TransformerBlock`` (reused verbatim
from ``tt_transformers``) or a ``ShortConv``-based layer (custom to this model), selected
per-layer via ``args.layer_types[layer_num]``.

``LfmDecoderLayer`` matches ``models.tt_transformers.tt.decoder.TransformerBlock``'s
constructor and ``forward`` signature exactly, so ``models.tt_transformers.tt.model.Transformer``
can build/drive a full hybrid stack unmodified -- see
``models.demos.multimodal.lfm25_vl.tt.e2e_model`` for how ``TransformerBlock`` is
monkeypatched to this class for the duration of ``Transformer.__init__``.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.multimodal.lfm25_vl.tt.short_conv import TtLfm2ShortConv

# Import the *original* TransformerBlock directly from its defining module so this class
# still works correctly even while `models.tt_transformers.tt.model.TransformerBlock` is
# monkeypatched to `LfmDecoderLayer` (see e2e_model.py).
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.decoder import TransformerBlock as OriginalTransformerBlock
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mlp import MLP


def _is_prefill(mode) -> bool:
    return mode == Mode.PREFILL or mode == "prefill" or str(mode) in ("Mode.PREFILL", "prefill")


class LfmDecoderLayer(LightweightModule):
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
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.layer_num = layer_num
        self.is_attention_layer = args.layer_types[layer_num] == "full_attention"

        if self.is_attention_layer:
            self.block = OriginalTransformerBlock(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                layer_num=layer_num,
                weight_cache_path=weight_cache_path,
                transformation_mats=transformation_mats,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                attention_class=attention_class,
                prefetcher=prefetcher,
            )
            return

        # --- ShortConv layer ---
        self.block = None
        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="attention_norm",
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            prefetcher=prefetcher,
            TG=args.is_galaxy,
            ag_config_key="ATTN_LN_AG_CONFIG",
        )
        self.conv = TtLfm2ShortConv(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("ShortConv", layer_num),
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
        )
        self.ff_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="ffn_norm",
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            prefetcher=prefetcher,
            TG=args.is_galaxy,
            ag_config_key="FFN_LN_AG_CONFIG",
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=args.get_model_config(),
            prefetcher=prefetcher,
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
        if self.is_attention_layer:
            return self.block.forward(
                x,
                current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )

        # rot_mats_*, page_table*, chunk_*, kv_cache are attention-only concepts and are unused here;
        # accepted anyway so Transformer.forward can call every layer identically.
        residual = x
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)

        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        normed = self.attention_norm(x, mode, norm_config=attn_norm_config)

        if batch_size > 1:
            normed = ttnn.reshape(normed, [batch_size, 1, normed.shape[-2] // batch_size, -1])

        conv_mode = Mode.PREFILL if _is_prefill(mode) else Mode.DECODE
        conv_out = self.conv(normed, mode=conv_mode, user_id=user_id)

        if _is_prefill(mode) and batch_size > 1:
            residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1])

        conv_out = ttnn.to_memory_config(conv_out, skip_mem_cfg)
        hidden_states = ttnn.add(residual, conv_out, memory_config=skip_mem_cfg)
        residual = hidden_states
        if _is_prefill(mode):
            x.deallocate(True)

        ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
        hidden_states = self.ff_norm(hidden_states, mode, norm_config=ff_norm_config)
        ttnn.deallocate(conv_out)

        hidden_states = self.feed_forward.forward(hidden_states, mode)

        out = ttnn.add(residual, hidden_states, memory_config=skip_mem_cfg)
        return out
