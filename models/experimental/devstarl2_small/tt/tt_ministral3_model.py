# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal TT ``Ministral3Model`` (text stack): embeddings -> decoder layers -> final RMSNorm.

Composes existing TT submodules from this experimental folder; no Torch fallback in forward.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.experimental.devstarl2_small.tt.tt_ministral3_decoder_layer import TtMinistral3DecoderLayer
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.embedding import Embedding


class TtMinistral3Model(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        model_args,
        meta_state_dict,
        weight_cache_path,
        dtype,
        transformation_mats,
        configuration,
        llama_4_scaling_beta=None,
        original_max_position_embeddings=None,
    ):
        super().__init__()
        self.args = model_args
        self.mesh_device = mesh_device
        self.n_layers = int(model_args.n_layers)

        self.embed_tokens = Embedding(
            mesh_device=mesh_device,
            args=model_args,
            weight_cache_path=weight_cache_path,
            state_dict=meta_state_dict,
            dtype=ttnn.bfloat16,
        )

        self.layers = [
            TtMinistral3DecoderLayer(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                model_args=model_args,
                meta_state_dict=meta_state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                dtype=dtype,
                transformation_mats=transformation_mats,
                configuration=configuration,
                llama_4_scaling_beta=llama_4_scaling_beta,
                original_max_position_embeddings=original_max_position_embeddings,
            )
            for i in range(self.n_layers)
        ]

        self.norm = RMSNorm(
            device=mesh_device,
            dim=model_args.dim,
            eps=model_args.norm_eps,
            state_dict=meta_state_dict,
            weight_key="norm",
            state_dict_prefix=model_args.get_state_dict_prefix("", None),
            weight_cache_path=None if model_args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            is_distributed=model_args.is_distributed_norm,
            add_unit_offset=model_args.rms_norm_add_unit_offset,
            ccl_topology=model_args.ccl_topology(),
            tt_ccl=tt_ccl,
        )

    def forward_prefill_from_embeddings(self, hidden_states_11SH: ttnn.Tensor, rot_mats, position_ids) -> ttnn.Tensor:
        h = hidden_states_11SH
        for layer in self.layers:
            h = layer.forward_prefill(h, rot_mats, position_ids=position_ids)
        return self.norm(h, Mode.PREFILL)

    def forward_prefill(self, input_ids_tt: ttnn.Tensor, rot_mats, position_ids) -> ttnn.Tensor:
        h = self.embed_tokens(input_ids_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self.forward_prefill_from_embeddings(h, rot_mats, position_ids)


__all__ = ["TtMinistral3Model"]
