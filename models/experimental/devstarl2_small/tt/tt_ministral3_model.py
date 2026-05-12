# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal TT ``Ministral3Model`` (text stack): embeddings -> decoder layers -> final RMSNorm.

Optionally owns :class:`TtMinistral3RotaryEmbedding` (device cos/sin from ``Ministral3Config`` via
``ministral_text_config``). If configured, ``forward_prefill`` / ``forward_prefill_from_embeddings`` may
omit ``rot_mats`` (``None``) and slice tables in-model; otherwise pass ``rot_mats`` as before.

Composes existing TT submodules from this experimental folder; no Torch fallback in forward.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional


# Load ``fp8_dequantize_compat`` by path so we do not import ``devstral_utils`` package ``__init__``
# (which pulls ``multimodal_demo_helpers`` and would circular-import this module).
def _ensure_fp8_scalar_compat() -> None:
    _mod_name = "_devstarl2_fp8_dequantize_compat_exec"
    if _mod_name in sys.modules:
        sys.modules[_mod_name].apply_fp8_dequantize_compat()
        return
    _path = Path(__file__).resolve().parent.parent / "devstral_utils" / "fp8_dequantize_compat.py"
    _spec = importlib.util.spec_from_file_location(_mod_name, _path)
    if _spec is None or _spec.loader is None:
        return
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules[_mod_name] = _mod
    _mod.apply_fp8_dequantize_compat()


_ensure_fp8_scalar_compat()

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.experimental.devstarl2_small.tt.tt_ministral3_decoder_layer import TtMinistral3DecoderLayer
from models.experimental.devstarl2_small.tt.tt_ministral_rotary_emb import TtMinistral3RotaryEmbedding
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.embedding import Embedding

if TYPE_CHECKING:
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config


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
        ministral_text_config: Optional["Ministral3Config"] = None,
        tt_rotary_embedding: Optional[TtMinistral3RotaryEmbedding] = None,
    ):
        super().__init__()
        self.args = model_args
        self.mesh_device = mesh_device
        self.n_layers = int(model_args.n_layers)

        if tt_rotary_embedding is not None and ministral_text_config is not None:
            raise ValueError("Pass at most one of tt_rotary_embedding and ministral_text_config.")

        if tt_rotary_embedding is not None:
            self.tt_rotary_embedding = tt_rotary_embedding
        elif ministral_text_config is not None:
            self.tt_rotary_embedding = TtMinistral3RotaryEmbedding(
                device=mesh_device,
                batch_size=model_args.max_batch_size,
                head_dim=model_args.head_dim,
                max_seq_len=model_args.max_seq_len,
                config=ministral_text_config,
                datatype=ttnn.bfloat16,
            )
        else:
            self.tt_rotary_embedding = None

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

    def forward_prefill_from_embeddings(
        self,
        hidden_states_11SH: ttnn.Tensor,
        rot_mats,
        position_ids,
        rope_start_pos: int = 0,
    ) -> ttnn.Tensor:
        if rot_mats is None:
            if self.tt_rotary_embedding is None:
                raise ValueError(
                    "rot_mats is required when tt_rotary_embedding is not set (pass ministral_text_config "
                    "or tt_rotary_embedding to TtMinistral3Model.__init__, or supply rot_mats explicitly)."
                )
            seq_len = int(hidden_states_11SH.shape[2])
            rot_mats = self.tt_rotary_embedding.slice_rot_mats_prefill(rope_start_pos, seq_len)
        h = hidden_states_11SH
        for layer in self.layers:
            h = layer.forward_prefill(h, rot_mats, position_ids=position_ids)
        return self.norm(h, Mode.PREFILL)

    def forward_prefill(
        self,
        input_ids_tt: ttnn.Tensor,
        position_ids,
        rot_mats=None,
        rope_start_pos: int = 0,
    ) -> ttnn.Tensor:
        if rot_mats is None:
            if self.tt_rotary_embedding is None:
                raise ValueError(
                    "rot_mats is required when tt_rotary_embedding is not set (pass ministral_text_config "
                    "or tt_rotary_embedding to TtMinistral3Model.__init__, or supply rot_mats explicitly)."
                )
            seq_len = int(input_ids_tt.shape[-1])
            rot_mats = self.tt_rotary_embedding.slice_rot_mats_prefill(rope_start_pos, seq_len)
        h = self.embed_tokens(input_ids_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = ttnn.unsqueeze_to_4D(h)
        return self.forward_prefill_from_embeddings(h, rot_mats, position_ids, rope_start_pos)


__all__ = ["TtMinistral3Model"]
