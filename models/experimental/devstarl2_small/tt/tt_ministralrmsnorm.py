# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Tenstorrent RMSNorm for Hugging Face Ministral3 text (``Ministral3RMSNorm``). Delegates to :class:`models.common.rmsnorm.RMSNorm` (``ttnn.rms_norm`` / distributed variants). Meta checkpoints use ``attention_norm`` / ``ffn_norm`` per layer after ``map_hf_to_meta_keys`` (HF ``input_layernorm`` / ``post_attention_layernorm``). Forward uses only ``ttnn`` ops; host tensors appear only when staging weights in ``__init__``.

from __future__ import annotations

import ttnn

from models.common.rmsnorm import RMSNorm


class TtMinistralRMSNorm(RMSNorm):
    # post_attention=False → attention_norm; True → ffn_norm.

    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        tt_ccl,
        *,
        post_attention: bool = False,
    ):
        weight_key = "ffn_norm" if post_attention else "attention_norm"
        super().__init__(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            weight_key=weight_key,
            layer_num=None,
            state_dict_prefix=args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            is_distributed=args.is_distributed_norm,
            add_unit_offset=args.rms_norm_add_unit_offset,
            ccl_topology=args.ccl_topology(),
            tt_ccl=tt_ccl,
        )


__all__ = ["TtMinistralRMSNorm"]
