# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One decoder layer.

Both layer types have an identical residual shape — norm, token mixer, add, norm, MLP, add — and
differ only in which mixer sits in the middle. So the mixer is chosen **once at construction** and
stored, rather than re-deciding on ``layer_type`` inside ``forward``: the two mixers share the
``mix(x, ctx)`` signature (see ``tt/context.py``) precisely so this composition has no branch in it.

The residual stream is full-emb **replicated** across TP and sequence-sharded across SP. Attention
and the MLP both close with a TP all-reduce, so every norm sees the full width and no collective is
needed to feed the column-parallel projections. The alternative — an ``emb/tp``-sharded residual
with one all-gather per norm — trades a collective for a collective and adds a second layout to
keep straight; at bring-up the simpler invariant is worth more than the op count.
"""

from __future__ import annotations

from typing import Optional, Protocol

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..config import MeshConfig
from ..reference.config import Qwen35TextConfig
from ..utils.general_utils import get_cache_file_name
from ..utils.substate import substate
from .attention.prefill import Attention
from .context import ChunkContext
from .gdn.prefill import GatedDeltaNet
from .mlp import MLP
from .rms_norm import RMSNorm


class TokenMixer(Protocol):
    """What ``DecoderLayer`` needs of a mixer. ``Attention`` and ``GatedDeltaNet`` both satisfy it."""

    def mix(self, x: ttnn.Tensor, ctx: ChunkContext) -> ttnn.Tensor:
        ...


class DecoderLayer(LightweightModule):
    #: state-dict sub-key and class for each ``layer_types`` value.
    MIXERS: dict[str, tuple[str, type]] = {
        "full_attention": ("self_attn", Attention),
        "linear_attention": ("linear_attn", GatedDeltaNet),
    }

    def __init__(
        self,
        mesh_device,
        cfg: Qwen35TextConfig,
        state_dict: dict,
        layer_idx: int,
        *,
        mesh_config: MeshConfig,
        ccl_manager,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat8_b,
        tensor_cache_path: Optional[str] = None,
    ) -> None:
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.mesh_config = mesh_config

        sub_key, mixer_cls = self.MIXERS[self.layer_type]
        self.mixer: TokenMixer = mixer_cls(
            mesh_device,
            cfg,
            substate(state_dict, sub_key),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            layer_idx=layer_idx,
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, sub_key),
            **({"cache_dtype": cache_dtype} if mixer_cls is Attention else {}),
        )
        self.mlp = MLP(
            mesh_device,
            cfg,
            substate(state_dict, "mlp"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
        )
        self.input_layernorm = RMSNorm(
            mesh_device,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            substate(state_dict, "input_layernorm"),
            mesh_config=mesh_config,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            substate(state_dict, "post_attention_layernorm"),
            mesh_config=mesh_config,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
        )

    def forward(self, x: ttnn.Tensor, ctx: ChunkContext) -> ttnn.Tensor:
        residual = x
        h = self.input_layernorm(x)
        h = self.mixer.mix(h, ctx)
        x = ttnn.add(residual, h, output_tensor=h)
        residual.deallocate(True)

        residual = x
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        out = ttnn.add(residual, h, output_tensor=h)
        residual.deallocate(True)
        return out
