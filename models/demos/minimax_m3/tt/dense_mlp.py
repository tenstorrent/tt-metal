# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M3 dense MLP for the first 3 layers (moe_layer_freq == 0).

A plain clamped-"swigluoai" SwiGLU FFN: down(swiglu(gate(x), up(x))), at
dense_intermediate_size (12288). Weights load from mlp.{gate,up,down}_proj. gate/up are
column-parallel (shard the intermediate dim across TP); down is row-parallel followed by a
TP all-reduce. The activation is the same clamped SwiGLU as the MoE experts (see
moe/activation.apply_swiglu); anchor: transformers minimax_m3_vl MLP.
"""

from types import SimpleNamespace

import ttnn
from models.demos.minimax_m3.utils.general_utils import get_cache_file_name
from models.demos.minimax_m3.utils.substate import substate

from .moe.activation import apply_swiglu


class DenseMLP:
    """Dense SwiGLU FFN for MiniMax-M3 layers 0-2."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        weight_dtype=ttnn.bfloat16,
        tensor_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        # apply_swiglu only reads .swiglu_limit and .alpha.
        self.swiglu_cfg = SimpleNamespace(
            swiglu_limit=getattr(hf_config, "swiglu_limit", 7.0),
            alpha=getattr(hf_config, "swiglu_alpha", 1.702),
        )

        col_mapper = mesh_config.column_parallel(mesh_device)  # shard output (intermediate) dim
        row_mapper = mesh_config.row_parallel(mesh_device)  # shard input (intermediate) dim

        def _load(name, weight, mapper):
            # weight is None in cache-only mode (empty state_dict) — still build so ttnn.as_tensor loads
            # the tilized tensor straight from the cache. A dense FFN always has all three projections,
            # so return None only when there's no cache path to load from.
            if weight is None and not tensor_cache_path:
                return None
            return ttnn.as_tensor(
                weight,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, name),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if state_dict:
            # HF stores Linear weight as [out, in]; ttnn.linear wants [in, out] -> transpose.
            def _prep(key):
                return substate(state_dict, key)["weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

            gate_w, up_w, down_w = _prep("gate_proj"), _prep("up_proj"), _prep("down_proj")
        else:
            gate_w = up_w = down_w = None

        self.gate_proj = _load("gate_proj", gate_w, col_mapper)
        self.up_proj = _load("up_proj", up_w, col_mapper)
        self.down_proj = _load("down_proj", down_w, row_mapper)

    def __call__(self, x):
        gate = ttnn.linear(x, self.gate_proj, dtype=ttnn.bfloat16)
        up = ttnn.linear(x, self.up_proj, dtype=ttnn.bfloat16)
        act = apply_swiglu(gate, up, self.swiglu_cfg)  # clamped swigluoai (M3)
        out = ttnn.linear(act, self.down_proj, dtype=ttnn.bfloat16)
        act.deallocate(True)
        # down is row-parallel: each TP device holds a partial sum over the intermediate
        # shard -> all-reduce to complete the hidden output.
        if self.mesh_config.tp > 1:
            out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
        return out
