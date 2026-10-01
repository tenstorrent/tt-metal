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
from models.demos.minimax_m3.utils.profiler_utils import FINE, zone
from models.demos.minimax_m3.utils.substate import substate

from .attention.operations import assert_sharded_residual_unpadded
from .moe.activation import swiglu
from .residual import use_sharded_residual


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
        scatter_output=None,
    ):
        """scatter_output: True => close with a TP reduce-scatter (output emb/tp, the sharded-residual
        contract); False => all-reduce (output full emb). None derives it from the residual scheme.
        Both call sites want the derived value — the dense layers 0-2 and the MoE layers' shared
        expert — which is exactly why the shared expert stops paying its own all-gather when sharded."""
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.scatter_output = use_sharded_residual() if scatter_output is None else scatter_output
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
        with zone("gate_up_proj", FINE):
            gate = ttnn.linear(x, self.gate_proj, dtype=ttnn.bfloat16)
            up = ttnn.linear(x, self.up_proj, dtype=ttnn.bfloat16)
        with zone("swiglu", FINE):
            act = swiglu(gate, up, self.swiglu_cfg)  # clamped swigluoai (M3); consumes gate and up
        with zone("down_proj", FINE):
            out = ttnn.linear(act, self.down_proj, dtype=ttnn.bfloat16)
        act.deallocate(True)
        # down is row-parallel: each TP device holds a partial sum over the intermediate shard, so a TP
        # collective is required either way. Sharded residual -> reduce-scatter only (emb/tp out, which
        # the caller adds straight into its residual); replicated residual -> full all-reduce (RS + AG).
        if self.mesh_config.tp > 1:
            if self.scatter_output:
                # Same guard attention's apply_reduce_scatter runs: a non-tile-aligned hidden/tp would
                # land output-dim padding inside one TP column's residual slice after the scatter.
                assert_sharded_residual_unpadded(self.mesh_config, self.hidden_size)
                with zone("tp_reduce_scatter"):
                    scattered = self.mesh_config.reduce_scatter(
                        out, self.ccl_manager, dim=3, axis=self.mesh_config.tp_axis
                    )
                out.deallocate(True)
                out = scattered
            else:
                with zone("tp_allreduce"):
                    out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
        return out
