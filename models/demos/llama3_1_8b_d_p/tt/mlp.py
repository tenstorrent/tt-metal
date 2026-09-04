# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B dense SwiGLU MLP.

Copied from ``minimax_m3/tt/dense_mlp.py`` — the only dense-FFN donor in the prefill packages
(``gpt_oss_d_p/tt/mlp.py`` is an EP-MoE wrapper and does not transfer). What carries over is the
column/row-parallel split and where the CCL lands:

  * ``gate_proj`` / ``up_proj`` are **column-parallel** — the intermediate dim (14336) shards across
    the TP axis, 3584 per chip at TP=4.
  * ``down_proj`` is **row-parallel** — each chip holds a partial sum over its intermediate shard, so
    a TP collective is mandatory. This package uses the replicated-residual contract: a full
    all-reduce, output in full emb. (M3's sharded-residual reduce-scatter variant is a perf
    optimisation and is deliberately out of bring-up scope.)

What does NOT carry over is the activation: M3 uses clamped ``swigluoai`` (alpha / limit); Llama is
plain SwiGLU, ``silu(gate) * up``. No biases (``mlp_bias: false``).
"""

import ttnn
from models.demos.llama3_1_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama3_1_8b_d_p.utils.substate import substate


class MLP:
    """Dense gated SwiGLU FFN: ``down(silu(gate(x)) * up(x))``."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        weight_dtype=ttnn.bfloat4_b,
        tensor_cache_path=None,
    ):
        """
        Args:
            mesh_device: TTNN mesh device
            hf_config: config carrying ``hidden_size`` / ``intermediate_size``
            state_dict: ``mlp.*`` substate with gate_proj/up_proj/down_proj ``weight``. Empty dict =>
                cache-only load (every tilized tensor comes from ``tensor_cache_path``).
            mesh_config: MeshConfig (TP axis carries the intermediate-dim shard)
            ccl_manager: CCLManager, required when ``mesh_config.tp > 1``
            weight_dtype: bfloat4_b per the spec's ``numerics.dense_mlp_weights``
            tensor_cache_path: optional weight-cache dir
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        col_mapper = mesh_config.column_parallel(mesh_device)  # shard output (intermediate) dim
        row_mapper = mesh_config.row_parallel(mesh_device)  # shard input (intermediate) dim

        def _load(name, weight, mapper):
            # weight is None in cache-only mode — still build so ttnn.as_tensor loads the tilized
            # tensor straight from the cache. A dense FFN always has all three projections, so return
            # None only when there is no cache path to load from either.
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
        """x: [1, 1, S, hidden] (full emb, replicated across TP) -> [1, 1, S, hidden]."""
        gate = ttnn.linear(x, self.gate_proj, dtype=ttnn.bfloat16)
        up = ttnn.linear(x, self.up_proj, dtype=ttnn.bfloat16)
        act = ttnn.multiply(ttnn.silu(gate), up)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(act, self.down_proj, dtype=ttnn.bfloat16)
        act.deallocate(True)
        # down_proj is row-parallel: each TP chip holds a partial sum over its intermediate shard.
        if self.mesh_config.tp > 1:
            out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
        return out
