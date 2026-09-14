# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dense SwiGLU FFN: 4096 -> 14336 -> 4096, no biases.

There is no ``ttnn.mlp``, so the block is composed from the ops that do exist (recipe D3 step 3):
two column-parallel matmuls, a fused SiLU-multiply, one row-parallel matmul, one TP all-reduce.

The **structure** (which projection is column- vs row-parallel and where the collective lands) is
M3's dense MLP; the **math** is not. M3 and gpt-oss use the clamped ``swigluoai`` activation with an
alpha and a clamp limit — Llama uses plain ``silu(gate) * up``, so the activation is written fresh.
Porting the source's activation is exactly the "the activation variant differs — port the structure,
never the math" case in the recipe's reject column.

``intermediate_size`` 14336 / tp 4 = 3584 per device, which is tile-aligned (112 tiles), so no
padding is needed on the column-parallel split.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule

from .compute import matmul_compute_config
from ..utils.general import cache_name, substate


def swiglu(gate, up):
    """``silu(gate) * up`` as one op — the SiLU is fused onto the multiply's first input.

    Same call shape as ``tt_transformers/tt/mlp.py`` and ``common/modules/mlp/mlp_1d.py``; the
    activation is plain SILU (no alpha, no clamp — those are gpt-oss / M3 swigluoai, not Llama).
    """
    return ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])


class MLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg,
        mesh_config,
        ccl_manager,
        state_dict: Optional[dict] = None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = cfg.hidden_size
        self.compute_kernel_config = matmul_compute_config(mesh_device)

        col = mesh_config.column_parallel(mesh_device)  # shard the intermediate (output) dim
        row = mesh_config.row_parallel(mesh_device)  # shard the intermediate (input) dim

        def prep(name):
            # HF stores nn.Linear as [out, in]; ttnn.linear wants [in, out].
            return substate(state_dict, name)["weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        has_sd = bool(state_dict)
        self.gate_proj = ttnn.as_tensor(
            prep("gate_proj") if has_sd else None,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=col,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(tensor_cache_path, "gate_proj"),
        )
        self.up_proj = ttnn.as_tensor(
            prep("up_proj") if has_sd else None,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=col,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(tensor_cache_path, "up_proj"),
        )
        self.down_proj = ttnn.as_tensor(
            prep("down_proj") if has_sd else None,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=row,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(tensor_cache_path, "down_proj"),
        )

    def forward(self, x):
        """``[1, 1, s_local, hidden]`` (TP-replicated) -> same shape, TP-reduced."""
        gate = ttnn.linear(x, self.gate_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
        up = ttnn.linear(x, self.up_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
        act = swiglu(gate, up)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(act, self.down_proj, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
        act.deallocate(True)
        # down_proj is row-parallel: each TP device holds a partial sum over its intermediate shard,
        # so the all-reduce is what makes the result a result, not an optimization.
        if self.mesh_config.tp > 1:
            out = self.mesh_config.all_reduce(out, self.ccl_manager)
        return out
