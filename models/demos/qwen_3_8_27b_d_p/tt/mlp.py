# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dense SiLU-SwiGLU MLP — every one of the 64 layers has one; there are no experts.

``gate_proj`` / ``up_proj`` are column-parallel (the intermediate dim shards across TP) and
``down_proj`` is row-parallel, so each chip holds a partial sum over its intermediate shard and
the layer closes with a TP all-reduce back to the replicated full-emb residual.

The activation is plain ``silu(gate) * up``. The structure here is MiniMax-M3's ``dense_mlp.py``;
its *activation* is a clamped swigluoai with an alpha and a limit, and porting that math across
would be a silent PCC loss rather than an error — hence ``ttnn.silu`` written out here.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..config import MeshConfig
from ..reference.config import Qwen35TextConfig
from ..utils.general_utils import get_cache_file_name
from ..utils.substate import substate
from .compute import matmul_compute_config


class MLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg: Qwen35TextConfig,
        state_dict: dict[str, torch.Tensor],
        *,
        mesh_config: MeshConfig,
        ccl_manager,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        tensor_cache_path: Optional[str] = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size
        self.activation_dtype = activation_dtype
        self.compute_config = matmul_compute_config(mesh_device)
        assert (
            cfg.intermediate_size % (mesh_config.tp * ttnn.TILE_SIZE) == 0
        ), f"intermediate_size {cfg.intermediate_size} must split tile-aligned across tp={mesh_config.tp}"

        col = mesh_config.column_parallel(mesh_device)
        row = mesh_config.row_parallel(mesh_device)

        def _prep(name: str) -> Optional[torch.Tensor]:
            if not state_dict:
                return None  # cache-only: ttnn.as_tensor loads the tilized tensor straight from disk
            # HF stores Linear weight as [out, in]; ttnn.linear wants [in, out].
            return substate(state_dict, name)["weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        def _load(name: str, mapper) -> ttnn.Tensor:
            return ttnn.as_tensor(
                _prep(name),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=weight_dtype,
                mesh_mapper=mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, name),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.gate_proj = _load("gate_proj", col)
        self.up_proj = _load("up_proj", col)
        self.down_proj = _load("down_proj", row)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``x`` full-emb replicated -> full-emb replicated."""
        gate = ttnn.linear(x, self.gate_proj, dtype=self.activation_dtype, compute_kernel_config=self.compute_config)
        up = ttnn.linear(x, self.up_proj, dtype=self.activation_dtype, compute_kernel_config=self.compute_config)
        act = ttnn.silu(gate)
        gate.deallocate(True)
        act = ttnn.multiply(act, up, output_tensor=act)
        up.deallocate(True)
        out = ttnn.linear(act, self.down_proj, dtype=self.activation_dtype, compute_kernel_config=self.compute_config)
        act.deallocate(True)
        if self.mesh_config.tp > 1:
            out = self.mesh_config.allreduce(out, self.ccl_manager, axis=self.mesh_config.tp_axis)
        return out
