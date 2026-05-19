# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SwiGLU MLP for Devstral-2 / Ministral3 with simple TP=mesh-axis-N.

HF ``Ministral3MLP``::

    down_proj(silu(gate_proj(x)) * up_proj(x))

We follow ``Ministral3Config.base_model_tp_plan``:
  - ``gate_proj``, ``up_proj``: **column-parallel** (split ``intermediate_size`` across devices).
  - ``down_proj``: **row-parallel** (split ``intermediate_size`` across devices), then **all-reduce**
    along the TP axis.

Activations entering / leaving this module are **replicated** along the TP axis (full ``hidden_size``
on every device). That keeps the boundary contract symmetric with HF and avoids the need to
all-gather inside the layer. Inside the layer, intermediate activations of width
``intermediate_size / TP`` live on each device.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from models.experimental.devstral2_large.tt.ccl_helpers import all_reduce_replicate
from models.experimental.devstral2_large.tt.mem_config import get_compute_kernel_config, get_linear_program_config
from models.experimental.devstral2_large.tt.model_args import Devstral2Args
from models.experimental.devstral2_large.tt.weight_loading import resolve_weight_cache_path, upload_matmul_weight

__all__ = ["TtMLP"]


class TtMLP:
    """Two-stage SwiGLU MLP with column-then-row TP and a single all-reduce.

    ``hidden_size``-wide activations come in replicated across TP; ``hidden_size``-wide activations
    leave replicated across TP (post all-reduce).
    """

    def __init__(
        self,
        args: Devstral2Args,
        mesh_device,
        state_dict: dict,
        layer_idx: int,
        tt_ccl,
        *,
        dtype: Optional[ttnn.DataType] = None,
        weight_cache_path: Optional[str] = None,
    ) -> None:
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dtype = dtype or args.weight_dtype

        prefix = args.state_dict_prefix("mlp", layer_idx)
        gate_w = state_dict[prefix + "gate_proj.weight"]
        up_w = state_dict[prefix + "up_proj.weight"]
        down_w = state_dict[prefix + "down_proj.weight"]

        wp = resolve_weight_cache_path(weight_cache_path, args)
        self.gate_proj = upload_matmul_weight(
            gate_w,
            mesh_device,
            args,
            dtype=self.dtype,
            shard_dim=-1,
            weight_cache_path=wp,
            cache_key=f"{prefix}gate_proj",
        )
        self.up_proj = upload_matmul_weight(
            up_w, mesh_device, args, dtype=self.dtype, shard_dim=-1, weight_cache_path=wp, cache_key=f"{prefix}up_proj"
        )
        self.down_proj = upload_matmul_weight(
            down_w,
            mesh_device,
            args,
            dtype=self.dtype,
            shard_dim=-2,
            weight_cache_path=wp,
            cache_key=f"{prefix}down_proj",
        )

        self._compute_kernel_config = get_compute_kernel_config(mesh_device)

    def __call__(self, x: ttnn.Tensor, *, mode: str = "decode") -> ttnn.Tensor:
        act_mem = self.args.get_activation_mem_config(mode, self.mesh_device)
        seq_len = max(1, int(x.shape[-2]))

        def _linear(inp, weight, *, kind: str, activation: Optional[str] = None) -> ttnn.Tensor:
            return ttnn.linear(
                inp,
                weight,
                dtype=self.args.activation_dtype,
                memory_config=act_mem,
                activation=activation,
                program_config=get_linear_program_config(
                    self.args,
                    self.mesh_device,
                    mode=mode,
                    kind=kind,
                    seq_len=seq_len,
                    k=int(weight.shape[-2]),
                    n=int(weight.shape[-1]),
                ),
                compute_kernel_config=self._compute_kernel_config,
            )

        gate = _linear(x, self.gate_proj, kind="gate", activation="silu")
        up = _linear(x, self.up_proj, kind="up")
        inner = ttnn.mul(gate, up, memory_config=act_mem)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        down = _linear(inner, self.down_proj, kind="down")
        ttnn.deallocate(inner)

        return all_reduce_replicate(
            down,
            mesh_device=self.mesh_device,
            tt_ccl=self.tt_ccl,
            dim=3,
            cluster_axis=self.args.cluster_axis,
            topology=self.args.ccl_topology,
            memory_config=self.args.get_ccl_output_mem_config(mode, self.mesh_device),
        )

    def forward(self, x: ttnn.Tensor, *, mode: str = "decode") -> ttnn.Tensor:
        return self(x, mode=mode)
