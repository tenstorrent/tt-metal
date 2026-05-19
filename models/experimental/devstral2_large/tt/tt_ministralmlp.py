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

import torch
import ttnn

from models.experimental.devstral2_large.tt.ccl_helpers import all_reduce_replicate
from models.experimental.devstral2_large.tt.mem_config import get_compute_kernel_config, get_linear_program_config
from models.experimental.devstral2_large.tt.model_args import Devstral2Args

__all__ = ["TtMLP"]


def _shard_dim_n(t: torch.Tensor, dim: int, n: int) -> list[torch.Tensor]:
    """Split ``t`` along ``dim`` into ``n`` contiguous shards."""
    sz = t.shape[dim]
    if sz % n != 0:
        raise ValueError(f"Cannot shard dim {dim} of size {sz} into {n} pieces")
    return list(torch.chunk(t, n, dim=dim))


def _load_colwise(
    weight: torch.Tensor,  # HF shape: (out_features, in_features)
    mesh_device,
    args: Devstral2Args,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    """Column-parallel: shard along ``out_features`` (HF dim 0); upload as ``(in, out_per_dev)`` per device."""
    # ttnn.linear expects weights in (in, out) layout, so transpose post-shard.
    w = weight.to(torch.bfloat16).T.contiguous()  # (in, out)
    return ttnn.from_torch(
        w,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, -1) if args.cluster_axis == 1 else (-1, None),
            mesh_shape=args.mesh_shape,
        ),
    )


def _load_rowwise(
    weight: torch.Tensor,  # HF shape: (out_features, in_features)
    mesh_device,
    args: Devstral2Args,
    dtype: ttnn.DataType,
) -> ttnn.Tensor:
    """Row-parallel: shard along ``in_features`` (HF dim 1); upload as ``(in_per_dev, out)`` per device."""
    w = weight.to(torch.bfloat16).T.contiguous()  # (in, out)
    return ttnn.from_torch(
        w,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, -2) if args.cluster_axis == 1 else (-2, None),
            mesh_shape=args.mesh_shape,
        ),
    )


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

        self.gate_proj = _load_colwise(gate_w, mesh_device, args, self.dtype)
        self.up_proj = _load_colwise(up_w, mesh_device, args, self.dtype)
        self.down_proj = _load_rowwise(down_w, mesh_device, args, self.dtype)

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
                    self.args, self.mesh_device, mode=mode, kind=kind, seq_len=seq_len
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
