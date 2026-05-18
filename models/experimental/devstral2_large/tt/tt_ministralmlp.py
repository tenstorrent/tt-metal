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

        self._compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Per-device shard width for the intermediate is ``intermediate_size / TP``.
        gate = ttnn.linear(
            x,
            self.gate_proj,
            dtype=self.args.activation_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        up = ttnn.linear(
            x,
            self.up_proj,
            dtype=self.args.activation_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        # Fused activation in the next matmul would skip the explicit mul, but ttnn.linear's
        # ``activation`` arg only takes ``silu``/``gelu`` on the *output* of THIS matmul, not on one
        # input. Keep the explicit silu+mul; one extra L1 traversal in exchange for clarity.
        gate = ttnn.silu(gate, memory_config=ttnn.L1_MEMORY_CONFIG)
        inner = ttnn.mul(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        down = ttnn.linear(
            inner,
            self.down_proj,
            dtype=self.args.activation_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self._compute_kernel_config,
        )
        ttnn.deallocate(inner)

        # All-reduce along the TP axis so every device leaves with the full sum (replicated).
        return all_reduce_replicate(
            down,
            mesh_device=self.mesh_device,
            tt_ccl=self.tt_ccl,
            dim=3,
            cluster_axis=self.args.cluster_axis,
            topology=self.args.ccl_topology,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self(x)
