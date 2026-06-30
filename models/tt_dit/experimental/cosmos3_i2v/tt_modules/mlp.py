# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""tt-symbiote adapter for the native Cosmos3VLTextMLP (SwiGLU).

Mirrors `tt_modules/joint_attention.py`: a `TTNNModule` that
`register_module_replacement_dict` can swap in for every
`Cosmos3VLTextMLP` instance in the HF transformer. Both `mlp` (und
expert) and `mlp_moe_gen` (gen expert) instances get the same swap —
they're separate `Cosmos3VLTextMLP` modules with disjoint weights.

Mesh-shape policy matches the attention adapter: TP along the larger
axis, SP not exercised yet. The MLP doesn't have a per-head divisibility
constraint, but to keep the trunk's TP factor consistent the same
{1,2,4,8} valid set applies (driven by the attention's KV-head limit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule

if TYPE_CHECKING:
    from torch import nn

    from models.tt_dit.experimental.cosmos3_i2v.model.mlp import Cosmos3VLTextMLP


class TTNNCosmos3VLTextMLP(TTNNModule):
    """Drop-in replacement for `Cosmos3VLTextMLP` under tt-symbiote."""

    def __init__(self, *, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self._config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        }
        self._captured_state_dict: dict | None = None
        self._inner: Cosmos3VLTextMLP | None = None

    @classmethod
    def from_torch(cls, mlp: nn.Module) -> TTNNCosmos3VLTextMLP:
        """Build adapter from a `Cosmos3VLTextMLP` instance.

        We sniff dimensions off the loaded sub-Linears since the reference
        class doesn't surface them as attributes.
        """
        hidden_size = mlp.gate_proj.in_features
        intermediate_size = mlp.gate_proj.out_features
        new = cls(hidden_size=hidden_size, intermediate_size=intermediate_size)
        new._fallback_torch_layer = mlp
        return new

    def preprocess_weights_impl(self) -> None:
        if self.torch_layer is None:
            msg = "TTNNCosmos3VLTextMLP.preprocess_weights_impl requires a fallback torch layer"
            raise RuntimeError(msg)
        self._captured_state_dict = self.torch_layer.state_dict()

    def move_weights_to_device_impl(self) -> None:
        # Local imports avoid circular imports via the package __init__ during replacement registration.
        from models.tt_dit.experimental.cosmos3_i2v.model.mlp import Cosmos3VLTextMLP
        from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
        from models.tt_dit.parallel.manager import CCLManager

        if self.device is None:
            msg = "TTNNCosmos3VLTextMLP.move_weights_to_device_impl requires a device"
            raise RuntimeError(msg)
        if self._captured_state_dict is None:
            msg = "preprocess_weights must run before move_weights_to_device"
            raise RuntimeError(msg)

        mesh_shape = tuple(self.device.shape)
        tp_axis = max(range(len(mesh_shape)), key=lambda i: mesh_shape[i])
        tp_factor = mesh_shape[tp_axis]
        sp_axis = 1 - tp_axis if len(mesh_shape) == 2 else 0
        sp_factor = mesh_shape[sp_axis] if len(mesh_shape) == 2 else 1

        parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(1, 0),
            sequence_parallel=ParallelFactor(sp_factor, sp_axis),
            tensor_parallel=ParallelFactor(tp_factor, tp_axis),
        )

        ccl_manager = (
            CCLManager(mesh_device=self.device, num_links=1, topology=ttnn.Topology.Linear)
            if tp_factor > 1 or sp_factor > 1
            else None
        )

        self._inner = Cosmos3VLTextMLP(
            mesh_device=self.device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            **self._config,
        )
        self._inner.load_torch_state_dict(self._captured_state_dict)

        self._captured_state_dict = None
        self._fallback_torch_layer = None

    def deallocate_weights_impl(self) -> None:
        if self._inner is not None:
            self._inner.deallocate_weights()
            self._inner = None

    @staticmethod
    def _to_4d(x: ttnn.Tensor) -> tuple[ttnn.Tensor, tuple[int, ...]]:
        """Reshape `[N, H]` or `[B, N, H]` to `[1, 1, N, H]`. Returns (reshaped, orig_shape)."""
        shape = tuple(x.shape)
        rank = len(shape)
        if rank == 4:
            return x, shape
        if rank == 2:
            n, h = shape
            return ttnn.reshape(x, (1, 1, n, h)), shape
        if rank == 3:
            b, n, h = shape
            return ttnn.reshape(x, (1, b, n, h)), shape
        msg = f"TTNNCosmos3VLTextMLP expects rank-2/3/4 input, got rank {rank}"
        raise ValueError(msg)

    @staticmethod
    def _ensure_tile(x: ttnn.Tensor) -> ttnn.Tensor:
        """tt-symbiote's host→device conversion lands tensors in ROW_MAJOR. The matmul kernels
        require TILE — convert here so the inner native module can assume TILE."""
        if x.layout != ttnn.TILE_LAYOUT:
            return ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._inner is None:
            msg = "TTNNCosmos3VLTextMLP.forward called before weights were moved to device"
            raise RuntimeError(msg)
        x_4d, orig_shape = self._to_4d(self._ensure_tile(x))
        out_4d = self._inner(x_4d)
        if len(orig_shape) != 4:
            return ttnn.reshape(out_4d, orig_shape)
        return out_4d
