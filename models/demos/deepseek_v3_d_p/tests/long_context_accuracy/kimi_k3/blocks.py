# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 device Blocks of the long-context cases (``common/prefill/long_context_accuracy/block.DeviceBlock``).

``KimiK3KdaBlock``: the KDA layer ``ttKDA`` of ``model.layers.0``, sequence-parallel over one mesh axis and
tensor-parallel over the other, run as consecutive full prefill calls from row 0 carrying its own state. After every
call it reads back the output and the recurrent state, so the scorer sees the state after each call.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

import ttnn
from models.demos.common.prefill.long_context_accuracy.block import BlockOutputs
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import make_actual_start
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    deallocate_state,
    make_kimi_k3_device_case,
    mla_row_permutation,
    reconstruct_sp_tp_tensor,
    to_sp_input,
)


class KimiK3KdaBlock:
    """One Kimi-K3 KDA layer (``config``, host ``weights`` with content digest ``weights_sha256``) on the device.

    ``forward`` takes the layer's input rows ``[chunk, hidden]`` in natural order and returns ``BlockOutputs`` with
    ``output`` ``[chunk, hidden]`` in natural order and ``extras["recurrent"]`` ``[heads, k, v]``, the recurrent state
    after the call (held by the last sequence rank). Every call is a full ``chunk``: a context length that is not a
    multiple of it is refused at ``build``. No carried state beyond what each call returns."""

    def __init__(
        self, config: KDAConfig, weights: dict[str, torch.Tensor], weights_sha256: str, tensor_parallel_axis: int
    ) -> None:
        self.config = config
        self.weights = weights
        self.weights_sha256 = weights_sha256
        self.tp_axis = tensor_parallel_axis
        self.sp_axis = 1 - tensor_parallel_axis
        self._layer = None

    def build(self, mesh: ttnn.MeshDevice, context_len: int, chunk: int) -> None:
        if context_len % chunk:
            raise ValueError(f"KimiK3KdaBlock runs full {chunk}-row calls; context length {context_len} is not one")
        self.mesh, self.chunk = mesh, chunk
        self.sp_size = tuple(mesh.shape)[self.sp_axis]
        self.permutation = mla_row_permutation(0, self.sp_size, chunk // self.sp_size)
        # The layer is sized from a sample input of one call; its values are never used.
        sample = torch.zeros(1, chunk, self.config.hidden_size, dtype=torch.bfloat16)
        self._layer, sample_tt = make_kimi_k3_device_case(
            mesh,
            KimiK3TestCase(self.config, self.weights, sample, self.weights_sha256),
            tensor_parallel_axis=self.tp_axis,
            cache_weights=False,
        )
        ttnn.deallocate(sample_tt)
        self._start_tt = make_actual_start(self._layer.device, 0)
        self._end_tt = make_actual_start(self._layer.device, chunk)
        self._state = self._layer.allocate_state(batch_size=1)

    def forward(self, x: torch.Tensor, lo: int, hi: int) -> BlockOutputs:
        if hi - lo != self.chunk:
            raise ValueError(f"rows [{lo}, {hi}) are not one full {self.chunk}-row call")
        hidden_tt = to_sp_input(x[None][:, self.permutation, :], self.mesh, self.sp_axis)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output_tt, self._state = self._layer.forward(hidden_tt, self._state, self._start_tt, self._end_tt)
        rotated = reconstruct_sp_tp_tensor(output_tt, self.mesh, self.sp_axis, self.tp_axis, tp_dim=2, sp_dim=1)
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(output_tt)
        output = torch.empty_like(rotated)
        output[:, self.permutation, :] = rotated
        return BlockOutputs(output[0], {"recurrent": self._last_rank_recurrent()[0]})

    def _last_rank_recurrent(self) -> torch.Tensor:
        """The recurrent state ``[1, heads, k, v]`` held by the last sequence rank: its tensor-parallel head shards,
        concatenated. Reads back only those shards (``reconstruct_state_at_sp_rank`` reads every device's)."""
        shards = ttnn.get_device_tensors(self._state.recurrent)
        columns = tuple(self.mesh.shape)[1]
        tp_size = tuple(self.mesh.shape)[self.tp_axis]
        last = self.sp_size - 1
        coordinates = [(last, tp) if self.sp_axis == 0 else (tp, last) for tp in range(tp_size)]
        return torch.cat([ttnn.to_torch(shards[row * columns + column]) for row, column in coordinates], dim=1)

    def carried_state(self) -> Mapping[str, torch.Tensor]:
        return {}

    def release(self) -> None:
        if self._layer is None:
            return
        deallocate_state(self._state)
        ttnn.deallocate(self._start_tt)
        ttnn.deallocate(self._end_tt)
        self._layer = None
