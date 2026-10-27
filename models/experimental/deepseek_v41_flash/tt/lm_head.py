# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash ``lm_head``, vocabulary-parallel over a 1xTP stage.

A replicated ``[D, V]`` head has every rank of the output stage read the whole ~370 MB weight
(bfloat4_b at ``V = 129280``) each step. Here each rank holds and multiplies ``V / TP`` columns, and
an all-gather along the TP axis puts the whole ``[..., V]`` logits row back on every rank, the
output a replicated :class:`~models.experimental.deepseek_v4_flash.tt.layers.Linear` gives.
"""

import ttnn

from models.experimental.deepseek_v4_flash.tt.common import CCL_TOPOLOGY, DeepSeekV4Module
from models.experimental.deepseek_v4_flash.tt.decode.attention import _tp_cluster_axis
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.weight_cache import _CachePath


class DeepSeekV41LMHead(DeepSeekV4Module):
    """``x [A, B, T, D]`` (TILE) -> logits ``[A, B, T, V]`` (TILE), replicated over ``device``.

    ``weight`` is the checkpoint's ``[V, D]`` head or a thunk. On a multi-device ``device`` (a 1xTP
    mesh with fabric) the weight is split along ``V``, so ``V / TP`` has to be whole tiles; its cache
    file then carries the split (``<name>.tp<TP>``), since a cache key does not record the mesh mapping.
    """

    def __init__(self, weight, device: ttnn.MeshDevice, cache_file_name=None, dtype: ttnn.DataType = ttnn.bfloat16):
        self.device = device
        self.tp_size = device.get_num_devices()
        mesh_mapper = None
        if self.tp_size > 1:
            mesh_mapper = ttnn.ShardTensorToMesh(device, dim=-1)
            if cache_file_name is not None:
                cache_file_name = _CachePath(
                    f"{cache_file_name}.tp{self.tp_size}", getattr(cache_file_name, "require_cache", False)
                )
        self.linear = Linear(weight, device, cache_file_name, dtype=dtype, mesh_mapper=mesh_mapper)
        self.weight = self.linear.weight
        local_vocab = self.weight.shape[-1]
        if local_vocab % ttnn.TILE_SIZE:
            raise ValueError(f"lm_head's {local_vocab} columns per rank are not whole tiles over TP{self.tp_size}")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        logits = self.linear(x)
        if self.tp_size == 1:
            return logits
        gathered = ttnn.all_gather(
            logits,
            dim=3,
            cluster_axis=_tp_cluster_axis(self.device),
            num_links=1,
            topology=CCL_TOPOLOGY,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(logits)
        return gathered
