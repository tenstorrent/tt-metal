# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of the Qwen2.5-VL `lm_head` (Linear 3584 -> 152064, no bias).

TP scheme (as models/tt_transformers/tt/lm_head.py): column-parallel over the vocab. Each chip on the
TP axis owns vocab/TP output columns, computes its logits slice locally, and an all_gather on the
last dim over the TP axis reassembles the full logits on every chip. Any DP mesh axis replicates.
"""

from __future__ import annotations

import ttnn


def _mesh_shape(device):
    try:
        shape = tuple(device.shape)
        if len(shape) == 2:
            return shape
    except Exception:
        pass
    n = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    return (1, n)


def _shard_mapper(device, dim):
    """Shard `dim` across the TP (column) axis of the mesh, replicate across any DP (row) axis."""
    rows, cols = _mesh_shape(device)
    if rows == 1:
        return ttnn.ShardTensorToMesh(device, dim=dim)
    return ttnn.ShardTensor2dMesh(device, mesh_shape=(rows, cols), dims=(None, dim))


class TtDecoderHead:
    def __init__(self, device, torch_module):
        self.device = device
        _, self.tp = _mesh_shape(device)
        w = torch_module.weight.detach()  # [vocab, hidden]
        self.vocab = w.shape[0]
        assert self.vocab % (self.tp * 32) == 0, f"vocab {self.vocab} not tile-divisible by TP={self.tp}"
        self.weight = ttnn.from_torch(
            w.t().contiguous(),  # [hidden, vocab] -> vocab columns sharded
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=_shard_mapper(device, -1),
        )
        self.compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x, **kwargs):
        logits = ttnn.linear(x, self.weight, compute_kernel_config=self.compute_cfg)
        if self.tp > 1:
            logits = ttnn.all_gather(logits, dim=-1, cluster_axis=1, topology=ttnn.Topology.Linear)
        return logits


def build(device, torch_module=None):
    return TtDecoderHead(device, torch_module)


def decoder_head(device, torch_module=None):
    return TtDecoderHead(device, torch_module)
