# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native, tensor-parallel TTNN port of the output head (`proj_out`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)` (FLUX.2 Klein 9B).

`Flux2Transformer2DModel.proj_out` is the DiT's decoder head: a single
`nn.Linear(inner_dim=4096 -> patch_size**2 * out_channels = 128, bias=False)`
run on the image tokens after `norm_out`. It is the structural analogue of a
causal-LM `lm_head`, which is why the planner pointed this component at
`models/tt_transformers/tt/lm_head.py`.

TENSOR-PARALLEL SCHEME (TP = devices in the mesh; validated at TP=8)
-------------------------------------------------------------------
ROW-parallel (split the INPUT features, all_reduce the partial sums).

`models/tt_transformers/tt/lm_head.py` splits its weight COLUMN-wise, over the
vocabulary: for an LM head the output (padded_vocab_size, ~128k) is the large,
tile-aligned axis and the input (model dim) is the small one. Here the shape is
the exact opposite -- 4096 in, **128** out -- so the reference's axis is the
wrong one to copy. Applying the same underlying principle ("split the large
matmul axis; never split an axis into ragged, sub-tile shards") to THIS shape
selects the input axis:

  * column-parallel would hand each of 8 chips 128/8 = **16** output columns.
    16 < the 32-wide tile, so every shard would be half padding and the gather
    dim would carry tile padding -- all_gather then has to fall back to its
    composite all-broadcast path to stay correct, and the matmul wastes half of
    every tile it computes.
  * row-parallel hands each chip 4096/8 = **512** input features: 16 whole
    tiles, no padding, and the output stays a full [.., 128] on every chip.

So the weight is sharded on its INPUT axis (`ShardTensorToMesh(dim=2)` of the
`[1, 1, in, out]` ttnn-transposed weight). The activation arrives REPLICATED
from the harness, so `ttnn.mesh_partition(dim=-1)` -- the documented inverse of
all_gather -- turns it into the matching K-shard on each chip. Each chip then
holds a PARTIAL sum over the full output, and one `ttnn.all_reduce` (SUM) makes
every chip hold the single-device answer. The arithmetic is unchanged: a
row-parallel matmul is just the K-loop of the same dot product, split.

A bias, if a future config adds one, is added AFTER the all_reduce -- adding it
before would count it TP times.

At TP=1 the mesh ops are skipped and this is a plain `ttnn.linear`. The forward
is pure ttnn: no torch math, no device->host readback.
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32


def _mesh_shape(device):
    """(rows, cols) of `device` as a mesh, or None if it is a single device."""
    shape = getattr(device, "shape", None)
    if shape is None:
        return None
    try:
        dims = [int(d) for d in shape]
    except TypeError:
        return None
    if len(dims) == 1:
        dims = [1, dims[0]]
    if len(dims) != 2:
        return None
    return dims[0], dims[1]


def _num_devices(device) -> int:
    shape = _mesh_shape(device)
    return shape[0] * shape[1] if shape is not None else 1


def _replicate_mapper(device):
    return ttnn.ReplicateTensorToMesh(device) if _num_devices(device) > 1 else None


def _shard_mapper(device, dim: int):
    return ttnn.ShardTensorToMesh(device, dim=dim) if _num_devices(device) > 1 else None


class TtDecoderHead:
    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("decoder_head needs the torch reference module for weights")

        self.device = device
        self.tp = _num_devices(device)
        self.mesh = _mesh_shape(device)

        # torch keeps [out, in]; ttnn wants [in, out].
        w = torch_module.weight.detach().to(torch.float32).t().contiguous()
        self.in_features, self.out_features = int(w.shape[0]), int(w.shape[1])

        # Row-parallel only when every chip gets a whole number of tiles of K.
        self.row_parallel = self.tp > 1 and self.in_features % (TILE * self.tp) == 0

        self.weight = ttnn.from_torch(
            w.reshape(1, 1, self.in_features, self.out_features),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_shard_mapper(device, 2) if self.row_parallel else _replicate_mapper(device),
        )

        bias = getattr(torch_module, "bias", None)
        self.bias = (
            None
            if bias is None
            else ttnn.from_torch(
                bias.detach().to(torch.float32).reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(device),
            )
        )

        self.kernel_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _to_device(self, tensor):
        if isinstance(tensor, ttnn.Tensor):
            return tensor
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(self.device),
        )

    def _all_reduce(self, x):
        if self.mesh is not None and self.mesh[0] > 1:
            # (DP, TP) mesh: reduce only along the tensor-parallel axis.
            return ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.all_reduce(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _partition(self, x, dim):
        if self.mesh is not None and self.mesh[0] > 1:
            return ttnn.mesh_partition(x, dim, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.mesh_partition(x, dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, x, **kwargs):
        x = self._to_device(x)
        shape = list(x.shape)
        seq = int(shape[-2])
        batch = int(shape[0]) if len(shape) >= 3 else 1
        x4 = x if len(shape) == 4 else ttnn.reshape(x, [batch, 1, seq, self.in_features])

        if self.row_parallel:
            x4 = self._partition(x4, 3)

        out = ttnn.linear(
            x4,
            self.weight,
            compute_kernel_config=self.kernel_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.row_parallel:
            out = self._all_reduce(out)
        if self.bias is not None:
            # After the reduce: a replicated bias added before it would be
            # summed TP times.
            out = ttnn.add(out, self.bias)

        if len(shape) != 4:
            out = ttnn.reshape(out, shape[:-1] + [self.out_features])
        return out


def build(device, torch_module=None):
    return TtDecoderHead.build(device, torch_module)


def decoder_head(device, torch_module=None):
    return TtDecoderHead.build(device, torch_module)
