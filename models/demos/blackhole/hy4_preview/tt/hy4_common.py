# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Small device helpers shared by the Hy4 attention modules (upload with mesh mappers, compute configs, rope on device)."""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_rope import rope_tables, rotation_matrix

SENTINEL = 0xFFFFFFFF


def mesh_rc(md):
    return int(md.shape[0]), int(md.shape[1])


def upload(md, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard_dim=None, row_dim=None):
    """host torch -> mesh. shard_dim: tensor dim split over the TP columns (None = replicate); row_dim: tensor dim split over the mesh rows (None = replicate)."""
    r, c = mesh_rc(md)
    if shard_dim is None and row_dim is None:
        mapper = ttnn.ReplicateTensorToMesh(md)
    else:
        mapper = ttnn.ShardTensor2dMesh(md, dims=(row_dim, shard_dim), mesh_shape=(r, c))
    return ttnn.from_torch(
        t, device=md, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )


def to_host(t, md, col=0, row=0):
    """one device's shard of a mesh tensor -> torch (device (row, col))."""
    r, c = mesh_rc(md)
    return ttnn.to_torch(ttnn.get_device_tensors(t)[row * c + col])


def compute_cfg(md, fidelity="hifi4", fp32_acc=True):
    return ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity={"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2}[fidelity],
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


class RopeDev:
    """cos / sin / rotation matrix on device for one pairing; ``tables(positions)`` -> (cos, sin) [1,1,n,64] bf16 TILE replicated (rows may carry different positions
    when ``positions`` is [rows, n]: then the tables are sharded over the mesh rows)."""

    def __init__(self, md, dim=64, theta=1e7, interleave=True):
        self.md, self.dim, self.theta, self.interleave = md, dim, theta, interleave
        self.P = upload(md, rotation_matrix(dim, interleave).reshape(1, 1, dim, dim))

    def tables(self, positions):
        pos = torch.as_tensor(positions)
        if pos.dim() == 1:
            c, s = rope_tables(pos, self.dim, self.theta, self.interleave)
            return upload(self.md, c.reshape(1, 1, -1, self.dim)), upload(self.md, s.reshape(1, 1, -1, self.dim))
        r = pos.shape[0]
        cs = [rope_tables(pos[i], self.dim, self.theta, self.interleave) for i in range(r)]
        c = torch.stack([x[0] for x in cs]).reshape(r, 1, -1, self.dim)
        s = torch.stack([x[1] for x in cs]).reshape(r, 1, -1, self.dim)
        return upload(self.md, c, row_dim=0), upload(self.md, s, row_dim=0)

    def apply(self, x, cos, sin, ckc=None):
        """x [1,H,n,64] -> x * cos + (x @ P) * sin (cos/sin [1,1,n,64] broadcast over heads)."""
        if self.interleave and x.shape[2] >= 256 and os.environ.get("HY4_B49_ROPE_LLAMA", "0") == "1":
            # B49: prefill chunks: ONE rotary_embedding_llama (adjacent-pair rotation) instead of matmul + 2 multiply + add (+ 3 deallocations)
            if getattr(self, "_tm", None) is None:
                from models.tt_transformers.tt.common import get_rot_transformation_mat

                self._tm = upload(self.md, get_rot_transformation_mat())
            return ttnn.experimental.rotary_embedding_llama(
                x, cos, sin, self._tm, is_decode_mode=False, compute_kernel_config=ckc
            )
        rot = ttnn.matmul(x, self.P, compute_kernel_config=ckc)
        a = ttnn.multiply(x, cos)
        b = ttnn.multiply(rot, sin)
        out = ttnn.add(a, b)
        for t in (rot, a, b):
            ttnn.deallocate(t)
        return out
