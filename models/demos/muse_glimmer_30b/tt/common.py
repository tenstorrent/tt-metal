# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers: compute configs, mesh mappers, as_tt, RoPE tables."""

import functools

import torch

import ttnn
from models.demos.muse_glimmer_30b.tt.model_config import LOW_PRECISION


def _cfg(mesh_device, fidelity, packer_l1_acc, fp32_acc=True):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def matmul_compute_config(mesh_device):
    if LOW_PRECISION:
        return _cfg(mesh_device, ttnn.MathFidelity.LoFi, True, fp32_acc=False)
    return _cfg(mesh_device, ttnn.MathFidelity.HiFi2, True)


def norm_compute_config(mesh_device):
    return _cfg(mesh_device, ttnn.MathFidelity.HiFi4, False)


def sdpa_compute_config(mesh_device):
    if LOW_PRECISION:
        return _cfg(mesh_device, ttnn.MathFidelity.HiFi2, True)
    return _cfg(mesh_device, ttnn.MathFidelity.HiFi4, False)


def decode_memcfg(width, mesh_device):
    """L1 width-sharded config for a [32, width] decode activation: row-major over the device grid,
    the same core order as decode matmul outputs."""
    g = mesh_device.compute_with_storage_grid_size()
    return _decode_memcfg(width, g.x, g.y)


@functools.lru_cache(maxsize=None)
def _decode_memcfg(width, gx, gy):
    tiles = width // 32
    n = max(c for c in range(1, 65) if tiles % c == 0)
    cores = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(gx, gy), row_wise=True)
    spec = ttnn.ShardSpec(cores, [32, width // n], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)


def is_decode(x):
    """True for a 4-D activation with 32 rows."""
    return len(x.shape) == 4 and x.shape[-2] == 32


def replicate_mapper(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def shard_mapper(mesh_device, dim):
    return ttnn.ShardTensorToMesh(mesh_device, dim=dim)


def as_tt(
    t: torch.Tensor,
    mesh_device,
    *,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    mesh_mapper=None,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cache_file_name=None,
):
    """Thin wrapper over ttnn.as_tensor (default: replicated, DRAM, optional weight cache)."""
    if mesh_mapper is None:
        mesh_mapper = replicate_mapper(mesh_device)
    return ttnn.as_tensor(
        t,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=mesh_mapper,
        memory_config=memory_config,
        cache_file_name=cache_file_name,
    )


def rope_cos_sin(head_dim: int, theta: float, max_seq_len: int):
    """HF-style (NeoX, cat(freqs, freqs)) cos/sin tables [max_seq_len, head_dim], bfloat16."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


class RopeTables:
    """Device RoPE tables for sliding layers (one set shared by all layers)."""

    def __init__(self, mesh_device, head_dim, theta, max_seq_len):
        self.mesh_device = mesh_device
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        cos, sin = rope_cos_sin(head_dim, theta, max_seq_len)
        kw = dict(dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # prefill: [1, 1, max_seq_len, head_dim] TILE
        self.cos_4d = as_tt(cos[None, None], mesh_device, layout=ttnn.TILE_LAYOUT, **kw)
        self.sin_4d = as_tt(sin[None, None], mesh_device, layout=ttnn.TILE_LAYOUT, **kw)
        # decode: [max_seq_len, head_dim] ROW_MAJOR (weight of ttnn.embedding)
        self.cos_2d = as_tt(cos, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
        self.sin_2d = as_tt(sin, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)

    def prefill(self, start: int, seq_len: int):
        """cos/sin [1, 1, seq_len, head_dim] for positions start..start+seq_len."""
        end = start + seq_len
        cos = self.cos_4d[:, :, start:end, :]
        sin = self.sin_4d[:, :, start:end, :]
        return cos, sin

    def decode(self, rope_idx_tt, memory_config=None):
        """Per-user cos/sin [1, B_pad, 1, head_dim] TILE from uint32 [1, B_pad] positions (trace-safe)."""
        outs = []
        for table in (self.cos_2d, self.sin_2d):
            g = ttnn.embedding(rope_idx_tt, table, layout=ttnn.TILE_LAYOUT)  # [1, B, hd]
            g = ttnn.unsqueeze_to_4D(g)  # [1, 1, B, hd]
            t = ttnn.transpose(g, 1, 2)  # [1, B, 1, hd]
            if memory_config is not None:
                t_i, t = t, ttnn.to_memory_config(t, memory_config)
                ttnn.deallocate(t_i)
            outs.append(t)
        return outs[0], outs[1]
