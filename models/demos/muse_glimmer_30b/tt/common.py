# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers: compute configs, mesh mappers, as_tt, RoPE tables."""

import torch

import ttnn


def _cfg(mesh_device, fidelity, packer_l1_acc):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def matmul_compute_config(mesh_device):
    return _cfg(mesh_device, ttnn.MathFidelity.HiFi2, True)


def norm_compute_config(mesh_device):
    return _cfg(mesh_device, ttnn.MathFidelity.HiFi4, False)


def sdpa_compute_config(mesh_device):
    return _cfg(mesh_device, ttnn.MathFidelity.HiFi4, False)


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

    def decode(self, rope_idx_tt):
        """Per-user cos/sin [1, B_pad, 1, head_dim] TILE from uint32 [1, B_pad] positions (trace-safe)."""
        outs = []
        for table in (self.cos_2d, self.sin_2d):
            g = ttnn.embedding(rope_idx_tt, table, layout=ttnn.TILE_LAYOUT)  # [1, B, hd]
            g = ttnn.unsqueeze_to_4D(g)  # [1, 1, B, hd]
            outs.append(ttnn.transpose(g, 1, 2))  # [1, B, 1, hd]
        return outs[0], outs[1]
