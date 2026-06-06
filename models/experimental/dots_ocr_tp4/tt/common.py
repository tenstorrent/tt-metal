# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the from-scratch dots.ocr TP4 prefill rebuild.

Design: standard Megatron tensor-parallelism with a *replicated* hidden
stream. The hidden state is the full ``hidden_size`` on every chip between
sublayers, so RMSNorm is a local (per-chip) exact op and the only collectives
are the two all-reduces after ``o_proj`` and ``down_proj``.
"""

from dataclasses import dataclass

import torch
import ttnn


@dataclass
class DotsOCRConfig:
    """Subset of the dots.ocr text-decoder config needed for prefill."""

    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = True
    max_position_embeddings: int = 131072

    @classmethod
    def from_hf(cls, hf_config):
        head_dim = getattr(hf_config, "head_dim", None) or (hf_config.hidden_size // hf_config.num_attention_heads)
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:
            rp = getattr(hf_config, "rope_parameters", {}) or {}
            rope_theta = rp.get("rope_theta", 1000000.0)
        return cls(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_theta=float(rope_theta),
            attention_bias=getattr(hf_config, "attention_bias", True),
            max_position_embeddings=getattr(hf_config, "max_position_embeddings", 131072),
        )

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


def mesh_num_devices(mesh_device) -> int:
    if mesh_device is None or not hasattr(mesh_device, "get_num_devices"):
        return 1
    return int(mesh_device.get_num_devices())


def tp_cluster_axis(mesh_device) -> int:
    """Cluster axis along which the TP shards live (the >1 dim of a 1xN/Nx1 mesh)."""
    shape = [int(x) for x in mesh_device.shape]
    # For a (1, N) mesh the device dim is axis 1; for (N, 1) it is axis 0.
    return 0 if (len(shape) == 2 and shape[0] > 1 and shape[1] == 1) else 1


def all_reduce(t: ttnn.Tensor, mesh_device, num_links: int = 1) -> ttnn.Tensor:
    """All-reduce a [.., N] partial-sum tensor along its last dim across the TP axis.

    Implemented as reduce_scatter + all_gather (the proven 1x4-ring pattern on
    this host). Input/output last dim N must be divisible by num_devices and
    tile-aligned. Returns a replicated full-N tensor.
    """
    nd = mesh_num_devices(mesh_device)
    if nd <= 1:
        return t
    orig_shape = list(t.shape)
    work = t
    if len(work.shape) < 4:
        work = ttnn.reshape(work, [1] * (4 - len(work.shape)) + orig_shape)
    axis = tp_cluster_axis(mesh_device)
    scattered = ttnn.reduce_scatter(
        work,
        dim=3,
        num_links=num_links,
        cluster_axis=axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
    gathered = ttnn.all_gather(
        scattered,
        dim=3,
        num_links=num_links,
        cluster_axis=axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
    ttnn.deallocate(scattered)
    if len(orig_shape) < 4:
        gathered = ttnn.reshape(gathered, orig_shape[:-1] + [int(gathered.shape[-1])])
    return gathered


def all_gather_last_dim(t: ttnn.Tensor, mesh_device, num_links: int = 1) -> ttnn.Tensor:
    """All-gather a [.., N/ndev] column-sharded tensor along its last dim across
    the TP axis -> full replicated [.., N]. Used by the column-parallel LM head."""
    nd = mesh_num_devices(mesh_device)
    if nd <= 1:
        return t
    orig_shape = list(t.shape)
    work = t
    if len(work.shape) < 4:
        work = ttnn.reshape(work, [1] * (4 - len(work.shape)) + orig_shape)
    axis = tp_cluster_axis(mesh_device)
    gathered = ttnn.all_gather(
        work,
        dim=3,
        num_links=num_links,
        cluster_axis=axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
    if len(orig_shape) < 4:
        gathered = ttnn.reshape(gathered, orig_shape[:-1] + [int(gathered.shape[-1])])
    return gathered


def to_replicated(torch_tensor: torch.Tensor, mesh_device, dtype, layout=ttnn.TILE_LAYOUT):
    """Send a torch tensor to device replicated on every chip."""
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_num_devices(mesh_device) > 1 else None
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def shard_to_mesh(torch_tensor: torch.Tensor, mesh_device, dim: int, dtype, layout=ttnn.TILE_LAYOUT):
    """Send a torch tensor to device sharded along ``dim`` across the chips."""
    if mesh_num_devices(mesh_device) <= 1:
        return ttnn.from_torch(
            torch_tensor, dtype=dtype, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return ttnn.from_torch(
        torch_tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=dim),
    )


def from_replicated_to_torch(t: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Read back a replicated device tensor as a single torch tensor (chip 0)."""
    if mesh_num_devices(mesh_device) <= 1:
        return ttnn.to_torch(t)
    full = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # Replicated: every chip identical. Take the first replica slice.
    return full[: int(t.shape[0])]
