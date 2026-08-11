# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side helpers for the Muse-Glimmer functional-decoder tests.

Everything in here is an explicit *test boundary*: building device input tensors,
page tables and position tensors, and reading results back for PCC. The decoder's own
prefill/decode paths never call torch or ttnn.from_torch/to_torch (see
``test_no_runtime_host_fallback``).
"""

from __future__ import annotations

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import FunctionalDecoder


def replicate(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def to_device(tensor: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate(mesh_device),
    )


def from_device(tensor) -> torch.Tensor:
    return ttnn.to_torch(tensor).to(torch.float32)


def prefill_input(hidden: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16):
    """``[batch, seq, hidden]`` torch -> ``[batch, 1, seq, hidden]`` device tensor."""
    batch, seq, hidden_size = hidden.shape
    return to_device(hidden.reshape(batch, 1, seq, hidden_size), mesh_device, dtype=dtype)


def decode_input(hidden: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16):
    """``[batch, 1, hidden]`` torch -> ``[1, 1, batch, hidden]`` device tensor."""
    batch, one, hidden_size = hidden.shape
    assert one == 1
    return to_device(hidden.reshape(1, 1, batch, hidden_size), mesh_device, dtype=dtype)


def host_decode_input(hidden: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16):
    """Host-side twin of :func:`decode_input`, for ``ttnn.copy_host_to_device_tensor``."""
    batch, one, hidden_size = hidden.shape
    assert one == 1
    return ttnn.from_torch(
        hidden.reshape(1, 1, batch, hidden_size),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate(mesh_device),
    )


def prefill_output(tensor) -> torch.Tensor:
    """``[batch, 1, seq, hidden]`` device tensor -> ``[batch, seq, hidden]`` torch."""
    out = from_device(tensor)
    return out.reshape(out.shape[0], out.shape[2], out.shape[3])


def decode_output(tensor) -> torch.Tensor:
    """``[1, 1, batch, hidden]`` device tensor -> ``[batch, 1, hidden]`` torch."""
    out = from_device(tensor)
    return out.reshape(out.shape[2], 1, out.shape[3])


def build_page_table(
    *,
    batch: int,
    blocks_per_seq: int,
    total_blocks: int | None = None,
    seed: int | None = 1234,
) -> torch.Tensor:
    """Logical-to-physical block map, ``[batch, blocks_per_seq]`` int32.

    With ``seed`` set the physical blocks are a random permutation of a *larger* pool, so
    a page-table bug cannot hide behind identity addressing (the common
    ``physical == logical`` accident). ``seed=None`` gives the identity table.
    """
    needed = batch * blocks_per_seq
    total_blocks = total_blocks if total_blocks is not None else needed
    if total_blocks < needed:
        raise ValueError(f"total_blocks {total_blocks} < needed {needed}")
    if seed is None:
        order = torch.arange(needed, dtype=torch.int32)
    else:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(total_blocks, generator=generator)[:needed].to(torch.int32)
    return order.reshape(batch, blocks_per_seq)


def page_table_to_device(page_table: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        page_table.to(torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate(mesh_device),
    )


def position_tensors(positions, mesh_device, *, device=True):
    """``(current_pos int32 [batch], rope_idxs uint32 [1, batch])`` for decode.

    ``device=False`` keeps them on host so they can be copied into traced input tensors
    with ``ttnn.copy_host_to_device_tensor``.
    """
    positions = torch.tensor(list(positions), dtype=torch.int32)
    target = mesh_device if device else None
    current_pos = ttnn.from_torch(
        positions,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=target,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
        mesh_mapper=replicate(mesh_device),
    )
    rope_idxs = ttnn.from_torch(
        positions.to(torch.int32).reshape(1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=target,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
        mesh_mapper=replicate(mesh_device),
    )
    return current_pos, rope_idxs


def read_paged_cache(cache, page_table: torch.Tensor, *, block_size: int, seq_len: int) -> torch.Tensor:
    """Un-page a device K/V cache into ``[batch, kv_heads, seq_len, head_dim]`` (host, test-only)."""
    paged = from_device(cache)  # [num_blocks, kv_heads, block_size, head_dim]
    batch, blocks_per_seq = page_table.shape
    kv_heads, head_dim = paged.shape[1], paged.shape[3]
    out = torch.zeros(batch, kv_heads, blocks_per_seq * block_size, head_dim, dtype=paged.dtype)
    for b in range(batch):
        for logical in range(blocks_per_seq):
            physical = int(page_table[b, logical])
            out[b, :, logical * block_size : (logical + 1) * block_size] = paged[physical]
    return out[:, :, :seq_len]


def pcc(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Pearson correlation of the flattened pair (same definition as ``comp_pcc``)."""
    a = golden.reshape(-1).to(torch.float64)
    b = calculated.reshape(-1).to(torch.float64)
    if a.numel() != b.numel():
        raise ValueError(f"shape mismatch {golden.shape} vs {calculated.shape}")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def build_decoder(
    *,
    mesh_device,
    hf_config,
    layer_idx: int,
    state_dict,
    block_size: int = 64,
    prefill_chunk_size: int = 8192,
    rope_max_seq_len: int | None = None,
    cache_dtype=ttnn.bfloat16,
) -> FunctionalDecoder:
    return FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        block_size=block_size,
        prefill_chunk_size=prefill_chunk_size,
        rope_max_seq_len=rope_max_seq_len,
        cache_dtype=cache_dtype,
    )
