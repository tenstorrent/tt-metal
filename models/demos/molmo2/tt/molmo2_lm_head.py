# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Split language-model head for Molmo2: large vocab × hidden as several smaller
``ttnn.linear`` calls over column chunks, then full vocab along the last dim.

**Single device or replicated mesh (default)**
Weights use ``ReplicateTensorToMesh``; each device runs the same matmuls (redundant
on mesh, but matches the original single-tensor LM head).

**Mesh column-parallel (optional, T3K / multi-device)**
When ``mesh_column_parallel=True`` and ``num_devices > 1``, each chunk weight is
laid out like tt_transformers LMHead1D: per-device column slices are concatenated
on the host and uploaded with ``ShardTensorToMesh(..., dim=3)`` so each chip holds
a disjoint slice of the vocab for that chunk. Activations stay **replicated**.

After each chunk linear, partial logits are composed across the mesh with
``ttnn.to_torch(..., mesh_composer=concat_mesh_to_tensor_composer(dim=3))``,
then chunk results are ``torch.cat`` on CPU and sent back with
``ttnn.from_torch(..., ReplicateTensorToMesh)``. That gather has host overhead but
avoids custom CCL and matches correctness tests on T3K.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch

import ttnn

# 152064 / 6 = 25344 (tile-aligned); keeps each matmul N smaller than full vocab.
DEFAULT_LM_HEAD_MAX_CHUNK_N = 25344


def _padded_vocab_for_mesh(vocab: int, num_devices: int, tile: int = 32) -> int:
    step = num_devices * tile
    return math.ceil(vocab / step) * step


def _align_max_chunk_n_for_mesh(max_chunk_n: int, num_devices: int, tile: int = 32) -> int:
    """Each chunk width must be divisible by ``num_devices * tile`` for column sharding."""
    step = num_devices * tile
    aligned = (max_chunk_n // step) * step
    return max(step, aligned)


def vocab_column_split_sizes(vocab_size: int, max_chunk_n: int) -> List[int]:
    """Split vocab columns so each chunk has at most ``max_chunk_n``."""
    sizes: List[int] = []
    start = 0
    while start < vocab_size:
        end = min(start + max_chunk_n, vocab_size)
        sizes.append(end - start)
        start = end
    return sizes


def build_lm_head_chunks(
    lm_head_weight: torch.Tensor,
    mesh_device,
    mesh_mapper,
    dtype,
    vocab_size: int,
    cache_name: Optional[Callable[[str], object]],
    max_chunk_n: int = DEFAULT_LM_HEAD_MAX_CHUNK_N,
    mesh_column_parallel: bool = False,
) -> Tuple[List[ttnn.Tensor], List[int]]:
    """
    Split ``lm_head.weight`` [vocab, hidden] along vocab into TTNN tensors
    ``[1, 1, hidden, n_chunk]``.

    ``vocab_size`` is the model config size; checkpoints may store fewer rows
    (e.g. 151936 vs 152064). We use ``min(vocab_size, weight_vocab)`` then pad
    for mesh column-parallel when requested.
    """
    if mesh_column_parallel:
        num_devices = mesh_device.get_num_devices()
        if num_devices <= 1:
            mesh_column_parallel = False
    else:
        num_devices = 1

    w_full = torch.transpose(lm_head_weight, -2, -1).contiguous()
    vocab_in_weights = w_full.shape[1]
    effective_vocab = min(vocab_size, vocab_in_weights)
    w_full = w_full[:, :effective_vocab]

    if mesh_column_parallel:
        chunk_n = _align_max_chunk_n_for_mesh(max_chunk_n, num_devices)
        padded_vocab = _padded_vocab_for_mesh(effective_vocab, num_devices)
        nchunks = math.ceil(padded_vocab / chunk_n)
        padded_vocab = nchunks * chunk_n
        if padded_vocab > w_full.shape[1]:
            w_full = torch.nn.functional.pad(w_full, (0, padded_vocab - w_full.shape[1]))
        # Equal chunk widths so each is divisible by num_devices (required for sharding).
        split_sizes = [chunk_n] * nchunks
        shard_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
        chunks: List[ttnn.Tensor] = []
        col = 0
        for i, n_cols in enumerate(split_sizes):
            assert n_cols % num_devices == 0, f"chunk width {n_cols} not divisible by num_devices {num_devices}"
            subw = n_cols // num_devices
            device_splits = []
            for d in range(num_devices):
                cs = col + d * subw
                device_splits.append(w_full[:, cs : cs + subw])
            combined = torch.cat(device_splits, dim=-1)
            col += n_cols
            t = combined.unsqueeze(0).unsqueeze(0)
            name = f"lm_head.weight.chunk_{i}_n{n_cols}_meshshard"
            chunks.append(
                ttnn.as_tensor(
                    t,
                    dtype=dtype,
                    device=mesh_device,
                    mesh_mapper=shard_mapper,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name(name) if cache_name else None,
                )
            )
        return chunks, split_sizes

    # Replicated weights (single device or full mesh copy)
    split_sizes = vocab_column_split_sizes(effective_vocab, max_chunk_n)
    chunks = []
    col = 0
    for i, n_cols in enumerate(split_sizes):
        piece = w_full[:, col : col + n_cols]
        col += n_cols
        t = piece.unsqueeze(0).unsqueeze(0)
        name = f"lm_head.weight.chunk_{i}_n{n_cols}"
        chunks.append(
            ttnn.as_tensor(
                t,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(name) if cache_name else None,
            )
        )
    return chunks, split_sizes


def forward_split_lm_head(
    x: ttnn.Tensor,
    lm_head_chunks: Sequence[ttnn.Tensor],
    compute_kernel_config: ttnn.WormholeComputeKernelConfig,
    output_memory_config: Optional[object] = None,
    *,
    mesh_device=None,
    mesh_column_parallel: bool = False,
    logits_output_width: Optional[int] = None,
) -> ttnn.Tensor:
    """Run split LM head linears; on mesh column-parallel, gather logits via host compositor.

    ``logits_output_width``: If set, slice the last dim to this size. Mesh column-parallel
    pads vocab for alignment; must match ``min(config_vocab, lm_head.rows)`` for PCC tests.
    """
    if output_memory_config is None:
        output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    if mesh_column_parallel and mesh_device is not None and mesh_device.get_num_devices() > 1:
        composer = ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=3)
        cpu_parts: List[torch.Tensor] = []
        for w in lm_head_chunks:
            out = ttnn.linear(
                x,
                w,
                compute_kernel_config=compute_kernel_config,
                memory_config=output_memory_config,
            )
            cpu_parts.append(ttnn.to_torch(out, mesh_composer=composer))
            ttnn.deallocate(out)

        full_cpu = torch.cat(cpu_parts, dim=-1)
        if logits_output_width is not None and full_cpu.shape[-1] > logits_output_width:
            full_cpu = full_cpu[..., :logits_output_width]
        if full_cpu.dtype != torch.bfloat16:
            full_cpu = full_cpu.to(torch.bfloat16)
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        return ttnn.from_torch(
            full_cpu,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=output_memory_config,
            mesh_mapper=mesh_mapper,
        )

    parts: List[ttnn.Tensor] = []
    for w in lm_head_chunks:
        out = ttnn.linear(
            x,
            w,
            compute_kernel_config=compute_kernel_config,
            memory_config=output_memory_config,
        )
        parts.append(out)

    if len(parts) == 1:
        logits = parts[0]
    else:
        logits = ttnn.concat(parts, dim=-1, memory_config=output_memory_config)
        for p in parts:
            ttnn.deallocate(p)

    if logits_output_width is not None and logits.shape[-1] > logits_output_width:
        logits = ttnn.slice(
            logits,
            (0, 0, 0, 0),
            (1, 1, logits.shape[2], logits_output_width),
        )
    return logits
