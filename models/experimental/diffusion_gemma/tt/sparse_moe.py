# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma-local TRUE-SPARSE **prefill** MoE: ragged, zero-drop top-8.

The dense prefill MoE (gemma4 expert path) computes ALL 128 experts for every 32-token tile-group
and then zeros 120/128 via the routing weights. Only the top-8 experts per token are active, so
~16x of that compute is wasted.

This module replaces it, for prefill, with a **ragged** dispatch: pack each expert's routed tokens
into compact expert-homogeneous tile groups (zero-drop — there is no capacity to overflow), gather
the rows with ``ttnn.embedding``, run gate/up/down as ``ttnn.sparse_matmul`` per group, then scatter
back weighted by the routing weights and all-reduce. Bit-identical to the shared 128-expert path,
and chunked along the token dim so it scales past the DRAM / int32-index limits of a single
full-S call.

Why this is legal on the (1,4) TP mesh: the input is REPLICATED across TP (experts are TP-sharded
on the intermediate dim, NOT expert-parallel), so gather/scatter over the token dim is LOCAL per
device. Only the down-projection needs the existing TP all-reduce. No cross-device token dispatch
(unlike deepseek's all_to_all, which needs a 16-row ring).

**This module is prefill-only.** The denoise MoE lives entirely in tt/concat_moe.py; nothing in
this file is on the denoise path.

NEVER edits gemma4 — composes over ``moe.router`` and ``moe.experts.weights`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import ttnn
from models.experimental.diffusion_gemma.tt.ccl import ccl_allreduce
from models.experimental.diffusion_gemma.tt.expert_operations import apply_geglu

TILE = 32


RAGGED_MAX_M_BLOCKS = 4

# Default token-dim chunk length for long-prompt ragged prefill (see
# ``chunked_ragged_sparse_prefill_forward``). Matches the single-call ceiling.
RAGGED_PREFILL_CHUNK = 4096

# Segment-count ladder for the ragged groups. WHY THIS EXISTS: each group is handed to
# ``ttnn.sparse_matmul`` as [1, group_size, m_blocks*TILE, H] with nnz=group_size, and
# ``group_size`` is the number of expert-segments that happen to have that m_blocks -- i.e. it is
# ROUTING-dependent. A new prompt routes differently, so without the ladder every sparse_matmul
# geometry misses the program cache and gets compiled on the host. Rounding group_size onto this
# ladder collapses the shape space to at most len(_GROUP_LADDER) x RAGGED_MAX_M_BLOCKS programs,
# all reused across prompts and warmable at startup.
#
# The ladder is finer than powers of two on purpose: padded segments cost real device work (their
# rows are zeroed by slot_valid, so they compute 0 x W = 0), so the ceiling on waste matters.
# Powers of two waste up to 100%; this wastes at most 50% below 8 and at most 33% above it.
_GROUP_LADDER = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)


def _ladder_group_size(group_size: int) -> int:
    """Round a segment count up onto ``_GROUP_LADDER`` (next power of two beyond it)."""
    for step in _GROUP_LADDER:
        if group_size <= step:
            return step
    size = _GROUP_LADDER[-1]
    while size < group_size:
        size *= 2
    return size


def _ragged_ladder_enabled() -> bool:
    return os.environ.get("DG_PREFILL_RAGGED_LADDER", "1") == "1"


def _quantize_ragged_groups(groups, token_slot, packed_rows):
    """Pad each group's segment count onto the ladder, keeping the output identical.

    A padded segment gets ``slot_token=0`` and ``slot_valid=0``, so its gathered input rows are
    exactly zero and its expert output is exactly zero; ``token_slot`` never points at them, so
    nothing downstream reads them. Real segments keep their row order inside the group, and
    ``sparse_matmul`` is batched per segment with no cross-segment reduction, so every real row is
    bit-identical to the unpadded call.

    ``sparsity`` must keep exactly one non-zero per group row: ``nnz`` is passed as the (now padded)
    ``group_size``, and ttnn documents that ``nnz != count_nonzero(sparsity)`` HANGS the kernel
    (the receiver loops nnz times while the sender multicasts once per non-zero). Padded rows are
    therefore pointed at expert 0 rather than left all-zero.

    ``token_slot`` holds absolute row indices into the concatenated groups, so padding a group
    shifts every later group's base and the indices must be rebased.
    """
    import torch  # module-local, matching _ragged_metadata_host: no global torch import here

    if not groups:
        return groups, token_slot, packed_rows

    quantized = []
    rebase = []
    old_offset = 0
    new_offset = 0
    for m_blocks, group_size, slot_token, slot_valid, sparsity in groups:
        rows_per_segment = m_blocks * TILE
        old_rows = group_size * rows_per_segment
        padded_size = _ladder_group_size(group_size)
        pad_rows = (padded_size - group_size) * rows_per_segment
        if pad_rows:
            slot_token = torch.cat([slot_token.reshape(-1), torch.zeros(pad_rows, dtype=slot_token.dtype)])
            slot_valid = torch.cat([slot_valid.reshape(-1, 1), torch.zeros((pad_rows, 1), dtype=slot_valid.dtype)])
            padded_sparsity = torch.zeros((1, 1, padded_size, sparsity.shape[-1]), dtype=sparsity.dtype)
            padded_sparsity[0, 0, :group_size] = sparsity[0, 0]
            padded_sparsity[0, 0, group_size:, 0] = 1
            sparsity = padded_sparsity
        rebase.append((old_offset, old_rows, new_offset))
        quantized.append((m_blocks, padded_size, slot_token, slot_valid, sparsity))
        old_offset += old_rows
        new_offset += padded_size * rows_per_segment

    if new_offset != old_offset:
        rebased = token_slot.clone()
        for group_old, old_rows, group_new in rebase:
            if group_new == group_old:
                continue
            in_group = (token_slot >= group_old) & (token_slot < group_old + old_rows)
            rebased[in_group] = token_slot[in_group] + (group_new - group_old)
        token_slot = rebased

    return quantized, token_slot, new_offset


@dataclass
class RaggedRouting:
    values: object
    indices: object
    per_expert_scale: object | None


# Host-side cache of the router per-expert scale, keyed by device tensor id.
#
# The value keeps a STRONG REFERENCE to the device tensor it was read from; that pin is
# what makes ``id()`` a sound key (``id()`` is unique only among LIVE objects, so caching
# the host scale alone would silently serve a stale scale after address reuse). The pin
# is cheap: the tensor is a ``[num_experts]`` router weight that lives for the model's
# lifetime.
_ROUTER_SCALE_HOST_CACHE = {}


def ragged_router_forward(router, hidden_states):
    """Router forward that retains compact top-k metadata instead of scattering dense."""
    normed = router.norm.forward(hidden_states)
    scaled = ttnn.mul(normed, router.scale)
    normed.deallocate(True)
    scaled = ttnn.mul(scaled, router.scalar_root_size)
    expert_scores = ttnn.linear(scaled, router.proj_weight)
    scaled.deallocate(True)
    router_probs = ttnn.softmax(expert_scores, dim=-1)
    expert_scores.deallocate(True)
    top_k_values, top_k_indices = ttnn.topk(router_probs, k=router.top_k, dim=-1)
    router_probs.deallocate(True)
    top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
    normalized_values = ttnn.div(top_k_values, top_k_sum)
    top_k_values.deallocate(True)
    top_k_sum.deallocate(True)
    return RaggedRouting(normalized_values, top_k_indices, router.per_expert_scale)


try:
    import numba as _numba
except ImportError:  # pragma: no cover - exercised in minimal runtime environments
    _numba = None


if _numba is not None:
    import numpy as np

    @_numba.njit(cache=True)
    def _pack_ragged_assignments(expert_index, num_experts, max_m_blocks):
        sequence_length, top_k = expert_index.shape
        capacity_rows = max_m_blocks * TILE
        max_segments = (sequence_length + capacity_rows - 1) // capacity_rows
        counts = np.zeros(num_experts, np.int32)
        for token in range(sequence_length):
            for k_index in range(top_k):
                counts[expert_index[token, k_index]] += 1

        segment_m_blocks = np.zeros((num_experts, max_segments), np.int32)
        group_counts = np.zeros(max_m_blocks, np.int32)
        for expert in range(num_experts):
            num_segments = (counts[expert] + capacity_rows - 1) // capacity_rows
            for segment in range(num_segments):
                count = min(capacity_rows, counts[expert] - segment * capacity_rows)
                m_blocks = (count + TILE - 1) // TILE
                segment_m_blocks[expert, segment] = m_blocks
                group_counts[m_blocks - 1] += 1

        group_start = np.zeros(max_m_blocks, np.int32)
        total_rows = 0
        for m_blocks in range(1, max_m_blocks + 1):
            group_start[m_blocks - 1] = total_rows
            total_rows += group_counts[m_blocks - 1] * m_blocks * TILE

        group_experts = np.full((max_m_blocks, num_experts * max_segments), -1, np.int32)
        segment_local = np.zeros((num_experts, max_segments), np.int32)
        local_counts = np.zeros(max_m_blocks, np.int32)
        for expert in range(num_experts):
            for segment in range(max_segments):
                m_blocks = segment_m_blocks[expert, segment]
                if m_blocks != 0:
                    local = local_counts[m_blocks - 1]
                    local_counts[m_blocks - 1] += 1
                    segment_local[expert, segment] = local
                    group_experts[m_blocks - 1, local] = expert

        slot_token = np.zeros(total_rows, np.int32)
        slot_valid_bits = np.zeros(total_rows, np.uint16)
        token_slot = np.empty((sequence_length, top_k), np.int32)
        expert_rank = np.zeros(num_experts, np.int32)
        for token in range(sequence_length):
            for k_index in range(top_k):
                expert = expert_index[token, k_index]
                rank = expert_rank[expert]
                expert_rank[expert] += 1
                segment = rank // capacity_rows
                row = rank % capacity_rows
                m_blocks = segment_m_blocks[expert, segment]
                packed_row = group_start[m_blocks - 1] + segment_local[expert, segment] * m_blocks * TILE + row
                slot_token[packed_row] = token
                slot_valid_bits[packed_row] = 0x3F80  # BF16 1.0
                token_slot[token, k_index] = packed_row
        return slot_token, slot_valid_bits, token_slot, group_counts, group_experts, group_start

else:
    _pack_ragged_assignments = None


def _ragged_prefill_program_config(m_blocks, output_width):
    if output_width == 192:
        grid_x, grid_y, block_w, per_core_n = 6, 1, 44, 1
    elif output_width == 2816:
        grid_x, grid_y, block_w, per_core_n = 11, 4, 3, 2
    else:
        raise ValueError(f"unsupported ragged prefill output width: {output_width}")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=m_blocks,
        out_block_w=per_core_n,
        per_core_M=m_blocks,
        per_core_N=per_core_n,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _ragged_metadata_host(dense_routing, num_experts, top_k, max_m_blocks=RAGGED_MAX_M_BLOCKS):
    """Pack routed assignments into zero-drop, expert-homogeneous tile groups.

    This CPU metadata builder is intentionally vectorized; a per-assignment
    Python loop is orders of magnitude too slow for long prompts.
    """
    import torch

    if isinstance(dense_routing, RaggedRouting):
        values = dense_routing.values
        indices = dense_routing.indices
        if values.device().get_num_devices() > 1:
            route_weight = ttnn.to_torch(ttnn.get_device_tensors(values)[0])[0, 0]
            expert_index = ttnn.to_torch(ttnn.get_device_tensors(indices)[0])[0, 0].long()
        else:
            route_weight = ttnn.to_torch(values)[0, 0]
            expert_index = ttnn.to_torch(indices)[0, 0].long()
        values.deallocate(True)
        indices.deallocate(True)
        S = route_weight.shape[0]
        per_token_order = torch.argsort(expert_index, dim=-1)
        expert_index = torch.gather(expert_index, -1, per_token_order)
        route_weight = torch.gather(route_weight, -1, per_token_order)
        if dense_routing.per_expert_scale is not None:
            scale_tensor = dense_routing.per_expert_scale
            cached = _ROUTER_SCALE_HOST_CACHE.get(id(scale_tensor))
            # ``cached[0] is scale_tensor`` rejects an entry whose tensor died and whose
            # address was recycled; without it a stale scale is served silently.
            if cached is not None and cached[0] is scale_tensor:
                scale = cached[1]
            else:
                if scale_tensor.device().get_num_devices() > 1:
                    scale = ttnn.to_torch(ttnn.get_device_tensors(scale_tensor)[0]).reshape(-1)
                else:
                    scale = ttnn.to_torch(scale_tensor).reshape(-1)
                _ROUTER_SCALE_HOST_CACHE[id(scale_tensor)] = (scale_tensor, scale)
            route_weight = route_weight * scale[expert_index]
    else:
        if dense_routing.device().get_num_devices() > 1:
            routing = ttnn.to_torch(ttnn.get_device_tensors(dense_routing)[0])[0, 0]
        else:
            routing = ttnn.to_torch(dense_routing)[0, 0]
        S = routing.shape[0]
        active_mask = routing != 0
        active_entries = torch.nonzero(active_mask)
        if active_entries.shape[0] == S * top_k:
            # The router contract is exactly top_k nonzero entries per token.
            # nonzero() is row-major, so expert ids are already in the reduction
            # order that matches the dense path.
            expert_index = active_entries[:, -1].reshape(S, top_k)
            route_weight = routing[active_mask].reshape(S, top_k)
        else:
            # Defensive fallback for a future router that can emit exact zero for
            # an active slot or otherwise violates the fixed-top-k contract.
            route_weight, expert_index = torch.topk(routing, top_k, dim=-1)
            per_token_order = torch.argsort(expert_index, dim=-1)
            expert_index = torch.gather(expert_index, -1, per_token_order)
            route_weight = torch.gather(route_weight, -1, per_token_order)

    capacity_rows = max_m_blocks * TILE
    max_segments_per_expert = (S + capacity_rows - 1) // capacity_rows

    if _pack_ragged_assignments is not None:
        (
            slot_token_np,
            slot_valid_bits_np,
            token_slot_np,
            group_counts_np,
            group_experts_np,
            group_start_np,
        ) = _pack_ragged_assignments(
            expert_index.contiguous().numpy(),
            num_experts,
            max_m_blocks,
        )
        groups = []
        for m_blocks in range(1, max_m_blocks + 1):
            group_size = int(group_counts_np[m_blocks - 1])
            if group_size == 0:
                continue
            start = int(group_start_np[m_blocks - 1])
            total_rows = group_size * m_blocks * TILE
            slot_token = torch.from_numpy(slot_token_np[start : start + total_rows].copy())
            slot_valid_bits = torch.from_numpy(slot_valid_bits_np[start : start + total_rows].copy())
            slot_valid = slot_valid_bits.view(torch.bfloat16).reshape(total_rows, 1)
            group_experts = torch.from_numpy(group_experts_np[m_blocks - 1, :group_size].copy()).long()
            sparsity = torch.zeros((1, 1, group_size, num_experts), dtype=torch.bfloat16)
            sparsity[0, 0, torch.arange(group_size), group_experts] = 1
            groups.append((m_blocks, group_size, slot_token, slot_valid, sparsity))
        token_slot = torch.from_numpy(token_slot_np.copy())
        packed_rows = len(slot_token_np)
        if _ragged_ladder_enabled():
            groups, token_slot, packed_rows = _quantize_ragged_groups(groups, token_slot, packed_rows)
        return (
            groups,
            token_slot,
            route_weight.reshape(S, top_k, 1),
            packed_rows,
        )

    flat_expert = expert_index.reshape(-1)
    flat_token = torch.arange(S).repeat_interleave(top_k)
    flat_k = torch.arange(top_k).repeat(S)
    assignment_order = torch.argsort(flat_expert, stable=True)
    sorted_expert = flat_expert[assignment_order]
    sorted_token = flat_token[assignment_order]
    sorted_k = flat_k[assignment_order]

    expert_counts = torch.bincount(sorted_expert, minlength=num_experts)
    expert_starts = torch.cumsum(expert_counts, 0) - expert_counts
    rank_in_expert = torch.arange(S * top_k) - expert_starts[sorted_expert]
    segment_key = sorted_expert * max_segments_per_expert + rank_in_expert // capacity_rows
    segment_keys, segment_counts = torch.unique_consecutive(segment_key, return_counts=True)
    assignment_segment = torch.repeat_interleave(torch.arange(len(segment_keys)), segment_counts)
    row_in_segment = rank_in_expert % capacity_rows
    segment_m_blocks = (segment_counts + TILE - 1) // TILE

    token_slot = torch.empty((S, top_k), dtype=torch.int32)
    groups = []
    output_offset = 0
    for m_blocks in range(1, max_m_blocks + 1):
        segment_ids = torch.nonzero(segment_m_blocks == m_blocks).flatten()
        if len(segment_ids) == 0:
            continue
        group_size = len(segment_ids)
        rows_per_segment = m_blocks * TILE
        total_rows = group_size * rows_per_segment
        segment_to_group = torch.full((len(segment_keys),), -1, dtype=torch.int64)
        segment_to_group[segment_ids] = torch.arange(group_size)
        assignment_mask = segment_to_group[assignment_segment] >= 0
        packed_row = (
            segment_to_group[assignment_segment[assignment_mask]] * rows_per_segment + row_in_segment[assignment_mask]
        )

        slot_token = torch.zeros(total_rows, dtype=torch.int32)
        slot_valid = torch.zeros((total_rows, 1), dtype=torch.bfloat16)
        slot_token[packed_row] = sorted_token[assignment_mask].to(torch.int32)
        slot_valid[packed_row] = 1
        token_slot[sorted_token[assignment_mask], sorted_k[assignment_mask]] = output_offset + packed_row.to(
            torch.int32
        )

        group_experts = segment_keys[segment_ids] // max_segments_per_expert
        sparsity = torch.zeros((1, 1, group_size, num_experts), dtype=torch.bfloat16)
        sparsity[0, 0, torch.arange(group_size), group_experts] = 1
        groups.append((m_blocks, group_size, slot_token, slot_valid, sparsity))
        output_offset += total_rows

    if _ragged_ladder_enabled():
        groups, token_slot, output_offset = _quantize_ragged_groups(groups, token_slot, output_offset)
    return groups, token_slot, route_weight.reshape(S, top_k, 1), output_offset


def ragged_sparse_prefill_forward(
    hidden_states,
    routing_weights,
    weights,
    config,
    prefill_sparsity,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
):
    """Zero-drop sparse prefill with compact ragged expert batches.

    Bit-identical to the shared 128-expert path (logits + KV cache), at a small
    fraction of its prefill latency.
    """
    del prefill_sparsity
    mesh = mesh_device or hidden_states.device()
    S = hidden_states.shape[2]
    E = config.num_experts
    H = config.hidden_size
    I = weights.intermediate_size_per_device
    groups, token_slot_host, route_weight_host, packed_rows = _ragged_metadata_host(routing_weights, E, config.top_k)
    mapper = ttnn.ReplicateTensorToMesh(mesh) if hasattr(mesh, "shape") else None

    def upload(host_tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            host_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh,
            mesh_mapper=mapper,
        )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    hidden_flat = ttnn.reshape(hidden_states, (S, H))
    down_groups = []
    for m_blocks, group_size, slot_token_host, slot_valid_host, sparsity_host in groups:
        group_rows = group_size * m_blocks * TILE
        slot_token = upload(slot_token_host.reshape(1, group_rows), ttnn.uint32)
        slot_valid = upload(slot_valid_host.reshape(1, group_rows, 1), ttnn.bfloat16)
        sparsity = upload(sparsity_host, ttnn.bfloat16)

        gathered = ttnn.embedding(
            slot_token,
            hidden_flat,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered_valid = ttnn.mul(gathered, slot_valid)
        grouped_input = ttnn.reshape(gathered_valid, (1, group_size, m_blocks * TILE, H))
        gate_output = ttnn.empty(
            [1, group_size, m_blocks * TILE, I],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        up_output = ttnn.empty(
            [1, group_size, m_blocks * TILE, I],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        common = {
            "sparsity": sparsity,
            "nnz": group_size,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "compute_kernel_config": compute_kernel_config,
            "dtype": ttnn.bfloat16,
        }
        gate = ttnn.sparse_matmul(
            grouped_input,
            weights.gate_proj,
            program_config=_ragged_prefill_program_config(m_blocks, I),
            optional_output_tensor=gate_output,
            **common,
        )
        up = ttnn.sparse_matmul(
            grouped_input,
            weights.up_proj,
            program_config=_ragged_prefill_program_config(m_blocks, I),
            optional_output_tensor=up_output,
            **common,
        )
        down_input = apply_geglu(gate, up)
        down_output = ttnn.empty(
            [1, group_size, m_blocks * TILE, H],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
        )
        down = ttnn.sparse_matmul(
            down_input,
            weights.down_proj,
            program_config=_ragged_prefill_program_config(m_blocks, H),
            optional_output_tensor=down_output,
            **common,
        )
        down_groups.append(ttnn.reshape(down, (group_rows, H)))

        for tensor in (
            slot_token,
            slot_valid,
            sparsity,
            gathered,
            gathered_valid,
            gate,
            up,
            down_input,
        ):
            tensor.deallocate(True)

    packed_down = down_groups[0] if len(down_groups) == 1 else ttnn.concat(down_groups, dim=0)
    if len(down_groups) > 1:
        for tensor in down_groups:
            tensor.deallocate(True)
    assert packed_down.shape[0] == packed_rows

    # Embedding accepts a 2-D index matrix. Store top-k in the leading
    # dimension so fast_reduce_nc can consume it directly without a device
    # permute of the large [S, K, H] selected-expert tensor.
    token_slot = upload(token_slot_host.transpose(0, 1).contiguous(), ttnn.uint32)
    route_weight_transposed = route_weight_host.transpose(0, 1).contiguous()
    route_weight = upload(route_weight_transposed, ttnn.bfloat16)
    selected = ttnn.embedding(
        token_slot,
        packed_down,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weighted = ttnn.mul(selected, route_weight)
    weighted = ttnn.reshape(weighted, (1, config.top_k, S, H))
    out = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(weighted, dims=[1]))
    out = ttnn.reshape(out, (1, 1, S, H))

    for tensor in (packed_down, token_slot, route_weight, selected, weighted):
        tensor.deallocate(True)
    if mesh_config is not None and mesh_config.tp > 1:
        out = ccl_allreduce(out, mesh_config, ccl_manager)
    return out


def ragged_prefill_chunk_size():
    """Token-dim chunk length for long-prompt ragged prefill (``DG_PREFILL_RAGGED_CHUNK``).

    Defaults to ``RAGGED_PREFILL_CHUNK`` (4096), the single-call ceiling for the ragged path.
    Must be a positive multiple of the tile height so every chunk (and the
    32-multiple-padded tail) is a legal ragged prefill shape."""
    raw = os.environ.get("DG_PREFILL_RAGGED_CHUNK")
    if raw is None or not raw.strip():
        return RAGGED_PREFILL_CHUNK
    value = int(raw)
    if value <= 0 or value % TILE != 0:
        raise ValueError(f"DG_PREFILL_RAGGED_CHUNK must be a positive multiple of {TILE}, got {value}")
    return value


def chunked_ragged_sparse_prefill_forward(
    hidden_states,
    routing_weights,
    weights,
    config,
    prefill_sparsity,
    mesh_config=None,
    mesh_device=None,
    ccl_manager=None,
):
    """Ragged sparse prefill for prompts longer than one chunk.

    MoE is per-token, so a long prefill is processed in ``ragged_prefill_chunk_size()``-token
    slices along the sequence dim: the full-S ``RaggedRouting`` (computed once by the router hook)
    and ``hidden_states`` are sliced by the same boundaries, each slice runs
    ``ragged_sparse_prefill_forward`` UNCHANGED (including its per-slice TP all-reduce), and the
    per-chunk ``[1, 1, chunk, H]`` outputs are concatenated on the token dim. This is bit-identical
    to a single full-S ragged call — the router (RMSNorm/softmax/top-k) is strictly per-token so a
    sliced ``RaggedRouting`` equals per-token routing, the ragged FFN is per-token, and TP all-reduce
    is per-element (grouping tokens into chunks cannot change any element). It also keeps every
    intermediate bounded to the single-chunk footprint (the [top_k, S, H] combine reduction and the
    ``top_k*S*H`` index volumes both scale with the chunk length, not the full context), which is
    what lets prefill scale past the ~64K DRAM / ~128K int32-index limits of a single full-S call.

    Drop-in for ``ragged_sparse_prefill_forward`` (identical signature). For ``S <= chunk`` — or a
    non-``RaggedRouting`` argument — it delegates straight through, so the single-chunk path is
    byte-for-byte the validated behavior.
    """
    S = hidden_states.shape[2]
    chunk = ragged_prefill_chunk_size()
    if S <= chunk or not isinstance(routing_weights, RaggedRouting):
        return ragged_sparse_prefill_forward(
            hidden_states,
            routing_weights,
            weights,
            config,
            prefill_sparsity,
            mesh_config=mesh_config,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

    values = routing_weights.values  # [1, 1, S, top_k]
    indices = routing_weights.indices  # [1, 1, S, top_k]
    scale = routing_weights.per_expert_scale  # [1, 1, 1, E] — shared across chunks, NOT sliced
    top_k = values.shape[-1]
    H = hidden_states.shape[3]

    chunk_outputs = []
    for start in range(0, S, chunk):
        end = min(start + chunk, S)  # start is chunk-aligned; S is a 32-multiple upstream
        hidden_chunk = ttnn.slice(hidden_states, [0, 0, start, 0], [1, 1, end, H])
        values_chunk = ttnn.slice(values, [0, 0, start, 0], [1, 1, end, top_k])
        indices_chunk = ttnn.slice(indices, [0, 0, start, 0], [1, 1, end, top_k])
        routing_chunk = RaggedRouting(values_chunk, indices_chunk, scale)
        # ragged_sparse_prefill_forward deallocates values_chunk/indices_chunk (its RaggedRouting
        # input) inside _ragged_metadata_host; it never touches hidden_chunk, so we free that here.
        chunk_outputs.append(
            ragged_sparse_prefill_forward(
                hidden_chunk,
                routing_chunk,
                weights,
                config,
                prefill_sparsity,
                mesh_config=mesh_config,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
        )
        hidden_chunk.deallocate(True)

    values.deallocate(True)
    indices.deallocate(True)
    out = chunk_outputs[0] if len(chunk_outputs) == 1 else ttnn.concat(chunk_outputs, dim=2)
    if len(chunk_outputs) > 1:
        for tensor in chunk_outputs:
            tensor.deallocate(True)
    return out
