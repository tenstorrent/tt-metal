# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-request striped attention inside a shared projection batch."""

import ttnn
from models.demos.gemma4_d_p.tt.attention.ring_prefill import ring_prefill_program_config, ring_sdpa_chunk_sizes


def striped_prefill_attention(attention, q, k, v, requests, compute_config):
    mesh, ccl = attention.mesh_config, attention.ccl_manager
    cache = attention.ring_kv_cache
    global_attention = not attention.config.is_sliding
    cache_k = cache.kv if global_attention else cache.k
    keys = k if k.dtype == cache_k.dtype else ttnn.typecast(k, cache_k.dtype)
    values = None if global_attention else (v if v.dtype == cache.v.dtype else ttnn.typecast(v, cache.v.dtype))
    allocate_output = (
        ttnn.empty_like if sum(length for length, _ in requests) == q.shape[2] * mesh.cp_degree else ttnn.zeros_like
    )
    output = allocate_output(q, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    stats = ttnn.empty(
        (q.shape[0], q.shape[1], 2 * max(length for length, _ in requests) // mesh.cp_degree, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for length, metadata in requests:
        local_length = length // mesh.cp_degree
        pairs = ((cache.kv, keys),) if global_attention else ((cache.k, keys), (cache.v, values))
        for target, source in pairs:
            ttnn.experimental.gemma4.update_cache(
                target,
                source,
                request_descriptors=metadata.descriptors,
                request_index=metadata.index,
                request_local_tokens=local_length,
                layer_idx=attention.ring_layer_idx,
                num_layers=attention.ring_num_layers,
                cluster_axis=mesh.cp_axis,
            )

        q_chunk = next(size for size in (128, 64, 32) if local_length % size == 0)
        k_chunk = next(
            size for size in ((256, 128, 64, 32) if global_attention else (32,)) if cache_k.shape[2] % size == 0
        )
        grid = ccl.compute_grid_size
        num_cores = (grid.x - 1) * grid.y
        splits = 3 if q.shape[1] * (local_length // q_chunk) * 3 <= num_cores else 1
        segmented = False
        # Keep ragged slabs on divisor-Q tuning; legacy's policy is validated on 128-row-aligned slabs.
        if global_attention and local_length % 128 == 0:
            q_chunk, _, splits, segmented = ring_sdpa_chunk_sizes(
                local_length, sliding=False, num_heads=q.shape[1], num_cores=num_cores
            )
        config = ring_prefill_program_config(
            mesh.device,
            ccl,
            attention.config.head_dim,
            q_chunk,
            k_chunk,
            max_k_splits=splits,
            matmul_math_fidelity=ttnn.MathFidelity.LoFi if global_attention else ttnn.MathFidelity.HiFi2,
            segmented_accumulation=segmented,
        )
        key = "striped_global" if global_attention else "striped_sliding_k"
        gather_k = ccl.get_ring_gather_buffer(
            key,
            cache_k.shape[1],
            cache_k.shape[2] * mesh.cp_degree,
            cache_k.shape[3],
            cache_k.dtype,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        sliding_args = {}
        if not global_attention:
            gather_v = ccl.get_ring_gather_buffer(
                "striped_sliding_v",
                cache.v.shape[1],
                cache.v.shape[2] * mesh.cp_degree,
                cache.v.shape[3],
                cache.v.dtype,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            sliding_args = dict(
                input_tensor_v=cache.v,
                persistent_output_buffer_v=gather_v,
                sliding_window_size=attention.config.sliding_window_size,
                communication="halo",
            )
        ttnn.experimental.gemma4.rjsdpa(
            q,
            cache_k,
            persistent_output_buffer_kv=gather_k,
            head_dim_v=attention.config.head_dim,
            logical_n=attention.ring_max_seq_len,
            program_config=config,
            scale=1.0,
            compute_kernel_config=compute_config,
            dim=2,
            multi_device_global_semaphore=ccl.ring_attention_ccl_semaphore_handles,
            num_links=ccl.num_links,
            cluster_axis=mesh.cp_axis,
            mesh_device=mesh.device,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=ttnn.CoreCoord(*ccl.ring_attention_ccl_core_grid_offset),
            use_column_major_ccl=True,
            is_balanced=False,
            stripe_size=32,
            request_descriptors=metadata.descriptors,
            request_index=metadata.index,
            request_local_tokens=local_length,
            output_tensor=output,
            stats_tensor=stats,
            kv_cache_num_layers=attention.ring_num_layers,
            kv_cache_layer_idx=attention.ring_layer_idx,
            **sliding_args,
        )
    stats.deallocate(True)
    if keys is not k:
        keys.deallocate(True)
    if values is not None and values is not v:
        values.deallocate(True)
    return output
