# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-request striped attention inside a shared projection batch."""

import ttnn
from models.demos.gemma4_d_p.tt.attention.ring_prefill import ring_prefill_program_config


def _rows(tensor, start, length):
    return ttnn.slice(
        tensor,
        (0, 0, start, 0),
        (tensor.shape[0], tensor.shape[1], start + length, tensor.shape[3]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def striped_prefill_attention(attention, q, k, v, requests, compute_config):
    mesh, ccl = attention.mesh_config, attention.ccl_manager
    cache = attention.ring_kv_cache
    global_attention = not attention.config.is_sliding
    outputs = []
    offset = 0
    for length, metadata in requests:
        local_length = length // mesh.cp_degree
        query = _rows(q, offset, local_length)
        keys = _rows(k, offset, local_length)
        values = None if global_attention else _rows(v, offset, local_length)
        pairs = ((cache.kv, keys),) if global_attention else ((cache.k, keys), (cache.v, values))
        for target, source in pairs:
            converted = source if source.dtype == target.dtype else ttnn.typecast(source, target.dtype)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                target,
                converted,
                slot_idx=metadata.slot_idx,
                kv_actual_global=metadata.kv_actual_global,
                valid_global=metadata.valid_global,
                layer_idx=attention.ring_layer_idx,
                num_layers=attention.ring_num_layers,
                cluster_axis=mesh.cp_axis,
                stripe_size=32,
            )
            if converted is not source:
                converted.deallocate(True)
        keys.deallocate(True)
        if values is not None:
            values.deallocate(True)

        q_chunk = next(size for size in (128, 64, 32) if local_length % size == 0)
        cache_k = cache.kv if global_attention else cache.k
        k_chunk = next(
            size for size in ((256, 128, 64, 32) if global_attention else (32,)) if cache_k.shape[2] % size == 0
        )
        grid = ccl.compute_grid_size
        splits = 3 if query.shape[1] * (local_length // q_chunk) * 3 <= (grid.x - 1) * grid.y else 1
        config = ring_prefill_program_config(
            mesh.device,
            ccl,
            attention.config.head_dim,
            q_chunk,
            k_chunk,
            max_k_splits=splits,
            matmul_math_fidelity=ttnn.MathFidelity.LoFi if global_attention else ttnn.MathFidelity.HiFi2,
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
        output, stats = ttnn.experimental.gemma4.rjsdpa(
            query,
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
            slot_id=metadata.slot_idx,
            kv_actual_isl_tensor=metadata.kv_actual_global,
            kv_cache_num_layers=attention.ring_num_layers,
            kv_cache_layer_idx=attention.ring_layer_idx,
            **sliding_args,
        )
        query.deallocate(True)
        stats.deallocate(True)
        outputs.append(output)
        offset += local_length
    if offset < q.shape[2]:
        padding = _rows(q, offset, q.shape[2] - offset)
        outputs.append(ttnn.zeros_like(padding))
        padding.deallocate(True)
    if len(outputs) == 1:
        return outputs[0]
    result = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for output in outputs:
        output.deallocate(True)
    return result
