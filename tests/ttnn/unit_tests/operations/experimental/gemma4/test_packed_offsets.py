# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor offsets, cache writes, and shared outputs across CP8 trace replays."""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.experimental.gemma4.test_rjsdpa import (
    CP,
    PackedCase,
    runtime as runtime,
)
from tests.ttnn.unit_tests.operations.experimental.gemma4.test_rjsdpa_sliding import SlidingCase
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


@pytest.mark.parametrize("mode", ["global", "all_gather", "halo"])
@pytest.mark.parametrize("q_chunk,k_chunk,splits", [(32, 64, 1), (64, 128, 3)])
def test_packed_offsets_trace(runtime, mode, q_chunk, k_chunk, splits):
    global_attention = mode == "global"
    case = (
        PackedCase(runtime, tokens=4096, capacity=4096, nd_sharded=splits > 1)
        if global_attention
        else SlidingCase(runtime, tokens=4096, capacity=4096, communication=mode, nd_sharded=splits > 1)
    )
    mesh = case.mesh
    local_capacity = 4096 // CP
    spans, valid_lengths = (512, 1024), (257, 769)
    gen = torch.Generator().manual_seed(814)
    source_q = [torch.randn(1, 2 * case.tp, span, 64, generator=gen) for span in spans]
    width = 128 if global_attention else 64
    source_k = [torch.randint(-7, 8, (1, case.tp, span, width), generator=gen).float() / 8 for span in spans]
    source_v = [torch.randint(-7, 8, (1, case.tp, span, 64), generator=gen).float() / 8 for span in spans]
    packed_k = case.upload(torch.zeros(1, case.tp, 4096, width), ttnn.bfloat8_b)
    packed_v = None if global_attention else case.upload(torch.zeros(1, case.tp, 4096, 64), ttnn.bfloat8_b)
    if splits > 1:
        packed_k = ttnn.to_memory_config(packed_k, ttnn.L1_MEMORY_CONFIG)
        case.allocated.append(packed_k)
        if packed_v is not None:
            packed_v = ttnn.to_memory_config(packed_v, ttnn.L1_MEMORY_CONFIG)
            case.allocated.append(packed_v)
    output = case.upload(torch.zeros(1, 2 * case.tp, 4096, 64), ttnn.bfloat16)
    stats = case.upload(torch.zeros(1, 2 * case.tp, 2 * 1024, 1), ttnn.bfloat16)
    mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(1, 2))

    def copy_rows(data, target, dtype):
        host = ttnn.from_torch(data.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        ttnn.copy_host_to_device_tensor(host, target)

    def descriptor_host(rows):
        return ttnn.from_torch(
            torch.tensor(rows, dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    descriptors = ttnn.to_device(descriptor_host([[0] * 8 for _ in range(3)]), mesh)
    case.allocated.append(descriptors)
    expected_cache_k = (case.kv if global_attention else case.k).clone()
    expected_cache_v = None if global_attention else case.v.clone()

    def stage(offsets, prefixes, slots):
        q = torch.zeros(1, 2 * case.tp, 4096, 64)
        k = torch.zeros(1, case.tp, 4096, width)
        v = torch.zeros(1, case.tp, 4096, 64)
        descriptors_rows = [[0] * 8]
        references = []
        for i, (span, valid, offset, prefix, slot) in enumerate(zip(spans, valid_lengths, offsets, prefixes, slots)):
            descriptors_rows.append([1, slot, prefix, valid, offset, span // 256, 0, 0])
            padded_valid = (valid + 31) // 32 * 32
            positions = torch.arange(prefix, prefix + span)
            indices, valid_positions = [], []
            for rank in range(CP):
                source_rows = torch.where(positions.div(32, rounding_mode="floor") % CP == rank)[0]
                target_rows = rank * local_capacity + offset * 32 + torch.arange(len(source_rows))
                q[:, :, target_rows] = source_q[i][:, :, source_rows]
                k[:, :, target_rows] = source_k[i][:, :, source_rows]
                v[:, :, target_rows] = source_v[i][:, :, source_rows]
                keep = source_rows < valid
                indices.append(target_rows[keep])
                valid_positions.append(positions[source_rows[keep]])
            indices, valid_positions = torch.cat(indices), torch.cat(valid_positions)
            batch = slot * 2 + 1
            expected_cache_k[batch : batch + 1, :, prefix : prefix + padded_valid] = source_k[i][:, :, :padded_valid]
            if not global_attention:
                expected_cache_v[batch : batch + 1, :, prefix : prefix + padded_valid] = source_v[i][
                    :, :, :padded_valid
                ]
            references.append((indices, valid_positions, prefix, valid, batch))
        copy_rows(q, case.tt_q, ttnn.bfloat16)
        copy_rows(k, packed_k, ttnn.bfloat8_b)
        if packed_v is not None:
            copy_rows(v, packed_v, ttnn.bfloat8_b)
        copy_rows(torch.zeros_like(q), output, ttnn.bfloat16)
        ttnn.copy_host_to_device_tensor(descriptor_host(descriptors_rows), descriptors)
        return q, references

    def run():
        for i, span in enumerate(spans, start=1):
            descriptor_args = dict(request_descriptors=descriptors, request_index=i, request_local_tokens=span // CP)
            caches = ((case.tt_kv, packed_k),) if global_attention else ((case.tt_k, packed_k), (case.tt_v, packed_v))
            for target, source in caches:
                ttnn.experimental.gemma4.update_cache(
                    target, source, layer_idx=1, num_layers=2, cluster_axis=runtime.sp_axis, **descriptor_args
                )
            kwargs = dict(
                persistent_output_buffer_kv=case.buffer if global_attention else case.buffer_k,
                head_dim_v=64,
                logical_n=4096,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=runtime.sdpa_compute_grid if splits > 1 else ttnn.CoreCoord(2, 2),
                    q_chunk_size=q_chunk,
                    k_chunk_size=k_chunk,
                    max_k_splits=splits,
                    exp_approx_mode=False,
                ),
                compute_kernel_config=runtime.compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                cluster_axis=runtime.sp_axis,
                mesh_device=mesh,
                topology=runtime.topology,
                subdevice_id=runtime.worker_sub_device_id,
                ccl_core_grid_offset=(runtime.ccl_column, 0),
                use_column_major_ccl=True,
                stripe_size=32,
                kv_cache_num_layers=2,
                kv_cache_layer_idx=1,
                output_tensor=output,
                stats_tensor=stats,
                **descriptor_args,
            )
            if not global_attention:
                kwargs.update(
                    input_tensor_v=case.tt_v,
                    persistent_output_buffer_v=case.buffer_v,
                    sliding_window_size=case.window,
                    communication=mode,
                )
            result, scratch = ttnn.experimental.gemma4.rjsdpa(
                case.tt_q, case.tt_kv if global_attention else case.tt_k, **kwargs
            )
            assert result.buffer_address() == output.buffer_address()
            assert scratch.buffer_address() == stats.buffer_address()

    try:
        for iteration, (offsets, prefixes, slots) in enumerate(
            (((1, 5), (288, 1248), (0, 1)), ((8, 1), (160, 2080), (1, 0)))
        ):
            q, references = stage(offsets, prefixes, slots)
            if iteration == 0:
                run()
                ttnn.synchronize_device(mesh)
                case.trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                run()
                ttnn.end_trace_capture(mesh, case.trace_id, cq_id=0)
            ttnn.execute_trace(mesh, case.trace_id, cq_id=0, blocking=True)
            actual = ttnn.to_torch(output, mesh_composer=case.composer).float()
            for indices, positions, prefix, valid, batch in references:
                cache_k = expected_cache_k[batch : batch + 1, :, : prefix + valid]
                keys = cache_k[..., :64]
                values = (
                    cache_k[..., -64:] if global_attention else expected_cache_v[batch : batch + 1, :, : prefix + valid]
                )
                key_positions = torch.arange(prefix + valid)
                mask = key_positions[None, :] <= positions[:, None]
                if not global_attention:
                    mask &= key_positions[None, :] > positions[:, None] - case.window
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q[:, :, indices].bfloat16().float(),
                    keys.repeat_interleave(2, dim=1),
                    values.repeat_interleave(2, dim=1),
                    attn_mask=mask,
                )
                got = actual[:, :, indices]
                passed, pcc = comp_pcc(expected, got, 0.99)
                assert passed, pcc
                assert (expected - got).square().mean().sqrt() < 0.025
            untouched = torch.ones(4096, dtype=torch.bool)
            for offset, span in zip(offsets, spans):
                for rank in range(CP):
                    start = rank * local_capacity + offset * 32
                    untouched[start : start + span // CP] = False
            assert torch.count_nonzero(actual[:, :, untouched]) == 0
            for target, expected in (
                ((case.tt_kv, expected_cache_k),)
                if global_attention
                else ((case.tt_k, expected_cache_k), (case.tt_v, expected_cache_v))
            ):
                got = ttnn.to_torch(target, mesh_composer=case.composer).float()[
                    :, :, torch.argsort(case.cache_positions)
                ]
                torch.testing.assert_close(got, expected, atol=0, rtol=0)
    finally:
        case.close()
