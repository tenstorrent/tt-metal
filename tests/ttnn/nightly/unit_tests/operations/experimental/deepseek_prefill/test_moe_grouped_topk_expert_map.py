# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused physical IDs must retain logical selection, weights, sentinels and cached bindings."""

import pytest
import torch

import ttnn


def _gate(inputs, k, groups, layout, score_func, **kwargs):
    return ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        *inputs,
        n_groups=groups,
        summed_experts_per_group=2 if groups == 8 else 1,
        topk_groups=4 if groups == 8 else 1,
        n_activated_experts=k,
        stable_sort=True,
        route_scale=2.5,
        weights_layout=layout,
        score_func=score_func,
        **kwargs,
    )


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1048576}], indirect=True
)
def test_fused_expert_map_exact_and_cached(mesh_device, expect_error):
    rng = torch.Generator().manual_seed(9047)
    cases = [
        (640, 256, 8, 1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "sigmoid"),
        (65, 256, 8, 1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "sigmoid"),
        (33, 128, 4, 1, ttnn.float32, ttnn.TILE_LAYOUT, "sigmoid"),
        (640, 128, 4, 1, ttnn.float32, ttnn.TILE_LAYOUT, "sigmoid"),
        (65, 256, 17, 1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "sigmoid"),
        (33, 256, 32, 1, ttnn.float32, ttnn.TILE_LAYOUT, "sigmoid"),
        (65, 256, 8, 8, ttnn.float32, ttnn.TILE_LAYOUT, "sqrtsoftplus"),
        (33, 384, 8, 1, ttnn.bfloat16, ttnn.TILE_LAYOUT, "sigmoid"),
        # Kimi-K3 gate shape: 896 experts, top-16, one group, 640 rows per chip.
        (640, 896, 16, 1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "sigmoid"),
        (65, 896, 16, 1, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "sigmoid"),
    ]
    for case_index, (rows, experts, k, groups, dtype, layout, score_func) in enumerate(cases):
        scores = torch.randn((8 * rows, experts), generator=rng)
        bias = torch.randn((8 * rows, experts), generator=rng) * 0.1
        # Exact ties must still be resolved in logical expert order before mapping.
        scores[::17] = 0
        bias[::17] = 0
        inputs = tuple(
            ttnn.from_torch(
                x,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
            )
            for x in (scores, bias)
        )
        padding = ttnn.from_torch(
            torch.tensor([[0 if row == 3 else max(0, rows - row * 3), row % 2] for row in range(8)], dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
        )
        options = dict(padding_config=None if rows == 640 else padding, memory_config=ttnn.L1_MEMORY_CONFIG)
        baseline = _gate(inputs, k, groups, layout, score_func, **options)
        expected = [[ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(t)] for t in baseline]
        assert torch.equal(expected[1][0][0].long(), torch.arange(k))
        permutations = [torch.arange(experts), torch.arange(experts).flip(0), torch.randperm(experts, generator=rng)]
        retained = []
        cached_programs = None
        for which in (0, 1, 2, 0, 2):
            mapping = permutations[which]
            lut = ttnn.from_torch(
                mapping.reshape(1, -1).int(),
                device=mesh_device,
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG if case_index % 2 == 0 else ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            result = _gate(inputs, k, groups, layout, score_func, expert_id_map=lut, **options)
            assert all(a.tensor_topology() == b.tensor_topology() for a, b in zip(result, baseline))
            if cached_programs is None:
                cached_programs = mesh_device.num_program_cache_entries()
            else:
                assert mesh_device.num_program_cache_entries() == cached_programs
            for chip, (w, ids) in enumerate(
                zip(ttnn.get_device_tensors(result[0]), ttnn.get_device_tensors(result[1]))
            ):
                assert torch.equal(ttnn.to_torch(w), expected[0][chip])
                physical = expected[1][chip].long().clone()
                real = physical < experts
                physical[real] = mapping[physical[real]]
                assert torch.equal(ttnn.to_torch(ids).long(), physical)
            # Keep buffers alive so later calls cannot accidentally reuse their addresses.
            retained.append((lut, result))
        # A trace must keep its original map after a different map reuses the cached program.
        ttnn.synchronize_device(mesh_device)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced = _gate(inputs, k, groups, layout, score_func, expert_id_map=retained[0][0], **options)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        other = _gate(inputs, k, groups, layout, score_func, expert_id_map=retained[-1][0], **options)
        zeros = ttnn.from_host_shards(
            [ttnn.from_torch(torch.zeros_like(x), dtype=ttnn.bfloat16, layout=layout) for x in expected[0]],
            mesh_device.shape,
        )
        zeros.update_tensor_topology(traced[0].tensor_topology())
        ttnn.copy_host_to_device_tensor(zeros, traced[0])
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        for index, tensor in enumerate(traced):
            for chip, shard in enumerate(ttnn.get_device_tensors(tensor)):
                assert torch.equal(ttnn.to_torch(shard), expected[index][chip])
        ttnn.release_trace(mesh_device, trace)
        for tensor in (*inputs, padding, *baseline, *traced, *other):
            ttnn.deallocate(tensor)
        for lut, pair in retained:
            for tensor in (lut, *pair):
                ttnn.deallocate(tensor)

    inputs = tuple(
        ttnn.from_torch(
            torch.zeros((32, 128)),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(2)
    )
    for shape, dtype, layout, message in [
        ((1, 96), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, "shape"),
        ((1, 128), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, "UINT16"),
        ((1, 128), ttnn.uint16, ttnn.TILE_LAYOUT, "ROW_MAJOR"),
    ]:
        lut = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.int32),
            device=mesh_device,
            dtype=dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        with expect_error(RuntimeError, message):
            _gate(inputs, 4, 1, ttnn.TILE_LAYOUT, "sigmoid", expert_id_map=lut)
        ttnn.deallocate(lut)
    for tensor in inputs:
        ttnn.deallocate(tensor)
