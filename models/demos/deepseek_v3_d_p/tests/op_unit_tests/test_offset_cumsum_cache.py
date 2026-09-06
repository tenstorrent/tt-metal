# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independent dispatch groups must keep distinct data with shared cumsum programs."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [torus_xy_device_params(fabric_payload_size=6144, l1_small_size=1216, trace_region_size=8000000)],
    indirect=True,
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_offset_cumsum_distinct_groups_cache_and_trace(mesh_device, cluster_axis, memory_config):
    rows, cols = tuple(mesh_device.shape)
    width, experts_per_chip = 256, 8
    retained = []
    mesh_device.enable_program_cache()

    def make_input(seed):
        generator = torch.Generator().manual_seed(seed)
        histograms = torch.randint(0, 64, (rows, cols, width), dtype=torch.int32, generator=generator)
        tensor = ttnn.from_torch(
            histograms,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, 1)),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        retained.append(tensor)
        return histograms, tensor

    def run(tensor):
        return ttnn.experimental.deepseek_prefill.offset_cumsum(
            tensor,
            cluster_axis=cluster_axis,
            num_links=2,
            experts_per_chip=experts_per_chip,
            memory_config=memory_config,
            use_l1_small_for_semaphores=True,
        )

    def check(histograms, outputs):
        per_device = [ttnn.get_device_tensors(t) for t in outputs]
        assert all(len(tensors) == rows * cols for tensors in per_device)
        for chip in range(rows * cols):
            row, col = divmod(chip, cols)
            data = histograms[:, col, :] if cluster_axis == 0 else histograms[row, :, :]
            position = row if cluster_axis == 0 else col
            total = data.sum(0)
            aligned = ((total + 31) // 32 * 32).reshape(-1, experts_per_chip)
            region = (aligned.cumsum(-1) - aligned).reshape(width)
            offset = data[:position].sum(0) + region
            for reference, tensors in zip((offset, total, region), per_device):
                actual = ttnn.to_torch(tensors[chip]).reshape(-1).to(torch.int64)
                assert torch.equal(actual, reference), (cluster_axis, row, col)

    original, original_input = make_input(1234)
    warm = run(original_input)
    check(original, warm)
    ttnn.synchronize_device(mesh_device)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    captured = run(original_input)
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    retained.extend([*warm, *captured])
    try:
        for iteration in range(3):
            fresh, fresh_input = make_input(1235 + iteration)
            before = mesh_device.num_program_cache_entries()
            outputs = run(fresh_input)
            assert mesh_device.num_program_cache_entries() == before
            check(fresh, outputs)
            retained.extend(outputs)

            zeros = ttnn.from_torch(
                torch.zeros((1, width), dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            for output in captured:
                for shard in ttnn.get_device_tensors(output):
                    ttnn.copy_host_to_device_tensor(zeros, shard)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            check(original, captured)
            check(fresh, outputs)
    finally:
        ttnn.release_trace(mesh_device, trace)
