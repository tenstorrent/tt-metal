# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wrapped GroupNorm reductions through families, including cached buffer rebinding."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics


@pytest.mark.parametrize("use_welford", [False, True], ids=["legacy", "welford"])
@pytest.mark.parametrize("column_major", [False, True], ids=["rows", "columns"])
def test_group_norm_wrapped_family_cache(device, use_welford, column_major):
    # Seven batches across 63 height shards: nine consecutive cores per reduction.
    # In row order on 7x9, group 3 is (6,3), the full row y=4, and (0,5).
    # Column order on 9x7 exercises the transposed staircase. The sender is its
    # first singleton, so the broadcast combines a local record and remote rectangles.
    width, height = (9, 7) if column_major else (7, 9)
    available = device.compute_with_storage_grid_size()
    if available.x < width or available.y < height:
        pytest.skip("requires a 7x9 or 9x7 worker grid")
    grid = ttnn.CoreGrid(x=width, y=height)
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))])
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ranges,
            [32, 128],
            ttnn.ShardOrientation.COL_MAJOR if column_major else ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    torch.manual_seed(7)
    expected = []
    inputs = []
    # Keep all buffers live, forcing different addresses on cache hits.
    for _ in range(3):
        source = torch.rand(7, 128, 1, 288, dtype=torch.bfloat16)
        expected.append(torch.nn.functional.group_norm(source.float(), 16).permute(0, 2, 3, 1))
        inputs.append(
            ttnn.from_torch(
                source.permute(0, 2, 3, 1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=memory,
            )
        )
    mask = ttnn.to_device(ttnn.create_group_norm_input_mask(128, 16, 1, ttnn.bfloat8_b), device)
    outputs = []
    cache_entries = None
    for source in inputs:
        outputs.append(
            ttnn.group_norm(
                source,
                num_groups=16,
                epsilon=1e-5,
                input_mask=mask,
                core_grid=grid,
                memory_config=memory,
                output_layout=ttnn.TILE_LAYOUT,
                inplace=False,
                use_welford=use_welford,
            )
        )
        ttnn.synchronize_device(device)
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
            assert cache_entries > 0
        else:
            assert device.num_program_cache_entries() == cache_entries
    for reference, output in zip(expected, outputs):
        actual = ttnn.to_torch(ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)).float()
        assert_numeric_metrics(reference, actual, pcc_threshold=0.999, rtol=0.01, atol=0.09, frobenius_threshold=0.035)


@pytest.mark.parametrize("use_welford", [False, True], ids=["legacy", "welford"])
@pytest.mark.parametrize("batches", [1, 8], ids=["multicast", "local"])
def test_group_norm_interleaved_family_cache(device, use_welford, batches):
    available = device.compute_with_storage_grid_size()
    if available.x < 8 or available.y < 4:
        pytest.skip("requires an 8x4 worker grid")
    grid = ttnn.CoreGrid(x=8, y=4)
    spatial = 512 if batches == 1 else 64
    torch.manual_seed(11)
    inputs, expected = [], []
    for _ in range(3):
        source = torch.rand(batches, 128, 1, spatial, dtype=torch.bfloat16)
        expected.append(torch.nn.functional.group_norm(source.float(), 16).permute(0, 2, 3, 1))
        inputs.append(
            ttnn.from_torch(
                source.permute(0, 2, 3, 1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    (weight, bias), mask = ttnn.dram_group_norm_params_from_torch(
        [torch.ones(128), torch.zeros(128)], 128, 16, device, core_grid=grid, return_mask=True
    )
    outputs = []
    cache_entries = None
    for source in inputs:
        outputs.append(
            ttnn.group_norm(
                source,
                num_groups=16,
                epsilon=1e-5,
                input_mask=mask,
                weight=weight,
                bias=bias,
                core_grid=grid,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_layout=ttnn.TILE_LAYOUT,
                inplace=False,
                num_out_blocks=1,
                use_welford=use_welford,
            )
        )
        ttnn.synchronize_device(device)
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
            assert cache_entries > 0
        else:
            assert device.num_program_cache_entries() == cache_entries
    for reference, output in zip(expected, outputs):
        assert_numeric_metrics(
            reference,
            ttnn.to_torch(output).float(),
            pcc_threshold=0.999,
            rtol=0.01,
            atol=0.09,
            frobenius_threshold=0.035,
        )
