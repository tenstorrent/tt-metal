# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.quasar.all_gather: intra-chip all-gather of a sharded tensor.

Oracle: every output shard is the whole input, i.e. output == input.unsqueeze(0).expand(N, ...).
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tt_py_test_utils_common.utility_functions import is_quasar

pytestmark = pytest.mark.skipif(not is_quasar(), reason="Quasar-native op: ttnn.experimental.quasar.all_gather")


def _rank_grid(device, num_ranks):
    grid = device.compute_with_storage_grid_size()
    if num_ranks > grid.x * grid.y:
        pytest.skip(f"needs {num_ranks} clusters, device has {grid.x * grid.y}")
    return ttnn.num_cores_to_corerangeset(num_ranks, grid, True)


def _sharded_input(device, shape, num_ranks, layout, dtype):
    torch.manual_seed(2026)
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    torch_input = torch.randn(shape, dtype=torch_dtype)
    height = shape[-2] * (torch_input.numel() // (shape[-2] * shape[-1]))
    width = shape[-1]
    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_shape = (height // num_ranks, width)
    else:
        shard_shape = (height, width // num_ranks)
    spec = ttnn.ShardSpec(_rank_grid(device, num_ranks), shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, spec)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    return torch_input, tt_input


def _check(torch_input, tt_output, num_ranks):
    assert list(tt_output.shape) == [num_ranks] + list(torch_input.shape)
    expected = torch_input.unsqueeze(0).expand(num_ranks, *torch_input.shape)
    assert_equal(ttnn.to_torch(tt_output), expected)


@pytest.mark.parametrize("num_ranks", [2, 4])
@pytest.mark.parametrize(
    "shape, layout",
    [
        ((1, 1, 128, 64), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ((1, 1, 64, 256), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((2, 1, 64, 96), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
    ids=["h128x64", "w64x256", "b2h64x96"],
)
@pytest.mark.parametrize("threads", [1, 2, 6])
def test_all_gather_direct(device, num_ranks, shape, layout, threads):
    torch_input, tt_input = _sharded_input(device, shape, num_ranks, layout, ttnn.bfloat16)
    tt_output = ttnn.experimental.quasar.all_gather(tt_input, topology="direct", reader_threads=threads)
    _check(torch_input, tt_output, num_ranks)


@pytest.mark.parametrize("num_ranks", [2, 3, 4])
@pytest.mark.parametrize(
    "shape, layout",
    [
        ((1, 1, 384, 64), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        ((1, 1, 64, 384), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    ],
    ids=["h384x64", "w64x384"],
)
@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("chunks_per_sync", [1, 2])
def test_all_gather_ring(device, num_ranks, shape, layout, threads, chunks_per_sync):
    torch_input, tt_input = _sharded_input(device, shape, num_ranks, layout, ttnn.bfloat16)
    tt_output = ttnn.experimental.quasar.all_gather(
        tt_input,
        topology="ring",
        reader_threads=threads,
        writer_threads=threads,
        chunks_per_sync=chunks_per_sync,
    )
    _check(torch_input, tt_output, num_ranks)


def test_all_gather_ring_two_workers_per_rank(device):
    num_ranks = 4
    torch_input, tt_input = _sharded_input(
        device, (1, 1, 128, 128), num_ranks, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.bfloat16
    )
    tt_output = ttnn.experimental.quasar.all_gather(tt_input, topology="ring", num_workers_per_rank=2)
    _check(torch_input, tt_output, num_ranks)


@pytest.mark.parametrize("topology", ["direct", "ring"])
def test_all_gather_program_cache_relaunch(device, topology):
    """Semaphores and DFB credits must come back clean on a cached relaunch."""
    num_ranks = 4
    for _ in range(3):
        torch_input, tt_input = _sharded_input(
            device, (1, 1, 128, 64), num_ranks, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.bfloat16
        )
        tt_output = ttnn.experimental.quasar.all_gather(tt_input, topology=topology)
        _check(torch_input, tt_output, num_ranks)
        ttnn.deallocate(tt_output)
        ttnn.deallocate(tt_input)


@pytest.mark.parametrize("topology, threads", [("direct", 1), ("direct", 6), ("ring", 1), ("ring", 3)])
def test_all_gather_large(device, topology, threads):
    """8 ranks, 72 tiles per slice: many packets per thread, a short tail packet, every DM hart."""
    num_ranks = 8
    torch_input, tt_input = _sharded_input(
        device, (1, 1, 256, 288), num_ranks, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.bfloat16
    )
    tt_output = ttnn.experimental.quasar.all_gather(
        tt_input, topology=topology, reader_threads=threads, writer_threads=threads, tiles_per_packet=3
    )
    _check(torch_input, tt_output, num_ranks)


def test_all_gather_fp32(device):
    num_ranks = 2
    torch_input, tt_input = _sharded_input(
        device, (1, 1, 64, 64), num_ranks, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.float32
    )
    tt_output = ttnn.experimental.quasar.all_gather(tt_input)
    _check(torch_input, tt_output, num_ranks)
