# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal

TILE_HEIGHT = 32


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([1, 4, 4, 2], [1, 4, 128, 2], 2),
        ([8, 8, 8, 8], [8, 8, 8, 8], -1),
        ([32, 64, 128], [32, 64, 128], -1),
        ([64, 128, 256], [64, 128, 128], -1),
        ([1, 2048, 1, 64], [1, 2048, 1, 32], -1),
        ([1, 1, 1, 1], [1, 1, 1, 1], -1),
        ([4, 4], [4, 4], 1),
        ([128, 64], [128, 32], 1),
        ([16, 16, 16], [16, 16, 16], 0),
        ([1, 1, 1, 1], [1, 1, 1, 1], 1),
        ([64, 128, 256], [64, 128, 128], 1),
        ([256, 2, 32], [160, 2, 32], 1),
        ([2, 256, 2, 32], [2, 128, 2, 32], 1),
        ([2, 32, 96], [2, 32, 32], 1),
        ([128, 128], [128, 64], 1),
        ([1, 2, 128, 1, 768], [1, 2, 8, 1, 768], 2),
        ([1, 2, 8, 1, 768], [1, 2, 8, 1, 128], -1),
        ([1, 2, 8, 2, 768], [1, 2, 8, 2, 128], -1),
        ([1, 1, 2, 8, 2, 768], [1, 1, 2, 8, 2, 128], -2),
    ],
)
def test_gather_general(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([8, 8, 8, 8], [8, 8, 8, 8], -1),
        ([32, 64, 128], [32, 64, 128], -1),
        ([64, 128, 256], [64, 128, 128], -1),
        ([1, 2048, 1, 64], [1, 2048, 1, 32], -1),
        ([1, 1, 1, 1], [1, 1, 1, 1], -1),
    ],
)
def test_gather_preallocated_output(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)
    output = torch.zeros_like(index, dtype=torch_dtype)

    torch_gather = torch.gather(input, dim, index, out=output)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)
    ttnn_output = ttnn.from_torch(output, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    ttnn.gather(ttnn_input, dim, index=ttnn_index, out=ttnn_output)

    assert ttnn_output.shape == index.shape

    assert_allclose(torch_gather, ttnn.to_torch(ttnn_output))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([1, 1, 512, 64], [1, 1, 512, 32], -1),  # 16 cores
        ([1, 1, 2048, 64], [1, 1, 2048, 32], -1),  # 64 cores
        ([1, 1, 2240, 64], [1, 1, 2240, 32], -1),  # 70 cores
    ],
)
def test_gather_multicore_cases(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim, torch_input_datatype, ttnn_input_datatype, ttnn_index_datatype",
    [
        ([1, 1, 512, 64], [1, 1, 512, 32], -1, torch.float32, ttnn.float32, ttnn.uint16),
        ([128, 64], [128, 32], 1, torch.bfloat16, ttnn.bfloat16, ttnn.uint16),
        ([2, 32, 96], [2, 32, 32], -1, torch.float32, ttnn.float32, ttnn.uint32),
    ],
)
def test_gather_datatype_cases(
    input_shape, index_shape, dim, torch_input_datatype, ttnn_input_datatype, ttnn_index_datatype, device
):
    torch.manual_seed(0)

    input = torch.randn(input_shape, dtype=torch_input_datatype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn_input_datatype, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn_index_datatype, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([32, 256 * TILE_HEIGHT], [32, 64 * TILE_HEIGHT], -1),
        ([1, 1, 32, 256 * TILE_HEIGHT], [1, 1, 32, 128 * TILE_HEIGHT], -1),
        ([1, 1, 32, 63 * TILE_HEIGHT], [1, 1, 32, 63 * TILE_HEIGHT], -1),
        ([1, 1, 32, 20 * TILE_HEIGHT], [1, 1, 32, 20 * TILE_HEIGHT], -1),
        ([1, 1, 32, 96 * TILE_HEIGHT], [1, 1, 32, 96 * TILE_HEIGHT], -1),
        ([1, 1, 32, 256 * TILE_HEIGHT], [1, 1, 32, 256 * TILE_HEIGHT], -1),
        ([1, 151936], [1, 151936], -1),
        ([1, 128256], [1, 128256], -1),
    ],
)
def test_gather_long_tensor(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    max_uint32 = np.iinfo(np.uint32).max
    max_idx_val = min(input_shape[dim], max_uint32)
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, max_idx_val, index_shape, dtype=torch.int64)  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim, runs",
    [
        ([64, 64], [64, 32], -1, 10),
        ([1, 1, 32, 2048 * TILE_HEIGHT], [1, 1, 32, 2048 * TILE_HEIGHT], -1, 2),
        ([32, 128], [32, 128], -1, 5),
    ],
)
def test_gather_cache_run(input_shape, index_shape, dim, runs, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16

    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, input_shape[dim], index_shape, dtype=torch.int64)

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    for _ in range(runs):
        ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)
        assert ttnn_gather.shape == index.shape
        assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([32, 64, 128], [32, 64, 128], -1),
        ([32, 8192], [32, 2048], -1),
    ],
)
def test_gather_sub_core_grids(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(
        0, input_shape[dim], index_shape, dtype=torch.int64
    )  # torch.int64 is required for torch.gather

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
        ]
    )

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index, sub_core_grids=sub_core_grids)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


@pytest.mark.parametrize(
    "input_shape, index_shape, dim",
    [
        ([32, 32, 64 * TILE_HEIGHT], [32, 32, 64 * TILE_HEIGHT], -1),
        ([64, 64, 128 * TILE_HEIGHT], [64, 64, 128 * TILE_HEIGHT], -1),
    ],
)
def test_gather_multirow(input_shape, index_shape, dim, device):
    torch.manual_seed(0)

    torch_dtype = torch.bfloat16
    max_uint32 = np.iinfo(np.uint32).max
    max_idx_val = min(input_shape[dim], max_uint32)
    input = torch.randn(input_shape, dtype=torch_dtype)
    index = torch.randint(0, max_idx_val, index_shape, dtype=torch.int64)

    torch_gather = torch.gather(input, dim, index)

    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.TILE, device=device)

    ttnn_gather = ttnn.gather(ttnn_input, dim, index=ttnn_index)

    assert ttnn_gather.shape == index.shape
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))


# --- Codegen-path coverage ---
#
# ttnn.gather routes gate-supported cases to codegen and the rest to native, and offers no way to
# ask for one: the verification-only entries below live in the private module for that reason (see
# gather_force.hpp). The nightly routing suite only asserts the *rejected* cases fall back to
# native, so nothing there fails if a codegen kernel itself breaks -- these pin codegen and compare
# it against native on the same input. That comparison is exact because gather only moves values,
# so any mismatch is a real kernel bug rather than tolerance.
#
# One case per factory select_program_factory() can pick:
#   Wt_index == 1                    -> interleaved, one buffered index row per core
#   Wt_index >= 2 and Ht < the grid  -> tiled, index tile columns split across the cores
# The streaming factory is reached by L1 pressure rather than by shape, so its witness row has to be
# computed from the device; it gets its own test below.
_CODEGEN_CASES = [
    # (input_shape, index_shape, dim)
    ([1, 1, 32, 64], [1, 1, 32, 32], -1),
    ([1, 1, 128, 256], [1, 1, 128, 192], -1),
]
_CODEGEN_CASE_IDS = ["interleaved", "tiled"]

_force_native = ttnn._ttnn.operations.data_movement.gather_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.gather_force_codegen

_TILE_BYTES = 32 * 32 * 2


def _codegen_tensors(device, input_shape, index_shape, dim):
    """The gathered axis fixes the index dtype: uint16 cannot name a position past 65535."""
    axis_len = input_shape[dim]
    xt = ttnn.from_torch(
        torch.rand(input_shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    index = torch.randint(0, axis_len, index_shape, dtype=torch.int32)
    index_dtype = ttnn.uint16 if axis_len <= 65535 else ttnn.uint32
    return xt, ttnn.from_torch(index, dtype=index_dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("input_shape, index_shape, dim", _CODEGEN_CASES, ids=_CODEGEN_CASE_IDS)
def test_gather_codegen(device, input_shape, index_shape, dim):
    torch.manual_seed(42)
    xt, it = _codegen_tensors(device, input_shape, index_shape, dim)

    golden = _force_native(xt, dim, it)
    output = _force_codegen(xt, dim, it)

    assert output.shape == golden.shape, f"Output shape {output.shape} does not match native shape {golden.shape}"
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(output))


def test_gather_codegen_streaming(device):
    # An input row the per-core L1 budget can only just hold streams as ONE block whose depth is the
    # row itself; a wider row splits into two half-depth blocks and fits under any budget. So the
    # witness row is the budget in tile pages net of the index and output pages sharing it, and it
    # has to come from the device -- Wormhole's L1 is 36 pages smaller than Blackhole's, and a row
    # hardcoded for one is not brimful on the other. Wt_index then sizes the interleaved plan's
    # max(4, Wt_index)-deep output CB past the same budget, so selection reaches streaming.
    torch.manual_seed(42)
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
    wt_input = (budget - 2 * _TILE_BYTES) // _TILE_BYTES
    xt, it = _codegen_tensors(device, [1, 1, 64, 32 * wt_input], [1, 1, 64, 32 * 16], -1)

    golden = _force_native(xt, -1, it)
    output = _force_codegen(xt, -1, it)

    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(output))


@pytest.mark.parametrize("input_shape, index_shape, dim", _CODEGEN_CASES, ids=_CODEGEN_CASE_IDS)
def test_pc_gather_codegen(device, input_shape, index_shape, dim):
    torch.manual_seed(42)
    num_iters = 3
    # A distinct allocation per iteration: the cached program has to rebind its Buffer*s rather than
    # replay the first dispatch's addresses.
    tensors = [_codegen_tensors(device, input_shape, index_shape, dim) for _ in range(num_iters)]
    goldens = [ttnn.to_torch(_force_native(xt, dim, it)) for xt, it in tensors]

    for i in range(num_iters):
        xt, it = tensors[i]
        with device.cache_entries_counter.measure():
            output = _force_codegen(xt, dim, it)

        assert_equal(goldens[i], ttnn.to_torch(output))
        if i == 0:
            base_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"
