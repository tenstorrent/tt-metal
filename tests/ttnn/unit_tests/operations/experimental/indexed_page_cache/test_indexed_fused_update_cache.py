# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_slow_dispatch


def _device_tensor(tensor, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=None):
    return ttnn.from_torch(
        tensor,
        device=device,
        dtype=dtype,
        layout=layout,
        tile=tile,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _replicated_device_tensor(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _sharded_device_tensor(tensor, mesh_device, shard_dim, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
    )


def _mapped_2d_device_tensor(
    tensor, mesh_device, mesh_shape, mesh_dims, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=mesh_dims),
    )


def _apply_reference(cache, packed, positions):
    expected = cache.clone()
    rows_per_page = cache.shape[2]
    total_rows = cache.shape[0] * rows_per_page
    for source_row, physical_row in enumerate(positions):
        if physical_row < 0 or physical_row >= total_rows:
            continue
        page = physical_row // rows_per_page
        row_in_page = physical_row % rows_per_page
        expected[page, :, row_in_page, :] = packed[0, :, source_row, :]
    return expected


@pytest.mark.parametrize("cache_shape", [(1, 2, 128, 64), (4, 2, 64, 64)])
def test_indexed_fused_update_cache_scatter_rows(device, cache_shape):
    torch.manual_seed(17)
    source_rows = 9
    input_shape = (1, cache_shape[1], source_rows, cache_shape[3])
    cache1 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    cache2 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    input1 = torch.randn(input_shape, dtype=torch.bfloat16)
    input2 = torch.randn(input_shape, dtype=torch.bfloat16)

    # Several rows target the same physical page. Negative and out-of-range
    # entries exercise the trace-safe per-row skip path.
    total_rows = cache_shape[0] * cache_shape[2]
    positions = [1, 3, 31, 32, 33, 63, 64, -1, total_rows]
    positions_torch = torch.tensor([positions], dtype=torch.int32)

    cache1_tt = _device_tensor(cache1, device)
    cache2_tt = _device_tensor(cache2, device)
    input1_tt = _device_tensor(input1, device)
    input2_tt = _device_tensor(input2, device)
    positions_tt = _device_tensor(
        positions_torch,
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    output1_tt, output2_tt = ttnn.experimental.indexed_fused_update_cache(
        cache1_tt, input1_tt, cache2_tt, input2_tt, positions_tt
    )

    assert output1_tt.buffer_address() == cache1_tt.buffer_address()
    assert output2_tt.buffer_address() == cache2_tt.buffer_address()
    assert torch.equal(ttnn.to_torch(output1_tt), _apply_reference(cache1, input1, positions))
    assert torch.equal(ttnn.to_torch(output2_tt), _apply_reference(cache2, input2, positions))


def test_indexed_fused_update_cache_program_cache_reuses_runtime_positions(device):
    torch.manual_seed(23)
    cache_shape = (3, 1, 64, 128)
    input_shape = (1, 1, 4, 128)
    device.enable_program_cache()
    device.clear_program_cache()

    try:
        cache1_tt = _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device)
        cache2_tt = _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device)

        first_input1 = torch.randn(input_shape, dtype=torch.bfloat16)
        first_input2 = torch.randn(input_shape, dtype=torch.bfloat16)
        first_positions = [0, 1, 64, 65]
        ttnn.experimental.indexed_fused_update_cache(
            cache1_tt,
            _device_tensor(first_input1, device),
            cache2_tt,
            _device_tensor(first_input2, device),
            _device_tensor(
                torch.tensor([first_positions], dtype=torch.int32),
                device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
        cache_entries = device.num_program_cache_entries()

        second_input1 = torch.randn(input_shape, dtype=torch.bfloat16)
        second_input2 = torch.randn(input_shape, dtype=torch.bfloat16)
        second_positions = [2, 3, 128, -1]
        output1_tt, output2_tt = ttnn.experimental.indexed_fused_update_cache(
            cache1_tt,
            _device_tensor(second_input1, device),
            cache2_tt,
            _device_tensor(second_input2, device),
            _device_tensor(
                torch.tensor([second_positions], dtype=torch.int32),
                device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )

        assert cache_entries > 0
        assert device.num_program_cache_entries() == cache_entries
        expected1 = _apply_reference(
            _apply_reference(torch.zeros(cache_shape, dtype=torch.bfloat16), first_input1, first_positions),
            second_input1,
            second_positions,
        )
        expected2 = _apply_reference(
            _apply_reference(torch.zeros(cache_shape, dtype=torch.bfloat16), first_input2, first_positions),
            second_input2,
            second_positions,
        )
        assert torch.equal(ttnn.to_torch(output1_tt), expected1)
        assert torch.equal(ttnn.to_torch(output2_tt), expected2)
    finally:
        device.disable_and_clear_program_cache()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
@skip_for_slow_dispatch()
def test_indexed_fused_update_cache_trace_replay(device):
    torch.manual_seed(27)
    cache_shape = (2, 2, 64, 64)
    input_shape = (1, 2, 5, 64)

    cache1_tt = _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device)
    cache2_tt = _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device)
    input1_tt = _device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), device)
    input2_tt = _device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), device)
    positions_tt = _device_tensor(
        torch.zeros((1, input_shape[2]), dtype=torch.int32),
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Compile before capture. Trace replay must subsequently use the same device
    # buffers while observing newly copied inputs and physical positions.
    ttnn.experimental.indexed_fused_update_cache(cache1_tt, input1_tt, cache2_tt, input2_tt, positions_tt)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    output1_tt, output2_tt = ttnn.experimental.indexed_fused_update_cache(
        cache1_tt, input1_tt, cache2_tt, input2_tt, positions_tt
    )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    try:
        for positions in ([0, 31, 32, -1, 128], [2, 63, 64, 65, 66]):
            input1 = torch.randn(input_shape, dtype=torch.bfloat16)
            input2 = torch.randn(input_shape, dtype=torch.bfloat16)
            cache1 = torch.zeros(cache_shape, dtype=torch.bfloat16)
            cache2 = torch.zeros(cache_shape, dtype=torch.bfloat16)

            for host_tensor, device_tensor, dtype, layout in (
                (cache1, cache1_tt, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (cache2, cache2_tt, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (input1, input1_tt, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (input2, input2_tt, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (
                    torch.tensor([positions], dtype=torch.int32),
                    positions_tt,
                    ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
            ):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(host_tensor, dtype=dtype, layout=layout),
                    device_tensor,
                )

            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

            assert output1_tt.buffer_address() == cache1_tt.buffer_address()
            assert output2_tt.buffer_address() == cache2_tt.buffer_address()
            assert torch.equal(ttnn.to_torch(output1_tt), _apply_reference(cache1, input1, positions))
            assert torch.equal(ttnn.to_torch(output2_tt), _apply_reference(cache2, input2, positions))
    finally:
        ttnn.release_trace(device, trace_id)


def test_indexed_fused_update_cache_multi_tile_rows_and_worker_stride(device):
    torch.manual_seed(29)
    # 8 heads * 16 width tiles = 128 workers. This is larger than the 12x10
    # P150 compute grid and verifies that a core can own more than one worker;
    # the three physical pages also cover num_cores being much larger than pages.
    # Three source tiles force the two-slot scratch pipeline to wrap and wait
    # before reusing slot zero.
    cache_shape = (3, 8, 64, 512)
    source_rows = 96
    input_shape = (1, cache_shape[1], source_rows, cache_shape[3])
    cache1 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    cache2 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    input1 = torch.randn(input_shape, dtype=torch.bfloat16)
    input2 = torch.randn(input_shape, dtype=torch.bfloat16)
    positions = list(range(60, 60 + source_rows))
    # Reuse one destination at both source-tile boundaries. Writes tagged with
    # different transaction IDs must still preserve last-source-row-wins order.
    positions[32] = positions[31]
    positions[64] = positions[31]

    output1_tt, output2_tt = ttnn.experimental.indexed_fused_update_cache(
        _device_tensor(cache1, device),
        _device_tensor(input1, device),
        _device_tensor(cache2, device),
        _device_tensor(input2, device),
        _device_tensor(
            torch.tensor([positions], dtype=torch.int32),
            device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    )

    assert torch.equal(ttnn.to_torch(output1_tt), _apply_reference(cache1, input1, positions))
    assert torch.equal(ttnn.to_torch(output2_tt), _apply_reference(cache2, input2, positions))


def test_indexed_fused_update_cache_golden_function():
    cache1 = torch.zeros((2, 2, 32, 64), dtype=torch.bfloat16)
    cache2 = torch.zeros_like(cache1)
    input1 = torch.arange(2 * 4 * 64, dtype=torch.float32).reshape(1, 2, 4, 64).to(torch.bfloat16)
    input2 = -input1
    positions = torch.tensor([[0, 33, -1, 64]], dtype=torch.int32)

    golden_function = ttnn.get_golden_function(ttnn.experimental.indexed_fused_update_cache)
    output1, output2 = golden_function(cache1, input1, cache2, input2, positions)

    assert torch.equal(output1, _apply_reference(cache1, input1, positions.reshape(-1).tolist()))
    assert torch.equal(output2, _apply_reference(cache2, input2, positions.reshape(-1).tolist()))
    assert torch.count_nonzero(cache1) == 0
    assert torch.count_nonzero(cache2) == 0


@pytest.mark.parametrize(
    "invalid_case, expected_message",
    [
        ("cache_dtype", "cache_tensor1 must use BFLOAT16"),
        ("positions_dtype", "physical_update_idxs_tensor must use INT32"),
        ("positions_layout", "physical_update_idxs_tensor must use ROW_MAJOR layout"),
        ("positions_too_short", "physical_update_idxs_tensor has fewer entries"),
        ("cache_shape_mismatch", "cache_tensor1 and cache_tensor2 must have identical shapes"),
        ("input_shape_mismatch", "input_tensor1 and input_tensor2 must have identical shapes"),
        ("cache_requires_padding", "cache_tensor1 rows per physical page must be tile aligned"),
        ("too_many_source_rows", "input_tensor1 supports at most 256 source rows"),
        ("cache1_custom_tile", "cache_tensor1 must use the default non-transposed 32x32 tile"),
        ("cache2_custom_tile", "cache_tensor2 must use the default non-transposed 32x32 tile"),
        ("input1_custom_tile", "input_tensor1 must use the default non-transposed 32x32 tile"),
        ("input2_transposed_tile", "input_tensor2 must use the default non-transposed 32x32 tile"),
        ("positions_too_wide", "physical_update_idxs_tensor supports at most 256 entries"),
    ],
)
def test_indexed_fused_update_cache_validation(device, invalid_case, expected_message, expect_error):
    cache_shape = (2, 2, 64, 64)
    input_shape = (1, 2, 4, 64)
    cache1 = _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device)
    cache2 = _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device)
    input1 = _device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), device)
    input2 = _device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), device)
    positions = _device_tensor(
        torch.arange(input_shape[2], dtype=torch.int32).reshape(1, -1),
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    if invalid_case == "cache_dtype":
        cache1 = _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device, dtype=ttnn.bfloat8_b)
    elif invalid_case == "positions_dtype":
        positions = _device_tensor(
            torch.arange(input_shape[2], dtype=torch.int32).reshape(1, -1),
            device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    elif invalid_case == "positions_layout":
        positions = _device_tensor(
            torch.arange(input_shape[2], dtype=torch.int32).reshape(1, -1),
            device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
        )
    elif invalid_case == "positions_too_short":
        positions = _device_tensor(
            torch.arange(input_shape[2] - 1, dtype=torch.int32).reshape(1, -1),
            device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    elif invalid_case == "cache_shape_mismatch":
        cache2 = _device_tensor(torch.zeros((3, 2, 64, 64), dtype=torch.bfloat16), device)
    elif invalid_case == "input_shape_mismatch":
        input2 = _device_tensor(torch.zeros((1, 2, 5, 64), dtype=torch.bfloat16), device)
    elif invalid_case == "cache_requires_padding":
        cache1 = _device_tensor(torch.zeros((2, 2, 33, 64), dtype=torch.bfloat16), device)
        cache2 = _device_tensor(torch.zeros((2, 2, 33, 64), dtype=torch.bfloat16), device)
    elif invalid_case == "too_many_source_rows":
        input1 = _device_tensor(torch.zeros((1, 2, 257, 64), dtype=torch.bfloat16), device)
        input2 = _device_tensor(torch.zeros((1, 2, 257, 64), dtype=torch.bfloat16), device)
        positions = _device_tensor(
            torch.arange(257, dtype=torch.int32).reshape(1, -1),
            device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    elif invalid_case == "cache1_custom_tile":
        cache1 = _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device, tile=ttnn.Tile((16, 32)))
    elif invalid_case == "cache2_custom_tile":
        cache2 = _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device, tile=ttnn.Tile((16, 32)))
    elif invalid_case == "input1_custom_tile":
        input1 = _device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), device, tile=ttnn.Tile((16, 32)))
    elif invalid_case == "input2_transposed_tile":
        input2 = _device_tensor(
            torch.zeros(input_shape, dtype=torch.bfloat16),
            device,
            tile=ttnn.Tile((32, 32), transpose_tile=True),
        )
    elif invalid_case == "positions_too_wide":
        positions = _device_tensor(
            torch.arange(257, dtype=torch.int32).reshape(1, -1),
            device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    with expect_error(RuntimeError, expected_message):
        ttnn.experimental.indexed_fused_update_cache(cache1, input1, cache2, input2, positions)


@pytest.mark.parametrize(
    "first_name,second_name",
    [
        ("cache_tensor1", "input_tensor1"),
        ("cache_tensor1", "cache_tensor2"),
        ("cache_tensor1", "input_tensor2"),
        ("cache_tensor1", "physical_update_idxs_tensor"),
        ("input_tensor1", "cache_tensor2"),
        ("input_tensor1", "input_tensor2"),
        ("input_tensor1", "physical_update_idxs_tensor"),
        ("cache_tensor2", "input_tensor2"),
        ("cache_tensor2", "physical_update_idxs_tensor"),
        ("input_tensor2", "physical_update_idxs_tensor"),
    ],
)
def test_indexed_fused_update_cache_rejects_bound_tensor_alias_before_cache_insert(
    device, first_name, second_name, expect_error
):
    cache_shape = (2, 2, 64, 64)
    input_shape = (1, 2, 4, 64)
    tensors = {
        "cache_tensor1": _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device),
        "input_tensor1": _device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), device),
        "cache_tensor2": _device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), device),
        "input_tensor2": _device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), device),
        "physical_update_idxs_tensor": _device_tensor(
            torch.arange(input_shape[2], dtype=torch.int32).reshape(1, -1),
            device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    }
    tensors[second_name] = tensors[first_name]

    device.enable_program_cache()
    device.clear_program_cache()
    try:
        with expect_error(
            RuntimeError,
            f"{first_name} and {second_name} must not alias the same device buffer",
        ):
            ttnn.experimental.indexed_fused_update_cache(
                tensors["cache_tensor1"],
                tensors["input_tensor1"],
                tensors["cache_tensor2"],
                tensors["input_tensor2"],
                tensors["physical_update_idxs_tensor"],
            )
        assert device.num_program_cache_entries() == 0
    finally:
        device.disable_and_clear_program_cache()


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_indexed_fused_update_cache_replicated_mesh(mesh_device):
    torch.manual_seed(31)
    cache_shape = (2, 2, 64, 64)
    input_shape = (1, 2, 6, 64)
    cache1 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    cache2 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    input1 = torch.randn(input_shape, dtype=torch.bfloat16)
    input2 = torch.randn(input_shape, dtype=torch.bfloat16)
    positions = [0, 31, 32, 63, 64, -1]

    output1, output2 = ttnn.experimental.indexed_fused_update_cache(
        _replicated_device_tensor(cache1, mesh_device),
        _replicated_device_tensor(input1, mesh_device),
        _replicated_device_tensor(cache2, mesh_device),
        _replicated_device_tensor(input2, mesh_device),
        _replicated_device_tensor(
            torch.tensor([positions], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    )

    expected1 = _apply_reference(cache1, input1, positions)
    expected2 = _apply_reference(cache2, input2, positions)
    for shard in ttnn.get_device_tensors(output1.cpu()):
        assert torch.equal(ttnn.to_torch(shard), expected1)
    for shard in ttnn.get_device_tensors(output2.cpu()):
        assert torch.equal(ttnn.to_torch(shard), expected2)


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
@pytest.mark.parametrize(
    "shard_dim,cache_shape",
    [
        pytest.param(1, (2, 4, 64, 64), id="heads"),
        pytest.param(3, (2, 2, 64, 128), id="head_dim"),
    ],
)
def test_indexed_fused_update_cache_tensor_parallel_mesh(mesh_device, shard_dim, cache_shape):
    torch.manual_seed(37 + shard_dim)
    source_rows = 6
    input_shape = (1, cache_shape[1], source_rows, cache_shape[3])
    cache1 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    cache2 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    first_input1 = torch.randn(input_shape, dtype=torch.bfloat16)
    first_input2 = torch.randn(input_shape, dtype=torch.bfloat16)
    first_positions = [0, 31, 32, 63, 64, -1]

    cache1_tt = _sharded_device_tensor(cache1, mesh_device, shard_dim)
    cache2_tt = _sharded_device_tensor(cache2, mesh_device, shard_dim)
    mesh_device.enable_program_cache()
    mesh_device.clear_program_cache()

    try:
        output1, output2 = ttnn.experimental.indexed_fused_update_cache(
            cache1_tt,
            _sharded_device_tensor(first_input1, mesh_device, shard_dim),
            cache2_tt,
            _sharded_device_tensor(first_input2, mesh_device, shard_dim),
            _replicated_device_tensor(
                torch.tensor([first_positions], dtype=torch.int32),
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
        cache_entries = mesh_device.num_program_cache_entries()

        assert output1.tensor_topology() == cache1_tt.tensor_topology()
        assert output2.tensor_topology() == cache2_tt.tensor_topology()
        assert torch.equal(
            ttnn.to_torch(output1, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim)),
            _apply_reference(cache1, first_input1, first_positions),
        )
        assert torch.equal(
            ttnn.to_torch(output2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim)),
            _apply_reference(cache2, first_input2, first_positions),
        )

        # A cache hit must rebind new sharded inputs and replicated positions
        # while preserving the in-place cache tensors and their topology.
        second_input1 = torch.randn(input_shape, dtype=torch.bfloat16)
        second_input2 = torch.randn(input_shape, dtype=torch.bfloat16)
        second_positions = [2, 3, 65, 95, -1, cache_shape[0] * cache_shape[2]]
        output1, output2 = ttnn.experimental.indexed_fused_update_cache(
            cache1_tt,
            _sharded_device_tensor(second_input1, mesh_device, shard_dim),
            cache2_tt,
            _sharded_device_tensor(second_input2, mesh_device, shard_dim),
            _replicated_device_tensor(
                torch.tensor([second_positions], dtype=torch.int32),
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )

        assert cache_entries > 0
        assert mesh_device.num_program_cache_entries() == cache_entries
        expected1 = _apply_reference(
            _apply_reference(cache1, first_input1, first_positions), second_input1, second_positions
        )
        expected2 = _apply_reference(
            _apply_reference(cache2, first_input2, first_positions), second_input2, second_positions
        )
        assert output1.tensor_topology() == cache1_tt.tensor_topology()
        assert output2.tensor_topology() == cache2_tt.tensor_topology()
        assert torch.equal(
            ttnn.to_torch(output1, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim)), expected1
        )
        assert torch.equal(
            ttnn.to_torch(output2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim)), expected2
        )
    finally:
        mesh_device.disable_and_clear_program_cache()


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 2), id="2x2")], indirect=True)
def test_indexed_fused_update_cache_2d_tensor_parallel_mesh(mesh_device):
    torch.manual_seed(43)
    mesh_shape = (2, 2)
    shard_dims = (1, 3)
    cache_shape = (2, 4, 64, 128)
    input_shape = (1, 4, 6, 128)
    cache1 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    cache2 = torch.zeros(cache_shape, dtype=torch.bfloat16)
    input1 = torch.randn(input_shape, dtype=torch.bfloat16)
    input2 = torch.randn(input_shape, dtype=torch.bfloat16)
    positions = [0, 31, 32, 63, 64, -1]

    cache1_tt = _mapped_2d_device_tensor(cache1, mesh_device, mesh_shape, shard_dims)
    cache2_tt = _mapped_2d_device_tensor(cache2, mesh_device, mesh_shape, shard_dims)
    output1, output2 = ttnn.experimental.indexed_fused_update_cache(
        cache1_tt,
        _mapped_2d_device_tensor(input1, mesh_device, mesh_shape, shard_dims),
        cache2_tt,
        _mapped_2d_device_tensor(input2, mesh_device, mesh_shape, shard_dims),
        _mapped_2d_device_tensor(
            torch.tensor([positions], dtype=torch.int32),
            mesh_device,
            mesh_shape,
            (None, None),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    )

    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=shard_dims)
    assert output1.tensor_topology() == cache1_tt.tensor_topology()
    assert output2.tensor_topology() == cache2_tt.tensor_topology()
    assert torch.equal(ttnn.to_torch(output1, mesh_composer=composer), _apply_reference(cache1, input1, positions))
    assert torch.equal(ttnn.to_torch(output2, mesh_composer=composer), _apply_reference(cache2, input2, positions))


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_indexed_fused_update_cache_rejects_sharded_positions(mesh_device, expect_error):
    cache_shape = (2, 2, 64, 64)
    input_shape = (1, 2, 4, 64)
    positions = torch.arange(input_shape[2], dtype=torch.int32).reshape(1, -1)

    cache1 = _replicated_device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), mesh_device)
    cache2 = _replicated_device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), mesh_device)
    input1 = _replicated_device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), mesh_device)
    input2 = _replicated_device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), mesh_device)
    sharded_positions = ttnn.from_torch(
        positions,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    with expect_error(RuntimeError, "physical_update_idxs_tensor must be replicated across mesh axes"):
        ttnn.experimental.indexed_fused_update_cache(cache1, input1, cache2, input2, sharded_positions)


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_indexed_fused_update_cache_rejects_mesh_row_sharding(mesh_device, expect_error):
    cache_shape = (2, 2, 128, 64)
    input_shape = (1, 2, 8, 64)
    positions = torch.arange(input_shape[2], dtype=torch.int32).reshape(1, -1)

    with expect_error(RuntimeError, "cache page/row sharding requires physical-index remapping"):
        ttnn.experimental.indexed_fused_update_cache(
            _sharded_device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), mesh_device, 2),
            _sharded_device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), mesh_device, 2),
            _sharded_device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), mesh_device, 2),
            _sharded_device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), mesh_device, 2),
            _replicated_device_tensor(
                positions,
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )


@pytest.mark.parametrize("mesh_device", [2], indirect=True)
def test_indexed_fused_update_cache_rejects_mismatched_data_topology(mesh_device, expect_error):
    cache_shape = (2, 4, 64, 128)
    input_shape = (1, 4, 4, 128)
    positions = torch.arange(input_shape[2], dtype=torch.int32).reshape(1, -1)

    with expect_error(RuntimeError, "must use identical mesh topology"):
        ttnn.experimental.indexed_fused_update_cache(
            _sharded_device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), mesh_device, 1),
            _sharded_device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), mesh_device, 3),
            _sharded_device_tensor(torch.zeros(cache_shape, dtype=torch.bfloat16), mesh_device, 1),
            _sharded_device_tensor(torch.zeros(input_shape, dtype=torch.bfloat16), mesh_device, 3),
            _replicated_device_tensor(
                positions,
                mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
        )
