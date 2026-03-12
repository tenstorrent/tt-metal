# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.utils_for_testing import assert_with_pcc
import torch
import ttnn
import pytest
import numpy as np
from ttnn import ReplicateTensorToMesh, ShardTensorToMesh, ShardTensor2dMesh, ConcatMeshToTensor


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


TORCH_FLOAT_TYPES = [torch.float16, torch.float32, torch.float64]


def get_types_from_binding_framework():
    if hasattr(ttnn.DataType, "__entries"):
        # pybind
        ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]
    elif hasattr(ttnn.DataType, "_member_map_"):
        # nanobind
        ALL_TYPES = [dtype for _, dtype in ttnn.DataType._member_map_.items() if dtype != ttnn.DataType.INVALID]
    else:
        raise Exception("ttnn.DataType has unexpected way of holding values. Not matching pybind/nanobind.")

    return ALL_TYPES


ALL_TYPES = get_types_from_binding_framework()
FLOAT_TYPES = [dtype for dtype in ALL_TYPES if is_ttnn_float_type(dtype)]
TTNN_MUST_TILE_TYPES = [ttnn.bfloat8_b, ttnn.bfloat4_b]
NUMPY_FLOAT_TYPES = [np.float16, np.float32, np.float64]


def get_expected_conversion_pcc(ttnn_dtype, other_dtype):
    ttnn_is_float = ttnn_dtype in [ttnn.float32, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b]
    other_is_float = other_dtype in TORCH_FLOAT_TYPES or other_dtype in NUMPY_FLOAT_TYPES

    if (ttnn_dtype == ttnn.float32 and other_dtype == torch.float32) or (
        ttnn_dtype == ttnn.int32 and other_dtype == torch.int32
    ):
        return 1

    elif ttnn_dtype == ttnn.bfloat4_b:
        return 0.960

    elif ttnn_is_float != other_is_float:
        return 0.98

    elif other_dtype == torch.bfloat16:
        if ttnn_dtype == ttnn.bfloat16 or ttnn_dtype == ttnn.bfloat8_b:
            return 0.9999
        elif ttnn_dtype == ttnn.bfloat4_b:
            return 0.989
        else:
            return 0.999

    else:
        return 0.999


def create_from_torch_test_tensors(
    device,
    shape,
    ttnn_dtype,
    torch_dtype,
    ttnn_layout,
    convert_with_device,
    min_range=0,
    max_range=100,
    memory_config=None,
):
    torch.manual_seed(0)
    if torch_dtype in TORCH_FLOAT_TYPES:
        torch_input_tensor = torch.rand(shape, dtype=torch_dtype) * max_range

    else:
        torch_input_tensor = torch.randint(min_range, max_range, shape, dtype=torch_dtype)

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device if convert_with_device else None,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT if (ttnn_dtype in TTNN_MUST_TILE_TYPES) else ttnn_layout,
        memory_config=memory_config,
    )

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting torch tensor {torch_input_tensor.dtype}"

    return torch_input_tensor, ttnn_result_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.int32,
    ],
)
@pytest.mark.parametrize(
    "torch_dtype",
    [
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("convert_with_device", [True, False])
def test_from_torch_conversion(device, shape, ttnn_dtype, torch_dtype, ttnn_layout, convert_with_device):
    torch.manual_seed(205)
    torch_input_tensor, ttnn_result_tensor = create_from_torch_test_tensors(
        device,
        shape,
        ttnn_dtype,
        torch_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=0,
        max_range=10 if torch_dtype in TORCH_FLOAT_TYPES else 100,
    )

    assert_with_pcc(
        expected_pytorch_result=torch_input_tensor,
        actual_pytorch_result=ttnn_result_tensor.cpu().to_torch(),
        pcc=get_expected_conversion_pcc(ttnn_dtype, torch_dtype),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.int32,
    ],
)
@pytest.mark.parametrize(
    "torch_dtype",
    [
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("convert_with_device", [True, False])
def test_to_torch_conversion(device, shape, ttnn_dtype, torch_dtype, ttnn_layout, convert_with_device):
    ttnn_dtype_has_random = ttnn_dtype not in [ttnn.uint8, ttnn.int32]
    if ttnn_dtype_has_random:
        for store_input_on_device in [True, False]:
            ttnn_input_tensor = ttnn.rand(
                shape,
                dtype=ttnn_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT if (ttnn_dtype in TTNN_MUST_TILE_TYPES) else ttnn_layout,
            )

            if not store_input_on_device:
                ttnn_input_tensor = ttnn.from_device(ttnn_input_tensor)

            torch_result_tensor = ttnn.to_torch(
                ttnn_input_tensor, dtype=torch_dtype, device=device if convert_with_device else None
            )
            assert (
                torch_result_tensor.dtype == torch_dtype
            ), f"Expected result {torch_dtype}, got result tensor {torch_result_tensor.dtype} when converting TTNN tensor {ttnn_input_tensor.dtype}"

        assert_with_pcc(
            expected_pytorch_result=torch_result_tensor,
            actual_pytorch_result=ttnn_input_tensor.cpu().to_torch(),
            pcc=get_expected_conversion_pcc(ttnn_dtype, torch_dtype),
        )


@pytest.mark.parametrize("seed", list(range(6)))
@pytest.mark.parametrize(
    "shape,ttnn_dtype,torch_dtype,ttnn_layout,convert_with_device,value_ranges,pcc_override,memory_config",
    [
        ((4, 4), ttnn.float32, torch.float64, ttnn.TILE_LAYOUT, True, (0, 100), None, None),
        ((32, 32), ttnn.bfloat16, torch.int64, ttnn.ROW_MAJOR_LAYOUT, False, (0, 255), None, None),
        ((32, 32, 64, 64), ttnn.bfloat8_b, torch.float16, ttnn.TILE_LAYOUT, True, (0, 127), None, None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (0, 123123), 1, None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (-2147482786, 2147482213), 1, None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.ROW_MAJOR_LAYOUT, True, (-2147482786, 2147482213), 1, None),
        ((1, 1, 32, 1024), ttnn.float32, torch.float32, ttnn.TILE_LAYOUT, True, (0, 123123), 1, None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), 1, None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), 1, ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_from_torch_conversion_with_fixed_edge_case_params(
    seed,
    device,
    shape,
    ttnn_dtype,
    torch_dtype,
    ttnn_layout,
    convert_with_device,
    value_ranges,
    pcc_override,
    memory_config,
):
    """
    Test `from_torch` conversion with a fixed set of parameters for various edge cases
    """
    torch.manual_seed(seed)
    torch_input_tensor, ttnn_result_tensor = create_from_torch_test_tensors(
        device,
        shape,
        ttnn_dtype,
        torch_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=value_ranges[0],
        max_range=value_ranges[1],
        memory_config=memory_config,
    )

    assert_with_pcc(
        expected_pytorch_result=torch_input_tensor,
        actual_pytorch_result=ttnn_result_tensor.cpu().to_torch(),
        pcc=pcc_override or get_expected_conversion_pcc(ttnn_dtype, torch_dtype),
    )


@pytest.mark.parametrize("seed", list(range(6)))
@pytest.mark.parametrize(
    "shape,ttnn_dtype,torch_dtype,ttnn_layout,convert_with_device,value_ranges,memory_config",
    [
        ((4, 4), ttnn.float32, torch.float64, ttnn.TILE_LAYOUT, True, (0, 100), None),
        ((32, 32), ttnn.bfloat16, torch.int64, ttnn.ROW_MAJOR_LAYOUT, False, (0, 255), None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (0, 123123), None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.TILE_LAYOUT, True, (-2147482786, 2147482213), None),
        ((32, 32, 64, 64), ttnn.int32, torch.int32, ttnn.ROW_MAJOR_LAYOUT, True, (-2147482786, 2147482213), None),
        ((1, 1, 32, 1024), ttnn.float32, torch.float32, ttnn.TILE_LAYOUT, True, (0, 123123), None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), None),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((2, 256, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((2, 64, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), None),
        ((2, 64, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), None),
        ((2, 64, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((2, 64, 64, 64), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((4, 4, 4, 4), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), None),
        ((4, 4, 4, 4), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), None),
        ((4, 4, 4, 4), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, True, (-100, 100), ttnn.L1_MEMORY_CONFIG),
        ((4, 4, 4, 4), ttnn.bfloat16, torch.int64, ttnn.TILE_LAYOUT, False, (-100, 100), ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_from_torch_conversion_close(
    seed, device, shape, ttnn_dtype, torch_dtype, ttnn_layout, convert_with_device, value_ranges, memory_config
):
    """
    Test `from_torch` conversion with a fixed set of parameters for various edge cases
    """
    torch.manual_seed(seed)
    torch_tensor, ttnn_tensor = create_from_torch_test_tensors(
        device,
        shape,
        ttnn_dtype,
        torch_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=value_ranges[0],
        max_range=value_ranges[1],
        memory_config=memory_config,
    )

    torch.testing.assert_close(torch.Tensor(torch_tensor.tolist()), torch.Tensor(ttnn_tensor.to_list()))


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        return torch.rand(shape, dtype=torch.bfloat16)
    raise ValueError(f"Unsupported dtype for random_torch_tensor: {dtype}")


@pytest.mark.parametrize(
    "shape", [[1, 1, 32, 32], [2, 2, 32, 32], [32, 32, 32, 32], [1, 1, 64, 64], [2, 2, 64, 64], [32, 32, 64, 64]]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_dtype_conversion_pcc(device, shape, dtype):
    torch.manual_seed(2005)
    torch_tensor = random_torch_tensor(dtype, shape)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    assert_with_pcc(torch_tensor, torch.Tensor(input_tensor.to_list()), 0.999999)


def create_from_numpy_test_tensors(
    device,
    shape,
    ttnn_dtype,
    numpy_dtype,
    ttnn_layout,
    convert_with_device,
    min_range=0,
    max_range=100,
    memory_config=None,
):
    if numpy_dtype in NUMPY_FLOAT_TYPES:
        numpy_input_tensor = np.random.rand(*shape).astype(numpy_dtype) * max_range
    else:
        numpy_input_tensor = np.random.randint(min_range, max_range, shape, dtype=numpy_dtype)

    ttnn_result_tensor = ttnn.from_torch(
        numpy_input_tensor,
        device=device if convert_with_device else None,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT if (ttnn_dtype in TTNN_MUST_TILE_TYPES) else ttnn_layout,
        memory_config=memory_config,
    )

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting numpy tensor {numpy_input_tensor.dtype}"

    return numpy_input_tensor, ttnn_result_tensor


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.float32,
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.int32,
    ],
)
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("convert_with_device", [True, False])
def test_from_numpy_conversion(device, shape, ttnn_dtype, numpy_dtype, ttnn_layout, convert_with_device):
    np.random.seed(205)
    numpy_input_tensor, ttnn_result_tensor = create_from_numpy_test_tensors(
        device,
        shape,
        ttnn_dtype,
        numpy_dtype,
        ttnn_layout,
        convert_with_device,
        min_range=0,
        max_range=10 if numpy_dtype in NUMPY_FLOAT_TYPES else 100,
    )

    # Convert numpy tensor to torch for comparison
    torch_input_tensor = torch.from_numpy(numpy_input_tensor)
    torch_computed = ttnn_result_tensor.cpu().to_torch()

    assert isinstance(torch_input_tensor, torch.Tensor)
    assert isinstance(torch_computed, torch.Tensor)

    assert_with_pcc(
        expected_pytorch_result=torch_input_tensor.to(torch.float64),
        actual_pytorch_result=torch_computed.to(torch.float64),
        pcc=get_expected_conversion_pcc(ttnn_dtype, numpy_dtype),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.uint16,
        ttnn.uint32,
    ],
)
@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_torch_conversion_unsigned_edge_cases_random(device, shape, ttnn_dtype, ttnn_layout, seed):
    torch.manual_seed(seed)

    if ttnn_dtype == ttnn.uint16:
        low = np.iinfo(np.uint16).min
        high = np.iinfo(np.uint16).max

    elif ttnn_dtype == ttnn.uint32:
        low = np.iinfo(np.uint32).min
        high = np.iinfo(np.uint32).max

    ttnn_input_tensor = ttnn.rand(
        shape,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn_layout,
        low=low,
        high=high,
    )

    torch_result_tensor: torch.Tensor = ttnn.to_torch(ttnn_input_tensor)

    torch.testing.assert_close(torch.tensor(torch_result_tensor.tolist()), torch.tensor(ttnn_input_tensor.to_list()))


@pytest.mark.parametrize(
    "tensor_data,ttnn_dtype,torch_input_type",
    [
        ([np.iinfo(np.uint16).max], ttnn.uint16, torch.uint16),
        ([np.iinfo(np.uint16).min], ttnn.uint16, torch.uint16),
        ([np.iinfo(np.uint32).max], ttnn.uint32, torch.uint32),
        ([np.iinfo(np.uint32).min], ttnn.uint32, torch.uint32),
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("with_device", [True, False])
def test_torch_conversion_unsigned_edge_cases(
    device, tensor_data, ttnn_dtype, torch_input_type, ttnn_layout, with_device
):
    torch_input_tensor = torch.tensor(tensor_data, dtype=torch_input_type)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn_dtype,
        layout=ttnn_layout,
        device=device if with_device else None,
    )

    torch_result_tensor: torch.Tensor = ttnn.to_torch(ttnn_input_tensor)

    torch.testing.assert_close(torch.tensor(torch_input_tensor.tolist()), torch.tensor(ttnn_input_tensor.to_list()))
    torch.testing.assert_close(torch.tensor(torch_result_tensor.tolist()), torch.tensor(ttnn_input_tensor.to_list()))
    torch.testing.assert_close(torch_input_tensor, torch_result_tensor)


@pytest.mark.parametrize(
    "tensor_data,ttnn_dtype,numpy_input_type",
    [
        ([np.iinfo(np.uint16).max], ttnn.uint16, np.uint16),
        ([np.iinfo(np.uint16).min], ttnn.uint16, np.uint16),
        ([np.iinfo(np.uint32).max], ttnn.uint32, np.uint32),
        ([np.iinfo(np.uint32).min], ttnn.uint32, np.uint32),
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("with_device", [True, False])
def test_numpy_conversion_unsigned_edge_cases_fixed(
    device, tensor_data, ttnn_dtype, numpy_input_type, ttnn_layout, with_device
):
    numpy_input_tensor = np.array(tensor_data, dtype=numpy_input_type)
    ttnn_input_tensor = ttnn.from_torch(
        numpy_input_tensor,
        dtype=ttnn_dtype,
        layout=ttnn_layout,
        device=device if with_device else None,
    )
    numpy_result_tensor = ttnn_input_tensor.cpu().to_numpy()
    np.testing.assert_allclose(np.array(numpy_input_tensor.tolist()), np.array(ttnn_input_tensor.to_list()))
    np.testing.assert_allclose(np.array(numpy_result_tensor.tolist()), np.array(ttnn_input_tensor.to_list()))
    np.testing.assert_allclose(numpy_input_tensor, numpy_result_tensor)


@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,shape,shard_shape,num_cores,shard_strategy",
    [
        (torch.bfloat16, ttnn.bfloat16, [1, 7, 1, 128], [7 * 32, 32], 4, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (torch.float32, ttnn.bfloat16, [1, 7, 1, 128], [7 * 32, 32], 4, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (torch.bfloat16, ttnn.bfloat16, [1, 3, 1, 64], [3 * 32, 32], 2, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        (torch.bfloat16, ttnn.bfloat16, [2, 7, 64, 128], [2 * 32 * 2, 128], 7, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
)
def test_from_torch_sharded_tile_layout_non_tile_aligned_height(
    device, torch_dtype, ttnn_dtype, shape, shard_shape, num_cores, shard_strategy
):
    """
    Regression test: from_torch with TILE_LAYOUT and a sharded memory config
    whose shard dimensions are computed for the tile-padded physical shape.

    Previously crashed with TT_FATAL in TensorSpec validation because the
    internal ROW_MAJOR TensorSpec used for the can_construct_on_device check
    had a physical height that did not match the shard height (which was
    designed for the TILE-padded shape).
    """
    torch.manual_seed(0)
    torch_tensor = torch.rand(shape, dtype=torch_dtype)

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, num_cores - 1))})
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=core_grid,
        strategy=(
            ttnn.ShardStrategy.WIDTH
            if shard_strategy == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            else ttnn.ShardStrategy.HEIGHT
        ),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    ttnn_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )

    assert ttnn_tensor.dtype == ttnn_dtype
    assert ttnn_tensor.layout == ttnn.TILE_LAYOUT
    assert ttnn_tensor.memory_config().memory_layout == shard_strategy


@pytest.mark.parametrize("fast_approx", [True, False])
@pytest.mark.parametrize("use_mesh_mapper", [True, False])
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.bfloat16),
    ],
)
def test_from_torch_fast_approx_preserves_sharded_memory_config(
    device, fast_approx, use_mesh_mapper, torch_dtype, ttnn_dtype
):
    """
    Validate that fast_approx=True preserves DRAM-sharded memory config.

    When fast_approx is used with a mesh_mapper, the tensor is initially created
    on the device with a default (interleaved) memory config and then converted
    in-place. This test ensures the final tensor has the requested sharded config.
    """
    dram_cores = device.dram_grid_size().x
    width = dram_cores * ttnn.TILE_SIZE
    shape = (1, 1, 32, width)

    torch.manual_seed(42)
    torch_tensor = torch.rand(shape, dtype=torch_dtype)

    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_cores - 1, 0),
            )
        }
    )
    shard_spec = ttnn.ShardSpec(
        dram_grid,
        (shape[-2], shape[-1] // dram_cores),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )

    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if use_mesh_mapper else None

    ttnn_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
        fast_approx=fast_approx,
    )

    assert (
        ttnn_tensor.memory_config() == memory_config
    ), f"Memory config mismatch: expected {memory_config}, got {ttnn_tensor.memory_config()}"
    assert ttnn_tensor.dtype == ttnn_dtype
    assert ttnn_tensor.layout == ttnn.TILE_LAYOUT


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["t3k"], indirect=True)
@pytest.mark.parametrize(
    "mapper_type,ttnn_dtype,memory_config_type,fast_approx",
    [
        ("replicate", ttnn.bfloat16, "DRAM", True),
        ("replicate", ttnn.bfloat16, "DRAM", False),
        ("replicate", ttnn.bfloat16, "L1", True),
        ("replicate", ttnn.bfloat8_b, "DRAM", True),
        ("shard_1d", ttnn.bfloat16, "DRAM", True),
        ("shard_1d", ttnn.bfloat16, "L1", True),
        ("shard_2d", ttnn.bfloat16, "DRAM", True),
        ("shard_2d", ttnn.bfloat16, "DRAM", False),
        ("shard_2d", ttnn.bfloat16, "L1", True),
        ("shard_2d", ttnn.bfloat8_b, "DRAM", True),
        ("shard_2d", ttnn.bfloat16, "DRAM_SHARDED", True),
        ("shard_2d", ttnn.bfloat8_b, "DRAM_SHARDED", True),
        ("replicate", ttnn.bfloat16, "DRAM_SHARDED", True),
        ("replicate", ttnn.bfloat8_b, "DRAM_SHARDED", True),
    ],
)
def test_from_torch_mesh_mapper_preserves_memory_config(
    mesh_device, mapper_type, ttnn_dtype, memory_config_type, fast_approx
):
    """
    Verify that from_torch with mesh_mapper on a mesh device preserves the
    requested memory_config, dtype, and layout.

    Reproduces a regression where the memory_config on the output tensor
    does not match what was passed in, observed in deepseek_v3 weight
    loading with ShardTensor2dMesh + DRAM-sharded memory config +
    fast_approx=True.
    """
    torch.manual_seed(42)
    mesh_shape = tuple(mesh_device.shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    layout = ttnn.TILE_LAYOUT

    if memory_config_type == "DRAM_SHARDED":
        dram_cores = mesh_device.dram_grid_size().x
        per_device_width = dram_cores * ttnn.TILE_SIZE
    else:
        per_device_width = 128

    per_device_height = 32

    if mapper_type == "replicate":
        shape = (1, 1, per_device_height, per_device_width)
        mesh_mapper = ReplicateTensorToMesh(mesh_device)
    elif mapper_type == "shard_1d":
        shape = (num_devices, 1, per_device_height, per_device_width)
        mesh_mapper = ShardTensorToMesh(mesh_device, dim=0)
    elif mapper_type == "shard_2d":
        shape = (mesh_shape[0], mesh_shape[1], per_device_height, per_device_width)
        mesh_mapper = ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(0, 1))

    if memory_config_type == "DRAM":
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    elif memory_config_type == "L1":
        memory_config = ttnn.L1_MEMORY_CONFIG
    elif memory_config_type == "DRAM_SHARDED":
        dram_cores = mesh_device.dram_grid_size().x
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_cores - 1, 0),
                )
            }
        )
        shard_spec = ttnn.ShardSpec(
            dram_grid,
            (per_device_height, per_device_width // dram_cores),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            shard_spec,
        )

    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
        fast_approx=fast_approx,
    )

    assert (
        ttnn_tensor.memory_config() == memory_config
    ), f"Memory config mismatch: expected {memory_config}, got {ttnn_tensor.memory_config()}"
    assert ttnn_tensor.dtype == ttnn_dtype, f"Dtype mismatch: expected {ttnn_dtype}, got {ttnn_tensor.dtype}"
    assert ttnn_tensor.layout == layout, f"Layout mismatch: expected {layout}, got {ttnn_tensor.layout}"
