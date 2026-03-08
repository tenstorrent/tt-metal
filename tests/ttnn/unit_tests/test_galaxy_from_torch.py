# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.utils_for_testing import assert_with_pcc
import torch
import ttnn
import pytest
import numpy as np


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


def generate_bfloat4_b_exact_tensor(shape, seed=0):
    """
    Generate a float32 torch tensor whose values survive a round-trip through
    bfloat4_b without any precision loss.

    bfloat4_b is a block floating-point format where every 16-element block
    (one face row in tile layout) shares a single 8-bit exponent, and each
    element stores 1 sign bit + 3 mantissa bits (with an implicit leading 1).

    To guarantee exact round-trip, this function ensures:
     - All 16 elements in each block share the same power-of-two exponent,
        so the shared exponent equals every element's exponent (no alignment
        shift, no mantissa bit loss).
     - Each element's mantissa is one of the 8 exactly representable values:
        {1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875}
        which correspond to the 3-bit patterns 000..111 with hidden bit.

    The last two dimensions of *shape* must be multiples of 32 (tile layout).
    """
    torch.manual_seed(seed)

    EXACT_MANTISSAS = torch.tensor([1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875], dtype=torch.float32)
    FACE_ROW_SIZE = 16

    assert len(shape) >= 2, "Shape must have at least 2 dimensions"
    H, W = shape[-2], shape[-1]
    assert H % 32 == 0 and W % 32 == 0, f"Last two dims must be multiples of 32 for tile layout, got ({H}, {W})"

    total_elements = 1
    for d in shape:
        total_elements *= d
    total_rows = total_elements // W
    num_blocks_per_row = W // FACE_ROW_SIZE
    total_blocks = total_rows * num_blocks_per_row

    exponents = torch.randint(-4, 5, (total_blocks, 1), dtype=torch.float32)
    mantissa_indices = torch.randint(0, 8, (total_blocks, FACE_ROW_SIZE))
    mantissas = EXACT_MANTISSAS[mantissa_indices]
    signs = torch.where(
        torch.randint(0, 2, (total_blocks, FACE_ROW_SIZE)).bool(),
        torch.ones(1, dtype=torch.float32),
        -torch.ones(1, dtype=torch.float32),
    )

    values = signs * torch.pow(2.0, exponents) * mantissas
    return values.reshape(shape)


def quantize_to_bf4(tensor, exp_bits=2, mant_bits=1):
    """
    Simulates a 4-bit float roundtrip.
    Default: 1 sign bit, 2 exponent bits, 1 mantissa bit (E2M1).
    """
    # 1. Capture the sign
    sign = torch.sign(tensor)
    abs_tensor = torch.abs(tensor)

    # 2. Handle zeros to avoid log errors
    abs_tensor[abs_tensor == 0] = 1e-8

    # 3. Log-scale to find the exponent
    # bfloat4 has a very narrow range (usually 2^exponent)
    exponent = torch.floor(torch.log2(abs_tensor))

    # Clip exponent to fit in 2 bits (e.g., range -1 to 2)
    max_exp = 2 ** (exp_bits - 1)
    exponent = torch.clamp(exponent, -max_exp, max_exp)

    # 4. Quantize the mantissa
    # With 1 mantissa bit, we only have two levels: 1.0 and 1.5
    mantissa = abs_tensor / (2**exponent)
    mantissa = torch.round(mantissa * (2**mant_bits)) / (2**mant_bits)
    mantissa = torch.clamp(mantissa, 1.0, 2.0 - (1 / (2**mant_bits)))

    # 5. Reconstruct the "crushed" value
    bf4_simulated = sign * mantissa * (2**exponent)

    return bf4_simulated.to(torch.float32)


@pytest.mark.parametrize("shape", [(4, 18432, 7168)])
@pytest.mark.parametrize("shard_shape", [(2304, 608)])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat4_b])
@pytest.mark.parametrize("torch_dtype", [torch.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])  # ttnn.ROW_MAJOR_LAYOUT
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_from_torch_conversion_deep_seek(mesh_device, shape, shard_shape, ttnn_dtype, torch_dtype, layout):
    torch.manual_seed(0)
    # torch.rand(shape, dtype=torch_dtype)
    torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
    torch_input_tensor = quantize_to_bf4(torch_input_tensor)

    # assert mesh_device.shape == (4, 8), f"Mesh device shape must be (4, 8), got {mesh_device.shape}"

    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        dtype=ttnn_dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=mesh_device.shape),
    )

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting torch tensor {torch_input_tensor.dtype}"

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_device.shape)
    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor, mesh_composer=mesh_composer)
    # torch.testing.assert_close(torch_input_tensor, torch_result_tensor)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


@pytest.mark.parametrize("shape", [(1, 7168, 2304)])
@pytest.mark.parametrize("shard_shape", [(7168, 192)])
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat4_b])
@pytest.mark.parametrize("torch_dtype", [torch.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])  # ttnn.ROW_MAJOR_LAYOUT
def test_from_torch_conversion_deep_seek_mc(device, shape, shard_shape, ttnn_dtype, torch_dtype, layout):
    torch.manual_seed(0)
    # torch.rand(shape, dtype=torch_dtype)
    torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
    torch_input_tensor = quantize_to_bf4(torch_input_tensor)

    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))]
    )  # TODO: Choose DRAM cores
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,  # ttnn.DRAM_MEMORY_CONFIG
    )

    print("to_layout")
    ttnn_result_tensor = ttnn.to_layout(ttnn_result_tensor, layout)
    ttnn.synchronize_device(device)

    print("typecast")
    ttnn_result_tensor = ttnn.typecast(ttnn_result_tensor, ttnn_dtype)
    ttnn.synchronize_device(device)

    assert (
        ttnn_result_tensor.dtype == ttnn_dtype
    ), f"Expected result {ttnn_dtype}, got result tensor {ttnn_result_tensor.dtype} when converting torch tensor {torch_input_tensor.dtype}"

    print("to_torch")
    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor)
    torch.testing.assert_close(torch_input_tensor, torch_result_tensor)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


DRAM_CORE_GRID_12 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))])


@pytest.mark.parametrize(
    "shape,shard_shape,memory_layout",
    [
        # wq_b: WIDTH_SHARDED, shard [1536, 256], 12 DRAM banks, shard_dims=(0, -1)
        ((1, 1536, 3072), (1536, 256), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        # wo: WIDTH_SHARDED, shard [16384, 96], 12 DRAM banks, shard_dims=(0, -1)
        ((1, 16384, 1152), (16384, 96), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        # wq_kv_a: WIDTH_SHARDED, shard [896, 192], 12 DRAM banks, shard_dims=(0, -2)
        ((1, 896, 2304), (896, 192), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        # wkv_b1: HEIGHT_SHARDED, shard [256, 512], 12 DRAM banks, shard_dims=(0, -3)
        ((1, 16, 128, 512), (256, 512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        # wkv_b2: HEIGHT_SHARDED, shard [5632, 128], 12 DRAM banks, shard_dims=(0, None)
        # ((4, 128,512, 128), (5632, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
    ids=["wq_b", "wo", "wq_kv_a", "wkv_b1"],  # , "wkv_b2"
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b])
def test_from_torch_deep_seek_sharded_weights_single_device(device, shape, shard_shape, memory_layout, ttnn_dtype):
    torch.manual_seed(0)
    if ttnn_dtype == ttnn.bfloat4_b or ttnn_dtype == ttnn.bfloat8_b:
        torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
        torch_input_tensor = quantize_to_bf4(torch_input_tensor)
    else:
        torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)  # float32?

    memory_config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_CORE_GRID_12, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )

    assert ttnn_result_tensor.dtype == ttnn_dtype, f"Expected {ttnn_dtype}, got {ttnn_result_tensor.dtype}"

    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


# TODO: Increase shape size according to shard_dims
@pytest.mark.parametrize(
    "shape,shard_shape,memory_layout,shard_dims",
    [
        ([1, 1536, 3072], (1536, 256), ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, -1)),
        ([1, 16384, 1152], (16384, 96), ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, -1)),
        ([1, 896, 2304], (896, 192), ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, -2)),
        ([1, 16, 128, 512], (256, 512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, -3)),
        # ([1, 128,512, 128], (5632, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, None)),
    ],
    ids=["wq_b", "wo", "wq_kv_a", "wkv_b1"],  # , "wkv_b2"
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_from_torch_deep_seek_sharded_weights_galaxy(
    mesh_device, shape, shard_shape, memory_layout, shard_dims, ttnn_dtype
):
    torch.manual_seed(0)
    if shard_dims[0] is not None:
        shape[shard_dims[0]] = shape[shard_dims[0]] * mesh_device.shape[0]

    if shard_dims[1] is not None:
        shape[shard_dims[1]] = shape[shard_dims[1]] * mesh_device.shape[1]

    print(f"[PY_DEBUG]shard_shape: {shard_shape}")
    print(f"[PY_DEBUG]mesh_device.shape: {mesh_device.shape}")
    print(f"[PY_DEBUG]shape: {shape}")
    if ttnn_dtype == ttnn.bfloat4_b or ttnn_dtype == ttnn.bfloat8_b:
        torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
        torch_input_tensor = quantize_to_bf4(torch_input_tensor)
    else:
        torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)  # float32?

    memory_config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_CORE_GRID_12, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    mesh_shape = (mesh_device.shape[0], mesh_device.shape[1])

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )

    assert ttnn_result_tensor.dtype == ttnn_dtype, f"Expected {ttnn_dtype}, got {ttnn_result_tensor.dtype}"

    # mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_shape)

    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[x for x in shard_dims if x is not None],
            mesh_shape_override=ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1]),
        ),
    )

    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor, mesh_composer=mesh_composer)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


# # TODO: Increase shape size according to shard_dims
# @pytest.mark.parametrize(
#     "shape,shard_shape,memory_layout,shard_dims",
#     [
#         ([1, 67584, 128], (5632, 128), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (0, None)),
#     ],
#     ids=["wkv_b2"],
# )
# @pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b])
# @pytest.mark.parametrize("mesh_device", [ (4, 8) ], indirect=True)
# def test_from_torch_deep_seek_sharded_replicated_galaxy(mesh_device, shape, shard_shape, memory_layout, shard_dims, ttnn_dtype):
#     torch.manual_seed(0)
#     if shard_dims[0] is not None and shard_dims[1] is not None:
#         shape[shard_dims[0]] = shape[shard_dims[0]] * mesh_device.shape[0]
#         shape[shard_dims[1]] = shape[shard_dims[1]] * mesh_device.shape[1]

#     print(f"[PY_DEBUG]shard_shape: {shard_shape}")
#     print(f"[PY_DEBUG]mesh_device.shape: {mesh_device.shape}")
#     print(f"[PY_DEBUG]shape: {shape}")
#     if ttnn_dtype == ttnn.bfloat4_b or ttnn_dtype == ttnn.bfloat8_b:
#         torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
#         torch_input_tensor = quantize_to_bf4(torch_input_tensor)
#     else:
#         torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)  # float32?

#     memory_config = ttnn.MemoryConfig(
#         memory_layout,
#         ttnn.BufferType.DRAM,
#         ttnn.ShardSpec(DRAM_CORE_GRID_12, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
#     )

#     mesh_shape = (mesh_device.shape[0], mesh_device.shape[1])

#     ttnn_result_tensor = ttnn.from_torch(
#         torch_input_tensor,
#         device=mesh_device,
#         dtype=ttnn_dtype,
#         layout=ttnn.TILE_LAYOUT,
#         memory_config=memory_config,
#         mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
#     )

#     assert ttnn_result_tensor.dtype == ttnn_dtype, f"Expected {ttnn_dtype}, got {ttnn_result_tensor.dtype}"

#     mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_shape)
#     torch_result_tensor = ttnn.to_torch(ttnn_result_tensor, mesh_composer=mesh_composer)
#     assert torch.equal(torch_input_tensor, torch_result_tensor)


# @pytest.mark.parametrize("shape", [(1, 1, 224, 32)])
# def test_from_torch_deep_seek_interleaved(device, shape):
#     torch.manual_seed(0)
#     torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

#     memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

#     ttnn_result_tensor = ttnn.from_torch(
#         torch_input_tensor,
#         device=device,
#         dtype=ttnn.bfloat16,
#         layout=ttnn.ROW_MAJOR_LAYOUT,
#         memory_config=memory_config,
#     )

#     assert ttnn_result_tensor.dtype == ttnn.bfloat16, f"Expected {ttnn.bfloat16}, got {ttnn_result_tensor.dtype}"

#     torch_result_tensor = ttnn.to_torch(ttnn_result_tensor)
#     assert torch.equal(torch_input_tensor, torch_result_tensor)


### TOP PRIO
@pytest.mark.parametrize(
    "shape",
    [
        ([1, 256, 7168, 2048]),
        # ([1, 256, 2048, 7168]),
    ],
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat4_b])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_from_torch_deep_seek_interleaved_weights_galaxy(mesh_device, shape, ttnn_dtype):
    torch.manual_seed(0)
    torch_input_tensor = generate_bfloat4_b_exact_tensor(shape)
    torch_input_tensor = quantize_to_bf4(torch_input_tensor)

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    ttnn.synchronize_device(mesh_device)
    assert ttnn_result_tensor.dtype == ttnn_dtype, f"Expected {ttnn_dtype}, got {ttnn_result_tensor.dtype}"
    assert (
        ttnn_result_tensor.storage_type() == ttnn.StorageType.DEVICE
    ), f"Expected {ttnn.StorageType.DEVICE}, got {ttnn_result_tensor.storage_type()}"

    mesh_composer = ttnn.concat_mesh_to_tensor_composer(mesh_device, dim=1)

    print("ttnn_result_tensor storage_type: ", ttnn_result_tensor.storage_type())
    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor, mesh_composer=mesh_composer)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


@pytest.mark.parametrize(
    "shape,shard_shape,memory_layout,shard_dims",
    [
        ([4, 2048, 7168], (256, 608), ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, 1)),
    ],
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat4_b])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_from_torch_deep_seek_padded_4b_weights_galaxy(
    mesh_device, shape, shard_shape, memory_layout, shard_dims, ttnn_dtype
):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    memory_config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_CORE_GRID_12, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    mesh_shape = (mesh_device.shape[0], mesh_device.shape[1])

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
    )

    assert ttnn_result_tensor.dtype == ttnn_dtype, f"Expected {ttnn_dtype}, got {ttnn_result_tensor.dtype}"

    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[x for x in shard_dims if x is not None],
            mesh_shape_override=ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1]),
        ),
    )

    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor, mesh_composer=mesh_composer)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


# Replicate at index 0 (distributed_tensor.cpp:167)
# Shard dimension: -1 at index 1 (distributed_tensor.cpp:165)
@pytest.mark.parametrize(
    "shape,shard_shape,memory_layout,shard_dims",
    [
        ([1, 16384, 7168], (256, 608), ttnn.TensorMemoryLayout.WIDTH_SHARDED, (0, 1)),
    ],
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_from_torch_deep_seek_padded_8b_weights_galaxy(
    mesh_device, shape, shard_shape, memory_layout, shard_dims, ttnn_dtype
):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    memory_config = ttnn.MemoryConfig(
        memory_layout,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_CORE_GRID_12, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )

    mesh_shape = (mesh_device.shape[0], mesh_device.shape[1])

    ttnn_result_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=1),  # ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    assert ttnn_result_tensor.dtype == ttnn_dtype, f"Expected {ttnn_dtype}, got {ttnn_result_tensor.dtype}"

    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[x for x in shard_dims if x is not None],
            mesh_shape_override=ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1]),
        ),
    )

    torch_result_tensor = ttnn.to_torch(ttnn_result_tensor, mesh_composer=mesh_composer)
    assert torch.equal(torch_input_tensor, torch_result_tensor)


# TOP PRIO
#  15      58      9.4590  [1, 256, 7168, 2048]         TILE         BFLOAT16     BFLOAT4_B    INTERLEAVED        N/A              Shard(dim=1)@0
#  16      58      8.7260  [1, 256, 2048, 7168]         TILE         BFLOAT16     BFLOAT4_B    INTERLEAVED        N/A              Shard(dim=1)@0
#  17      58      5.3185  [1, 256, 7168, 2048]         TILE         BFLOAT16     BFLOAT8_B    INTERLEAVED        N/A              Shard(dim=1)@0
