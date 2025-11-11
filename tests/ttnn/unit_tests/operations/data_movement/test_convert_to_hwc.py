# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.test_utils import round_up

# CHANNEL_TEST_CASES = [1, 2, 3, 4, 8, 12, 15, 16, 32]
CHANNEL_TEST_CASES = [1, 2, 3, 4]
BATCH_TEST_CASES = [1, 2, 4, 8]


@pytest.mark.parametrize("B", BATCH_TEST_CASES)
@pytest.mark.parametrize("C", CHANNEL_TEST_CASES)
@pytest.mark.parametrize("provide_memory_config", [True])
@pytest.mark.parametrize(
    "HW, core_grid, padded_sharded_dim",
    (
        (
            32,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                }
            ),
            32,
        ),
        (
            128,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                }
            ),
            128,
        ),
        (
            128,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                }
            ),
            64,
        ),
        (
            256,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                }
            ),
            64,
        ),
        (
            8192,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7)),
                }
            ),
            128,
        ),
    ),
)
def test_convert_to_hwc_with_l1_input(device, B, C, HW, core_grid, padded_sharded_dim, provide_memory_config):
    device_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    requested_num_cores = core_grid.num_cores()
    if device_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {device_num_cores})")

    # Verify this is even sharding
    assert (
        padded_sharded_dim * core_grid.num_cores() == HW
    ), f"Expected even sharding but got uneven: {padded_sharded_dim} * {core_grid.num_cores()} != {HW}"

    input_tensor = torch.concat(
        [
            torch.concat(
                [torch.full([1, 1, 1, HW], c + ((b + 1) * 100), dtype=torch.bfloat16) for c in range(C)], dim=2
            )
            for b in range(B)
        ],
        dim=1,
    )
    input_tensor = torch.randn([1, B, C, HW], dtype=torch.bfloat16)
    print(input_tensor)

    expected = input_tensor.transpose(2, 3).reshape(1, 1, B * HW, C)

    input_shard_shape = (B * C, padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    print(input_tensor)
    print(
        input_tensor.shape,
        input_tensor.memory_config().shard_spec,
        " cores=",
        input_tensor.memory_config().shard_spec.num_cores(),
    )

    if provide_memory_config:
        output_shard_shape = (B * padded_sharded_dim, round_up(C, 8))
        output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )
        actual = ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)
    else:
        actual = ttnn.experimental.convert_to_hwc(input_tensor, dtype=ttnn.bfloat16)

    actual = ttnn.to_torch(actual)

    print(expected)
    print(actual[:, :, :, : expected.shape[-1]])

    passed, message = assert_equal(
        expected, actual[:, :, :, : expected.shape[-1]]
    )  # slice off padding that is applied when C % 8 != 0
    assert passed, message


@pytest.mark.parametrize("B", [1])  # Only B=1 is currently supported for uneven sharding
@pytest.mark.parametrize("C", CHANNEL_TEST_CASES)
@pytest.mark.parametrize("provide_memory_config", [True])
@pytest.mark.parametrize(
    "HW, core_grid, padded_sharded_dim",
    (
        (
            30,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                }
            ),
            32,
        ),
        (
            60,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                }
            ),
            32,
        ),
        (
            168960,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                }
            ),
            2688,
        ),  # UNet Shallow
    ),
)
def test_convert_to_hwc_with_l1_input_uneven_sharding(
    device, B, C, HW, core_grid, padded_sharded_dim, provide_memory_config
):
    device_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    requested_num_cores = core_grid.num_cores()
    if device_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {device_num_cores})")

    # Verify this is uneven sharding
    assert (
        padded_sharded_dim * core_grid.num_cores() > HW
    ), f"Expected uneven sharding but got even: {padded_sharded_dim} * {core_grid.num_cores()} <= {HW}"

    # Only B=1 is supported for uneven sharding
    assert B == 1, f"Uneven sharding is only supported when B=1 (was {B})"

    input_tensor = torch.concat(
        [
            torch.concat(
                [torch.full([1, 1, 1, HW], c + ((b + 1) * 100), dtype=torch.bfloat16) for c in range(C)], dim=2
            )
            for b in range(B)
        ],
        dim=1,
    )
    input_tensor = torch.randn([1, B, C, HW], dtype=torch.bfloat16)

    expected = input_tensor.transpose(2, 3).reshape(1, 1, B * HW, C)

    input_shard_shape = (B * C, padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    print(input_tensor)
    print(
        input_tensor.shape,
        input_tensor.memory_config().shard_spec,
        " cores=",
        input_tensor.memory_config().shard_spec.num_cores(),
    )

    if provide_memory_config:
        output_shard_shape = (B * padded_sharded_dim, round_up(C, 8))
        output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )
        actual = ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)
    else:
        actual = ttnn.experimental.convert_to_hwc(input_tensor, dtype=ttnn.bfloat16)

    actual = ttnn.to_torch(actual)

    print(expected)
    print(actual[:, :, :, : expected.shape[-1]])

    passed, message = assert_equal(
        expected, actual[:, :, :, : expected.shape[-1]]
    )  # slice off padding that is applied when C % 8 != 0
    assert passed, message


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("C", [1, 2])
@pytest.mark.parametrize(
    "HW, input_core_grid, output_core_grid, input_padded_sharded_dim, output_padded_sharded_dim",
    (
        (
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            32,
            32,
        ),
        (
            64,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            64,
            32,
        ),
    ),
)
def test_convert_to_hwc_with_l1_input_resharded(
    device, B, C, HW, input_core_grid, output_core_grid, input_padded_sharded_dim, output_padded_sharded_dim
):
    device_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    requested_num_cores = output_core_grid.num_cores()
    if device_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {device_num_cores})")

    is_uneven = input_padded_sharded_dim * input_core_grid.num_cores() > HW
    if is_uneven and B > 1:
        pytest.skip(f"Uneven sharding is not supported when B > 1 (was {B})")

    input_tensor = torch.concat(
        [
            torch.concat(
                [torch.full([1, 1, 1, HW], c + ((b + 1) * 100), dtype=torch.bfloat16) for c in range(C)], dim=2
            )
            for b in range(B)
        ],
        dim=1,
    )
    input_tensor = torch.randn([1, B, C, HW], dtype=torch.bfloat16)

    expected = input_tensor.transpose(2, 3).reshape(1, 1, B * HW, C)

    input_shard_shape = (B * C, input_padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    print(input_tensor)
    print(
        input_tensor.shape,
        input_tensor.memory_config().shard_spec,
        " cores=",
        input_tensor.memory_config().shard_spec.num_cores(),
    )

    output_shard_shape = (B * output_padded_sharded_dim, round_up(C, 8))
    output_shard_spec = ttnn.ShardSpec(output_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    print(output_mem_config.shard_spec, " cores=", output_mem_config.shard_spec.num_cores())

    actual = ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)
    actual = ttnn.to_torch(actual)

    print(expected)
    print(actual[:, :, :, : expected.shape[-1]])

    passed, message = assert_equal(
        expected, actual[:, :, :, : expected.shape[-1]]
    )  # slice off padding that is applied when C % 8 != 0
    assert passed, message


# @pytest.mark.parametrize("C", CHANNEL_TEST_CASES)
@pytest.mark.parametrize("C", [1, 2])
@pytest.mark.parametrize(
    "HW, input_core_grid, output_core_grid, input_padded_sharded_dim, output_padded_sharded_dim",
    (
        (
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            32,
            32,
        ),
        (
            64,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            64,
            64,
        ),
    ),
)
def test_convert_to_hwc_dram(
    device, C, HW, input_core_grid, output_core_grid, input_padded_sharded_dim, output_padded_sharded_dim
):
    worker_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    requested_num_cores = output_core_grid.num_cores()
    if worker_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {worker_num_cores})")

    dram_num_cores = device.dram_grid_size().x * device.dram_grid_size().y
    requested_num_dram_cores = input_core_grid.num_cores()
    if dram_num_cores < requested_num_dram_cores:
        pytest.skip(
            f"Not enough DRAM cores to run test case (need {requested_num_dram_cores} but have {dram_num_cores})"
        )

    # DRAM test function uses B=1 implicitly, so uneven sharding is supported
    is_uneven = input_padded_sharded_dim * input_core_grid.num_cores() > HW

    input_tensor = torch.randn([1, 1, C, HW], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3)

    input_shard_shape = (C, input_padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, input_shard_spec)

    output_shard_shape = (output_padded_sharded_dim, round_up(C, 8))
    output_shard_spec = ttnn.ShardSpec(output_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    actual = ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)
    actual = ttnn.to_torch(actual)

    passed, message = assert_equal(
        expected, actual[:, :, :, : expected.shape[-1]]
    )  # slice off padding that is applied when C % 8 != 0
    assert passed, message


@pytest.mark.parametrize("C", [1, 2])
@pytest.mark.parametrize(
    "HW, input_core_grid, output_core_grid, input_padded_sharded_dim, output_padded_sharded_dim",
    (
        (
            168960,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                }
            ),
            14080,
            2688,
        ),
    ),
)
def test_convert_to_hwc_dram_uneven_sharding(
    device, C, HW, input_core_grid, output_core_grid, input_padded_sharded_dim, output_padded_sharded_dim
):
    worker_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    requested_num_cores = output_core_grid.num_cores()
    if worker_num_cores < requested_num_cores:
        pytest.skip(f"Not enough cores to run test case (need {requested_num_cores} but have {worker_num_cores})")

    dram_num_cores = device.dram_grid_size().x * device.dram_grid_size().y
    requested_num_dram_cores = input_core_grid.num_cores()
    if dram_num_cores < requested_num_dram_cores:
        pytest.skip(
            f"Not enough DRAM cores to run test case (need {requested_num_dram_cores} but have {dram_num_cores})"
        )

    # Uneven sharding along B*HW for output
    assert input_padded_sharded_dim * input_core_grid.num_cores() == HW
    assert output_padded_sharded_dim * output_core_grid.num_cores() > HW

    input_tensor = torch.randn([1, 1, C, HW], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3)

    input_shard_shape = (C, input_padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, input_shard_spec)

    output_shard_shape = (output_padded_sharded_dim, round_up(C, 8))
    output_shard_spec = ttnn.ShardSpec(output_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    actual = ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)
    actual = ttnn.to_torch(actual)

    passed, message = assert_equal(
        expected, actual[:, :, :, : expected.shape[-1]]
    )  # slice off padding that is applied when C % 8 != 0
    assert passed, message


def test_convert_to_hwc_dram_input_without_memory_config_should_fail(device):
    C = 4
    HW = 32
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    padded_sharded_dim = 32

    input_tensor = torch.randn([1, 1, C, HW], dtype=torch.bfloat16)

    input_shard_shape = (C, padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, input_shard_spec)
    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    with pytest.raises(
        RuntimeError, match="When input tensor is in DRAM, output memory_config must be explicitly specified"
    ):
        ttnn.experimental.convert_to_hwc(input_tensor, dtype=ttnn.bfloat16)

    # Create an output shard that is not padded up to nearest aligned width
    output_shard_shape = (padded_sharded_dim, C)
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    with pytest.raises(RuntimeError):
        ttnn.experimental.convert_to_hwc(input_tensor, dtype=ttnn.bfloat16, memory_config=output_mem_config)
