# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.test_utils import round_up

CHANNEL_TEST_CASES = [1, 2, 3, 4]
BATCH_TEST_CASES = [1, 2, 4, 8]


@pytest.mark.parametrize("B", BATCH_TEST_CASES)
@pytest.mark.parametrize("C", CHANNEL_TEST_CASES)
@pytest.mark.parametrize("provide_memory_config", [True, False])
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

    input_tensor = torch.randn([1, B, C, HW], dtype=torch.bfloat16)

    expected = input_tensor.transpose(2, 3).reshape(1, 1, B * HW, C)

    input_shard_shape = (B * C, padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
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

    passed, message = assert_equal(
        expected, actual[:, :, :, : expected.shape[-1]]
    )  # slice off padding that is applied when C % 8 != 0
    assert passed, message


@pytest.mark.parametrize("B", [1])  # Only B=1 is currently supported for uneven sharding
@pytest.mark.parametrize("C", CHANNEL_TEST_CASES)
@pytest.mark.parametrize("provide_memory_config", [True, False])
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

    input_tensor = torch.randn([1, B, C, HW], dtype=torch.bfloat16)

    expected = input_tensor.transpose(2, 3).reshape(1, 1, B * HW, C)

    input_shard_shape = (B * C, padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
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

    passed, message = assert_equal(
        expected, actual[:, :, :, : expected.shape[-1]]
    )  # slice off padding that is applied when C % 8 != 0
    assert passed, message


@pytest.mark.parametrize("B", [1, 2, 4])
@pytest.mark.parametrize("C", [1, 2, 3, 4])
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
        (
            128,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            128,
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

    input_tensor = torch.randn([1, B, C, HW], dtype=torch.bfloat16)

    expected = input_tensor.transpose(2, 3).reshape(1, 1, B * HW, C)

    input_shard_shape = (B * C, input_padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    output_shard_shape = (B * output_padded_sharded_dim, round_up(C, 8))
    output_shard_spec = ttnn.ShardSpec(output_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    actual = ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)
    actual = ttnn.to_torch(actual)
    passed, message = assert_equal(
        expected, actual[:, :, :, : expected.shape[-1]]
    )  # slice off padding that is applied when C % 8 != 0
    assert passed, message


@pytest.mark.parametrize("B", [1, 2, 4])
@pytest.mark.parametrize("C", CHANNEL_TEST_CASES)
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
        (
            128,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
            128,
            64,
        ),
    ),
)
def test_convert_to_hwc_dram(
    device, B, C, HW, input_core_grid, output_core_grid, input_padded_sharded_dim, output_padded_sharded_dim
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

    input_tensor = torch.randn([1, B, C, HW], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3).reshape(1, 1, B * HW, C)

    input_shard_shape = (B * C, input_padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, input_shard_spec)

    output_shard_shape = (B * output_padded_sharded_dim, round_up(C, 8))
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


@pytest.mark.parametrize("C", CHANNEL_TEST_CASES)
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


def test_convert_to_hwc_dram_input_without_memory_config_should_fail(device, expect_error):
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

    with expect_error(RuntimeError, r"When input tensor is in DRAM, output memory_config must be explicitly specified"):
        ttnn.experimental.convert_to_hwc(input_tensor, dtype=ttnn.bfloat16)

    # Create an output shard that is not padded up to nearest aligned width
    output_shard_shape = (padded_sharded_dim, C)
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    with expect_error(RuntimeError, r"Output shard width must be rounded up to next multiple of 8"):
        ttnn.experimental.convert_to_hwc(input_tensor, dtype=ttnn.bfloat16, memory_config=output_mem_config)


# 2112 and 2240 on 1 core produce 2 blocks of 33 and 35 tiles (odd tile count)
@pytest.mark.parametrize("C", CHANNEL_TEST_CASES)
@pytest.mark.parametrize(
    "HW, core_grid, padded_sharded_dim",
    (
        (
            2112,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            2112,
        ),
        (
            2240,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            2240,
        ),
    ),
)
def test_convert_to_hwc_with_l1_input_odd_tiles_per_block(device, C, HW, core_grid, padded_sharded_dim):
    B = 1
    input_tensor = torch.randn([1, B, C, HW], dtype=torch.bfloat16)
    expected = input_tensor.transpose(2, 3).reshape(1, 1, B * HW, C)

    input_shard_shape = (B * C, padded_sharded_dim)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.Tensor(
        input_tensor, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
    )

    output_shard_shape = (B * padded_sharded_dim, round_up(C, 8))
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    actual = ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)
    actual = ttnn.to_torch(actual)

    passed, message = assert_equal(
        expected, actual[:, :, :, : expected.shape[-1]]
    )  # slice off padding that is applied when C % 8 != 0
    assert passed, message


def test_convert_to_hwc_dram_program_cache(device):
    B, C, HW = 1, 4, 32
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})

    input_shard_shape = (B * C, HW)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, input_shard_spec)

    output_shard_shape = (B * HW, round_up(C, 8))
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    device.enable_program_cache()
    device.clear_program_cache()

    # Keep prior tensors alive so each convert allocates at a new DRAM address
    keep_alive = []
    entries = None
    for i in range(4):
        torch_input = torch.randn([1, B, C, HW], dtype=torch.bfloat16)
        expected = torch_input.transpose(2, 3).reshape(1, 1, B * HW, C)
        tt_input = ttnn.Tensor(
            torch_input, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
        )
        tt_output = ttnn.experimental.convert_to_hwc(tt_input, memory_config=output_mem_config, dtype=ttnn.bfloat16)
        keep_alive += [tt_input, tt_output]
        actual = ttnn.to_torch(tt_output)
        passed, message = assert_equal(expected, actual[:, :, :, : expected.shape[-1]])
        assert passed, message
        if i == 0:
            entries = device.num_program_cache_entries()
        else:
            assert (
                device.num_program_cache_entries() == entries
            ), "convert_to_hwc must reuse the cached program on a hit"

    assert entries >= 1, "convert_to_hwc should cache at least one program"
    device.disable_and_clear_program_cache()


def test_convert_to_hwc_program_cache_rebinds_dram_and_l1(device):
    """A second call with the same spec must read the new allocation.

    DRAM refreshes the writer Buffer* binding. L1 refreshes the sharded circular-buffer base.
    Prior tensors stay alive so the allocator cannot hand the first address back; a frozen
    binding would then fail the golden compare.
    """
    B, C, HW = 1, 4, 32
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    padded_sharded_dim = 32
    num_dispatches = 2

    def run_binding(buffer_type, label):
        input_shard_shape = (B * C, padded_sharded_dim)
        input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type, input_shard_spec)

        output_shard_shape = (B * padded_sharded_dim, round_up(C, 8))
        output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )

        device.cache_entries_counter.reset()
        kept_alive = []
        input_addrs = set()
        for i in range(num_dispatches):
            torch_input = torch.arange(B * C * HW, dtype=torch.float32).reshape(1, B, C, HW)
            torch_input = (torch_input + float(i + 1)).to(dtype=torch.bfloat16)
            expected = torch_input.transpose(2, 3).reshape(1, 1, B * HW, C)

            tt_input = ttnn.Tensor(
                torch_input, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=input_mem_config
            )
            # Count only convert_to_hwc. to_torch on a sharded tensor can dispatch its own programs.
            with device.cache_entries_counter.measure():
                tt_output = ttnn.experimental.convert_to_hwc(
                    tt_input, memory_config=output_mem_config, dtype=ttnn.bfloat16
                )
            input_addrs.add(tt_input.buffer_address())
            actual = ttnn.to_torch(tt_output)
            passed, message = assert_equal(expected, actual[:, :, :, : expected.shape[-1]])
            assert passed, f"{label} dispatch {i}: {message}"
            kept_alive.append((tt_input, tt_output))

        assert len(input_addrs) == num_dispatches, f"{label} inputs reused an address: {sorted(input_addrs)}"
        assert device.cache_entries_counter.total == 1, (
            f"{label} expected 1 program-cache entry across {num_dispatches} dispatches, "
            f"got {device.cache_entries_counter.total}"
        )

    run_binding(ttnn.BufferType.DRAM, "DRAM")
    run_binding(ttnn.BufferType.L1, "L1")


def _width_sharded_hwc_input(device, C, HW, buffer_type):
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(core_grid, (C, HW), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer_type, shard_spec)
    torch_input = torch.randn([1, 1, C, HW], dtype=torch.bfloat16)
    return core_grid, ttnn.Tensor(
        torch_input, ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, mem_config=mem_config
    )


# An explicit output config without a shard spec used to be dereferenced before any check ran.
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_convert_to_hwc_rejects_output_config_without_shard_spec(device, output_mem_config, expect_error):
    _, input_tensor = _width_sharded_hwc_input(device, 4, 32, ttnn.BufferType.DRAM)
    with expect_error(RuntimeError, "Output memory config must be height sharded with a shard spec"):
        ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.bfloat16)


# The kernels size the output from the input dtype, so a different output dtype returned garbage.
def test_convert_to_hwc_rejects_dtype_change(device, expect_error):
    core_grid, input_tensor = _width_sharded_hwc_input(device, 3, 128, ttnn.BufferType.L1)
    output_shard_spec = ttnn.ShardSpec(core_grid, (128, 8), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    with expect_error(RuntimeError, "convert_to_hwc does not convert dtypes"):
        ttnn.experimental.convert_to_hwc(input_tensor, memory_config=output_mem_config, dtype=ttnn.float32)
