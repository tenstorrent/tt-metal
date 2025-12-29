import ttnn
import tests.ttnn.unit_tests.operations.maxpool.utils as utils


def _main(input):
    input_0 = input[0]
    ttnn_to_layout_0 = ttnn.to_layout(
        input_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_0, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 2227200, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_reshape_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_typecast_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_max_pool2d_0 = ttnn.max_pool2d(
        ttnn_to_layout_1,
        6,
        464,
        800,
        64,
        [3, 3],
        [2, 2],
        [1, 1],
        [1, 1],
        ceil_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        applied_shard_scheme=None,
        reallocate_halo_output=False,
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_max_pool2d_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_max_pool2d_0, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_to_layout_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_typecast_1,
        [6, 232, 400, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_1,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    util_create_list_0 = [ttnn_permute_1]
    return util_create_list_0


def create_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([6, 64, 464, 800]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=utils_DeviceGetter_get_device_0,
    )
    util_create_list_1 = [ttnn_ones_0]
    return util_create_list_1


def test_maxpool2d():
    create_inputs_for__main_0 = create_inputs_for__main()
    tt_output = _main(create_inputs_for__main_0)
