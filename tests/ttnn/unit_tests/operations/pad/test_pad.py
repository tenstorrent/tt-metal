import ttnn
import tests.ttnn.unit_tests.operations.pad.utils as utils


def _main(input):
    input_0 = input[0]
    ttnn_to_layout_0 = ttnn.to_layout(
        input_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_0, False)
    ttnn_pad_0 = ttnn.pad(
        ttnn_to_layout_0,
        [[0, 0], [0, 0], [-16, -16], [-16, -16]],
        0.0,
        use_multicore=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_pad_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_pad_0, False)
    util_create_list_0 = [ttnn_to_layout_1]
    return util_create_list_0


def load_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_1 = [utils_load_tensor_0]
    return util_create_list_1


def test_pad():
    load_inputs_for__main_0 = load_inputs_for__main()
    ttnn_output = _main(load_inputs_for__main_0)

    return ttnn_output
