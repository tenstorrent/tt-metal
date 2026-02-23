import ttnn
from cumsum_sanity import utils

_CONST_EVAL_CACHE = {}


def _main(input):
    ttnn_to_layout_0 = ttnn.to_layout(
        input[0],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input[0], False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_0,
        [1168640, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_moreh_cumsum_0 = ttnn.moreh_cumsum(
        ttnn_reshape_0,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_moreh_cumsum_0,
        [1168640],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_moreh_cumsum_0, False)
    util_create_list_0 = [ttnn_reshape_1]
    return util_create_list_0


def load_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "cumsum_sanity/tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_1 = [utils_load_tensor_0]
    return util_create_list_1


def test_main():
    load_inputs_for__main_0 = load_inputs_for__main()
    _main_0 = _main(load_inputs_for__main_0)
    print("_main_0 result is ", _main_0.shape)
    return const0_0


if __name__ == "__main__":
    test_main()
