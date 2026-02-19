import ttnn
import tests.ttnn.unit_tests.operations.sort.utils as utils

_CONST_EVAL_CACHE = {}


def _main(input):
    ttnn_to_layout_0 = ttnn.to_layout(
        input[0],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input[0], False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_to_layout_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    v_0, v_1 = ttnn.sort(
        ttnn_typecast_0,
        0,
        True,
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_0, False)
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_typecast_1 = ttnn.typecast(
        v_1,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_1, False)
    util_create_list_0 = [ttnn_typecast_1]
    return util_create_list_0


def load_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_1 = [utils_load_tensor_0]
    return util_create_list_1


def test_sort_org():
    load_inputs_for__main_0 = load_inputs_for__main()
    _main_0 = _main(load_inputs_for__main_0)
   