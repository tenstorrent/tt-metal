import ttnn
from groot_module import utils


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=32,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_0 = [ttnn_full_0]
    return util_create_list_0


def main_const_eval_1():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=0,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_1 = [ttnn_full_1]
    return util_create_list_1


def main_const_eval_2(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_0,
        [32, 1572864],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_reshape_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    util_create_list_2 = [ttnn_to_layout_1]
    return util_create_list_2


CACHED_main_const_eval_0 = None
CACHED_main_const_eval_1 = None
CACHED_main_const_eval_2 = None


def _main(input):
    global CACHED_main_const_eval_2
    global CACHED_main_const_eval_1
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    const_0 = main_const_eval_0
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(
        const_0, CACHED_main_const_eval_0
    )
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapperZeroArg_0
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_1 = main_const_eval_1
    utils_constEvalFuncWrapperZeroArg_1 = utils.constEvalFuncWrapperZeroArg(
        const_1, CACHED_main_const_eval_1
    )
    CACHED_main_const_eval_1 = utils_constEvalFuncWrapperZeroArg_1
    utils_constEvalFuncWrapperZeroArg_1_0 = utils_constEvalFuncWrapperZeroArg_1[0]
    const_2 = main_const_eval_2
    util_create_list_3 = [input_1]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(
        const_2, util_create_list_3, CACHED_main_const_eval_2
    )
    CACHED_main_const_eval_2 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    ttnn_to_layout_2 = ttnn.to_layout(
        input_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_lt_0 = ttnn.lt(
        ttnn_to_layout_2,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_to_layout_3 = ttnn.to_layout(
        input_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_0 = ttnn.add(
        ttnn_to_layout_3,
        utils_constEvalFuncWrapperZeroArg_0_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_lt_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_lt_0, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_add_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_to_layout_4 = ttnn.to_layout(
        input_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_0, False)
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_to_layout_4,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn_where_0 = ttnn.where(
        ttnn_typecast_0,
        ttnn_typecast_1,
        ttnn_typecast_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_where_0,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_where_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_typecast_3,
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_reshape_1,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_typecast_4,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_to_layout_5, utils_constEvalFuncWrapper_0_0, layout=ttnn.Layout.TILE
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_embedding_0,
        [1, 1024, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_embedding_0, False)
    util_create_list_4 = [ttnn_reshape_2]
    return util_create_list_4


def load_inputs_for__main():
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "groot_module/tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "groot_module/tensors/arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    util_create_list_5 = [utils_load_tensor_0, utils_load_tensor_1]
    return util_create_list_5


def test_groot_slice():
    load_inputs = load_inputs_for__main()
    tt_out = _main(load_inputs)
    


if __name__ == "__main__":
    main()
