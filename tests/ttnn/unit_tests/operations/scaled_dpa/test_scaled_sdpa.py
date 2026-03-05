import ttnn
import utils

_CONST_EVAL_CACHE = {}


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.0,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_0 = ttnn.repeat(ttnn_full_0, ttnn.Shape([2, 24, 4429, 4429]))
    ttnn.deallocate(ttnn_full_0, False)
    util_create_list_0 = [ttnn_repeat_0]
    return util_create_list_0


def main_const_eval_1():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_1 = [ttnn_full_1]
    return util_create_list_1


def main_const_eval_2():
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.35355338454246521,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_2 = [ttnn_full_2]
    return util_create_list_2


def main_const_eval_3():
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_3 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.35355338454246521,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_3 = [ttnn_full_3]
    return util_create_list_3


def _main(input):
    global _CONST_EVAL_CACHE
    const_0 = main_const_eval_0
    const_1 = "main_const_eval_0"
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(const_0, _CONST_EVAL_CACHE, const_1)
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_2 = main_const_eval_1
    const_3 = "main_const_eval_1"
    utils_constEvalFuncWrapperZeroArg_1 = utils.constEvalFuncWrapperZeroArg(const_2, _CONST_EVAL_CACHE, const_3)
    utils_constEvalFuncWrapperZeroArg_1_0 = utils_constEvalFuncWrapperZeroArg_1[0]
    const_4 = main_const_eval_2
    const_5 = "main_const_eval_2"
    utils_constEvalFuncWrapperZeroArg_2 = utils.constEvalFuncWrapperZeroArg(const_4, _CONST_EVAL_CACHE, const_5)
    utils_constEvalFuncWrapperZeroArg_2_0 = utils_constEvalFuncWrapperZeroArg_2[0]
    const_6 = main_const_eval_3
    const_7 = "main_const_eval_3"
    utils_constEvalFuncWrapperZeroArg_3 = utils.constEvalFuncWrapperZeroArg(const_6, _CONST_EVAL_CACHE, const_7)
    utils_constEvalFuncWrapperZeroArg_3_0 = utils_constEvalFuncWrapperZeroArg_3[0]
    ttnn_to_layout_0 = ttnn.to_layout(
        input[2],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input[2], False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_to_layout_0,
        utils_constEvalFuncWrapperZeroArg_3_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        input[1],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input[1], False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_1,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_permute_0,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_multiply_0,
        ttnn_multiply_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn_eq_0 = ttnn.eq(
        ttnn_matmul_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_logical_not_0 = ttnn.logical_not(
        ttnn_eq_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_eq_0, False)
    ttnn_sum_0 = ttnn.sum(
        ttnn_logical_not_0,
        [3],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_not_0, False)
    ttnn_logical_not_1 = ttnn.logical_not(
        ttnn_sum_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_0, False)
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_matmul_0,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_repeat_1 = ttnn.repeat(ttnn_logical_not_1, ttnn.Shape([1, 1, 1, 4429]))
    ttnn.deallocate(ttnn_logical_not_1, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_repeat_1,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_1, False)
    ttnn_where_0 = ttnn.where(
        ttnn_typecast_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_softmax_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn.deallocate(ttnn_softmax_0, False)
    ttnn_to_layout_2 = ttnn.to_layout(
        input[0],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input[0], False)
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_where_0,
        ttnn_to_layout_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn.deallocate(ttnn_where_0, False)
    util_create_list_4 = [ttnn_matmul_1]
    return util_create_list_4


def load_inputs_for__main():
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/scaled_dpa/tensors/arg0.tensorbinn",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_4,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/scaled_dpa/tensors/arg1.tensorbinn",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_4,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/scaled_dpa/tensors/arg2.tensorbinn",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_4,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_5 = [utils_load_tensor_0, utils_load_tensor_1, utils_load_tensor_2]
    return util_create_list_5


def main():
    load_inputs_for__main_0 = load_inputs_for__main()
    _main_0 = _main(load_inputs_for__main_0)
    const0_0 = 0
    return const0_0


if __name__ == "__main__":
    main()
