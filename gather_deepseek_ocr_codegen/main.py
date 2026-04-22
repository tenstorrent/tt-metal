import ttnn
from gather_deepseek_ocr_codegen import utils


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_arange_0 = ttnn.arange(
        0,
        1280,
        1,
        dtype=ttnn.DataType.UINT32,
        device=utils_DeviceGetter_get_device_0,
        layout=ttnn.Layout.TILE,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_arange_0,
        [1, 1280, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_arange_0, False)
    ttnn_repeat_0 = ttnn.repeat(
        ttnn_reshape_0,
        ttnn.Shape([913, 1, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    return [ttnn_repeat_0]


def main_const_eval_1():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_0 = ttnn.Tensor(
        [1280.0, 1.0],
        [2, 1],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_Tensor_0]


_cached__main = {}


def _main(input):
    global _cached__main
    _cached__main = consteval__main(_cached__main, input)
    var_0 = input[0]
    var_1 = input[1]
    ttnn_to_layout_0 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_0, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_to_layout_0,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_typecast_0,
        [913, 1280, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_concat_0 = ttnn.concat(
        [ttnn_reshape_1, _cached__main["main_const_eval_0"][0]],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_to_layout_1 = ttnn.to_layout(var_1, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_1, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_1,
        [1155840, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_concat_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_typecast_1,
        _cached__main["main_const_eval_1"][0],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_matmul_0,
        [1, 1168640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_reshape_3,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_3, False)
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_typecast_2, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_reshape_2, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_to_layout_2,
        ttnn_to_layout_3,
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_embedding_0,
        [913, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_embedding_0, False)
    return [ttnn_reshape_4]


def consteval__main(ce_cache, input_1):
    if not ce_cache:
        main_const_eval_0_0 = main_const_eval_0()
        ce_cache["main_const_eval_0"] = [main_const_eval_0_0[0]]
        main_const_eval_1_0 = main_const_eval_1()
        ce_cache["main_const_eval_1"] = [main_const_eval_1_0[0]]
    return ce_cache


def load_inputs_for__main():
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "gather_deepseek_ocr_codegen/tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "gather_deepseek_ocr_codegen/tensors/arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [utils_load_tensor_0, utils_load_tensor_1]


def main():
    load_inputs_for__main_0 = load_inputs_for__main()
    _main_0 = _main(load_inputs_for__main_0)
    return _main_0


if __name__ == "__main__":
    main()
