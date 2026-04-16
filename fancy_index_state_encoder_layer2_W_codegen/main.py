import ttnn
from fancy_index_state_encoder_layer2_W_codegen import utils


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=0,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_0]


def main_const_eval_1(input):
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_0 = ttnn.to_layout(ttnn_to_device_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_0,
        [32, 1572864],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(ttnn_reshape_0, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_0, False)
    return [ttnn_to_layout_1]


def main_const_eval_2():
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=32,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_1]


_cached__main = {}


def _main(input):
    global _cached__main
    _cached__main = consteval__main(_cached__main, input)
    var_0 = input[0]
    ttnn_to_layout_2 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn_gt_0 = ttnn.gt(
        _cached__main["main_const_eval_0"][0],
        ttnn_to_layout_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_0 = ttnn.add(
        ttnn_to_layout_2,
        _cached__main["main_const_eval_2"][0],
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_gt_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_gt_0, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_add_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_to_layout_3 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_0, False)
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_to_layout_3,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn_where_0 = ttnn.where(
        ttnn_typecast_0,
        ttnn_typecast_1,
        ttnn_typecast_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_where_0,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_typecast_3,
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_reshape_1,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_to_layout_4 = ttnn.to_layout(ttnn_typecast_4, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_typecast_4, False)
    print(
        "shapes are",
        ttnn_to_layout_4.shape,
        ttnn_to_layout_4.dtype,
        ttnn_to_layout_4.layout,
        ttnn_to_layout_4.memory_config(),
    )
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_to_layout_4,
        _cached__main["main_const_eval_1"][0],
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_embedding_0,
        [1, 1024, 1536],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_embedding_0, False)
    return [ttnn_reshape_2]


def consteval__main(ce_cache, input_1):
    if not ce_cache:
        main_const_eval_0_0 = main_const_eval_0()
        ce_cache["main_const_eval_0"] = [main_const_eval_0_0[0]]
        main_const_eval_1_0 = main_const_eval_1([input_1[1]])
        ce_cache["main_const_eval_1"] = [main_const_eval_1_0[0]]
        main_const_eval_2_0 = main_const_eval_2()
        ce_cache["main_const_eval_2"] = [main_const_eval_2_0[0]]
    return ce_cache


def load_inputs_for__main():
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "fancy_index_state_encoder_layer2_W_codegen/tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_3,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "fancy_index_state_encoder_layer2_W_codegen/tensors/arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    return [utils_load_tensor_0, utils_load_tensor_1]


def test_main():
    load_inputs_for__main_0 = load_inputs_for__main()
    _main_0 = _main(load_inputs_for__main_0)
    return _main_0
