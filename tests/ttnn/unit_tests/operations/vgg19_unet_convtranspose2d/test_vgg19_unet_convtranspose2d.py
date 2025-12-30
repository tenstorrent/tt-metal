import ttnn
import utils


def main_const_eval_0(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_0,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_reshape_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_from_device_0 = ttnn.from_device(ttnn_to_layout_1)
    ttnn.deallocate(ttnn_to_layout_1, False)
    util_create_list_0 = [ttnn_from_device_0]
    return util_create_list_0


def main_const_eval_1(input):
    input_0 = input[0]
    util_create_list_1 = [input_0]
    return util_create_list_1


CACHED_main_const_eval_0 = None
CACHED_main_const_eval_1 = None


def _main(input):
    global CACHED_main_const_eval_1
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    input_2 = input[2]
    const_0 = main_const_eval_0
    util_create_list_2 = [input_0]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(const_0, util_create_list_2, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_1 = main_const_eval_1
    util_create_list_3 = [input_1]
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(const_1, util_create_list_3, CACHED_main_const_eval_1)
    CACHED_main_const_eval_1 = utils_constEvalFuncWrapper_1
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_2 = ttnn.to_layout(
        input_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_2, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_2,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 4096, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_reshape_1,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_conv_transpose2d_0 = ttnn.conv_transpose2d(
        input_tensor=ttnn_to_layout_3,
        weight_tensor=utils_constEvalFuncWrapper_1_0,
        device=utils_DeviceGetter_get_device_1,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        output_padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        dtype=ttnn.DataType.FLOAT32,
        bias_tensor=utils_constEvalFuncWrapper_0_0,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_conv_transpose2d_0,
        [1, 128, 128, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv_transpose2d_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_2,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    util_create_list_4 = [ttnn_permute_1]
    return util_create_list_4


def load_inputs_for__main():
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/vgg19_unet_convtranspose2d/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_1 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/vgg19_unet_convtranspose2d/arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_2 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/vgg19_unet_convtranspose2d/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_2,
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
