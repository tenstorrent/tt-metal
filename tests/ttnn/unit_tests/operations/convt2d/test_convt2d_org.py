import ttnn
import tests.ttnn.unit_tests.operations.convt2d.utils as utils


def main_const_eval_0(input):
    input_0 = input[0]
    util_create_list_0 = [input_0]
    return util_create_list_0


CACHED_main_const_eval_0 = None


def _main(input):
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    const_0 = main_const_eval_0
    util_create_list_1 = [input_0]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(const_0, util_create_list_1, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_0 = ttnn.to_layout(
        input_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_1, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 13392, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_reshape_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_conv_transpose2d_0 = ttnn.conv_transpose2d(
        input_tensor=ttnn_to_layout_1,
        weight_tensor=utils_constEvalFuncWrapper_0_0,
        device=utils_DeviceGetter_get_device_0,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=124,
        input_width=108,
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        output_padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        dtype=ttnn.DataType.FLOAT32,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_conv_transpose2d_0,
        [1, 248, 216, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv_transpose2d_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_1,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    util_create_list_2 = [ttnn_permute_1]
    return util_create_list_2


def create_inputs_for__main():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([128, 128, 2, 2]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=None,
    )
    ttnn_ones_1 = ttnn.ones(
        shape=ttnn.Shape([1, 128, 124, 108]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=utils_DeviceGetter_get_device_1,
    )
    util_create_list_3 = [ttnn_ones_0, ttnn_ones_1]
    return util_create_list_3


def test_convt2d():
    create_inputs_for__main_0 = create_inputs_for__main()
    tt_output = _main(create_inputs_for__main_0)
