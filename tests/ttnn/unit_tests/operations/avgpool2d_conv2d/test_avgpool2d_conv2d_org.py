import ttnn
import tests.ttnn.unit_tests.operations.avgpool2d_conv2d.utils as utils

def main_const_eval_0(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_0,
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
        [1, 1, 1, 480],
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
    ttnn_from_device_0 = ttnn.from_device(ttnn_to_layout_1)
    ttnn.deallocate(ttnn_to_layout_1, False)
    util_create_list_0 = [ttnn_from_device_0]
    return util_create_list_0


CACHED_main_const_eval_0 = None


def _main(input):
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    input_2 = input[2]
    const_0 = main_const_eval_0
    util_create_list_1 = [input_0]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(
        const_0, util_create_list_1, CACHED_main_const_eval_0
    )
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_2 = ttnn.to_layout(
        input_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input_2, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_2,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 1120, 360],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_reshape_1,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_avg_pool2d_0 = ttnn.avg_pool2d(
        ttnn_to_layout_3,
        1,
        28,
        40,
        360,
        [2, 2],
        [1, 1],
        [0, 0],
        False,
        True,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        applied_shard_scheme=None,
        compute_kernel_config=None,
        reallocate_halo_output=False,
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_avg_pool2d_0,
        weight_tensor=input_1,
        device=utils_DeviceGetter_get_device_1,
        in_channels=360,
        out_channels=480,
        batch_size=1,
        input_height=27,
        input_width=39,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_0_0,
        conv_config=ttnn.Conv2dConfig(
            config_tensors_in_dram=True, enable_kernel_stride_folding=False
        ),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_avg_pool2d_0, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_conv2d_0,
        [1, 14, 20, 480],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_2,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    util_create_list_2 = [ttnn_permute_1]
    return util_create_list_2


def create_inputs_for__main():
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([480]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=None,
    )
    ttnn_ones_1 = ttnn.ones(
        shape=ttnn.Shape([480, 360, 3, 3]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=None,
    )
    ttnn_ones_2 = ttnn.ones(
        shape=ttnn.Shape([1, 360, 28, 40]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=utils_DeviceGetter_get_device_2,
    )
    util_create_list_3 = [ttnn_ones_0, ttnn_ones_1, ttnn_ones_2]
    return util_create_list_3


def test_avgpool2d_conv2d():
    create_inputs_for__main_0 = create_inputs_for__main()
    _main_0 = _main(create_inputs_for__main_0)
    
