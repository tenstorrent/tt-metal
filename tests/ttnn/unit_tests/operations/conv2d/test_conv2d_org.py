import ttnn
import tests.ttnn.unit_tests.operations.conv2d.utils as utils


def _main(input):
    var_0 = input[1]
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_0 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_0, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 144000, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_reshape_0, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_1,
        weight_tensor=input[0],
        device=utils_DeviceGetter_get_device_0,
        in_channels=128,
        out_channels=128,
        batch_size=6,
        input_height=120,
        input_width=200,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(
            config_tensors_in_dram=True, enable_kernel_stride_folding=False
        ),
        compute_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_conv2d_0,
        [6, 60, 100, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_1,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    return [ttnn_permute_1]


def create_inputs_for__main():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([128, 128, 3, 3]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=None,
    )
    ttnn_ones_1 = ttnn.ones(
        shape=ttnn.Shape([6, 128, 120, 200]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=utils_DeviceGetter_get_device_1,
    )
    return [ttnn_ones_0, ttnn_ones_1]


def test_conv2d():
    create_inputs_for__main_0 = create_inputs_for__main()
    tt_output = _main(create_inputs_for__main_0)
   