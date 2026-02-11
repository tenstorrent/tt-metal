import torch
import ttnn
import tests.ttnn.unit_tests.operations.conv_pcc_drop.utils as utils

_CONST_EVAL_CACHE = {}


def main_const_eval_0(input):
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_0 = ttnn.to_device(
        input[0],
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
        [128, 128, 1, 3],
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


def main_const_eval_1(input):
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_1 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_to_device_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_1, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_to_layout_2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_reshape_1,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_permute_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_from_device_1 = ttnn.from_device(ttnn_to_layout_3)
    ttnn.deallocate(ttnn_to_layout_3, False)
    util_create_list_1 = [ttnn_from_device_1]
    return util_create_list_1


def _main(input):
    global _CONST_EVAL_CACHE
    const_0 = main_const_eval_0
    util_create_list_2 = [input[1]]
    const_1 = "main_const_eval_0"
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(
        const_0, util_create_list_2, _CONST_EVAL_CACHE, const_1
    )
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_2 = main_const_eval_1
    util_create_list_3 = [input[0]]
    const_3 = "main_const_eval_1"
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(
        const_2, util_create_list_3, _CONST_EVAL_CACHE, const_3
    )
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_4 = ttnn.to_layout(
        input[2],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input[2], False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_4,
        [1, 128, 1, 8192],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_2,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_permute_1,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_5,
        weight_tensor=utils_constEvalFuncWrapper_0_0,
        device=utils_DeviceGetter_get_device_2,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=1,
        input_width=8192,
        kernel_size=[1, 3],
        stride=[1, 1],
        padding=[0, 0, 3, 3],
        dilation=[1, 3],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_1_0,
        conv_config=ttnn.Conv2dConfig(
            config_tensors_in_dram=True, enable_kernel_stride_folding=False
        ),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_conv2d_0,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_permute_2,
        [1, 128, 8192],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_2, False)
    util_create_list_4 = [ttnn_reshape_3]
    return util_create_list_4


def load_inputs_for__main():
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_3,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_5 = [utils_load_tensor_0, utils_load_tensor_1, utils_load_tensor_2]
    return util_create_list_5


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten(), y.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return ((vx @ vy) / denom).item()


class CustomConvModel(torch.nn.Module):
    """Custom Conv1d model that matches the TT operation."""
    def __init__(self):
        super().__init__()
        # Conv1d: in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=3, dilation=3
        self.conv = torch.nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3)

    def forward(self, x):
        return self.conv(x)


def main():
    # Load weight, bias, and input tensors 
    tt_conv_weight_torch = ttnn.to_torch(ttnn.load_tensor("./tensors/arg1.tensorbin"))
    tt_conv_bias_torch = ttnn.to_torch(ttnn.load_tensor("./tensors/arg0.tensorbin"))
    tt_input_torch = ttnn.to_torch(ttnn.load_tensor("./tensors/arg2.tensorbin"))
    
    # TT run
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)
    
    # CPU run
    model = CustomConvModel()
    model.eval()
    
    # Set model weights and bias to match TT exactly
    model.conv.weight.data = tt_conv_weight_torch.to(torch.bfloat16)
    model.conv.bias.data = tt_conv_bias_torch.to(torch.bfloat16)
    
    # CPU inference
    with torch.no_grad():
        cpu_output = model(tt_input_torch)
    
    # Verify weight and bias tensors are equal
    print(f"Conv weight equal (torch.equal): {torch.equal(tt_conv_weight_torch, model.conv.weight.data)}")
    print(f"Conv bias equal (torch.equal): {torch.equal(tt_conv_bias_torch, model.conv.bias.data)}")

    # Compare output using pcc
    tt_output_torch = ttnn.to_torch(tt_output[0])
    tt_output_fp64 = tt_output_torch.to(torch.float64)
    cpu_output_fp64 = cpu_output.to(torch.float64)
    
    pcc = compute_pcc(tt_output_fp64, cpu_output_fp64)
    print(f"Output PCC: {pcc}")



if __name__ == "__main__":
    main()
