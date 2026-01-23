import ttnn
import tests.ttnn.unit_tests.operations.conv2d_bn2d.utils as utils
import torch
import torch.nn as nn
from loguru import logger

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
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_reshape_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_permute_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_0, False)
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
    util_create_list_1 = [ttnn_reshape_1]
    return util_create_list_1


def main_const_eval_2(input):
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_2 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_to_device_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_2, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_3,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    util_create_list_2 = [ttnn_reshape_2]
    return util_create_list_2


def main_const_eval_3(input):
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_3 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_to_device_3,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_3, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_to_layout_4,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    util_create_list_3 = [ttnn_reshape_3]
    return util_create_list_3


def main_const_eval_4(input):
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_4 = ttnn.to_device(
        input[0],
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_to_device_4,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_4, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_to_layout_5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    util_create_list_4 = [ttnn_reshape_4]
    return util_create_list_4


def _main(input):
    global _CONST_EVAL_CACHE
    const_0 = main_const_eval_0
    util_create_list_5 = [input[4]]
    const_1 = "main_const_eval_0"
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(
        const_0, util_create_list_5, _CONST_EVAL_CACHE, const_1
    )
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_2 = main_const_eval_1
    util_create_list_6 = [input[2]]
    const_3 = "main_const_eval_1"
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(
        const_2, util_create_list_6, _CONST_EVAL_CACHE, const_3
    )
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_4 = main_const_eval_2
    util_create_list_7 = [input[3]]
    const_5 = "main_const_eval_2"
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(
        const_4, util_create_list_7, _CONST_EVAL_CACHE, const_5
    )
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_6 = main_const_eval_3
    util_create_list_8 = [input[0]]
    const_7 = "main_const_eval_3"
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(
        const_6, util_create_list_8, _CONST_EVAL_CACHE, const_7
    )
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    const_8 = main_const_eval_4
    util_create_list_9 = [input[1]]
    const_9 = "main_const_eval_4"
    utils_constEvalFuncWrapper_4 = utils.constEvalFuncWrapper(
        const_8, util_create_list_9, _CONST_EVAL_CACHE, const_9
    )
    utils_constEvalFuncWrapper_4_0 = utils_constEvalFuncWrapper_4[0]
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_6 = ttnn.to_layout(
        input[6],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input[6], False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_to_layout_6,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_6, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_permute_1,
        [1, 1, 784, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_reshape_5,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_5, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_7,
        weight_tensor=input[5],
        device=utils_DeviceGetter_get_device_5,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
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
    ttnn.deallocate(ttnn_to_layout_7, False)
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_conv2d_0,
        [1, 28, 28, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_6,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_6, False)
    ttnn_batch_norm_0 = ttnn.batch_norm(
        ttnn_permute_2,
        running_mean=utils_constEvalFuncWrapper_4_0,
        running_var=utils_constEvalFuncWrapper_3_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_2_0,
        bias=utils_constEvalFuncWrapper_1_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_2, False)
    util_create_list_10 = [ttnn_batch_norm_0]
    return util_create_list_10


def load_inputs_for__main():
    utils_DeviceGetter_get_device_6 = utils.DeviceGetter.get_device((1, 1))
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
        None,
        None,
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_6,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_11 = [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
    ]
    return util_create_list_11


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_float = x.to(torch.float32) if x.dtype != torch.float32 else x
    y_float = y.to(torch.float32) if y.dtype != torch.float32 else y

    x_flat, y_flat = x_float.flatten(), y_float.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    if denom == 0:
        return float("nan")
    else:
        return ((vx @ vy) / denom).item()


class CPUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        return x
    

def test_conv2d_bn2d_org():
    # TT run
    logger.info("Running TT inference...")
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)
    logger.info("tt_output: {}", tt_output)
    logger.info("tt_output[0].shape: {}", tt_output[0].shape)
    logger.info("tt_output[0].dtype: {}", tt_output[0].dtype)
    
    # CPU run
    logger.info("Running CPU inference...")
    cpu_model = CPUModel().to(torch.bfloat16)
    cpu_model.eval()
    
    # Load the weights from tensorbin files into CPU model
    logger.info("Loading weights from tensorbin files into CPU model...")
    
    # Load tensors and convert to torch
    tt_bn_var_torch = ttnn.to_torch(ttnn.load_tensor("./tensors/arg0.tensorbin"))
    tt_bn_mean_torch = ttnn.to_torch(ttnn.load_tensor("./tensors/arg1.tensorbin"))
    tt_bn_bias_torch = ttnn.to_torch(ttnn.load_tensor("./tensors/arg2.tensorbin"))
    tt_bn_weight_torch = ttnn.to_torch(ttnn.load_tensor("./tensors/arg3.tensorbin"))
    tt_conv_bias_torch = ttnn.to_torch(ttnn.load_tensor("./tensors/arg4.tensorbin"))
    tt_conv_weights_torch = ttnn.to_torch(ttnn.load_tensor("./tensors/arg5.tensorbin"))
    
    # Assign the same weights to CPU model
    cpu_model.conv.weight.data = tt_conv_weights_torch
    cpu_model.conv.bias.data = tt_conv_bias_torch
    cpu_model.bn.weight.data = tt_bn_weight_torch
    cpu_model.bn.bias.data = tt_bn_bias_torch
    cpu_model.bn.running_mean.data = tt_bn_mean_torch
    cpu_model.bn.running_var.data = tt_bn_var_torch
    
    logger.info("Weights loaded successfully!")
    
    cpu_input = torch.load("conv_ip.pt", map_location="cpu")
    
    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)
    
    # Verify inputs and parameters match between TT and CPU
    logger.info("\n=== Verification: Comparing TT and CPU tensors ===")
    
    # 1. Compare input
    tt_input_ttnn = ttnn.load_tensor("./tensors/arg6.tensorbin")
    tt_input_torch = ttnn.to_torch(tt_input_ttnn)
    
    inputs_match = torch.allclose(tt_input_torch, cpu_input)
    logger.info(f"Inputs match (allclose): {inputs_match}")
    assert inputs_match, "TT and CPU inputs do not match!"
    
    # 2. Compare conv weights (arg5)
    conv_weights_match = torch.allclose(tt_conv_weights_torch, cpu_model.conv.weight.data)
    logger.info(f"Conv weights match (allclose): {conv_weights_match}")
    assert conv_weights_match, "TT and CPU conv weights do not match!"
    
    # 3. Compare conv bias (arg4)
    conv_bias_match = torch.allclose(tt_conv_bias_torch, cpu_model.conv.bias.data)
    logger.info(f"Conv bias match (allclose): {conv_bias_match}")
    assert conv_bias_match, "TT and CPU conv biases do not match!"
    
    # 4. Compare BN weight (arg3)
    bn_weight_match = torch.allclose(tt_bn_weight_torch, cpu_model.bn.weight.data)
    logger.info(f"BN weight match (allclose): {bn_weight_match}")
    assert bn_weight_match, "TT and CPU BN weights do not match!"
    
    # 5. Compare BN bias (arg2)
    bn_bias_match = torch.allclose(tt_bn_bias_torch, cpu_model.bn.bias.data)
    logger.info(f"BN bias match (allclose): {bn_bias_match}")
    assert bn_bias_match, "TT and CPU BN biases do not match!"
    
    # 6. Compare BN running_mean (arg1)
    bn_mean_match = torch.allclose(tt_bn_mean_torch, cpu_model.bn.running_mean.data)
    logger.info(f"BN running_mean match (allclose): {bn_mean_match}")
    assert bn_mean_match, "TT and CPU BN running_mean do not match!"
    
    # 7. Compare BN running_var (arg0)
    bn_var_match = torch.allclose(tt_bn_var_torch, cpu_model.bn.running_var.data)
    logger.info(f"BN running_var match (allclose): {bn_var_match}")
    assert bn_var_match, "TT and CPU BN running_var do not match!"
    
    logger.info("✓ All input tensors and model parameters match between TT and CPU!")
    
    # Compare outputs
    logger.info("\n=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output[0])
    
    # Print both outputs for comparison
    logger.info("TT output tensor:")
    logger.info(f"{tt_output_torch}")
    logger.info(f"TT output shape: {tt_output_torch.shape}")
    logger.info(f"TT output dtype: {tt_output_torch.dtype}")
    
    logger.info("\nCPU output tensor:")
    logger.info(f"{cpu_output}")
    logger.info(f"CPU output shape: {cpu_output.shape}")
    logger.info(f"CPU output dtype: {cpu_output.dtype}")
    
    # Compute PCC
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")
