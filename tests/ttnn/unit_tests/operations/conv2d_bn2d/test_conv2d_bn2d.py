import ttnn

import tests.ttnn.unit_tests.operations.conv2d_bn2d.utils as utils
import torch
import torch.nn as nn
from loguru import logger
from pytorchcv.model_provider import get_model as ptcv_get_model


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
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    util_create_list_0 = [ttnn_reshape_0]
    return util_create_list_0


def main_const_eval_1(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_1 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_to_device_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_1, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_to_layout_1,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    util_create_list_1 = [ttnn_reshape_1]
    return util_create_list_1


def main_const_eval_2(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_2 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_to_device_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_2, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    util_create_list_2 = [ttnn_reshape_2]
    return util_create_list_2


def main_const_eval_3(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_device_3 = ttnn.to_device(
        input_0,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_to_device_3,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_3, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_to_layout_3,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    util_create_list_3 = [ttnn_reshape_3]
    return util_create_list_3


CACHED_main_const_eval_0 = None
CACHED_main_const_eval_1 = None
CACHED_main_const_eval_2 = None
CACHED_main_const_eval_3 = None


def _main(input):
    global CACHED_main_const_eval_3
    global CACHED_main_const_eval_2
    global CACHED_main_const_eval_1
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    input_2 = input[2]
    input_3 = input[3]
    input_4 = input[4]
    input_5 = input[5]
    const_0 = main_const_eval_0
    util_create_list_4 = [input_1]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(const_0, util_create_list_4, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_1 = main_const_eval_1
    util_create_list_5 = [input_0]
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(const_1, util_create_list_5, CACHED_main_const_eval_1)
    CACHED_main_const_eval_1 = utils_constEvalFuncWrapper_1
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_2 = main_const_eval_2
    util_create_list_6 = [input_3]
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(const_2, util_create_list_6, CACHED_main_const_eval_2)
    CACHED_main_const_eval_2 = utils_constEvalFuncWrapper_2
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_3 = main_const_eval_3
    util_create_list_7 = [input_2]
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(const_3, util_create_list_7, CACHED_main_const_eval_3)
    CACHED_main_const_eval_3 = utils_constEvalFuncWrapper_3
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_4 = ttnn.to_layout(
        input_5,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_5, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_4,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 12544, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_reshape_4,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_4, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_5,
        weight_tensor=input_4,
        device=utils_DeviceGetter_get_device_4,
        in_channels=32,
        out_channels=64,
        batch_size=1,
        input_height=112,
        input_width=112,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_conv2d_0,
        [1, 112, 112, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_5,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_5, False)
    ttnn_batch_norm_0 = ttnn.batch_norm(
        ttnn_permute_1,
        running_mean=utils_constEvalFuncWrapper_0_0,
        running_var=utils_constEvalFuncWrapper_1_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_2_0,
        bias=utils_constEvalFuncWrapper_3_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    util_create_list_8 = [ttnn_batch_norm_0]
    return util_create_list_8


def load_inputs_for__main():
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
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
        utils_DeviceGetter_get_device_5,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_9 = [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
    ]
    return util_create_list_9


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.s1_p_conv = model.encoder.backbone.stage1.unit1.pw_conv.conv
        self.s1_p_bn = model.encoder.backbone.stage1.unit1.pw_conv.bn

    def forward(self, x):
        x = self.s1_p_conv(x)
        x = self.s1_p_bn(x)
        return x


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


def test_conv2d_bn2d():
    # TT run
    logger.info("Running TT inference...")
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)
    logger.info("tt_output: {}", tt_output)
    logger.info("tt_output[0].shape: {}", tt_output[0].shape)
    logger.info("tt_output[0].dtype: {}", tt_output[0].dtype)

    # CPU run
    logger.info("Running CPU inference...")
    model = ptcv_get_model("lwopenpose2d_mobilenet_cmupan_coco", pretrained=True).to(torch.bfloat16)
    cpu_model = Wrapper(model)
    cpu_model.eval()

    cpu_input = torch.load("cpu_input.pt", map_location="cpu")
    logger.info("cpu_input shape: {}", cpu_input.shape)
    logger.info("cpu_input dtype: {}", cpu_input.dtype)

    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)

    logger.info("cpu_output: {}", cpu_output)
    logger.info("cpu_output shape: {}", cpu_output.shape)
    logger.info("cpu_output dtype: {}", cpu_output.dtype)

    # Verify inputs and parameters match between TT and CPU
    logger.info("\n=== Verification: Comparing TT and CPU tensors ===")

    # 1. Compare input
    tt_input_ttnn = ttnn.load_tensor("./tensors/arg5.tensorbin")
    tt_input_torch = ttnn.to_torch(tt_input_ttnn)

    inputs_match = torch.allclose(tt_input_torch, cpu_input)
    logger.info(f"Inputs match (allclose): {inputs_match}")
    assert inputs_match, "TT and CPU inputs do not match!"

    # 2. Compare conv weights (arg4)
    tt_conv_weights_ttnn = ttnn.load_tensor("./tensors/arg4.tensorbin")
    tt_conv_weights_torch = ttnn.to_torch(tt_conv_weights_ttnn)
    cpu_conv_weights = cpu_model.s1_p_conv.weight.data

    conv_weights_match = torch.allclose(tt_conv_weights_torch, cpu_conv_weights)
    logger.info(f"Conv weights match (allclose): {conv_weights_match}")
    assert conv_weights_match, "TT and CPU conv weights do not match!"

    # 3. Compare BN weight (arg3)
    tt_bn_weight_ttnn = ttnn.load_tensor("./tensors/arg3.tensorbin")
    tt_bn_weight_torch = ttnn.to_torch(tt_bn_weight_ttnn).squeeze()
    cpu_bn_weight = cpu_model.s1_p_bn.weight.data

    bn_weight_match = torch.allclose(tt_bn_weight_torch, cpu_bn_weight)
    logger.info(f"BN weight match (allclose): {bn_weight_match}")
    assert bn_weight_match, "TT and CPU BN weights do not match!"

    # 4. Compare BN bias (arg2)
    tt_bn_bias_ttnn = ttnn.load_tensor("./tensors/arg2.tensorbin")
    tt_bn_bias_torch = ttnn.to_torch(tt_bn_bias_ttnn).squeeze()
    cpu_bn_bias = cpu_model.s1_p_bn.bias.data

    bn_bias_match = torch.allclose(tt_bn_bias_torch, cpu_bn_bias)
    logger.info(f"BN bias match (allclose): {bn_bias_match}")
    assert bn_bias_match, "TT and CPU BN biases do not match!"

    # 5. Compare BN running_mean (arg1)
    tt_bn_mean_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    tt_bn_mean_torch = ttnn.to_torch(tt_bn_mean_ttnn).squeeze()
    cpu_bn_mean = cpu_model.s1_p_bn.running_mean.data

    bn_mean_match = torch.allclose(tt_bn_mean_torch, cpu_bn_mean)
    logger.info(f"BN running_mean match (allclose): {bn_mean_match}")
    assert bn_mean_match, "TT and CPU BN running_mean do not match!"

    # 6. Compare BN running_var (arg0)
    tt_bn_var_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    tt_bn_var_torch = ttnn.to_torch(tt_bn_var_ttnn).squeeze()
    cpu_bn_var = cpu_model.s1_p_bn.running_var.data

    bn_var_match = torch.allclose(tt_bn_var_torch, cpu_bn_var)
    logger.info(f"BN running_var match (allclose): {bn_var_match}")
    assert bn_var_match, "TT and CPU BN running_var do not match!"

    logger.info("✓ All input tensors and model parameters match between TT and CPU!")

    # Compare outputs
    logger.info("\n=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output[0])

    # Compute PCC
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")
