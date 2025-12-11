import ttnn
import tests.ttnn.unit_tests.operations.conv2d_sig.utils as utils
import torch
import torch.nn as nn
from loguru import logger
from .model import download_nvidia_model


def main_const_eval_0(input):
    input_0 = input[0]
    ttnn_reshape_0 = ttnn.reshape(
        input_0,
        [1, 1, 1, 720],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_reshape_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_from_device_0 = ttnn.from_device(ttnn_to_layout_0)
    ttnn.deallocate(ttnn_to_layout_0, False)
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
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(const_0, util_create_list_1, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_permute_0 = ttnn.permute(
        input_2,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(input_2, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 80, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_reshape_1,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_1,
        weight_tensor=input_1,
        device=utils_DeviceGetter_get_device_0,
        in_channels=256,
        out_channels=720,
        batch_size=1,
        input_height=8,
        input_width=10,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=utils_constEvalFuncWrapper_0_0,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID),
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_conv2d_0,
        [1, 8, 10, 720],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_2,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    util_create_list_2 = [ttnn_permute_1]
    return util_create_list_2


def load_inputs_for__main():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
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
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_3 = [utils_load_tensor_0, utils_load_tensor_1, utils_load_tensor_2]
    return util_create_list_3


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.cls_head_8 = model.cls_head[8]

    def forward(self, t):
        t = self.cls_head_8(t)
        out = t.sigmoid()
        return out


def test_conv2d_sig():
    # TT run
    logger.info("Running TT inference...")
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)
    logger.info("tt_output : {}", tt_output)
    logger.info("tt_output[0].shape : {}", tt_output[0].shape)
    logger.info("tt_output[0].dtype : {}", tt_output[0].dtype)

    # CPU run
    logger.info("Running CPU inference...")

    from .model import Model

    model = Model.load(download_nvidia_model(variant_name="retinanet_rn18fpn"))
    model.to(torch.bfloat16)
    cpu_model = Wrapper(model)

    cpu_input = torch.load("cpu_input.pt", map_location="cpu")

    logger.info("cpu_input ={}", cpu_input)
    logger.info("cpu_input.shape={}", cpu_input.shape)
    logger.info("cpu_input.dtype={}", cpu_input.dtype)

    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)

    logger.info("cpu_output: {}", cpu_output)
    logger.info("cpu_output.shape: {}", cpu_output.shape)
    logger.info("cpu_output.dtype: {}", cpu_output.dtype)

    # Verify inputs, weights, and bias are the same between TT and CPU
    logger.info("\n=== Verification: Comparing TT and CPU tensors ===")

    # 1. Compare inputs
    tt_input_ttnn = ttnn.load_tensor("./tensors/arg2.tensorbin")
    tt_input_torch = ttnn.to_torch(tt_input_ttnn)

    inputs_match = torch.allclose(tt_input_torch, cpu_input)
    logger.info(f"Inputs match (allclose): {inputs_match}")
    assert inputs_match, "TT and CPU inputs do not match!"

    # 2. Compare weights
    tt_weights_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    tt_weights_torch = ttnn.to_torch(tt_weights_ttnn)
    cpu_weights = cpu_model.cls_head_8.weight.data

    weights_match = torch.allclose(tt_weights_torch, cpu_weights)
    logger.info(f"Weights match (allclose): {weights_match}")
    assert weights_match, "TT and CPU weights do not match!"

    # 3. Compare bias
    tt_bias_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    tt_bias_torch = ttnn.to_torch(tt_bias_ttnn).squeeze()
    cpu_bias = cpu_model.cls_head_8.bias.data

    bias_match = torch.allclose(tt_bias_torch, cpu_bias)
    logger.info(f"Bias match (allclose): {bias_match}")
    assert bias_match, "TT and CPU biases do not match!"

    logger.info("✓ All input tensors (input, weights, bias) match between TT and CPU!")

    # 4. Compare outputs
    logger.info("\n=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output[0])

    # Compute PCC for outputs
    x_flat = tt_output_torch.flatten().float()
    y_flat = cpu_output.flatten().float()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    pcc = ((vx @ vy) / denom).item() if denom != 0 else float("nan")

    logger.info(f"Output PCC: {pcc:.6f}")
