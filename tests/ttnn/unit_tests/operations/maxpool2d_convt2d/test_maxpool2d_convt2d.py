import ttnn
import tests.ttnn.unit_tests.operations.maxpool2d_convt2d.utils as utils
import torch
from loguru import logger


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
        [1, 1, 1, 16],
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
        [1, 1, 196, 4],
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
    ttnn_max_pool2d_0 = ttnn.max_pool2d(
        ttnn_to_layout_3,
        1,
        14,
        14,
        4,
        [2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        ceil_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        applied_shard_scheme=None,
        reallocate_halo_output=False,
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn_conv_transpose2d_0 = ttnn.conv_transpose2d(
        input_tensor=ttnn_max_pool2d_0,
        weight_tensor=utils_constEvalFuncWrapper_1_0,
        device=utils_DeviceGetter_get_device_1,
        in_channels=4,
        out_channels=16,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        output_padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=utils_constEvalFuncWrapper_0_0,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_max_pool2d_0, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_conv_transpose2d_0,
        [1, 14, 14, 16],
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
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_5 = [utils_load_tensor_0, utils_load_tensor_1, utils_load_tensor_2]
    return util_create_list_5


class ConvAE(torch.nn.Module):
    def __init__(self, weights=None, bias=None):
        super().__init__()
        self.encoder_max_pool2d = torch.nn.MaxPool2d(2, 2)
        self.decoder_conv2d_1 = torch.nn.ConvTranspose2d(4, 16, 2, stride=2)
        # Load provided weights and bias if available
        if weights is not None:
            self.decoder_conv2d_1.weight.data = weights
        if bias is not None:
            self.decoder_conv2d_1.bias.data = bias

    def forward(self, act):
        act = self.encoder_max_pool2d(act)
        act = self.decoder_conv2d_1(act)
        return act


def test_maxpool2d_conv2dt():
    # tt run
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)

    # cpu run - load same weights and bias as ttnn
    cpu_input = torch.load("cpu_input.pt", map_location="cpu")
    logger.info("cpu_input={}", cpu_input)
    logger.info("cpu_input.shape={}", cpu_input.shape)

    # Load the actual weights and bias used by ttnn
    weights_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    weights_torch = ttnn.to_torch(weights_ttnn)

    bias_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    bias_torch = ttnn.to_torch(bias_ttnn).squeeze()

    logger.info(f"Loaded weights shape: {weights_torch.shape}, bias shape: {bias_torch.shape}")

    cpu_model = ConvAE(weights=weights_torch, bias=bias_torch).to(torch.bfloat16)

    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)

    # Verify inputs and parameters match between TT and CPU
    logger.info("\n=== Verification: Comparing TT and CPU tensors ===")

    # 1. Compare input
    tt_input_ttnn = ttnn.load_tensor("./tensors/arg2.tensorbin")
    tt_input_torch = ttnn.to_torch(tt_input_ttnn)

    inputs_match = torch.allclose(tt_input_torch, cpu_input)
    logger.info(f"Inputs match (allclose): {inputs_match}")

    # 2. Compare conv weights (arg1)
    conv_weights_match = torch.allclose(weights_torch, cpu_model.decoder_conv2d_1.weight.data)
    logger.info(f"Conv weights match (allclose): {conv_weights_match}")

    # 3. Compare conv bias (arg0)
    conv_bias_match = torch.allclose(bias_torch, cpu_model.decoder_conv2d_1.bias.data)
    logger.info(f"Conv bias match (allclose): {conv_bias_match}")

    # Compare outputs
    logger.info("\n=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output[0])

    logger.info(f"tt_output_torch shape: {tt_output_torch.shape}")
    logger.info(f"cpu_output shape: {cpu_output.shape}")

    # Compute PCC
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")

    logger.info("tt_output={}", tt_output)
    logger.info("cpu_output={}", cpu_output)
