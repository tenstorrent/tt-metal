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


def _main_exp3(input_torch, weights_torch, bias_torch):
    device = utils.DeviceGetter.get_device((1, 1))

    # pre prcoessing
    input_nhwc = input_torch.permute(0, 2, 3, 1)
    input_flattened = input_nhwc.reshape(1, 1, 196, 4)
    bias_reshaped = bias_torch.reshape(1, 1, 1, 16)

    # torch -> ttnn tensor conversion
    input_ttnn = ttnn.from_torch(
        input_flattened,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    weights_ttnn = ttnn.from_torch(
        weights_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    bias_ttnn = ttnn.from_torch(
        bias_reshaped,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # ttnn ops
    maxpool_output_ttnn = ttnn.max_pool2d(
        input_ttnn,
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

    convt_output_ttnn = ttnn.conv_transpose2d(
        input_tensor=maxpool_output_ttnn,
        weight_tensor=weights_ttnn,
        device=device,
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
        bias_tensor=bias_ttnn,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # post processing
    maxpool_output_torch = ttnn.to_torch(maxpool_output_ttnn)

    logger.info("maxpool_output_torch before reshape & permute ")

    logger.info("maxpool_output_torch={}", maxpool_output_torch)
    logger.info("maxpool_output_torch.shape={}", maxpool_output_torch.shape)
    logger.info("maxpool_output_torch.dtype={}", maxpool_output_torch.dtype)
    # torch.save(maxpool_output_torch, "maxpool_output_torch.pt")

    convt_output_torch = ttnn.to_torch(convt_output_ttnn)

    convt_reshaped = convt_output_torch.reshape(1, 14, 14, 16)
    convt_nchw = convt_reshaped.permute(0, 3, 1, 2)
    return maxpool_output_torch, convt_nchw


def load_inputs_for__main():
    # Load bias (arg0)
    bias_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    bias_torch = ttnn.to_torch(bias_ttnn).squeeze()

    # Load weights (arg1)
    weights_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    weights_torch = ttnn.to_torch(weights_ttnn)

    # Load input (arg2)
    input_ttnn = ttnn.load_tensor("./tensors/arg2.tensorbin")
    input_torch = ttnn.to_torch(input_ttnn)
    return input_torch, weights_torch, bias_torch


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
        act1 = self.encoder_max_pool2d(act)
        act2 = self.decoder_conv2d_1(act1)
        return act1, act2


def test_experiment_1():
    # Load tensors
    input_torch, weights_torch, bias_torch = load_inputs_for__main()

    # tt run
    tt_maxpool_output, tt_convt_output = _main_exp3(input_torch, weights_torch, bias_torch)

    # CPU run

    # load same weights and bias as ttnn
    cpu_input = torch.load("cpu_input.pt", map_location="cpu")

    # Load the actual weights and bias used by ttnn
    weights_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    weights_torch = ttnn.to_torch(weights_ttnn)

    bias_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    bias_torch = ttnn.to_torch(bias_ttnn).squeeze()

    cpu_model = ConvAE(weights=weights_torch, bias=bias_torch).to(torch.bfloat16)

    with torch.no_grad():
        cpu_maxpool_output, cpu_convt_output = cpu_model(cpu_input)

    # ========================================================================
    # COMPARISON 1: MaxPool2d Output
    # ========================================================================

    tt_maxpool_reshaped = tt_maxpool_output.reshape(1, 7, 7, 4)
    tt_maxpool_nchw = tt_maxpool_reshaped.permute(0, 3, 1, 2)

    # Compute PCC for maxpool
    pcc_maxpool = compute_pcc(tt_maxpool_nchw, cpu_maxpool_output)
    logger.info(f"\nMaxPool2d PCC: {pcc_maxpool}")

    # ========================================================================
    # COMPARISON 2: Final Output
    # ========================================================================

    # Compute PCC for convt
    pcc_convt = compute_pcc(tt_convt_output, cpu_convt_output)
    logger.info(f"\nFinal PCC: {pcc_convt}")

    # ========================================================================
    # Verification: Comparing TT and CPU input tensors
    # ========================================================================

    # 1. Compare input
    inputs_match = torch.allclose(input_torch, cpu_input)
    logger.info(f"Inputs match (allclose): {inputs_match}")
    assert inputs_match, "TT and CPU inputs do not match!"

    # 2. Compare conv weights (arg1)
    conv_weights_match = torch.allclose(weights_torch, cpu_model.decoder_conv2d_1.weight.data)
    logger.info(f"Conv weights match (allclose): {conv_weights_match}")
    assert conv_weights_match, "TT and CPU conv weights do not match!"

    # 3. Compare conv bias (arg0)
    conv_bias_match = torch.allclose(bias_torch, cpu_model.decoder_conv2d_1.bias.data)
    logger.info(f"Conv bias match (allclose): {conv_bias_match}")
    assert conv_bias_match, "TT and CPU conv biases do not match!"

    logger.info("✓ All input tensors and model parameters match between TT and CPU!")

    logger.info("tt_maxpool_ouput={}", tt_maxpool_nchw)
    logger.info("cpu_maxpool_output={}", cpu_maxpool_output)

    logger.info("tt_convt_output={}", tt_convt_output)
    logger.info("cpu_convt_output={}", cpu_convt_output)
