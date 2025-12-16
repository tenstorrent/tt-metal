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


def _main_experiment_2(maxpool_output_torch, weights_torch, bias_torch):
    device = utils.DeviceGetter.get_device((1, 1))

    bias_reshaped = bias_torch.reshape(1, 1, 1, 16)

    # Transfer maxpool output to device
    maxpool_ttnn = ttnn.from_torch(
        maxpool_output_torch,
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

    # Run conv_transpose2d
    convt_output_ttnn = ttnn.conv_transpose2d(
        input_tensor=maxpool_ttnn,
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

    convt_output_torch = ttnn.to_torch(convt_output_ttnn)

    convt_reshaped = convt_output_torch.reshape(1, 14, 14, 16)
    convt_nchw = convt_reshaped.permute(0, 3, 1, 2)
    return convt_nchw


def load_inputs_for_experiment_2():
    # Load bias (arg0)
    bias_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    bias_torch = ttnn.to_torch(bias_ttnn).squeeze()

    # Load weights (arg1)
    weights_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    weights_torch = ttnn.to_torch(weights_ttnn)

    # Load saved maxpool output
    maxpool_output_torch = torch.load("maxpool_output_torch.pt", map_location="cpu")
    logger.info("convtranspose2d input={}", maxpool_output_torch)
    logger.info("convtranspose2d input shape={}", maxpool_output_torch.shape)
    logger.info("convtranspose2d input dtype={}", maxpool_output_torch.dtype)

    return maxpool_output_torch, weights_torch, bias_torch


class ConvAE(torch.nn.Module):
    def __init__(self, weights=None, bias=None):
        super().__init__()
        self.decoder_conv2d_1 = torch.nn.ConvTranspose2d(4, 16, 2, stride=2)
        if weights is not None:
            self.decoder_conv2d_1.weight.data = weights
        if bias is not None:
            self.decoder_conv2d_1.bias.data = bias

    def forward(self, act):
        act = self.decoder_conv2d_1(act)
        return act


def test_experiment_2():
    # Load tensors
    maxpool_output_torch, weights_torch, bias_torch = load_inputs_for_experiment_2()

    # tt run (only conv_transpose2d)
    tt_convt_output = _main_experiment_2(maxpool_output_torch, weights_torch, bias_torch)

    # CPU run
    cpu_input_flattened = torch.load("maxpool_output_torch.pt", map_location="cpu")

    # Convert from flattened [1, 1, 49, 4] to NCHW [1, 4, 7, 7] for CPU ConvTranspose2d
    cpu_input_reshaped = cpu_input_flattened.reshape(1, 7, 7, 4)
    cpu_input_nchw = cpu_input_reshaped.permute(0, 3, 1, 2)

    # Load the actual weights and bias used by ttnn
    weights_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    weights_torch = ttnn.to_torch(weights_ttnn)

    bias_ttnn = ttnn.load_tensor("./tensors/arg0.tensorbin")
    bias_torch = ttnn.to_torch(bias_ttnn).squeeze()

    cpu_model = ConvAE(weights=weights_torch, bias=bias_torch).to(torch.bfloat16)

    with torch.no_grad():
        cpu_convt_output = cpu_model(cpu_input_nchw)

    # ========================================================================
    # Comparison: Final Output
    # ========================================================================

    # Compute PCC for convt
    pcc_convt = compute_pcc(tt_convt_output, cpu_convt_output)
    logger.info(f"\nFinal PCC: {pcc_convt}")

    # ========================================================================
    # Verification: Comparing TT and CPU tensors
    # ========================================================================

    # 1. Compare conv_transpose2d inputs (should match since both loaded from same file)
    tt_convt_input_flattened = torch.load("maxpool_output_torch.pt", map_location="cpu")
    # Convert CPU input from NCHW back to flattened format for comparison
    cpu_convt_input_flattened = cpu_input_nchw.permute(0, 2, 3, 1).reshape(1, 1, 49, 4)

    inputs_match = torch.allclose(tt_convt_input_flattened, cpu_convt_input_flattened)
    logger.info(f"ConvTranspose2d inputs match (allclose): {inputs_match}")
    assert inputs_match, "TT and CPU conv_transpose2d inputs do not match!"

    # 2. Compare conv weights (arg1)
    conv_weights_match = torch.allclose(weights_torch, cpu_model.decoder_conv2d_1.weight.data)
    logger.info(f"Conv weights match (allclose): {conv_weights_match}")
    assert conv_weights_match, "TT and CPU conv weights do not match!"

    # 3. Compare conv bias (arg0)
    conv_bias_match = torch.allclose(bias_torch, cpu_model.decoder_conv2d_1.bias.data)
    logger.info(f"Conv bias match (allclose): {conv_bias_match}")
    assert conv_bias_match, "TT and CPU conv biases do not match!"

    logger.info("✓ All input tensors and model parameters match between TT and CPU!")
