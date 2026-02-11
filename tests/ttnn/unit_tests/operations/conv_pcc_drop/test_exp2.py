import torch
import ttnn
import tests.ttnn.unit_tests.operations.conv_pcc_drop.utils as utils
from loguru import logger


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_flat, y_flat = x.flatten(), y.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return ((vx @ vy) / denom).item()


def test_exp2():
    device = utils.DeviceGetter.get_device((1, 1))

    # Load all tensors 
    logger.info("Loading tensors...")
    conv_bias = ttnn.to_torch(utils.load_tensor("./tensors/arg0.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    conv_weight = ttnn.to_torch(utils.load_tensor("./tensors/arg1.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    input_tensor = ttnn.to_torch(utils.load_tensor("./tensors/arg2.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    
    logger.info("conv_bias.shape={}",conv_bias.shape)
    logger.info("conv_weight_bias.shape={}",conv_weight.shape)
    logger.info("input_tensor.shape={}",input_tensor.shape)

    # --- TT inference ---
    logger.info("Running TT inference...")

    # Preprocess input: permute NCL to NLC format for ttnn.conv1d
    # input_tensor is already [1, 128, 8192] in NCL format
    input_for_conv = input_tensor.permute(0, 2, 1)  # NLC format: [1, 8192, 128]
    input_ttnn = ttnn.from_torch(
        input_for_conv,
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    
    # Prepare weight for conv1d: keep 3D shape [out_channels, in_channels, kernel_size]
    conv_weight_ttnn = ttnn.from_torch(conv_weight, layout=ttnn.Layout.ROW_MAJOR)

    # Prepare bias for conv1d: reshape to [1, 1, 1, out_channels]
    conv_bias_reshaped = conv_bias.reshape(1, 1, 1, 128)
    conv_bias_ttnn = ttnn.from_torch(
        conv_bias_reshaped,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # TTNN CONV1D
    conv_config = ttnn.Conv1dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=True,
        config_tensors_in_dram=False,
    )
    
    tt_conv_output, out_length = ttnn.conv1d(
        input_tensor=input_ttnn,
        weight_tensor=conv_weight_ttnn,
        device=device,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_length=8192,
        kernel_size=3,
        stride=1,
        padding=3,
        dilation=3,
        groups=1,
        bias_tensor=conv_bias_ttnn,
        conv_config=conv_config,
        compute_config=None,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )

    # Postprocess output using torch: NLC -> NCL
    tt_output_torch = ttnn.to_torch(tt_conv_output)
    tt_output_torch = tt_output_torch.reshape(1, out_length, 128).permute(0, 2, 1)

    # --- CPU inference ---
    logger.info("Running CPU inference...")
    cpu_conv = torch.nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=3, dilation=3)
    logger.info("cpu_conv={}",cpu_conv)
    cpu_conv.weight.data = conv_weight
    cpu_conv.bias.data = conv_bias
    cpu_conv.eval()

    with torch.no_grad():
        cpu_output = cpu_conv(input_tensor)

    # Compare outputs
    logger.info("=== Output Comparison ===")
    pcc = compute_pcc(tt_output_torch.to(torch.float64), cpu_output.to(torch.float64))
    logger.info(f"Output PCC: {pcc}")
