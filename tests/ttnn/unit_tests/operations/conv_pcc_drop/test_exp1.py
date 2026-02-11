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


def test_exp1():
    device = utils.DeviceGetter.get_device((1, 1))

    # Load all tensors 
    logger.info("Loading tensors...")
    conv_bias = ttnn.to_torch(utils.load_tensor("./tensors/arg0.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    conv_weight = ttnn.to_torch(utils.load_tensor("./tensors/arg1.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    input_tensor = ttnn.to_torch(utils.load_tensor("./tensors/arg2.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))

    # --- TT inference ---
    logger.info("Running TT inference...")

    # Preprocess input using torch: reshape to 4D NCHW then permute to NHWC for ttnn.conv2d
    input_for_conv = input_tensor.reshape(1, 128, 1, 8192).permute(0, 2, 3, 1)
    input_ttnn = ttnn.from_torch(
        input_for_conv,
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    
    # Prepare weight using torch: [128, 128, 3] -> [128, 128, 1, 3] for conv2d
    conv_weight_4d = conv_weight.reshape(128, 128, 1, 3)
    conv_weight_ttnn = ttnn.from_torch(conv_weight_4d, layout=ttnn.Layout.ROW_MAJOR)

    # Prepare bias using torch: reshape to NHWC [1, 1, 1, 128]
    conv_bias_nhwc = conv_bias.reshape(1, 128, 1, 1).permute(0, 2, 3, 1)
    conv_bias_ttnn = ttnn.from_torch(
        conv_bias_nhwc,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # TTNN CONV2D
    tt_conv_output = ttnn.conv2d(
        input_tensor=input_ttnn,
        weight_tensor=conv_weight_ttnn,
        device=device,
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
        bias_tensor=conv_bias_ttnn,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # Postprocess output using torch: NHWC -> NCHW -> 3D
    tt_output_torch = ttnn.to_torch(tt_conv_output)
    tt_output_torch = tt_output_torch.permute(0, 3, 1, 2).reshape(1, 128, 8192)

    # --- CPU inference (original model uses Conv1d) ---
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
