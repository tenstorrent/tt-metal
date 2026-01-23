import ttnn
import tests.ttnn.unit_tests.operations.conv2d_bn2d.utils as utils
import torch
import torch.nn as nn
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


class CPUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
    def forward(self, x):
        x = self.conv(x)
        return x


def test_conv2d_exp2():
    device = utils.DeviceGetter.get_device((1, 1))
    
    # Load tensors
    logger.info("Loading tensors...")
    conv_bias = ttnn.to_torch(utils.load_tensor("./tensors/arg4.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    conv_weight_ttnn = utils.load_tensor("./tensors/arg5.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None)
    input_ttnn = utils.load_tensor("./tensors/arg6.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, device, 
                                    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
    
    # TT inference
    logger.info("Running TT inference (conv2d as ttnn op)...")
    
    # Preprocess input using torch
    input_torch = ttnn.to_torch(input_ttnn)
    input_torch = input_torch.permute(0, 2, 3, 1).reshape(1, 1, 784, 128)
    
    # Convert to ttnn for conv2d
    input_ttnn_preprocessed = ttnn.from_torch(
        input_torch, 
        layout=ttnn.Layout.ROW_MAJOR,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    
    # Prepare conv bias using torch
    conv_bias_prepared = conv_bias.reshape(1, 128, 1, 1).permute(0, 2, 3, 1)
    conv_bias_ttnn = ttnn.from_torch(
        conv_bias_prepared,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    
    # TTNN CONV2D operation
    tt_output = ttnn.conv2d(
        input_tensor=input_ttnn_preprocessed,
        weight_tensor=conv_weight_ttnn,
        device=device,
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
        bias_tensor=conv_bias_ttnn,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    
    logger.info("tt_output : {}", tt_output)
    logger.info("tt_output.shape : {}", tt_output.shape)
    logger.info("tt_output.dtype : {}", tt_output.dtype)
    
    # CPU inference
    logger.info("Running CPU inference...")
    cpu_model = CPUModel().to(torch.bfloat16)
    cpu_model.eval()
    
    # Load weights into CPU model
    conv_weight_torch = ttnn.to_torch(conv_weight_ttnn)
    cpu_model.conv.weight.data = conv_weight_torch
    cpu_model.conv.bias.data = conv_bias
    
    cpu_input = torch.load("conv_ip.pt", map_location="cpu")
    logger.info("CPU input shape: {}", cpu_input.shape)
    
    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)
    
    # Compare outputs
    logger.info("\n=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output)
    
    # Postprocess tt output using torch to match CPU output format
    tt_output_torch = tt_output_torch.reshape(1, 28, 28, 128).permute(0, 3, 1, 2)
    
    logger.info("tt_output_torch: {}", tt_output_torch)
    logger.info("tt_output_torch.shape: {}", tt_output_torch.shape)
    logger.info("tt_output_torch.dtype: {}", tt_output_torch.dtype)
    
    logger.info("cpu_output: {}", cpu_output)
    logger.info("cpu_output.shape: {}", cpu_output.shape)
    logger.info("cpu_output.dtype: {}", cpu_output.dtype)
    
    # Compute PCC
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")
