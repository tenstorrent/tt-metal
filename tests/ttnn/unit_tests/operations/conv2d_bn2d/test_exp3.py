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
        self.bn = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
    def forward(self, x):
        x = self.bn(x)
        return x


def test_batchnorm2d_exp3():
    device = utils.DeviceGetter.get_device((1, 1))
    
    # Load tensors
    logger.info("Loading tensors...")
    bn_running_var = ttnn.to_torch(utils.load_tensor("./tensors/arg0.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    bn_running_mean = ttnn.to_torch(utils.load_tensor("./tensors/arg1.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    bn_bias = ttnn.to_torch(utils.load_tensor("./tensors/arg2.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    bn_weight = ttnn.to_torch(utils.load_tensor("./tensors/arg3.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    
    # Load input for batch_norm
    cpu_input = torch.load("bn_ip.pt", map_location="cpu")
    logger.info("CPU input shape: {}", cpu_input.shape)
    logger.info("CPU input dtype: {}", cpu_input.dtype)
    
    # TT inference
    logger.info("Running TT inference (batch_norm as ttnn op)...")
    
    # Convert input to ttnn
    input_ttnn = ttnn.from_torch(
        cpu_input,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    
    # Prepare batch_norm parameters using torch
    bn_weight_prepared = bn_weight.reshape(1, 128, 1, 1)
    bn_bias_prepared = bn_bias.reshape(1, 128, 1, 1)
    bn_running_mean_prepared = bn_running_mean.reshape(1, 128, 1, 1)
    bn_running_var_prepared = bn_running_var.reshape(1, 128, 1, 1)
    
    bn_weight_ttnn = ttnn.from_torch(bn_weight_prepared, layout=ttnn.Layout.TILE, device=device,
                                      memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
    bn_bias_ttnn = ttnn.from_torch(bn_bias_prepared, layout=ttnn.Layout.TILE, device=device,
                                    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
    bn_running_mean_ttnn = ttnn.from_torch(bn_running_mean_prepared, layout=ttnn.Layout.TILE, device=device,
                                            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
    bn_running_var_ttnn = ttnn.from_torch(bn_running_var_prepared, layout=ttnn.Layout.TILE, device=device,
                                           memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None))
    
    # TTNN BATCH_NORM operation
    tt_output = ttnn.batch_norm(
        input_ttnn,
        running_mean=bn_running_mean_ttnn,
        running_var=bn_running_var_ttnn,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=bn_weight_ttnn,
        bias=bn_bias_ttnn,
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
    cpu_model.bn.weight.data = bn_weight
    cpu_model.bn.bias.data = bn_bias
    cpu_model.bn.running_mean.data = bn_running_mean
    cpu_model.bn.running_var.data = bn_running_var
    
    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)
    
    # Compare outputs
    logger.info("\n=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output)
    
    logger.info("tt_output_torch: {}", tt_output_torch)
    logger.info("tt_output_torch.shape: {}", tt_output_torch.shape)
    logger.info("tt_output_torch.dtype: {}", tt_output_torch.dtype)
    
    logger.info("cpu_output: {}", cpu_output)
    logger.info("cpu_output.shape: {}", cpu_output.shape)
    logger.info("cpu_output.dtype: {}", cpu_output.dtype)
    
    # Compute PCC
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")
