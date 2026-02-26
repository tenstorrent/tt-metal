import ttnn
import tests.ttnn.unit_tests.operations.conv2d_bn2d.utils as utils
import torch
import torch.nn as nn
from loguru import logger

DRAM_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_float = x.to(torch.float64) if x.dtype != torch.float64 else x
    y_float = y.to(torch.float64) if y.dtype != torch.float64 else y

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
        self.bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.bn(x)
        return x


def test_exp4():
    device = utils.DeviceGetter.get_device((1, 1))

    # Load BN parameters
    logger.info("Loading tensors...")
    bn_running_var = ttnn.to_torch(utils.load_tensor("arg0.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    bn_running_mean = ttnn.to_torch(utils.load_tensor("arg1.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    bn_bias = ttnn.to_torch(utils.load_tensor("arg2.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    bn_weight = ttnn.to_torch(utils.load_tensor("arg3.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))

    # Load TTNN conv2d intermediate output saved from exp2
    cpu_input = torch.load("ttnn_conv_output.pt", map_location="cpu")
    logger.info("Loaded ttnn_conv_output.pt shape: {}, dtype: {}", cpu_input.shape, cpu_input.dtype)

    # === TTNN BatchNorm on the TTNN conv2d output ===
    logger.info("Running TTNN batch_norm...")
    input_ttnn = ttnn.from_torch(cpu_input, layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)

    bn_running_mean_ttnn = ttnn.from_torch(bn_running_mean.reshape(1, 256, 1, 1), layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)
    bn_running_var_ttnn = ttnn.from_torch(bn_running_var.reshape(1, 256, 1, 1), layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)
    bn_weight_ttnn = ttnn.from_torch(bn_weight.reshape(1, 256, 1, 1), layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)
    bn_bias_ttnn = ttnn.from_torch(bn_bias.reshape(1, 256, 1, 1), layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)

    tt_output = ttnn.batch_norm(
        input_ttnn,
        running_mean=bn_running_mean_ttnn,
        running_var=bn_running_var_ttnn,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=bn_weight_ttnn,
        bias=bn_bias_ttnn,
        memory_config=DRAM_MEMCFG,
    )

    tt_output_torch = ttnn.to_torch(tt_output)
    logger.info("tt_output shape: {}, dtype: {}", tt_output_torch.shape, tt_output_torch.dtype)

    # === CPU BatchNorm on the same TTNN conv2d output ===
    logger.info("Running CPU batch_norm...")
    cpu_model = CPUModel().to(torch.bfloat16)
    cpu_model.eval()

    cpu_model.bn.weight.data = bn_weight
    cpu_model.bn.bias.data = bn_bias
    cpu_model.bn.running_mean.data = bn_running_mean
    cpu_model.bn.running_var.data = bn_running_var

    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)

    logger.info("cpu_output shape: {}, dtype: {}", cpu_output.shape, cpu_output.dtype)

    # === Compare ===
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"BatchNorm PCC (TTNN BN vs CPU BN, same TTNN conv input): {pcc}")
