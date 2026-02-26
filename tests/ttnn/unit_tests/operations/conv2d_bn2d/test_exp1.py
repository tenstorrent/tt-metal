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
        self.conv = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def test_exp1():
    device = utils.DeviceGetter.get_device((1, 1))

    # Load tensors and convert to torch
    # arg0 -> bn.running_var, arg1 -> bn.running_mean, arg2 -> bn.bias, arg3 -> bn.weight, arg4 -> conv.weight
    logger.info("Loading tensors...")
    bn_running_var = ttnn.to_torch(utils.load_tensor("arg0.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    bn_running_mean = ttnn.to_torch(utils.load_tensor("arg1.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    bn_bias = ttnn.to_torch(utils.load_tensor("arg2.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    bn_weight = ttnn.to_torch(utils.load_tensor("arg3.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    conv_weight_ttnn = utils.load_tensor("arg4.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None)
    input_ttnn = utils.load_tensor("arg5.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, device, DRAM_MEMCFG)

    # --- TT inference: conv2d + batch_norm only as TTNN ops ---
    logger.info("Running TT inference...")

    input_torch = ttnn.to_torch(input_ttnn)  
    input_torch = input_torch.permute(0, 2, 3, 1).reshape(1, 1, 65536, 128)

    input_for_conv = ttnn.from_torch(input_torch, layout=ttnn.Layout.ROW_MAJOR, device=device, memory_config=DRAM_MEMCFG)

    # [TTNN] Conv2d 
    conv_output = ttnn.conv2d(
        input_tensor=input_for_conv,
        weight_tensor=conv_weight_ttnn,
        device=device,
        in_channels=128,
        out_channels=256,
        batch_size=1,
        input_height=256,
        input_width=256,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        ),
        slice_config=None,
        memory_config=DRAM_MEMCFG,
    )


    conv_output_torch = ttnn.to_torch(conv_output)  
    conv_output_torch = conv_output_torch.reshape(1, 128, 128, 256).permute(0, 3, 1, 2)

    conv_output_for_bn = ttnn.from_torch(conv_output_torch, layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)

    bn_running_mean_ttnn = ttnn.from_torch(bn_running_mean.reshape(1, 256, 1, 1), layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)
    bn_running_var_ttnn = ttnn.from_torch(bn_running_var.reshape(1, 256, 1, 1), layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)
    bn_weight_ttnn = ttnn.from_torch(bn_weight.reshape(1, 256, 1, 1), layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)
    bn_bias_ttnn = ttnn.from_torch(bn_bias.reshape(1, 256, 1, 1), layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)

    # [TTNN] Batch norm 
    tt_output = ttnn.batch_norm(
        conv_output_for_bn,
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
    logger.info("TT output shape: {}", tt_output_torch.shape)
    logger.info("TT output dtype: {}", tt_output_torch.dtype)

    # --- CPU golden inference ---
    logger.info("Running CPU inference...")
    cpu_model = CPUModel().to(torch.bfloat16)
    cpu_model.eval()

    conv_weight_torch = ttnn.to_torch(conv_weight_ttnn)
    cpu_model.conv.weight.data = conv_weight_torch
    cpu_model.bn.weight.data = bn_weight
    cpu_model.bn.bias.data = bn_bias
    cpu_model.bn.running_mean.data = bn_running_mean
    cpu_model.bn.running_var.data = bn_running_var

    cpu_input = ttnn.to_torch(input_ttnn).to(torch.bfloat16)  
    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)

    # --- Compare ---
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")
    logger.info("CPU output shape: {}", cpu_output.shape)
