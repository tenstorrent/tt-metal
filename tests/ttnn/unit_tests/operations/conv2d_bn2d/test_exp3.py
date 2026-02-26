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

    def forward(self, x):
        x = self.conv(x)
        return x


def test_exp3():
    device = utils.DeviceGetter.get_device((1, 1))

    # Load tensors
    # arg4 -> conv.weight(256x128x3x3), arg5 -> input(1x128x256x256)
    logger.info("Loading tensors...")
    conv_weight_ttnn = utils.load_tensor("arg4.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None)
    input_ttnn = utils.load_tensor("arg5.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, device, DRAM_MEMCFG)

    # === TT inference (conv2d only) ===
    logger.info("Running TT inference (conv2d as ttnn op)...")

    # Preprocess input using torch: NCHW → NHWC → flatten spatial
    input_torch = ttnn.to_torch(input_ttnn)
    input_torch = input_torch.permute(0, 2, 3, 1).reshape(1, 1, 65536, 128)

    input_for_conv = ttnn.from_torch(input_torch, layout=ttnn.Layout.ROW_MAJOR, device=device, memory_config=DRAM_MEMCFG)

    # [TTNN] Conv2d
    tt_output = ttnn.conv2d(
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

    logger.info("tt_output: {}", tt_output)
    logger.info("tt_output.shape: {}", tt_output.shape)
    logger.info("tt_output.dtype: {}", tt_output.dtype)

    # === CPU inference ===
    logger.info("Running CPU inference...")
    cpu_model = CPUModel().to(torch.bfloat16)
    cpu_model.eval()

    conv_weight_torch = ttnn.to_torch(conv_weight_ttnn)
    cpu_model.conv.weight.data = conv_weight_torch

    cpu_input = ttnn.to_torch(input_ttnn).to(torch.bfloat16)
    with torch.no_grad():
        cpu_output = cpu_model(cpu_input)

    # === Compare outputs ===
    logger.info("\n=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output)

    # Postprocess: NHWC [1,1,16384,256] → NCHW [1,256,128,128]
    tt_output_torch = tt_output_torch.reshape(1, 128, 128, 256).permute(0, 3, 1, 2)

    logger.info("tt_output_torch.shape: {}", tt_output_torch.shape)
    logger.info("tt_output_torch.dtype: {}", tt_output_torch.dtype)
    logger.info("cpu_output.shape: {}", cpu_output.shape)
    logger.info("cpu_output.dtype: {}", cpu_output.dtype)

    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Conv2d Output PCC: {pcc}")
