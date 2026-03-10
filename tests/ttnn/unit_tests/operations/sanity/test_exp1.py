import ttnn
import tests.ttnn.unit_tests.operations.sanity.utils as utils
import torch
import torch.nn.functional as F
from loguru import logger

DRAM_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


class Cpu_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(768, 27, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, conv_ip, feat):
        selected = self.conv(conv_ip)
        selected = selected.flatten(2)
        attentions = F.softmax(selected, dim=-1)
        feat = torch.einsum("...si,...id->...sd", attentions, feat)
        return feat


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


def test_org():
    device = utils.DeviceGetter.get_device((1, 1))
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )

    # Load tensors
    # arg0 -> feat [1, 257, 768], arg1 -> conv.weight [27, 768, 1, 1], arg2 -> conv_ip [1, 768, 257, 1]
    logger.info("Loading tensors...")
    feat_ttnn = utils.load_tensor("arg0.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, device, DRAM_MEMCFG)
    conv_weight_ttnn = utils.load_tensor("arg1.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None)
    conv_ip_ttnn = utils.load_tensor("arg2.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, device, DRAM_MEMCFG)

    cpu_feat = ttnn.to_torch(feat_ttnn)
    cpu_conv_ip = ttnn.to_torch(conv_ip_ttnn)

    logger.info(f"feat (arg0) shape: {cpu_feat.shape}")
    logger.info(f"conv_ip (arg2) shape: {cpu_conv_ip.shape}")

    # --- TT inference ---
    logger.info("Running TT inference...")

    ttnn_to_layout_0 = ttnn.to_layout(conv_ip_ttnn, ttnn.Layout.TILE, None, memory_config=DRAM_MEMCFG)

    ttnn_permute_0 = ttnn.permute(ttnn_to_layout_0, [0, 2, 3, 1], memory_config=DRAM_MEMCFG, pad_value=0.0)

    ttnn_reshape_0 = ttnn.reshape(ttnn_permute_0, [1, 1, 257, 768], memory_config=DRAM_MEMCFG)

    ttnn_to_layout_1 = ttnn.to_layout(ttnn_reshape_0, ttnn.Layout.ROW_MAJOR, None, memory_config=DRAM_MEMCFG)

    conv_output = ttnn.conv2d(
        input_tensor=ttnn_to_layout_1,
        weight_tensor=conv_weight_ttnn,
        device=device,
        in_channels=768,
        out_channels=27,
        batch_size=1,
        input_height=257,
        input_width=1,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(config_tensors_in_dram=True, enable_kernel_stride_folding=False),
        compute_config=compute_config,
        slice_config=None,
        memory_config=DRAM_MEMCFG,
    )

    ttnn_reshape_1 = ttnn.reshape(conv_output, [1, 257, 1, 27], memory_config=DRAM_MEMCFG)

    ttnn_permute_1 = ttnn.permute(ttnn_reshape_1, [0, 3, 1, 2], memory_config=DRAM_MEMCFG, pad_value=0.0)

    ttnn_reshape_2 = ttnn.reshape(ttnn_permute_1, [1, 27, 257], memory_config=DRAM_MEMCFG)

    softmax_output = ttnn.softmax(
        ttnn_reshape_2, 2,
        numeric_stable=True,
        compute_kernel_config=compute_config,
        memory_config=DRAM_MEMCFG,
    )

    feat_tiled = ttnn.to_layout(feat_ttnn, ttnn.Layout.TILE, None, memory_config=DRAM_MEMCFG)

    tt_output = ttnn.matmul(
        softmax_output, feat_tiled,
        transpose_a=False,
        transpose_b=False,
        memory_config=DRAM_MEMCFG,
        dtype=None,
        program_config=None,
        activation=None,
        compute_kernel_config=compute_config,
    )

    tt_output_torch = ttnn.to_torch(tt_output)
    logger.info(f"TT output shape: {tt_output_torch.shape}")

    # --- CPU golden inference ---
    logger.info("Running CPU inference...")
    cpu_model = Cpu_model()
    cpu_model.eval()

    conv_weight_torch = ttnn.to_torch(conv_weight_ttnn)
    cpu_model.conv.weight.data = conv_weight_torch

    with torch.no_grad():
        cpu_output = cpu_model(cpu_conv_ip, cpu_feat)
    logger.info(f"CPU output shape: {cpu_output.shape}")

    # --- Compare ---
    logger.info("=== Output Comparison ===")
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"PCC: {pcc}")
