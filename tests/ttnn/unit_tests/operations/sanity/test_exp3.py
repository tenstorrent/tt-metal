import ttnn
import tests.ttnn.unit_tests.operations.sanity.utils as utils
import torch
import torch.nn.functional as F
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
    cpu_feat = ttnn.to_torch(utils.load_tensor("arg0.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    conv_weight = ttnn.to_torch(utils.load_tensor("arg1.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))
    cpu_conv_ip = ttnn.to_torch(utils.load_tensor("arg2.tensorbin", ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None))

    # --- Conv + flatten on CPU (torch) since PCC is good ---
    logger.info("Running conv + flatten on CPU...")
    conv = torch.nn.Conv2d(768, 27, kernel_size=(1, 1), stride=(1, 1), bias=False)
    conv.weight.data = conv_weight
    with torch.no_grad():
        conv_out = conv(cpu_conv_ip)
        flatten_out = conv_out.flatten(2)

    # --- CPU golden: softmax + matmul ---
    with torch.no_grad():
        cpu_softmax_out = F.softmax(flatten_out, dim=-1)
        cpu_matmul_out = torch.einsum("...si,...id->...sd", cpu_softmax_out, cpu_feat)

    # --- TT inference: softmax + matmul only ---
    logger.info("Running TT softmax + matmul...")

    flatten_ttnn = ttnn.from_torch(flatten_out, layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)

    softmax_output = ttnn.softmax(
        flatten_ttnn, 2,
        numeric_stable=True,
        compute_kernel_config=compute_config,
        memory_config=DRAM_MEMCFG,
    )

    feat_ttnn = ttnn.from_torch(cpu_feat, layout=ttnn.Layout.TILE, device=device, memory_config=DRAM_MEMCFG)

    tt_output = ttnn.matmul(
        softmax_output, feat_ttnn,
        transpose_a=False,
        transpose_b=False,
        memory_config=DRAM_MEMCFG,
        dtype=None,
        program_config=None,
        activation=None,
        compute_kernel_config=compute_config,
    )

    tt_softmax_torch = ttnn.to_torch(softmax_output)
    tt_matmul_torch = ttnn.to_torch(tt_output)

    # --- Compare ---
    logger.info("=== Output Comparison ===")

    pcc_softmax = compute_pcc(tt_softmax_torch, cpu_softmax_out)
    logger.info(f"[softmax] PCC: {pcc_softmax}")

    pcc_matmul = compute_pcc(tt_matmul_torch, cpu_matmul_out)
    logger.info(f"[matmul]  PCC: {pcc_matmul}")
