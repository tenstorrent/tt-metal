import ttnn
import tests.ttnn.unit_tests.operations.sanity.utils as utils
import torch
import torch.nn.functional as F
from loguru import logger

_CONST_EVAL_CACHE = {}


def _main(input):
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    compute_config = ttnn.init_device_compute_kernel_config(
        utils_DeviceGetter_get_device_0.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        input[2],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input[2], False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 257, 768],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_reshape_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_1,
        weight_tensor=input[1],
        device=utils_DeviceGetter_get_device_0,
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
        conv_config=ttnn.Conv2dConfig(
            config_tensors_in_dram=True, enable_kernel_stride_folding=False
        ),
        compute_config=compute_config,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_conv2d_0,
        [1, 257, 1, 27],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_1,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_permute_1,
        [1, 27, 257],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_reshape_2,
        2,
        numeric_stable=True,
        compute_kernel_config=compute_config,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_to_layout_2 = ttnn.to_layout(
        input[0],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input[0], False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_softmax_0,
        ttnn_to_layout_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=None,
        program_config=None,
        activation=None,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn.deallocate(ttnn_softmax_0, False)
    util_create_list_0 = [ttnn_matmul_0]
    return util_create_list_0


def load_inputs_for__main():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    utils_load_tensor_2 = utils.load_tensor(
        "arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_1 = [utils_load_tensor_0, utils_load_tensor_1, utils_load_tensor_2]
    return util_create_list_1


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
    # ------------------------------------------------------------------
    # 1. Load raw tensors to torch (same data used by both CPU and TT)
    # ------------------------------------------------------------------
    logger.info("Loading input tensors...")
    tt_arg0_host = ttnn.load_tensor("arg0.tensorbin")
    tt_arg1_host = ttnn.load_tensor("arg1.tensorbin")
    tt_arg2_host = ttnn.load_tensor("arg2.tensorbin")

    cpu_feat = ttnn.to_torch(tt_arg0_host)
    cpu_conv_ip = ttnn.to_torch(tt_arg2_host)

    logger.info(f"feat (arg0) shape: {cpu_feat.shape}")
    logger.info(f"conv_ip (arg2) shape: {cpu_conv_ip.shape}")

    # ------------------------------------------------------------------
    # 2. CPU inference
    # ------------------------------------------------------------------
    logger.info("Running CPU inference...")
    cpu_model = Cpu_model()
    cpu_model.eval()

    conv_weight_torch = ttnn.to_torch(tt_arg1_host)
    cpu_model.conv.weight.data = conv_weight_torch

    with torch.no_grad():
        cpu_output = cpu_model(cpu_conv_ip, cpu_feat)
    logger.info(f"CPU output shape: {cpu_output.shape}")

    # ------------------------------------------------------------------
    # 3. TT inference (uses the same tensorbin files)
    # ------------------------------------------------------------------
    logger.info("Running TT inference...")
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)

    tt_output_torch = ttnn.to_torch(tt_output[0])
    logger.info(f"TT output shape: {tt_output_torch.shape}")

    # ------------------------------------------------------------------
    # 4. Compare outputs
    # ------------------------------------------------------------------
    logger.info("=== Output Comparison ===")
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"PCC: {pcc}")