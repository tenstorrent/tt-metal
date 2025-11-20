import ttnn
from tests.ttnn.unit_tests.operations.conv.failing_conv import utils
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from loguru import logger
import pytest
from models.common.utility_functions import comp_pcc


def _main(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = utils.DeviceGetter.get_device((1, 1))
    v5 = ttnn.permute(
        v3,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(v3, False)
    v6 = ttnn.reshape(
        v5,
        [1, 1, 65536, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v5, False)
    v7 = ttnn.to_layout(
        v6,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v6, False)
    v8 = ttnn.conv2d(
        input_tensor=v7,
        weight_tensor=v2,
        device=v4,
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
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
            act_block_w_div=1,
            reshard_if_not_optimal=False,
            override_sharding_config=False,
            transpose_shards=False,
            output_layout=ttnn.Layout.TILE,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            in_place=False,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v7, False)
    v9 = ttnn.reshape(
        v8,
        [1, 128, 128, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v8, False)
    v10 = ttnn.permute(
        v9,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(v9, False)
    v11 = [v10]
    return v11


def load_inputs_for__main():
    v1 = utils.DeviceGetter.get_device((1, 1))
    v2 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/conv/failing_conv/tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    v3 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/conv/failing_conv/tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        v1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v4 = [v2, v3]
    return v4


def main():
    test_centernet_conv2d()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.B_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.B_conv(x)

        return x


def test_centernet_conv2d():
    ## tt run
    v1 = load_inputs_for__main()
    tt_output = _main(v1)

    logger.info("tt_output={}", tt_output)

    ## cpu run
    x1 = torch.load("tests/ttnn/unit_tests/operations/conv/failing_conv/input_for_problematic_conv.pt")
    w = torch.load("tests/ttnn/unit_tests/operations/conv/failing_conv/weight_for_problematic_conv.pt")

    cpu_model = Model()
    cpu_model.to(torch.bfloat16)
    cpu_model.B_conv.weight = torch.nn.Parameter(w)
    with torch.no_grad():
        cpu_output = cpu_model(x1)

    for i in range(len(tt_output)):
        tt_output[i] = ttnn.to_torch(tt_output[i])

    logger.info("cpu_output={}", cpu_output)
    # PCC & atol check
    pcc_values = comp_pcc(tt_output[0], cpu_output, 0.99)
    atol_delta = torch.max(torch.abs(tt_output[0] - cpu_output)).item()
    logger.info(f"PCC & atol value for output tensors: {pcc_values,atol_delta}")


if __name__ == "__main__":
    main()
