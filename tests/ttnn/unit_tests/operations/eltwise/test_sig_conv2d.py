import ttnn
import tests.ttnn.unit_tests.operations.eltwise.utils as utils
from loguru import logger

import torch.nn as nn
import torch

from scipy.stats import pearsonr


def main_const_eval_0(v1):
    v2 = v1[0]
    v3 = ttnn.reshape(
        v2,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v4 = ttnn.from_device(v3)
    ttnn.deallocate(v3, False)
    v5 = ttnn.to_layout(v4, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(v4, False)
    v6 = [v5]
    return v6


CACHED_main_const_eval_0 = None


def _main(v1):
    global CACHED_main_const_eval_0
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = main_const_eval_0
    v7 = [v3]
    v8 = utils.constEvalFuncWrapper(v6, v7, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = v8
    v9 = v8[0]
    v10 = utils.DeviceGetter.get_device((1, 1))
    v11 = ttnn.sigmoid(
        v2,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v2, False)
    v12 = ttnn.permute(
        v5,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(v5, False)
    v13 = ttnn.reshape(
        v12,
        [1, 1, 4096, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.from_device(v13)
    ttnn.deallocate(v13, False)
    v15 = ttnn.to_layout(v14, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(v14, False)
    v16 = ttnn.to_device(
        v15,
        device=v10,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v15, False)
    v17 = ttnn.conv2d(
        input_tensor=v16,
        weight_tensor=v4,
        device=v10,
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=64,
        input_width=64,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v9,
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
    ttnn.deallocate(v16, False)
    v18 = ttnn.reshape(
        v17,
        [1, 64, 64, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    v19 = ttnn.permute(
        v18,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(v18, False)
    v20 = [v11, v19]
    return v20


def load_inputs_for__main():
    v1 = utils.DeviceGetter.get_device((1, 1))
    v2 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        v1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v3 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        v1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v4 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    v5 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        v1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v6 = [v2, v3, v4, v5]
    return v6


class SigmoidConv(nn.Module):
    def __init__(self, weights=None, bias=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        # Load provided weights and bias if available
        if weights is not None:
            self.conv.weight.data = weights
        if bias is not None:
            self.conv.bias.data = bias

    def forward(self, x1, x2):
        y1 = torch.sigmoid(x2)
        y2 = self.conv(x1)
        return (y1, y2)


def compute_pcc_list(tt_list, cpu_list):
    pcc_values = []
    for t1, t2 in zip(tt_list, cpu_list):
        a = t1.detach().cpu().numpy().flatten()
        b = t2.detach().cpu().numpy().flatten()
        pcc, _ = pearsonr(a, b)
        pcc_values.append(pcc)
    return pcc_values


def test_sig_conv2d():
    ## tt run
    v1 = load_inputs_for__main()
    tt_output = _main(v1)

    logger.info("tt_output={}", tt_output)

    ## cpu run - load same weights as ttnn
    x1 = torch.load("input_1.pt")
    x2 = torch.load("input_2.pt")

    # Load the actual weights and bias used by ttnn
    weights_ttnn = ttnn.load_tensor("./tensors/arg2.tensorbin")
    weights_torch = ttnn.to_torch(weights_ttnn)

    bias_ttnn = ttnn.load_tensor("./tensors/arg1.tensorbin")
    bias_torch = ttnn.to_torch(bias_ttnn).squeeze()

    logger.info(f"Loaded weights shape: {weights_torch.shape}, bias shape: {bias_torch.shape}")

    cpu_model = SigmoidConv(weights=weights_torch, bias=bias_torch)

    with torch.no_grad():
        cpu_output = cpu_model(x1, x2)

    for i in range(len(tt_output)):
        tt_output[i] = ttnn.to_torch(tt_output[i])

    logger.info("cpu_output={}", cpu_output)

    # pcc check
    pcc_values = compute_pcc_list(tt_output, cpu_output)

    logger.info(f"PCC values per output tensor: {pcc_values}")
