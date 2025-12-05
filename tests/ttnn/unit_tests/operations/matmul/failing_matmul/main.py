import ttnn
from tests.ttnn.unit_tests.operations.matmul.failing_matmul import utils
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from loguru import logger
import pytest
from models.common.utility_functions import comp_pcc


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 12, 39, 39]),
        fill_value=0.08837890625,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_0 = [ttnn_full_0]
    return util_create_list_0


CACHED_main_const_eval_0 = None


def _main(input):
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    const_0 = main_const_eval_0
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(const_0, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapperZeroArg_0
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    ttnn_reshape_0 = ttnn.reshape(
        input_1,
        [12, 39, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_1, False)
    ttnn_permute_0 = ttnn.permute(
        input_0,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(input_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_permute_0,
        [12, 128, 39],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_reshape_0,
        ttnn_reshape_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_matmul_0,
        [1, 12, 39, 39],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_reshape_2,
        utils_constEvalFuncWrapperZeroArg_0_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    util_create_list_1 = [ttnn_multiply_0]
    return util_create_list_1


def load_inputs_for__main():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/matmul/failing_matmul/tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/matmul/failing_matmul/tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_2 = [utils_load_tensor_0, utils_load_tensor_1]
    return util_create_list_2


def main():
    test_qwen_2_5_matmul()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.B_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x, y):
        x = torch.matmul(x, y.transpose(2, 3)) * 0.08838834764831845

        return x


def test_qwen_2_5_matmul():
    ## tt run
    v1 = load_inputs_for__main()
    tt_output = _main(v1)

    logger.info("tt_output={}", tt_output)

    ## cpu run
    x1 = torch.load("tests/ttnn/unit_tests/operations/matmul/failing_matmul/key_states.pt")
    x2 = torch.load("tests/ttnn/unit_tests/operations/matmul/failing_matmul/query.pt")

    cpu_model = Model()
    cpu_model.to(torch.bfloat16)
    # cpu_model.B_conv.weight = torch.nn.Parameter(w)
    with torch.no_grad():
        cpu_output = cpu_model(x2, x1)

    for i in range(len(tt_output)):
        tt_output[i] = ttnn.to_torch(tt_output[i])

    logger.info("cpu_output={}", cpu_output)
    # PCC & atol check
    pcc_values = comp_pcc(tt_output[0], cpu_output, 0.99)
    atol_delta = torch.max(torch.abs(tt_output[0] - cpu_output)).item()
    logger.info(f"PCC & atol value for output tensors: {pcc_values,atol_delta}")


if __name__ == "__main__":
    main()
