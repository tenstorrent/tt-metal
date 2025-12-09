import ttnn
import utils
from models.common.utility_functions import comp_pcc
from loguru import logger
import torch
import torch.nn as nn


def _main(input):
    input_0 = input[0]
    input_1 = input[1]
    input_2 = input[2]
    input_3 = input[3]
    ttnn_typecast_0 = ttnn.typecast(
        input_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_2, False)
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_typecast_0,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_softmax_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_softmax_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_typecast_1,
        [32, 256, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn_reshape_1 = ttnn.reshape(
        input_1,
        [32, 256, 96],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_1, False)
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
        [1, 32, 256, 96],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
        ttnn_reshape_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_0,
        [256, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_0, False)
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_reshape_3,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_3, False)
    ttnn_add_0 = ttnn.add(
        ttnn_matmul_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn.deallocate(input_3, False)
    util_create_list_0 = [ttnn_add_0]
    return util_create_list_0


def load_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/phi3/model_phi3/tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/phi3/model_phi3/tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/phi3/model_phi3/tensors/arg2.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_3 = utils.load_tensor(
        "tests/ttnn/unit_tests/operations/phi3/model_phi3/tensors/arg3.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_1 = [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
    ]
    return util_create_list_1


def main():
    test_phi3()


class Phi3_CausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(32 * 96, 3072, bias=False)
        self.resid_attn_dropout = torch.nn.Dropout(p=0.0)

    def forward(self, attn_weights, value_states, residual):
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=False)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, 256, 3072)
        attn_output = self.o_proj(attn_output)
        hidden_states = residual + self.resid_attn_dropout(attn_output)
        return hidden_states


def test_phi3():
    v1 = load_inputs_for__main()
    tt_output = _main(v1)
    logger.info("tt_output={}", tt_output)

    model = Phi3_CausalLM()
    attn_weights = torch.load("tests/ttnn/unit_tests/operations/phi3/model_phi3/attn_weights.pt", map_location="cpu")
    o_proj_weight = torch.load("tests/ttnn/unit_tests/operations/phi3/model_phi3/o_proj_weight.pt", map_location="cpu")
    residual = torch.load("tests/ttnn/unit_tests/operations/phi3/model_phi3/residual.pt", map_location="cpu")
    value_states = torch.load("tests/ttnn/unit_tests/operations/phi3/model_phi3/value_states.pt", map_location="cpu")
    model.o_proj.weight = torch.nn.Parameter(o_proj_weight)
    model.to(torch.bfloat16)

    with torch.no_grad():
        cpu_output = model(attn_weights, value_states, residual)

    for i in range(len(tt_output)):
        tt_output[i] = ttnn.to_torch(tt_output[i])

    logger.info("cpu_output={}", cpu_output)
    # PCC & atol check
    pcc_values = comp_pcc(tt_output[0], cpu_output, 0.99)
    atol_delta = torch.max(torch.abs(tt_output[0] - cpu_output)).item()
    logger.info(f"PCC & atol value for output tensors: {pcc_values,atol_delta}")


if __name__ == "__main__":
    main()
