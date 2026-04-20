# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn.functional as F
from loguru import logger

import tests.ttnn.unit_tests.operations.sdpa_beit_base.utils as utils


def _main(input):
    var_0 = input[0]
    var_1 = input[1]
    var_2 = input[2]
    var_3 = input[3]
    ttnn_to_layout_0 = ttnn.to_layout(var_3, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_3, False)
    ttnn_to_layout_1 = ttnn.to_layout(var_2, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_2, False)
    ttnn_to_layout_2 = ttnn.to_layout(var_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_0, False)
    ttnn_to_layout_3 = ttnn.to_layout(var_1, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(var_1, False)
    ttnn_transformer_scaled_dot_product_attention_0 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_to_layout_0,
            ttnn_to_layout_1,
            ttnn_to_layout_2,
            attn_mask=ttnn_to_layout_3,
            is_causal=False,
            scale=0.125,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn.deallocate(ttnn_to_layout_0, False)
    return [ttnn_transformer_scaled_dot_product_attention_0]


def load_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_3 = utils.load_tensor(
        "arg3.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
    ]


class CpuSDPA(torch.nn.Module):

    def forward(self, query, key, value, attn_mask):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )


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


def test_sdpa():
    # TT run
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_output = _main(load_inputs_for__main_0)

    # CPU run 
    logger.info("Loading input tensors for CPU inference...")
    cpu_query = ttnn.to_torch(ttnn.load_tensor("arg3.tensorbin"))
    cpu_key = ttnn.to_torch(ttnn.load_tensor("arg2.tensorbin"))
    cpu_value = ttnn.to_torch(ttnn.load_tensor("arg0.tensorbin"))
    cpu_attn_mask = ttnn.to_torch(ttnn.load_tensor("arg1.tensorbin"))

    logger.info("cpu_query={}", cpu_query)
    logger.info("cpu_key={}", cpu_key)
    logger.info("cpu_value={}", cpu_value)
    logger.info("cpu_attn_mask={}", cpu_attn_mask)

    logger.info("cpu_query.shape={}", cpu_query.shape)
    logger.info("cpu_key.shape={}", cpu_key.shape)
    logger.info("cpu_value.shape={}", cpu_value.shape)
    logger.info("cpu_attn_mask.shape={}", cpu_attn_mask.shape)

    logger.info("cpu_query.dtype={}", cpu_query.dtype)
    logger.info("cpu_key.dtype={}", cpu_key.dtype)
    logger.info("cpu_value.dtype={}", cpu_value.dtype)
    logger.info("cpu_attn_mask.dtype={}", cpu_attn_mask.dtype)

    logger.info("Running CPU SDPA reference...")
    cpu_model = CpuSDPA()
    cpu_model.eval()
    
    with torch.no_grad():
        cpu_output = cpu_model(cpu_query, cpu_key, cpu_value, cpu_attn_mask)

    tt_output_torch = ttnn.to_torch(tt_output[0])

    logger.info("tt_output={}", tt_output_torch)
    logger.info("cpu_output={}", cpu_output)

    logger.info("tt_output.shape={}", tt_output_torch.shape)
    logger.info("cpu_output.shape={}", cpu_output.shape)

    logger.info("tt_output.dtype={}", tt_output_torch.dtype)
    logger.info("cpu_output.dtype={}", cpu_output.dtype)

    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")

