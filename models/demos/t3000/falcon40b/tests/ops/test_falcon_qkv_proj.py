# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.demos.t3000.falcon40b.tt.model_config import (
    get_model_config,
)

from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)

from models.demos.t3000.falcon40b.tt.model_utils import falcon_prefill_matmul


def run_test_FalconMLP_inference(
    pcc,
    device=None,
    device_mesh=None,
):
    # falcon 40b per chip: 8 chip setup
    if device is not None:
        num_devices = 1
    elif device_mesh is not None:
        num_devices = 8
    else:
        assert False

    num_heads = 16
    num_kv_heads = 1
    head_dim = 64
    hidden_size = 8192

    seqlen = 8192

    # Prepare input
    torch.manual_seed(0)
    model_input_shape = [1, seqlen]

    model_config = get_model_config("BFLOAT8_B-DRAM", "prefill", model_input_shape, 8)

    input_shape = [
        1,
        1,
        seqlen,
        hidden_size,
    ]
    weight_shape = [
        1,
        1,
        hidden_size,
        num_devices * (num_heads + num_kv_heads + num_kv_heads) * head_dim,
    ]
    input = (torch.rand(input_shape) * 2) - 1
    weight = (torch.rand(weight_shape) * 2) - 1

    # PyTorch output --------------------------------------------------------------------
    pytorch_reference = torch.matmul(input, weight)

    # TT hardware execution -------------------------------------------------------------
    print("Creating input tensor with shape: ", input.shape)
    if device is not None:
        input_host = torch2tt_tensor(input, None, tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B)
        tt_input = input_host.to(
            device,
            ttnn.experimental.tensor.MemoryConfig(
                ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
            ),
        )
    elif device_mesh is not None:
        tt_input = ttnn.as_tensor(
            tensor=input,
            dtype=model_config["ATTN_INPUT_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            memory_config=model_config["ATTN_INPUT_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
    else:
        assert False

    print("Creating weight tensor with shape: ", weight.shape)
    if device is not None:
        weight_host = torch2tt_tensor(weight, None, tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B)
        tt_weight = weight_host.to(
            device,
            ttnn.experimental.tensor.MemoryConfig(
                ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED, ttnn.experimental.tensor.BufferType.DRAM
            ),
        )
    elif device_mesh is not None:
        tt_weight = ttnn.as_tensor(
            tensor=weight,
            dtype=model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            memory_config=model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(device_mesh, dim=-1),
        )
    else:
        assert False

    max_mm_seq_len = 1024
    mm_seq_len_batched = 1024
    batch_dim = 1 if seqlen < max_mm_seq_len else seqlen // mm_seq_len_batched
    if batch_dim != 1:
        tt_input = ttnn.reshape(tt_input, (1, batch_dim, seqlen // batch_dim, -1))

    mm_output = falcon_prefill_matmul(
        tt_input,
        tt_weight,
        model_config["COMPUTE_KERNEL_CONFIG"],
        output_mem_config=model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
        output_dtype=model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
        grid=ttnn.CoreGrid(x=8, y=8) if seqlen >= 512 else ttnn.CoreGrid(x=8, y=min(seqlen // 32, 8)),
        overwrite_subblock_w=1,  # Workaround for non deterministic output/hang; issue: 7066
        overwrite_subblock_h=1,
        fuse_batch_mm2d=False,
    )

    print("MM output shape: ", mm_output.shape)

    # Reshape to compute long sequence lengths as multiple MM loops (reverse)
    if batch_dim != 1:
        mm_output = ttnn.reshape(mm_output, (1, 1, seqlen, -1))

    if device is not None:
        output = tt2torch_tensor(mm_output)
    elif device_mesh is not None:
        output = ttnn.to_torch(mm_output, device=device_mesh, mesh_composer=ConcatMeshToTensor(device_mesh, dim=3))
    else:
        assert False

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_reference, output, pcc)
    logger.info(f"PCC value: {output_pcc}")
    assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize("pcc", [(0.99)])
def test_FalconMatmul_inference_t3k(
    pcc,
    t3k_device_mesh,
):
    run_test_FalconMLP_inference(
        pcc,
        device_mesh=t3k_device_mesh,
    )


@pytest.mark.parametrize("pcc", [(0.99)])
def test_FalconMatmul_inference_wh(
    pcc,
    all_devices,
):
    device = all_devices[0]

    run_test_FalconMLP_inference(
        pcc,
        device=device,
    )
