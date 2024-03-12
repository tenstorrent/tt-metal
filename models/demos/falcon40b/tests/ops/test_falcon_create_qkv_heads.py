# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib as ttl
import ttnn
from models.demos.falcon40b.tt.ops.falcon_nlp_create_qkv_heads import TtFalconCreateQKVHeads
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.demos.falcon40b.tt.model_config import (
    get_model_config,
)

from models.demos.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)


class PytorchFalconCreateQKVHeads(torch.nn.Module):
    def __init__(self, num_heads=32, num_kv_heads=2, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, fused_qkv):
        fused_qkv = fused_qkv.squeeze(0)

        batch, seq_len, _ = fused_qkv.shape
        qkv = fused_qkv.view(
            batch,
            seq_len,
            -1,
            self.num_heads // self.num_kv_heads + 2,
            self.head_dim,
        )
        query = qkv[:, :, :, :-2]
        key = qkv[:, :, :, [-2]]
        value = qkv[:, :, :, [-1]]

        query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
        query, key, value = [x.permute(0, 2, 1, 3) for x in (query, key, value)]
        return query, key, value


def run_test_FalconMLP_inference(
    pcc,
    device,
    model_location_generator,
):
    # # mistral
    # num_heads = 32
    # num_kv_heads = 8
    # head_dim = 128

    # falcon 40b per chip: 8 chip setup
    num_heads = 16
    num_kv_heads = 1
    head_dim = 64

    # falcon 40b per chip: 4 chip setup
    num_heads = 32
    num_kv_heads = 2
    head_dim = 64

    seqlen = 32

    # Prepare input
    torch.manual_seed(0)
    model_input_shape = [1, seqlen]

    model_config = get_model_config("BFLOAT8_B-SHARDED", "prefill", model_input_shape, 4)

    input_shape = [
        1,
        1,
        seqlen,
        (num_heads + num_kv_heads + num_kv_heads) * head_dim,
    ]  # 2304 = (16 + 1 + 1) * 2 groups * 64 head_dim
    input = (torch.rand(input_shape) * 2) - 1

    model_version = "tiiuae/falcon-40b-instruct"
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, num_hidden_layers=1
    )

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconCreateQKVHeads_model = PytorchFalconCreateQKVHeads(
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    pytorch_q_out, pytorch_k_out, pytorch_v_out = pytorch_FalconCreateQKVHeads_model(input)

    # TT hardware execution -------------------------------------------------------------
    tt_Falconcreate_qkv_heads_model = TtFalconCreateQKVHeads(
        device, model_config=model_config, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )

    input_host = torch2tt_tensor(input, None, tt_dtype=ttl.tensor.DataType.BFLOAT16)
    input = input_host.to(
        device, ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    )

    q_tt_out, k_tt_out, v_tt_out = tt_Falconcreate_qkv_heads_model(input)

    q_tt_out = tt2torch_tensor(q_tt_out)
    k_tt_out = tt2torch_tensor(k_tt_out)
    v_tt_out = tt2torch_tensor(v_tt_out)

    # check outputs ----------------------------------------------------------------------
    all_pass = True
    does_pass, output_pcc = comp_pcc(pytorch_q_out, q_tt_out, pcc)
    all_pass = all_pass and does_pass
    logger.info(f"PCC Q value: {output_pcc}")

    does_pass, output_pcc = comp_pcc(pytorch_k_out, k_tt_out, pcc)
    all_pass = all_pass and does_pass
    logger.info(f"PCC K value: {output_pcc}")

    does_pass, output_pcc = comp_pcc(pytorch_v_out, v_tt_out, pcc)
    all_pass = all_pass and does_pass
    logger.info(f"PCC V value: {output_pcc}")

    if all_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert all_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("pcc", [(0.99)])
def test_FalconMatmul_inference(
    pcc,
    all_devices,
    model_location_generator,
):
    device = all_devices[0]

    run_test_FalconMLP_inference(
        pcc,
        device,
        model_location_generator,
    )
