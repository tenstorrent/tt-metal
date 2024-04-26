# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger

import tt_lib as ttl
import ttnn
from models.demos.t3000.falcon40b.tt.ops.falcon_softmax import TtFalconSoftmax
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.demos.t3000.falcon40b.tt.model_config import (
    get_model_config,
)

from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)

from torch.nn import functional as F


class PytorchFalconSoftmax(torch.nn.Module):
    def __init__(self, head_dim=64):
        super().__init__()
        self.scale = math.sqrt(head_dim)

    def forward(self, attention_scores, attention_mask):
        attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, -100000.0).to(torch.bfloat16)

        attention_scores /= self.scale

        attention_scores = F.softmax(
            attention_scores + attention_mask_float,
            dim=-1,
            dtype=torch.bfloat16,
        )
        return attention_scores


def run_test_FalconSoftmax_inference(
    pcc,
    device,
    model_location_generator,
):
    is_sharded = True

    head_dim = 64
    seqlen = 64
    num_chips = 8

    if num_chips == 8:
        num_attention_heads = 16
    elif num_chips == 4:
        num_attention_heads = 32
    else:
        assert False

    # Prepare input
    torch.manual_seed(0)
    model_input_shape = [1, seqlen]

    model_config = get_model_config("BFLOAT8_B-SHARDED", "prefill", model_input_shape, num_chips)

    input_shape = [1, num_attention_heads, seqlen, seqlen]
    input_torch = (torch.rand(input_shape) * 2) - 1
    input = torch2tt_tensor(input_torch, None, tt_dtype=ttl.tensor.DataType.BFLOAT16)
    input = input.to(device, model_config["DEFAULT_MEMCFG"])

    attention_mask_bool = torch.ones(1, 1, seqlen, seqlen, dtype=bool).triu(diagonal=1)

    if is_sharded:
        if num_attention_heads == 32:  # 4 chip setup
            shard_spec_cores_grid = ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0),
                        ttl.tensor.CoreCoord(7, 3),
                    ),
                }
            )
        elif num_attention_heads == 16:  # 8 chip setup
            shard_spec_cores_grid = ttl.tensor.CoreRangeSet(
                {
                    ttl.tensor.CoreRange(
                        ttl.tensor.CoreCoord(0, 0),
                        ttl.tensor.CoreCoord(7, 1),
                    ),
                }
            )
        else:
            assert False

        softmax_memcfg = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_cores_grid,
                [
                    seqlen,
                    seqlen,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        input = ttl.tensor.interleaved_to_sharded(input, sharded_mem_config=softmax_memcfg)

        attention_mask_bool = attention_mask_bool.expand(1, num_attention_heads, seqlen, seqlen)

        attention_mask_memconfig = ttl.tensor.MemoryConfig(
            ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttl.tensor.BufferType.L1,
            ttl.tensor.ShardSpec(
                shard_spec_cores_grid,
                [
                    seqlen,
                    seqlen,
                ],
                ttl.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

    attention_mask = attention_mask_bool * -100000

    tt_attention_mask_per_device_host = torch2tt_tensor(attention_mask, None, tt_dtype=ttl.tensor.DataType.BFLOAT16)
    tt_attention_mask_per_device = tt_attention_mask_per_device_host.to(device, model_config["DEFAULT_MEMCFG"])
    if is_sharded:
        tt_attention_mask_per_device = ttl.tensor.interleaved_to_sharded(
            tt_attention_mask_per_device, sharded_mem_config=attention_mask_memconfig
        )

    model_version = "tiiuae/falcon-40b-instruct"
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconSoftmax_model = PytorchFalconSoftmax(head_dim=head_dim)
    torch_out = pytorch_FalconSoftmax_model(input_torch, attention_mask_bool)

    print(attention_mask.shape)

    # TT hardware execution -------------------------------------------------------------
    tt_Falconcreate_qkv_heads_model = TtFalconSoftmax(
        device, model_config=model_config, head_dim=head_dim, seqlen=seqlen, is_sharded=is_sharded
    )

    tt_out = tt_Falconcreate_qkv_heads_model(input, tt_attention_mask_per_device)

    tt_out = tt2torch_tensor(tt_out)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(torch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("pcc", [(0.99)])
def test_FalconSoftmax_inference(
    pcc,
    all_devices,
    model_location_generator,
):
    device = all_devices[0]

    run_test_FalconSoftmax_inference(
        pcc,
        device,
        model_location_generator,
    )
