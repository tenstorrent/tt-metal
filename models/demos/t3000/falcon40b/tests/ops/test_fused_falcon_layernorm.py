# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger

import tt_lib as ttl
import ttnn
from models.demos.t3000.falcon40b.tt.ops.fused_layernorm import TtFusedFalconLayernorm
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

from models.demos.t3000.falcon40b.tt.model_utils import (
    convert_to_layout,
)


class PytorchFusedLayernorm(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num=0):
        super().__init__()
        self.ln_attn = hf_reference_model.transformer.h[layer_num].ln_attn
        self.ln_mlp = hf_reference_model.transformer.h[layer_num].ln_mlp

        self.ln_attn.eval()
        self.ln_mlp.eval()

    def forward(self, x):
        result1 = self.ln_attn(x)
        # result1 = x * self.ln_attn.weight
        result2 = self.ln_mlp(x)
        # result2 = x
        return result1, result2


def run_test_FalconLayernorm_inference(pcc, devices, model_location_generator, get_tt_cache_path):
    seqlen = 1024
    num_chips = 8

    # Prepare input
    torch.manual_seed(0)
    model_input_shape = [1, seqlen]

    model_version = "tiiuae/falcon-40b-instruct"

    model_config = get_model_config(
        "BFLOAT8_B-DRAM", "prefill", model_input_shape, num_chips
    )  # Block sharding for layernorm to work around PCC issue

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, num_hidden_layers=1
    )
    hugging_face_reference_model.eval()
    config = hugging_face_reference_model.config
    gamma1 = hugging_face_reference_model.transformer.h[0].ln_attn.weight
    beta1 = hugging_face_reference_model.transformer.h[0].ln_attn.bias
    gamma2 = hugging_face_reference_model.transformer.h[0].ln_mlp.weight
    beta2 = hugging_face_reference_model.transformer.h[0].ln_mlp.bias

    H = config.hidden_size  # H = 8192

    input_shape = [1, 1, seqlen, H]

    input_torch = (torch.rand(input_shape) * 2) - 1
    input = torch2tt_tensor(
        input_torch, None, tt_dtype=ttl.tensor.DataType.BFLOAT16
    )  # tt_dtype=ttl.tensor.DataType.BFLOAT16  # ttl.tensor.DataType.BFLOAT8_B
    input = input.to(devices[0], model_config["DEFAULT_MEMCFG"])

    # block sharded hardcoded for S=128 and 8x4 grid of cores
    shard_spec_cores_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0),
                ttl.tensor.CoreCoord(7, 7),
            ),
        }
    )
    block_sharded_memconfig = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        ttl.tensor.BufferType.L1,
        ttl.tensor.ShardSpec(
            shard_spec_cores_grid,
            [
                seqlen // 8,
                H // 8,
            ],
            ttl.tensor.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    # width_sharded_memconfig = ttl.tensor.MemoryConfig(
    #     ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
    #     ttl.tensor.BufferType.L1,
    #     ttl.tensor.ShardSpec(
    #         shard_spec_cores_grid,
    #         [
    #             seqlen,
    #             H // 64,
    #         ],
    #         ttl.tensor.ShardOrientation.ROW_MAJOR,
    #         False,
    #     ),
    # )
    input = ttl.tensor.interleaved_to_sharded(input, sharded_mem_config=block_sharded_memconfig)

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconLayernorm_model = PytorchFusedLayernorm(hugging_face_reference_model)
    torch_out1, torch_out2 = pytorch_FalconLayernorm_model(input_torch)

    # TT hardware execution -------------------------------------------------------------

    tt_Falcon_layernorm_model = TtFusedFalconLayernorm(
        devices[0], gamma1, beta1, gamma2, beta2, model_config, config, tt_cache_path
    )
    tt_out1, tt_out2 = tt_Falcon_layernorm_model(input)

    tt_out1 = tt2torch_tensor(tt_out1)
    tt_out2 = tt2torch_tensor(tt_out2)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc1 = comp_pcc(torch_out1, tt_out1, pcc)
    logger.info(f"PCC value: {output_pcc1}")

    if does_pass:
        logger.info("Layernorm output 1 Passed!")
    else:
        logger.warning("Layernorm output 1 Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"

    does_pass, output_pcc2 = comp_pcc(torch_out2, tt_out2, pcc)
    logger.info(f"PCC value: {output_pcc2}")

    if does_pass:
        logger.info("Layernorm output 2 Passed!")
    else:
        logger.warning("Layernorm output 2 Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("pcc", [(0.99)])
def test_FalconLayernorm_inference(
    pcc,
    all_devices,
    model_location_generator,
    get_tt_cache_path,
):
    devices = all_devices

    run_test_FalconLayernorm_inference(pcc, devices, model_location_generator, get_tt_cache_path)
