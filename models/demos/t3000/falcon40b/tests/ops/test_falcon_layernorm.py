# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger

import tt_lib as ttl
import ttnn
from models.demos.t3000.falcon40b.tt.ops.falcon_layernorm import TtFalconLayernorm
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


class PytorchFalconLayernorm(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num=0):
        super().__init__()
        self.ln_attn = hf_reference_model.transformer.h[layer_num].ln_attn

        # Disable dropout
        self.ln_attn.eval()

    def forward(self, x):
        result = self.ln_attn(x)
        return result


def run_test_FalconLayernorm_inference(pcc, devices, model_location_generator, get_tt_cache_path):
    is_sharded = True

    seqlen = 1024
    num_chips = 8

    # Prepare input
    torch.manual_seed(0)
    model_input_shape = [1, seqlen]

    model_version = "tiiuae/falcon-40b-instruct"

    if is_sharded:
        # model_config = get_model_config("BFLOAT8_B-SHARDED", "prefill", model_input_shape, num_chips) # Decode sharding
        model_config = get_model_config(
            "BFLOAT8_B-DRAM", "prefill", model_input_shape, num_chips
        )  # Block sharding for layernorm to work around PCC issue
    else:
        model_config = get_model_config("BFLOAT8_B-DRAM", "prefill", model_input_shape, num_chips)

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, num_hidden_layers=1
    )
    hugging_face_reference_model.eval()
    config = hugging_face_reference_model.config

    input_shape = [1, 1, seqlen, config.hidden_size]  # 4544 for 7b

    input_torch = (torch.rand(input_shape) * 2) - 1
    input = torch2tt_tensor(
        input_torch, None, tt_dtype=ttl.tensor.DataType.BFLOAT8_B
    )  # ttl.tensor.DataType.BFLOAT16 # TODO: should be BF16!!
    input = input.to(devices[0], model_config["DEFAULT_MEMCFG"])

    if is_sharded:
        # # Option1 : width sharded; produces bad PCC
        # shard_spec_32_cores_grid = ttl.tensor.CoreRangeSet(
        #     {
        #         ttl.tensor.CoreRange(
        #             ttl.tensor.CoreCoord(0, 0),
        #             ttl.tensor.CoreCoord(7, 3),
        #         ),
        #     }
        # )
        # input = ttl.tensor.interleaved_to_sharded(
        #     input,
        #     sharded_mem_config=ttl.tensor.MemoryConfig(
        #         ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttl.tensor.BufferType.L1,
        #         ttl.tensor.ShardSpec(
        #             shard_spec_32_cores_grid,
        #             [
        #                 seqlen,
        #                 config.hidden_size // 32,
        #             ],
        #             ttl.tensor.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        # )

        # # Option 2: block sharded hardcoded for S=128 and 8x4 grid of cores; produces good PCC!
        # shard_spec_32_cores_grid = ttl.tensor.CoreRangeSet(
        #         {
        #             ttl.tensor.CoreRange(
        #                 ttl.tensor.CoreCoord(0, 0),
        #                 ttl.tensor.CoreCoord(7, 3),
        #             ),
        #         }
        #     )
        # input = ttl.tensor.interleaved_to_sharded(
        #     input,
        #     sharded_mem_config=ttl.tensor.MemoryConfig(
        #     ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED,
        #     ttl.tensor.BufferType.L1,
        #     ttl.tensor.ShardSpec(
        #         shard_spec_32_cores_grid,
        #         [
        #             32,
        #             1024,
        #         ],
        #         ttl.tensor.ShardOrientation.ROW_MAJOR,
        #         False,
        #     ),
        #     )
        # )

        # # Version according to model_config for debug
        input = ttl.tensor.interleaved_to_sharded(
            input,
            sharded_mem_config=model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"],
        )

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconLayernorm_model = PytorchFalconLayernorm(hugging_face_reference_model)
    torch_out = pytorch_FalconLayernorm_model(input_torch)

    # TT hardware execution -------------------------------------------------------------
    tt_Falcon_layernorm_model = TtFalconLayernorm(devices, model_config, config, tt_cache_path, is_sharded=is_sharded)

    tt_out = tt_Falcon_layernorm_model(input)

    if is_sharded:
        tt_out = convert_to_layout(tt_out, model_config["LN_ATTN_OUTPUT_MEMCFG"], model_config["DEFAULT_MEMCFG"])

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
def test_FalconLayernorm_inference(
    pcc,
    all_devices,
    model_location_generator,
    get_tt_cache_path,
):
    devices = all_devices

    run_test_FalconLayernorm_inference(pcc, devices, model_location_generator, get_tt_cache_path)
