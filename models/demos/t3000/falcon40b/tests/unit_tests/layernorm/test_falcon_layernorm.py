# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import math
from torch import nn
from typing import List

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull
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


class TtFalconLayernorm:
    def __init__(self, device, gamma, beta, model_config, config, tt_cache_path, is_sharded=False):
        super().__init__()

        self.model_config = model_config
        self.is_sharded = is_sharded

        gamma_host = ttnn.Tensor(
            gamma.reshape([1, 1, -1, 32]),
            self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
        )
        self.gamma = gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])

        beta_host = ttnn.Tensor(
            beta.reshape([1, 1, -1, 32]),
            self.model_config["LN_ATTN_BIAS_DTYPE"],
        )
        self.beta = beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"])

        self.layernorm_eps = config.layer_norm_epsilon

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.is_sharded:
            row_height = x.get_legacy_shape()[2]
            shard_width_hidden_dim_across_32_cores = x.get_legacy_shape()[3] // 32
            shard_spec_32_cores_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 3),
                    ),
                }
            )
            # # Option1 : width sharded; produces bad PCC
            # out = ttnn.layer_norm(
            #     x,
            #     epsilon=self.layernorm_eps,
            #     weight=self.gamma,
            #     bias=self.beta,
            #     memory_config=ttnn.MemoryConfig(
            #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            #         ttnn.BufferType.L1,
            #         ttnn.ShardSpec(
            #             shard_spec_32_cores_grid,
            #             [
            #                 row_height,
            #                 shard_width_hidden_dim_across_32_cores,
            #             ],
            #             ttnn.ShardOrientation.ROW_MAJOR,
            #             False,
            #         ),
            #     ),
            #     program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            #         compute_with_storage_grid_size=[8, 4],
            #         subblock_w=8,
            #         block_h=row_height // 32,
            #         block_w=8,
            #         inplace=False,
            #     ),
            # )

            # option 2: block sharded hardcoded for S=128 and 8x4 grid of cores; produces good PCC!
            out = ttnn.layer_norm(
                x,
                epsilon=self.layernorm_eps,
                weight=self.gamma,
                bias=self.beta,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        shard_spec_32_cores_grid,
                        [
                            32,
                            1024,
                        ],
                        ttnn.ShardOrientation.ROW_MAJOR,
                        False,
                    ),
                ),
                program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[8, 4],
                    subblock_w=8,
                    block_h=1,
                    block_w=32,
                    inplace=False,
                ),
            )
        else:  # Interleaved does not work for falcon40b dims [32, 8192] since once one core per tile-height is used to process the whole row
            # Uses only one core; runs out of L1
            # E           Statically allocated circular buffers on core range {} grow to {} B which is beyond max L1 size of {} B
            # E           [(x=0,y=0) - (x=1,y=0)]
            out = ttnn.layer_norm(
                x,
                epsilon=self.layernorm_eps,
                weight=self.gamma,
                bias=self.beta,
            )

        return out


def run_test_FalconLayernorm_inference(pcc, device, model_location_generator, get_tt_cache_path):
    is_sharded = True

    seqlen = 128
    num_chips = 8

    # Prepare input
    torch.manual_seed(0)
    model_input_shape = [1, seqlen]

    model_version = "tiiuae/falcon-40b-instruct"
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
    gamma = hugging_face_reference_model.transformer.h[0].ln_attn.weight
    beta = hugging_face_reference_model.transformer.h[0].ln_attn.bias

    input_shape = [1, 1, seqlen, config.hidden_size]

    input_torch = (torch.rand(input_shape) * 2) - 1
    input = torch2tt_tensor(input_torch, None, tt_dtype=ttnn.bfloat8_b)  # ttnn.bfloat16 # TODO: should be BF16!!
    input = input.to(device, model_config["DEFAULT_MEMCFG"])

    if is_sharded:
        # # Option1 : width sharded; produces bad PCC
        # shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        #     {
        #         ttnn.CoreRange(
        #             ttnn.CoreCoord(0, 0),
        #             ttnn.CoreCoord(7, 3),
        #         ),
        #     }
        # )
        # input = ttnn.interleaved_to_sharded(
        #     input,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_32_cores_grid,
        #             [
        #                 seqlen,
        #                 config.hidden_size // 32,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        # )

        # Option 2: block sharded hardcoded for S=128 and 8x4 grid of cores; produces good PCC!
        shard_spec_32_cores_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 3),
                ),
            }
        )
        input = ttnn.interleaved_to_sharded(
            input,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_spec_32_cores_grid,
                    [
                        32,
                        1024,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
        )

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconLayernorm_model = PytorchFalconLayernorm(hugging_face_reference_model)
    torch_out = pytorch_FalconLayernorm_model(input_torch)

    # TT hardware execution -------------------------------------------------------------
    tt_Falcon_layernorm_model = TtFalconLayernorm(
        device, gamma, beta, model_config, config, tt_cache_path, is_sharded=is_sharded
    )

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

    run_test_FalconLayernorm_inference(pcc, devices[0], model_location_generator, get_tt_cache_path)
