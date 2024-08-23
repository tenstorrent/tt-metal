# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
from torch import nn

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


class PytorchFusedLayernorm(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num=0):
        super().__init__()
        self.ln_attn = hf_reference_model.transformer.h[layer_num].ln_attn
        self.ln_mlp = hf_reference_model.transformer.h[layer_num].ln_mlp

        self.ln_attn.eval()
        self.ln_mlp.eval()

    def forward(self, x):
        result1 = self.ln_attn(x)
        result2 = self.ln_mlp(x)
        return result1, result2


class TtFusedFalconLayernorm:
    def __init__(self, device, gamma1, beta1, gamma2, beta2, model_config, config, tt_cache_path):
        super().__init__()

        gamma_beta_rm = False

        self.model_config = model_config

        if gamma_beta_rm:
            ln_attn_gamma_host = ttnn.Tensor(
                gamma1.reshape([1, 1, 1, -1]),
                self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
            )
            self.ln_attn_gamma = ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])

            ln_attn_beta_host = ttnn.Tensor(
                beta1.reshape([1, 1, 1, -1]),
                self.model_config["LN_ATTN_BIAS_DTYPE"],
            )
            self.ln_attn_beta = ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"])

            ln_mlp_gamma_host = ttnn.Tensor(
                gamma2.reshape([1, 1, 1, -1]),
                self.model_config["LN_MLP_WEIGHTS_DTYPE"],
            )
            self.ln_mlp_gamma = ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"])

            ln_mlp_beta_host = ttnn.Tensor(
                beta2.reshape([1, 1, 1, -1]),
                self.model_config["LN_MLP_BIAS_DTYPE"],
            )
            self.ln_mlp_beta = ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"])

        else:
            ln_attn_gamma_torch = gamma1.reshape(1, 1, 1, -1)
            ln_attn_gamma_torch_padded = torch.cat(
                [ln_attn_gamma_torch, torch.zeros(1, 1, 31, ln_attn_gamma_torch.shape[-1])], dim=2
            )
            ln_attn_gamma_host = torch2tt_tensor(
                ln_attn_gamma_torch_padded,
                None,
                tt_layout=ttnn.TILE_LAYOUT,
                tt_memory_config=self.model_config["LN_ATTN_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["LN_ATTN_WEIGHTS_DTYPE"],
            )
            self.ln_attn_gamma = ln_attn_gamma_host.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])

            ln_attn_beta_torch = beta1.reshape(1, 1, 1, -1)
            ln_attn_beta_torch_padded = torch.cat(
                [ln_attn_beta_torch, torch.zeros(1, 1, 31, ln_attn_beta_torch.shape[-1])], dim=2
            )
            ln_attn_beta_host = torch2tt_tensor(
                ln_attn_beta_torch_padded,
                None,
                tt_layout=ttnn.experimental.tensor.Layout.TILE,
                tt_memory_config=self.model_config["LN_ATTN_BIAS_MEMCFG"],
                tt_dtype=self.model_config["LN_ATTN_BIAS_DTYPE"],
            )
            self.ln_attn_beta = ln_attn_beta_host.to(device, self.model_config["LN_ATTN_BIAS_MEMCFG"])

            ln_mlp_gamma_torch = gamma2.reshape(1, 1, 1, -1)
            ln_mlp_gamma_torch_padded = torch.cat(
                [ln_mlp_gamma_torch, torch.zeros(1, 1, 31, ln_mlp_gamma_torch.shape[-1])], dim=2
            )
            ln_mlp_gamma_host = torch2tt_tensor(
                ln_mlp_gamma_torch_padded,
                None,
                tt_layout=ttnn.experimental.tensor.Layout.TILE,
                tt_memory_config=self.model_config["LN_MLP_WEIGHTS_MEMCFG"],
                tt_dtype=self.model_config["LN_MLP_WEIGHTS_DTYPE"],
            )
            self.ln_mlp_gamma = ln_mlp_gamma_host.to(device, self.model_config["LN_MLP_WEIGHTS_MEMCFG"])

            ln_mlp_beta_torch = beta2.reshape(1, 1, 1, -1)
            ln_mlp_beta_torch_padded = torch.cat(
                [ln_mlp_beta_torch, torch.zeros(1, 1, 31, ln_mlp_beta_torch.shape[-1])], dim=2
            )
            ln_mlp_beta_host = torch2tt_tensor(
                ln_mlp_beta_torch_padded,
                None,
                tt_layout=ttnn.experimental.tensor.Layout.TILE,
                tt_memory_config=self.model_config["LN_MLP_BIAS_MEMCFG"],
                tt_dtype=self.model_config["LN_MLP_BIAS_DTYPE"],
            )
            self.ln_mlp_beta = ln_mlp_beta_host.to(device, self.model_config["LN_MLP_BIAS_MEMCFG"])

        self.layernorm_eps = config.layer_norm_epsilon

        shard_spec_cores_grid = ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(7, 7),
                ),
            }
        )
        H = config.hidden_size  # 8192
        self.sharded_memconfig = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            ttnn.experimental.tensor.ShardSpec(
                shard_spec_cores_grid,
                [self.model_config["SEQ_LEN"] // 8, H // 8],  # 1024
                ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        # self.width_sharded_memconfig = ttnn.experimental.tensor.MemoryConfig(
        #     ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED,
        #     ttnn.experimental.tensor.BufferType.L1,
        #     ttnn.experimental.tensor.ShardSpec(
        #         shard_spec_cores_grid,
        #         [
        #             self.model_config["SEQ_LEN"],
        #             H // 64,
        #         ],
        #         ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR,
        #         False,
        #     ),
        # )

        self.prg_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 8],
            subblock_w=min(8, H // 8),
            block_h=self.model_config["SEQ_LEN"] // 32 // 8,
            block_w=H // 32 // 8,
            inplace=False,
        )

        self.interleaved_memconfig = ttnn.experimental.tensor.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            ttnn.experimental.tensor.BufferType.L1,
        )

    def __call__(self, x: ttnn.experimental.tensor.Tensor) -> ttnn.experimental.tensor.Tensor:
        # # OG layernorm
        # out1 = ttnn.layer_norm(
        #     x, epsilon=self.layernorm_eps, weight=self.ln_attn_gamma, bias=self.ln_attn_beta, memory_config=self.sharded_memconfig, program_config=self.prg_config
        # )
        # out2 = ttnn.layer_norm(
        #     x, epsilon=self.layernorm_eps, weight=self.ln_mlp_gamma, bias=self.ln_mlp_beta, memory_config=self.sharded_memconfig, program_config=self.prg_config
        # )

        # all block sharded
        out2 = ttnn.layer_norm(
            x,
            epsilon=self.layernorm_eps,
            weight=None,
            bias=None,
            memory_config=self.sharded_memconfig,
            program_config=self.prg_config,
        )

        out1 = ttnn.bcast(
            out2,
            self.ln_attn_gamma,
            math_op=ttnn.BcastOpMath.MUL,
            dim=ttnn.BcastOpDim.H,
            memory_config=self.sharded_memconfig,
        )
        out1 = ttnn.bcast(
            out1,
            self.ln_attn_beta,
            math_op=ttnn.BcastOpMath.ADD,
            dim=ttnn.BcastOpDim.H,
            memory_config=self.sharded_memconfig,
        )

        out2 = ttnn.bcast(
            out2,
            self.ln_mlp_gamma,
            math_op=ttnn.BcastOpMath.MUL,
            dim=ttnn.BcastOpDim.H,
            memory_config=self.sharded_memconfig,
        )
        out2 = ttnn.bcast(
            out2,
            self.ln_mlp_beta,
            math_op=ttnn.BcastOpMath.ADD,
            dim=ttnn.BcastOpDim.H,
            memory_config=self.sharded_memconfig,
        )

        return out1, out2


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
    input = torch2tt_tensor(input_torch, None, tt_dtype=ttnn.bfloat16)  # tt_dtype=ttnn.bfloat16  # ttnn.bfloat8_b
    input = input.to(devices[0], model_config["DEFAULT_MEMCFG"])

    # block sharded hardcoded for S=128 and 8x4 grid of cores
    shard_spec_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 7),
            ),
        }
    )
    block_sharded_memconfig = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_cores_grid,
            [
                seqlen // 8,
                H // 8,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    # width_sharded_memconfig = ttnn.MemoryConfig(
    #     ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    #     ttnn.BufferType.L1,
    #     ttnn.ShardSpec(
    #         shard_spec_cores_grid,
    #         [
    #             seqlen,
    #             H // 64,
    #         ],
    #         ttnn.ShardOrientation.ROW_MAJOR,
    #         False,
    #     ),
    # )
    input = ttnn.interleaved_to_sharded(input, block_sharded_memconfig)

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
