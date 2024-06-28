# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger

import tt_lib as ttl
import ttnn
from models.demos.t3000.falcon40b.tt.ops.distributed_layernorm import TtDistributedLayernorm
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


class PytorchDistributedLayernorm(torch.nn.Module):
    def __init__(self, gammas, betas, epsilon=1e-5):
        super().__init__()
        self.gammas = gammas
        self.betas = betas
        self.epsilon = epsilon

    def forward(self, xs):
        num_chunks = len(xs)
        counts = []
        meanxs = []
        meanx2s = []
        # Distributed processing
        for chunk in xs:
            count_local = chunk.shape[-1]
            meanx_local = torch.mean(chunk, dim=-1, keepdim=True)
            meanx2_local = torch.mean(torch.square(chunk), dim=-1, keepdim=True)

            counts.append(count_local)
            meanxs.append(meanx_local)
            meanx2s.append(meanx2_local)

        count = torch.torch.FloatTensor(counts).sum(dim=0)
        meanxs = [meanxs[i] * counts[i] for i in range(num_chunks)]  # Weighting by chunk size
        meanx2s = [meanx2s[i] * counts[i] for i in range(num_chunks)]  # Weighting by chunk size

        # AllGather meanx, meanx2
        meanxs = torch.stack(meanxs, dim=0)
        meanx2s = torch.stack(meanx2s, dim=0)

        # Reduce
        mean = meanxs.sum(dim=0) / count
        meanx2 = meanx2s.sum(dim=0) / count
        var = meanx2 - torch.square(mean)

        # Distributed processing
        ys = []
        for i in range(num_chunks):
            # Normalize the input
            x_hat_local = (xs[i] - mean) / torch.sqrt(var + self.epsilon)

            # Scale and shift
            y_local = self.gammas[i] * x_hat_local + self.betas[i]
            ys.append(y_local)

        # Post processing: concat ys
        y = torch.cat(ys, dim=-1)

        return y


class PytorchLayernorm(torch.nn.Module):
    def __init__(self, gamma, beta, hidden_size=8192):
        super().__init__()
        self.ln = torch.nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.ln.weight = gamma
        self.ln.bias = beta

        self.ln.eval()

    def forward(self, x):
        result = self.ln(x)
        return result


def run_test_DistributedLayernorm_inference(pcc, devices, model_location_generator, get_tt_cache_path):
    S = 2048
    num_chips = 8
    epsilon = 1e-5

    # Prepare input
    torch.manual_seed(0)

    model_input_shape = [1, S]
    model_version = "tiiuae/falcon-40b-instruct"

    model_config = get_model_config("BFLOAT8_B-DRAM", "prefill", model_input_shape, num_chips)

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    if 0:
        model_version = "tiiuae/falcon-40b-instruct"

        model_name = model_location_generator(model_version, model_subdir="Falcon")

        hugging_face_reference_model = FalconForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, num_hidden_layers=1
        )
        hugging_face_reference_model.eval()
        config = hugging_face_reference_model.config

        hidden_size = config.hidden_size
        gamma = hugging_face_reference_model.transformer.h[0].ln_attn.weight
        beta = hugging_face_reference_model.transformer.h[0].ln_attn.bias
    else:
        hidden_size = 8192
        gamma = torch.nn.Parameter(torch.randn(hidden_size))  # Scale parameter
        beta = torch.nn.Parameter(torch.randn(hidden_size))  # Shift parameter

    input_shape = [1, 1, S, hidden_size]

    input_torch = (torch.rand(input_shape) * 2) - 1

    inputs_torch = torch.chunk(input_torch, len(devices), -1)
    gammas_torch = torch.chunk(gamma, len(devices), -1)
    betas_torch = torch.chunk(beta, len(devices), -1)

    tt_inputs = []
    for i in range(len(devices)):
        tt_input_host = torch2tt_tensor(inputs_torch[i], None, tt_dtype=ttl.tensor.DataType.BFLOAT16)
        tt_inputs.append(tt_input_host.to(devices[i], model_config["DEFAULT_MEMCFG"]))

    # PyTorch basic layernorm output --------------------------------------------------------------------
    pytorch_FalconLayernorm_model = PytorchLayernorm(gamma=gamma, beta=beta)
    torch_layernorm_output = pytorch_FalconLayernorm_model(input_torch)

    # PyTorch distributed layernorm output --------------------------------------------------------------------
    pytorch_FalconLayernorm_model = PytorchDistributedLayernorm(gammas=gammas_torch, betas=betas_torch)
    torch_distributed_layernorm_outputs = pytorch_FalconLayernorm_model(inputs_torch)
    torch_distributed_layernorm_output = torch.concat([torch_distributed_layernorm_outputs], -1)

    # check pytorch vs. distributed pytorch implementation---------------------------------------------------------
    does_pass, output_pcc = comp_pcc(torch_layernorm_output, torch_distributed_layernorm_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Pytorch distributed layernorm Passed!")
    else:
        logger.warning("Pytorch distributed layernorm Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"

    # TT hardware execution -------------------------------------------------------------
    tt_distributed_layernorm = TtDistributedLayernorm(devices, gammas_torch, betas_torch, epsilon, tt_cache_path)

    tt_outputs = tt_distributed_layernorm(tt_inputs)

    tt_out = torch.concat([tt2torch_tensor(tt_o) for tt_o in tt_outputs], -1)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(torch_layernorm_output, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("TT Distributed Layernorm Passed!")
    else:
        logger.warning("TT Distributed Layernorm Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("pcc", [(0.99)])
def test_DistributedLayernorm_inference(
    pcc,
    all_devices,
    model_location_generator,
    get_tt_cache_path,
):
    devices = get_devices_for_t3000(all_devices, 8)

    run_test_DistributedLayernorm_inference(pcc, devices, model_location_generator, get_tt_cache_path)
