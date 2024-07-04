# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger

import tt_lib as ttl
import ttnn
from models.demos.t3000.falcon40b.tt.ops.distributed_layernorm_dlnp1 import TtDistributedLayernormDLNP1
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


class PytorchDistributedLayernormDLNP1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        num_chunks = len(xs)
        S = xs[0].shape[2]
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

        meanxs = [meanxs[i] * counts[i] for i in range(num_chunks)]  # Weighting by chunk size
        meanx2s = [meanx2s[i] * counts[i] for i in range(num_chunks)]  # Weighting by chunk size

        # pad with zeros as for tiles
        output = []
        for i in range(num_chunks):
            output.append(
                torch.concat([meanxs[i], torch.zeros([1, 1, S, 31]), meanx2s[i], torch.zeros([1, 1, S, 31])], dim=-1)
            )

        return output


def run_test_DistributedLayernorm_inference(pcc, devices, model_location_generator, get_tt_cache_path):
    S = 2048

    # Prepare input
    torch.manual_seed(0)

    hidden_size = 8192
    input_shape = [1, 1, S, hidden_size]
    input_torch = (torch.rand(input_shape) * 2) - 1
    inputs_torch = torch.chunk(input_torch, len(devices), -1)

    dram_memcfg = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)

    tt_inputs = []
    for i in range(len(devices)):
        tt_input_host = torch2tt_tensor(inputs_torch[i], None, tt_dtype=ttl.tensor.DataType.BFLOAT16)
        tt_inputs.append(tt_input_host.to(devices[i], dram_memcfg))

    # PyTorch distributed layernorm output --------------------------------------------------------------------
    pytorch_FalconLayernorm_model = PytorchDistributedLayernormDLNP1()
    torch_output = pytorch_FalconLayernorm_model(inputs_torch)
    torch_output = torch.concat(torch_output, -1)

    # TT hardware execution -------------------------------------------------------------
    tt_distributed_layernorm = TtDistributedLayernormDLNP1()
    tt_output = tt_distributed_layernorm(tt_inputs)

    tt_output_host = torch.concat([tt2torch_tensor(tt_o) for tt_o in tt_output], -1)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(torch_output, tt_output_host, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("TT DLP1 Passed!")
    else:
        logger.warning("TT DLP1 Failed!")
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
