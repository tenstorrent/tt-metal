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
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000


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


class TtDistributedLayernormDLNP1:
    def __init__(self):
        super().__init__()

    def __call__(self, xs: ttnn.Tensor) -> ttnn.Tensor:
        num_devices = len(xs)

        counts = []
        total_count = 0
        meanxs = []

        # Each device computes local statistics mean(x) and mean(x^2)
        # meanx = torch.mean(xs, dim=-1, keepdim=True)
        for i in range(num_devices):
            count_local = xs[i].shape[-1]
            total_count += count_local
            counts.append(count_local)

            meanx_local = ttnn.sum(xs[i], dim=3, scaler=1.0 / counts[i])
            meanxs.append(meanx_local)

        # meanx2 = torch.mean(torch.square(xs), dim=-1, keepdim=True)
        meanx2s = []
        for i in range(num_devices):
            x2_local = ttnn.pow(xs[i], 2)
            meanx2_local = ttnn.sum(x2_local, dim=3, scaler=1.0 / counts[i])
            meanx2s.append(meanx2_local)

        # Weighted meanx to number of samples per device
        for i in range(num_devices):
            meanxs[i] = ttnn.multiply(meanxs[i], counts[i])

        # Weighted meanx2 to number of samples per device
        for i in range(num_devices):
            meanx2s[i] = ttnn.multiply(meanx2s[i], counts[i])

        output = []
        for i in range(num_devices):
            output.append(ttnn.concat([meanxs[i], meanx2s[i]], 3))

        return output


def run_test_DistributedLayernorm_inference(pcc, devices, model_location_generator, get_tt_cache_path):
    S = 2048

    # Prepare input
    torch.manual_seed(0)

    hidden_size = 8192
    input_shape = [1, 1, S, hidden_size]
    input_torch = (torch.rand(input_shape) * 2) - 1
    inputs_torch = torch.chunk(input_torch, len(devices), -1)

    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    tt_inputs = []
    for i in range(len(devices)):
        tt_input_host = torch2tt_tensor(inputs_torch[i], None, tt_dtype=ttnn.bfloat16)
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
