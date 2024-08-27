# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger

import ttnn
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


class TtDistributedLayernorm:
    def __init__(self, devices, gammas, betas, epsilon, tt_cache_path):
        super().__init__()

        self.devices = devices
        ln_weights_str = f"ln.weight"
        ln_bias_str = f"ln.bias"

        dtype = ttnn.bfloat16
        # dtype = ttnn.bfloat8_b
        dram_memcfg = ttnn.DRAM_MEMORY_CONFIG
        self.dram_memcfg = dram_memcfg

        num_devices = len(devices)

        self.ln_gamma = []
        self.ln_beta = []
        for i in range(num_devices):
            ln_weights_path = tt_cache_path / f"{ln_weights_str}_{dtype.name}_{i}_{num_devices}.bin"
            if (ln_weights_path).exists():
                ln_gamma_host = ttnn.load_tensor(str(ln_weights_path))
                self.ln_gamma.append(ln_gamma_host.to(devices[i], dram_memcfg))
            else:
                ln_gamma_host = torch2tt_tensor(
                    gammas[i],
                    None,
                    tt_layout=ttnn.ROW_MAJOR_LAYOUT,
                    tt_memory_config=dram_memcfg,
                    tt_dtype=dtype,
                )

                self.ln_gamma.append(ln_gamma_host.to(devices[i], dram_memcfg))

                ttnn.dump_tensor(
                    str(ln_weights_path),
                    ln_gamma_host,
                )

            ln_bias_path = tt_cache_path / f"{ln_bias_str}_{dtype.name}_{i}_{num_devices}.bin"
            if (ln_bias_path).exists():
                ln_beta_host = ttnn.load_tensor(str(ln_bias_path))
                self.ln_beta.append(ln_beta_host.to(devices[i], dram_memcfg))
            else:
                ln_beta_host = torch2tt_tensor(
                    betas[i],
                    None,
                    tt_layout=ttnn.ROW_MAJOR_LAYOUT,
                    tt_memory_config=dram_memcfg,
                    tt_dtype=dtype,
                )
                self.ln_beta.append(ln_beta_host.to(devices[i], dram_memcfg))

                ttnn.dump_tensor(
                    str(ln_bias_path),
                    ln_beta_host,
                )

        self.ln_eps = epsilon

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

            meanx_local = ttnn.sum(xs[i], 3, scaler=1.0 / counts[i])
            meanxs.append(meanx_local)

        # meanx2 = torch.mean(torch.square(xs), dim=-1, keepdim=True)
        meanx2s = []
        for i in range(num_devices):
            x2_local = ttnn.pow(xs[i], 2)
            meanx2_local = ttnn.sum(x2_local, dim=3, scaler=1.0 / counts[i])
            meanx2s.append(meanx2_local)

        # AllReduce meanx and meanx2
        # Weighted meanx to number of samples per device
        for i in range(num_devices):
            meanxs[i] = ttnn.multiply(meanxs[i], counts[i])
        # AllGather
        meanxs = ttnn.all_gather(
            meanxs,
            dim=3,
            num_links=1,
            output_mem_config=self.dram_memcfg,
        )
        # Mean over per-device meanx
        # mean = torch.stack(meanx, dim=0).sum(dim=0) / total_count
        mean = []
        for i in range(num_devices):
            mean.append(
                ttnn.sum(
                    meanxs[i],
                    dim=3,
                    scaler=1.0 / total_count,
                )
            )

        # Weighted meanx2 to number of samples per device
        for i in range(num_devices):
            meanx2s[i] = ttnn.multiply(meanx2s[i], counts[i])
        # AllGather
        meanx2s = ttnn.all_gather(
            meanx2s,
            dim=3,
            num_links=1,
            output_mem_config=self.dram_memcfg,
        )
        # Mean over per-device meanx2
        # meanx2 = torch.stack(meanx2, dim=0).sum(dim=0) / total_count
        meanx2 = []
        for i in range(num_devices):
            meanx2.append(
                ttnn.sum(
                    meanx2s[i],
                    dim=3,
                    scaler=1.0 / total_count,
                )
            )

        # Variance
        # var = meanx2 - torch.square(mean)
        var = []
        for i in range(num_devices):
            var.append(ttnn.pow(mean[i], 2))
        for i in range(num_devices):
            var[i] = ttnn.subtract(meanx2[i], var[i])
            meanx2[i].deallocate(True)

        # Normalize the input: x_hat = (xs[i] - mean) / torch.sqrt(var + epsilon)
        denominators = []
        for i in range(num_devices):
            denominators.append(ttnn.add(var[i], self.ln_eps))
        for i in range(num_devices):
            denominators[i] = ttnn.pow(denominators[i], 0.5)
        for i in range(num_devices):
            denominators[i] = ttnn.reciprocal(denominators[i])

        nominators = []
        for i in range(num_devices):
            nominators.append(ttnn.subtract(xs[i], mean[i]))

        x_hat = []
        for i in range(num_devices):
            x_hat.append(ttnn.multiply(nominators[i], denominators[i]))
            nominators[i].deallocate(True)
            denominators[i].deallocate(True)

        # Scale and shift: x_hat = self.gammas * x_hat + self.betas_torch
        for i in range(num_devices):
            x_hat[i] = ttnn.multiply(x_hat[i], self.ln_gamma[i])
        for i in range(num_devices):
            x_hat[i] = ttnn.add(x_hat[i], self.ln_beta[i])

        return x_hat


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
        tt_input_host = torch2tt_tensor(inputs_torch[i], None, tt_dtype=ttnn.bfloat16)
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
