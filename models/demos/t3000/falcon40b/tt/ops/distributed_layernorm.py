# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
import tt_lib as ttl
import ttnn

from typing import List
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class TtDistributedLayernorm:
    def __init__(self, devices, gammas, betas, epsilon, tt_cache_path):
        super().__init__()

        self.devices = devices
        ln_weights_str = f"ln.weight"
        ln_bias_str = f"ln.bias"

        dtype = ttl.tensor.DataType.BFLOAT16
        # dtype = ttl.tensor.DataType.BFLOAT8_B
        dram_memcfg = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
        self.dram_memcfg = dram_memcfg

        num_devices = len(devices)

        self.ln_gamma = []
        self.ln_beta = []
        for i in range(num_devices):
            ln_weights_path = tt_cache_path / f"{ln_weights_str}_{dtype.name}_{i}_{num_devices}.bin"
            if (ln_weights_path).exists():
                ln_gamma_host = ttl.tensor.load_tensor(str(ln_weights_path))
                self.ln_gamma.append(ln_gamma_host.to(devices[i], dram_memcfg))
            else:
                ln_gamma_host = torch2tt_tensor(
                    gammas[i],
                    None,
                    tt_layout=ttl.tensor.Layout.ROW_MAJOR,
                    tt_memory_config=dram_memcfg,
                    tt_dtype=dtype,
                )

                self.ln_gamma.append(ln_gamma_host.to(devices[i], dram_memcfg))

                ttl.tensor.dump_tensor(
                    str(ln_weights_path),
                    ln_gamma_host,
                )

            ln_bias_path = tt_cache_path / f"{ln_bias_str}_{dtype.name}_{i}_{num_devices}.bin"
            if (ln_bias_path).exists():
                ln_beta_host = ttl.tensor.load_tensor(str(ln_bias_path))
                self.ln_beta.append(ln_beta_host.to(devices[i], dram_memcfg))
            else:
                ln_beta_host = torch2tt_tensor(
                    betas[i],
                    None,
                    tt_layout=ttl.tensor.Layout.ROW_MAJOR,
                    tt_memory_config=dram_memcfg,
                    tt_dtype=dtype,
                )
                self.ln_beta.append(ln_beta_host.to(devices[i], dram_memcfg))

                ttl.tensor.dump_tensor(
                    str(ln_bias_path),
                    ln_beta_host,
                )

        self.ln_eps = epsilon

    def __call__(self, xs: ttl.tensor.Tensor) -> ttl.tensor.Tensor:
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

            meanx_local = ttl.tensor.reduce(
                xs[i], ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, scaler=1.0 / counts[i]
            )
            meanxs.append(meanx_local)

        # meanx2 = torch.mean(torch.square(xs), dim=-1, keepdim=True)
        meanx2s = []
        for i in range(num_devices):
            x2_local = ttl.tensor.pow(xs[i], 2)
            meanx2_local = ttl.tensor.reduce(
                x2_local, ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, scaler=1.0 / counts[i]
            )
            meanx2s.append(meanx2_local)

        # AllReduce meanx and meanx2
        # Weighted meanx to number of samples per device
        for i in range(num_devices):
            meanxs[i] = ttl.tensor.mul_unary(meanxs[i], counts[i])
        # AllGather
        meanxs = ttl.tensor.all_gather(
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
                ttl.tensor.reduce(
                    meanxs[i], ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, scaler=1.0 / total_count
                )
            )

        # Weighted meanx2 to number of samples per device
        for i in range(num_devices):
            meanx2s[i] = ttl.tensor.mul_unary(meanx2s[i], counts[i])
        # AllGather
        meanx2s = ttl.tensor.all_gather(
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
                ttl.tensor.reduce(
                    meanx2s[i], ttl.tensor.ReduceOpMath.SUM, ttl.tensor.ReduceOpDim.W, scaler=1.0 / total_count
                )
            )

        # Variance
        # var = meanx2 - torch.square(mean)
        var = []
        for i in range(num_devices):
            var.append(ttl.tensor.pow(mean[i], 2))
        for i in range(num_devices):
            var[i] = ttl.tensor.sub(meanx2[i], var[i])
            meanx2[i].deallocate(True)

        # Normalize the input: x_hat = (xs[i] - mean) / torch.sqrt(var + epsilon)
        denominators = []
        for i in range(num_devices):
            denominators.append(ttl.tensor.add_unary(var[i], self.ln_eps))
        for i in range(num_devices):
            denominators[i] = ttl.tensor.pow(denominators[i], 0.5)
        for i in range(num_devices):
            denominators[i] = ttl.tensor.recip(denominators[i])

        nominators = []
        for i in range(num_devices):
            nominators.append(
                ttl.tensor.bcast(xs[i], mean[i], math_op=ttl.tensor.BcastOpMath.SUB, dim=ttl.tensor.BcastOpDim.W)
            )

        x_hat = []
        for i in range(num_devices):
            x_hat.append(
                ttl.tensor.bcast(
                    nominators[i], denominators[i], math_op=ttl.tensor.BcastOpMath.MUL, dim=ttl.tensor.BcastOpDim.W
                )
            )
            nominators[i].deallocate(True)
            denominators[i].deallocate(True)

        # Scale and shift: x_hat = self.gammas * x_hat + self.betas_torch
        for i in range(num_devices):
            x_hat[i] = ttl.tensor.bcast(
                x_hat[i], self.ln_gamma[i], math_op=ttl.tensor.BcastOpMath.MUL, dim=ttl.tensor.BcastOpDim.H
            )
        for i in range(num_devices):
            x_hat[i] = ttl.tensor.bcast(
                x_hat[i], self.ln_beta[i], math_op=ttl.tensor.BcastOpMath.ADD, dim=ttl.tensor.BcastOpDim.H
            )

        return x_hat
