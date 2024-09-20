# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from typing import Callable

from models.demos.wormhole.mamba.reference.args import ModelArgs, ModelMode
from models.demos.wormhole.mamba.tt.cache import TensorCache
from models.demos.wormhole.mamba.tt.mamba_ssm import TtMambaSSM
from models.demos.wormhole.mamba.tt.mamba_conv import MambaConvConfig, MambaConv


class TtMambaBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args
        self.batch_size = args.batch_size
        self.configs = configs

        in_proj_weight_name = "mixer.in_proj.weight"

        # ssm wt
        self.ssm_in_proj_weights = load_fn(
            in_proj_weight_name,
            lambda x: x[: self.args.d_inner, :].transpose(-1, -2),
            postfix="ssm",
            tt_dtype=self.configs["dtype"]["weights"],
        )

        # mlp wt
        self.mlp_proj_weights = load_fn(
            in_proj_weight_name,
            lambda x: x[self.args.d_inner :, :].transpose(-1, -2),
            postfix="mlp",
            tt_dtype=self.configs["dtype"]["weights"],
        )

        # down proj wt
        out_proj_weight_name = "mixer.out_proj.weight"
        self.out_proj_weights = load_fn(
            out_proj_weight_name, lambda x: x.transpose(-1, -2), tt_dtype=self.configs["dtype"]["weights"]
        )

        # conv1d_weights (2E, 1, 4)->(2E, 1)->(2E, 1, 1)->(1, 1, 2E)
        conv1d_weight_name = "mixer.conv1d.weight"
        self.conv1d_weights = []
        for i in range(4):
            self.conv1d_weights.append(
                load_fn(
                    conv1d_weight_name,
                    lambda x: x[:, :, i]
                    .transpose(-1, -2)
                    .repeat(self.configs["num_users"], 1)
                    .unsqueeze(0)
                    .unsqueeze(0),
                    postfix=f"{i}_{self.configs['num_users']}",
                )
            )

        conv1d_bias_name = "mixer.conv1d.bias"
        self.conv1d_bias_prefill = load_fn(
            conv1d_bias_name,
            lambda x: x.repeat(self.configs["outer_dim"], 1),
            postfix=f"{self.configs['outer_dim']}",
        )
        self.conv1d_bias_decode = load_fn(
            conv1d_bias_name,
            lambda x: x.repeat(self.configs["num_users"], 1),
            postfix=f"{self.configs['num_users']}",
        )
        self.conv1d_bias = self.conv1d_bias_prefill

        self.conv_states = []
        for i in range(4):
            self.conv_states.append(
                load_fn(
                    f"conv_state{i}",
                    torch_tensor=torch.zeros(1, 1, self.batch_size, self.args.d_inner),
                    postfix=f"{args.batch_size}",
                )
            )
        self.convolution_cache = TensorCache(configs["num_users"], 4, self.args.d_inner, device)

        mamba_conv_config = MambaConvConfig(
            input_length=self.configs["outer_dim"] + (args.d_conv - 1),
            weights_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
        )
        self.mamba_conv = MambaConv(device, load_fn, mamba_conv_config)

        self.tt_ssm = TtMambaSSM(self.args, self.device, configs, load_fn)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def to_prefill(self, prefill_config):
        self.configs = prefill_config
        self.conv1d_bias = self.conv1d_bias_prefill
        self.tt_ssm.to_prefill(self.configs)

    def to_decode(self, decode_config):
        self.configs = decode_config
        self.conv1d_bias = self.conv1d_bias_decode
        self.conv_states = []
        for i in range(0, 4):
            self.conv_states.append(
                ttnn.typecast(self.convolution_cache.concat_users(i), self.configs["dtype"]["activations"])
            )
        self.tt_ssm.to_decode(self.configs)

    def forward(self, x):
        assert len(x.shape) == 4, "Mamba block expects inputs to be rank 4"

        residual = ttnn.linear(
            x,
            self.mlp_proj_weights,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=x.device().core_grid,
            dtype=self.configs["dtype"]["activations"],
            activation="silu",
        )

        x_ssm = ttnn.linear(
            x,
            self.ssm_in_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=x.device().core_grid,
            dtype=ttnn.bfloat16,  # convolution requires bfloat16
        )
        ttnn.deallocate(x)

        if self.configs["mode"] == ModelMode.DECODE:
            # shift the states leftward
            ttnn.deallocate(self.conv_states[0])
            for i in range(3):
                self.conv_states[i] = self.conv_states[i + 1]

            # update the last state and move it back to DRAM with all the other states
            self.conv_states[3] = ttnn.to_memory_config(x_ssm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x_ssm)

            # do the convolution
            conv_accumulator = ttnn.multiply(
                self.conv_states[0],
                self.conv1d_weights[0],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=self.configs["dtype"]["activations"],
            )

            for i in range(1, 4):
                prod = ttnn.mul(
                    self.conv_states[i],
                    self.conv1d_weights[i],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=self.configs["dtype"]["activations"],
                )

                conv_accumulator = ttnn.add(
                    conv_accumulator,
                    prod,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=self.configs["dtype"]["activations"],
                    output_tensor=conv_accumulator,
                )
                ttnn.deallocate(prod)

            conv_out_with_bias = ttnn.add(
                conv_accumulator,
                self.conv1d_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=self.configs["dtype"]["activations"],
            )
            ttnn.deallocate(conv_accumulator)

        elif self.configs["mode"] == ModelMode.PREFILL:
            x_ssm = ttnn.to_layout(x_ssm, ttnn.ROW_MAJOR_LAYOUT)
            x_ssm = ttnn.concat(
                [
                    self.convolution_cache.get(self.configs["current_user"], 1),
                    self.convolution_cache.get(self.configs["current_user"], 2),
                    self.convolution_cache.get(self.configs["current_user"], 3),
                    x_ssm,
                ],
                dim=-2,
            )  # (1, 1, L+3, E)

            for i in range(0, 4):
                slice_start = (0, 0, x_ssm.shape[2] - (4 - i), 0)
                slice_end = (1, 1, (x_ssm.shape[2] - (4 - i)) + 1, self.args.d_inner)
                entry = ttnn.slice(x_ssm, slice_start, slice_end)
                self.convolution_cache.set(self.configs["current_user"], i, entry)
                ttnn.deallocate(entry)

            conv_out_without_bias = self.mamba_conv(x_ssm)
            ttnn.deallocate(x_ssm)
            conv_out_with_bias = ttnn.add(
                conv_out_without_bias,
                self.conv1d_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=self.configs["dtype"]["activations"],
            )
            ttnn.deallocate(conv_out_without_bias)

        conv_out_after_silu = ttnn.silu(conv_out_with_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(conv_out_with_bias)

        out = self.tt_ssm(conv_out_after_silu)

        out = ttnn.multiply(
            out,
            residual,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=self.configs["dtype"]["activations"],
            output_tensor=out,
        )
        ttnn.deallocate(residual)

        out_proj = ttnn.linear(
            out,
            self.out_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=out.device().core_grid,
            dtype=self.configs["dtype"]["activations"],
        )
        ttnn.deallocate(out)

        return out_proj

    def reset(self):
        self.convolution_cache.reset()
        self.tt_ssm.reset()
