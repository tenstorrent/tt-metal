# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import tt_lib as ttl
from typing import Callable

from models.demos.mamba.reference.args import ModelArgs
from models.demos.mamba.tt.mamba_one_step_ssm import TtMambaSSM


class TtMambaBlock(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args
        self.batch_size = args.batch_size
        self.configs = configs

        assert self.batch_size == 32, "Batch size must be 32 for now"

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

        # conv states
        conv1d_weight_name = "mixer.conv1d.weight"
        self.conv1d_weights = []
        # conv1d_weights (2E, 1, 4)->(2E, 1)->(2E, 1, 1)->(1, 1, 2E)
        for i in range(4):
            self.conv1d_weights.append(
                load_fn(
                    conv1d_weight_name,
                    lambda x: x[:, :, i].transpose(-1, -2).repeat(self.batch_size, 1).unsqueeze(0).unsqueeze(0),
                    postfix=f"{i}_{args.batch_size}",
                )
            )

        conv1d_bias_name = "mixer.conv1d.bias"
        self.conv1d_bias = load_fn(
            conv1d_bias_name,
            lambda x: x.repeat(self.args.batch_size, 1),
            postfix=f"{args.batch_size}",
        )

        self.conv_states = []
        for i in range(4):
            self.conv_states.append(
                load_fn(
                    f"conv_state{i}",
                    torch_tensor=torch.zeros(1, 1, self.batch_size, self.args.d_inner),
                    postfix=f"{args.batch_size}",
                )
            )

        self.tt_ssm = TtMambaSSM(self.args, self.device, configs, load_fn)

        self.compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def forward(self, x):
        assert len(x.shape) == 4, "Mamba block expects inputs to be rank 4"

        residual_connection = x  # b, e=d_model

        x_ssm = ttnn.linear(
            x,
            self.ssm_in_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            dtype=self.configs["dtype"]["activations"],
        )

        # shift the states leftward
        ttnn.deallocate(self.conv_states[0])
        for i in range(3):
            self.conv_states[i] = self.conv_states[i + 1]

        # update the last state and move it back to DRAM with all the other states
        self.conv_states[3] = ttnn.to_memory_config(x_ssm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_ssm)

        # do the convolution
        conv1d_wt = ttnn.to_memory_config(self.conv1d_weights[0], memory_config=self.configs["sharded_d"])
        conv_state = ttnn.to_memory_config(self.conv_states[0], memory_config=self.configs["sharded_d"])
        conv_accumulator = ttnn.mul(
            conv_state, conv1d_wt, memory_config=self.configs["sharded_d"], dtype=self.configs["dtype"]["activations"]
        )
        ttnn.deallocate(conv1d_wt)
        ttnn.deallocate(conv_state)

        for i in range(1, 4):
            conv1d_wt = ttnn.to_memory_config(self.conv1d_weights[i], memory_config=self.configs["sharded_d"])
            conv_state = ttnn.to_memory_config(self.conv_states[i], memory_config=self.configs["sharded_d"])
            prod = ttnn.mul(
                conv_state,
                conv1d_wt,
                memory_config=self.configs["sharded_d"],
                dtype=self.configs["dtype"]["activations"],
            )
            ttnn.deallocate(conv1d_wt)
            ttnn.deallocate(conv_state)

            conv_out = ttnn.add(
                conv_accumulator,
                prod,
                memory_config=self.configs["sharded_d"],
                dtype=self.configs["dtype"]["activations"],
            )
            ttnn.deallocate(conv_accumulator)
            ttnn.deallocate(prod)
            conv_accumulator = conv_out

        conv1d_bias = ttnn.to_memory_config(self.conv1d_bias, memory_config=self.configs["sharded_d"])
        conv_out_with_bias = ttnn.add(
            conv_out, conv1d_bias, memory_config=self.configs["sharded_d"], dtype=self.configs["dtype"]["activations"]
        )
        ttnn.deallocate(conv_out)
        ttnn.deallocate(conv1d_bias)

        conv_out_with_bias_l1 = ttnn.to_memory_config(conv_out_with_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        conv_out_after_silu = ttnn.silu(conv_out_with_bias_l1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(conv_out_with_bias_l1)

        out = self.tt_ssm(conv_out_after_silu)

        residual = ttnn.linear(
            residual_connection,
            self.mlp_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            dtype=self.configs["dtype"]["activations"],
            activation="silu",
        )
        ttnn.deallocate(residual_connection)

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
            use_1d_systolic_array=True,
            dtype=self.configs["dtype"]["activations"],
        )
        ttnn.deallocate(out)

        return out_proj
