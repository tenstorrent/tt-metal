# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import tt_lib as ttl
from typing import Callable

from models.demos.mamba.reference.args import ModelArgs
from models.demos.mamba.tt.mamba_one_step_ssm import TtMambaSSM
from models.demos.mamba.reference.args import ModelMode


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
            lambda x: x.repeat(self.configs["outer_dim"], 1),
            postfix=f"{self.configs['outer_dim']}",
        )
        self.conv1d_bias_decode = load_fn(
            conv1d_bias_name,
            lambda x: x.repeat(self.configs["num_users"], 1),
            postfix=f"{self.configs['num_users']}",
        )

        if self.configs["mode"] == ModelMode.DECODE:
            self.conv_states = []
            for i in range(4):
                self.conv_states.append(
                    load_fn(
                        f"conv_state{i}",
                        torch_tensor=torch.zeros(1, 1, self.batch_size, self.args.d_inner),
                        postfix=f"{args.batch_size}",
                    )
                )
        elif self.configs["mode"] == ModelMode.PREFILL:
            self.conv_states = [torch.zeros(1, 1, self.configs["num_users"], 5120) for i in range(4)]

        self.use_torch_conv = True
        if self.use_torch_conv:
            self.torch_depthwise_conv1d = torch.nn.Conv1d(
                in_channels=self.args.d_inner,
                out_channels=self.args.d_inner,
                kernel_size=4,
                padding=0,
                groups=self.args.d_inner,
                bias=True,
            )
            self.torch_depthwise_conv1d.weight.data = load_fn(conv1d_weight_name, return_as_torch=True)
            self.torch_depthwise_conv1d.bias.data = load_fn(conv1d_bias_name, return_as_torch=True)

        self.tt_ssm = TtMambaSSM(self.args, self.device, configs, load_fn)

        self.compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def to_decode(self, decode_config):
        self.configs = decode_config
        self.tt_ssm.to_decode(self.configs)
        # deallocate prefill conv_bias, need to reinitialize when back in prefill mode
        ttnn.deallocate(self.conv1d_bias)
        self.conv1d_bias = self.conv1d_bias_decode
        for i in range(4):
            # self.conv_states[i] = torch.cat(self.conv_states[i], dim=2)
            self.conv_states[i] = ttnn.from_torch(
                self.conv_states[i],
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.configs["dtype"]["activations"],
            )

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
            if self.use_torch_conv:
                x_ssm_torch = ttnn.to_torch(x_ssm).to(torch.float32)
                ttnn.deallocate(x_ssm)

                x_ssm_torch = torch.concat(
                    [
                        self.conv_states[1][:, :, self.configs["current_user"]],
                        self.conv_states[2][:, :, self.configs["current_user"]],
                        self.conv_states[3][:, :, self.configs["current_user"]],
                        x_ssm_torch.squeeze(0),
                    ],
                    dim=-2,
                )  # (1, 1, L, E)

                for i in range(0, 4):
                    self.conv_states[i][:, :, self.configs["current_user"]] = x_ssm_torch[:, -(4 - i)]
                # self.conv_states[3][:, :, self.configs["current_user"]] = x_ssm_torch[:, :, 0]

                x_ssm_torch = x_ssm_torch.permute(0, 2, 1)
                conv_out_with_bias = self.torch_depthwise_conv1d(x_ssm_torch)

                x_ssm_torch.data = torch.tensor([])
                # omit the padding at the end
                # conv_out_with_bias = conv_out_with_bias[:, :, :-3]
                conv_out_with_bias = conv_out_with_bias.squeeze(0).permute(1, 0).unsqueeze(0).unsqueeze(0)
                conv_out_with_bias = ttnn.from_torch(
                    conv_out_with_bias,
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=self.configs["dtype"]["activations"],
                )

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
