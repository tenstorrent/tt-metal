# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import tt_lib as ttl
from typing import Callable

from models.demos.wormhole.mamba.reference.args import ModelArgs
from models.demos.wormhole.mamba.tt.transforms import MambaSsmBlockTransformer


class TtMambaSSM(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable, transformer: MambaSsmBlockTransformer):
        super().__init__()

        self.transformer = transformer

        self.device = device
        self.args = args

        # hidden state
        self.batch_size = args.batch_size
        self.hidden_size = args.d_inner
        self.configs = configs
        self.n = 16
        self.rank = self.args.dt_rank

        """
        We need to split up the x_proj weights because in the reference
        implementation they perform the linear operation for dt, B, and C in a
        single step. Here we can't do that because it would involve fallback op
        slicing, so we break up the weights ahead of time and do the linear ops
        separately.
        """

        x_proj_weight_name = "mixer.x_proj.weight"

        # delta_t_proj_weights
        self.delta_t_proj_weights = load_fn(
            x_proj_weight_name,
            lambda x: x[: self.args.dt_rank, :].transpose(-1, -2),
            postfix="delta_t",
            tt_dtype=ttnn.bfloat8_b,
        )

        # B_proj_weights
        def preprocess_B(x):
            x = x[self.args.dt_rank : (self.args.dt_rank + self.args.d_state), :]
            x = x.transpose(-1, -2)
            # x = F.pad(x, (0, 16), "constant", 0)
            return x

        self.B_proj_weights = load_fn(
            x_proj_weight_name,
            tm_fn=preprocess_B,
            postfix="B_proj",
            tt_dtype=ttnn.bfloat8_b,
        )

        # C_proj_weights
        def preprocess_C(x):
            x = x[(self.args.dt_rank + self.args.d_state) :, :].transpose(-1, -2)
            # x = F.pad(x, (0, 16), "constant", 0)
            return x

        self.C_proj_weights = load_fn(x_proj_weight_name, preprocess_C, postfix="C_proj", tt_dtype=ttnn.bfloat8_b)

        # dt_proj_weights
        dt_proj_weight_name = "mixer.dt_proj.weight"
        dt_proj_bias_name = "mixer.dt_proj.bias"
        self.dt_proj_weights = load_fn(dt_proj_weight_name, lambda x: x.transpose(-1, -2), tt_dtype=ttnn.bfloat8_b)
        self.dt_proj_bias = load_fn(dt_proj_bias_name, tt_dtype=ttnn.bfloat8_b)

        # B_intermediate_tranform_weights = torch.eye(self.n).repeat(1, self.hidden_size).unsqueeze(0).unsqueeze(0)

        # A weight
        A_weight_name = "mixer.A_log"

        def preprocess_A(x):
            x = -torch.exp(x.float())
            # padding with inf
            # x = F.pad(x, (0, 16), "constant", float("-inf"))
            x = x.reshape(1, self.hidden_size * self.n)  # (1, 2en)
            return x.repeat(self.batch_size, 1)  # b, 2en

        self.A = load_fn(A_weight_name, tm_fn=preprocess_A, postfix=f"A_{self.args.batch_size}")

        # D weight
        D_weight_name = "mixer.D"
        self.D = load_fn(
            D_weight_name,
            lambda x: x.repeat(self.args.batch_size, 1),
            postfix=f"D_{self.args.batch_size}",
        )

        # hidden state
        prev_hidden_states = torch.zeros((1, 1, self.batch_size, self.hidden_size * self.n))
        self.tt_hidden_state = load_fn(f"tt_hidden_state_{args.batch_size}", torch_tensor=prev_hidden_states)

        self.compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.compute_kernel_config_mask = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.core_grid_row = 5
        self.core_grid_col = 8

    def forward(self, x):
        assert len(x.shape) == 4, "SSM block expects inputs to be rank 4"

        # delta
        delta_t0 = ttnn.linear(
            x,
            self.delta_t_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
        )

        delta_t1 = ttnn.linear(
            delta_t0,
            self.dt_proj_weights,
            bias=self.dt_proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
        )
        ttnn.deallocate(delta_t0)

        delta_t2 = ttnn.softplus(delta_t1, parameter1=1.0, parameter2=20.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(delta_t1)

        # calculate abar
        delta_t3 = self.transformer.repeat_interleave(
            delta_t2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_mask,
            core_grid=ttnn.CoreGrid(y=7, x=8),
        )  # b,n

        ttnn.deallocate(delta_t2)

        # shard delta and A
        delta_t4 = ttnn.to_memory_config(delta_t3, memory_config=self.configs["sharded_dn"])
        abar0 = ttnn.to_memory_config(self.A, memory_config=self.configs["sharded_dn"])

        abar1 = ttnn.mul(delta_t4, abar0, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(abar0)
        ttnn.deallocate(delta_t4)

        abar2 = ttnn.to_memory_config(abar1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(abar1)
        abar3 = ttnn.exp(abar2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(abar2)

        abar4 = ttnn.to_memory_config(abar3, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(abar3)

        # multiply abar and hidden_state
        hidden_state0 = ttnn.to_memory_config(self.tt_hidden_state, memory_config=self.configs["sharded_dn"])
        amulh0 = ttnn.mul(abar4, hidden_state0, memory_config=self.configs["sharded_dn"])

        # deallocate abar and hidden_state
        ttnn.deallocate(abar4)
        ttnn.deallocate(hidden_state0)

        # B
        B0 = ttnn.linear(
            x,
            self.B_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
        )

        # repeat using mask+matmul instead of ttnn.repeat to avoid fallback
        B1 = self.transformer.repeat(
            B0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_mask,
            core_grid=ttnn.CoreGrid(y=7, x=8),
        )
        ttnn.deallocate(B0)

        # shard B
        B2 = ttnn.to_memory_config(B1, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(B1)

        # shard delta
        delta_t4 = ttnn.to_memory_config(delta_t3, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(delta_t3)

        # bbar
        bbar0 = ttnn.mul(delta_t4, B2, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(delta_t4)
        ttnn.deallocate(B2)

        # multiply bbar and x with mask instead of ttnn.repeat_interleave(x, self.n, dim=3)
        x0 = self.transformer.repeat_interleave(
            x,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_mask,
            core_grid=ttnn.CoreGrid(y=7, x=8),
        )  # b,n

        x1 = ttnn.to_memory_config(x0, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(x0)
        bmulx0 = ttnn.mul(bbar0, x1, memory_config=self.configs["sharded_dn"])

        # deallocate bbar
        ttnn.deallocate(bbar0)
        ttnn.deallocate(x1)

        # add amulh and bmulx
        hidden_state1 = ttnn.add(amulh0, bmulx0, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(self.tt_hidden_state)
        self.tt_hidden_state = ttnn.to_memory_config(hidden_state1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # deallocate amulh and bmulx
        ttnn.deallocate(amulh0)
        ttnn.deallocate(bmulx0)

        # compute C
        C0 = ttnn.linear(
            x,
            self.C_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            use_1d_systolic_array=True,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
        )  # b,n

        # repeat using mask+matmul instead of ttnn.repeat to avoid fallback
        C1 = self.transformer.repeat(
            C0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_mask,
            core_grid=ttnn.CoreGrid(y=7, x=8),
        )
        ttnn.deallocate(C0)

        # shard c
        C2 = ttnn.to_memory_config(C1, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(C1)

        C3 = ttnn.mul(hidden_state1, C2, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(hidden_state1)
        ttnn.deallocate(C2)

        # Reduction matmul
        C3 = ttnn.to_memory_config(C3, memory_config=ttnn.L1_MEMORY_CONFIG)
        C4 = self.transformer.reduce(
            C3,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_mask,
            core_grid=ttnn.CoreGrid(y=7, x=8),
        )  # b,n
        ttnn.deallocate(C3)

        # shard x, C
        x = ttnn.to_memory_config(x, memory_config=self.configs["sharded_d"])
        C5 = ttnn.to_memory_config(C4, memory_config=self.configs["sharded_d"])
        ttnn.deallocate(C4)

        # shard D
        D = ttnn.to_memory_config(self.D, memory_config=self.configs["sharded_d"])

        # x * D
        xD = ttnn.mul(x, D, memory_config=self.configs["sharded_d"])
        ttnn.deallocate(x)

        # add xD and x
        output = ttnn.add(xD, C5, memory_config=self.configs["sharded_d"])
        ttnn.deallocate(xD)
        ttnn.deallocate(C5)

        return output
