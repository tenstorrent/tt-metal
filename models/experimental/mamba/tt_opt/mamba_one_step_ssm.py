# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

import ttnn
from typing import Callable

from models.utility_functions import torch2tt_tensor
from models.helper_funcs import Linear
from models.experimental.mamba.reference.args import ModelArgs


class TtMambaSSM(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args

        # hidden state
        self.num_users = args.batch_size
        self.hidden_size = args.d_inner
        self.configs = configs
        self.n = 32
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
        self.delta_t_proj_weights = load_fn(x_proj_weight_name, lambda x: x[: self.args.dt_rank, :].transpose(-1, -2), postfix="delta_t")

        # B_proj_weights
        def preprocess_B(x):
            x = x[self.args.dt_rank : (self.args.dt_rank + self.args.d_state), :]
            x = x.transpose(-1, -2)
            x = F.pad(x, (0, 16), "constant", 0)
            return x

        self.B_proj_weights = load_fn(
            x_proj_weight_name,
            tm_fn=preprocess_B, postfix="B_proj",
        )

        # C_proj_weights
        def preprocess_C(x):
            x = x[(self.args.dt_rank + self.args.d_state) :, :].transpose(-1, -2)
            x = F.pad(x, (0, 16), "constant", 0)
            return x
        self.C_proj_weights = load_fn(
            x_proj_weight_name, preprocess_C, postfix="C_proj"
        )

        # dt_proj_weights
        dt_proj_weight_name = "mixer.dt_proj.weight"
        dt_proj_bias_name = "mixer.dt_proj.bias"
        self.dt_proj_weights = load_fn(dt_proj_weight_name, lambda x: x.transpose(-1, -2))
        self.dt_proj_bias = load_fn(dt_proj_bias_name)

        # B_intermediate_tranform_weights = torch.eye(self.n).repeat(1, self.hidden_size).unsqueeze(0).unsqueeze(0)

        # A weight
        A_weight_name = "mixer.A_log"
        def preprocess_A(x):
            x = -torch.exp(x.float())
            # padding with inf
            x = F.pad(x, (0, 16), "constant", float("-inf"))
            x = x.reshape(1, self.hidden_size*32)  # (1, 2en)
            return x.repeat(self.num_users, 1) # b, 2en

        self.A = load_fn(A_weight_name, tm_fn=preprocess_A, postfix=f"A_{self.args.batch_size}")

        # D weight
        D_weight_name = "mixer.D"
        self.D = load_fn(
            D_weight_name,
            lambda x: x.repeat(self.args.batch_size, 1),
            postfix=f"D_{self.args.batch_size}",
        )

        # hidden state
        prev_hidden_states = torch.zeros((1, 1, self.num_users, self.hidden_size*self.n))
        self.tt_hidden_state = load_fn(f"tt_hidden_state_{args.batch_size}", torch_tensor=prev_hidden_states)


    def forward(self, x):
        # delta
        delta_t_proj_weights = ttnn.to_memory_config(self.delta_t_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        delta_t0 = ttnn.linear(x, delta_t_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(delta_t_proj_weights)

        dt_proj_weights = ttnn.to_memory_config(self.dt_proj_weights, memory_config=self.configs["sharded_rank"])
        delta_t1 = ttnn.linear(
            delta_t0, self.dt_proj_weights, bias=self.dt_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(delta_t0)
        ttnn.deallocate(dt_proj_weights)

        delta_t2 = ttnn.softplus(delta_t1, parameter1=1.0, parameter2=20.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(delta_t1)

        # calculate abar
        delta_t3 = ttnn.repeat_interleave(delta_t2, self.n, dim=3)
        ttnn.deallocate(delta_t2)

        delta_t4 = ttnn.to_memory_config(delta_t3, memory_config=self.configs["sharded_large"])
        abar0 = ttnn.to_memory_config(self.A, memory_config=self.configs["sharded_large"])
        abar1 = ttnn.mul(delta_t4, abar0, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(abar0)
        ttnn.deallocate(delta_t4)

        abar2 = ttnn.to_memory_config(abar1, memory_config=ttnn.L1_MEMORY_CONFIG)
        abar3 = ttnn.exp(abar2, memory_config=ttnn.L1_MEMORY_CONFIG)
        abar4 = ttnn.to_memory_config(abar3, memory_config=self.configs["sharded_large"])

        # multiply abar and hidden_state
        hidden_state0 = ttnn.to_memory_config(self.tt_hidden_state, memory_config=self.configs["sharded_large"])
        amulh0 = ttnn.mul(abar4, hidden_state0, memory_config=self.configs["sharded_large"])

        # deallocate abar and hidden_state
        # ttnn.deallocate(abar1) # THIS CAUSES A CRASH
        ttnn.deallocate(abar2)
        # ttnn.deallocate(abar3) # THIS CAUSES A CRASH
        ttnn.deallocate(abar4)
        ttnn.deallocate(hidden_state0)

        # B
        B_proj_weights = ttnn.to_memory_config(self.B_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        B0 = ttnn.linear(x, B_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(B_proj_weights)
        B1 = ttnn.repeat(B0, ttnn.Shape([1, 1, 1, self.hidden_size], [1, 1, 32, self.hidden_size]))
        ttnn.deallocate(B0)
        B2 = ttnn.to_memory_config(B1, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(B1)

        # bbar
        delta_t4 = ttnn.to_memory_config(delta_t3, memory_config=self.configs["sharded_large"])
        delta_t3.deallocate()

        bbar0 = ttnn.mul(delta_t4, B2, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(delta_t4)
        ttnn.deallocate(B2)

        # multiply bbar and x
        x0 = ttnn.repeat_interleave(x, self.n, dim=3)
        x1 = ttnn.to_memory_config(x0, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(x0)
        bmulx0 = ttnn.mul(bbar0, x1, memory_config=self.configs["sharded_large"])

        # deallocate bbar
        ttnn.deallocate(bbar0)
        ttnn.deallocate(x1)

        # add amulh and bmulx
        hidden_state1 = ttnn.add(amulh0, bmulx0, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(self.tt_hidden_state)
        self.tt_hidden_state = ttnn.to_memory_config(hidden_state1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # deallocate amulh and bmulx
        ttnn.deallocate(amulh0)
        ttnn.deallocate(bmulx0)

        # compute C
        C_proj = ttnn.to_memory_config(self.C_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        C0 = ttnn.linear(x, C_proj, memory_config=ttnn.L1_MEMORY_CONFIG)  # b,n
        ttnn.deallocate(C_proj)
        C1 = ttnn.permute(C0, (0, 2, 3, 1))  # b,n,1
        ttnn.deallocate(C0)

        # hidden state @ C
        hidden_state1 = ttnn.to_memory_config(hidden_state1, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_state1 = ttnn.reshape(hidden_state1, (1, self.num_users, self.hidden_size, self.n))  # b, d, 32

        C2 = ttnn.matmul(hidden_state1, C1, memory_config=ttnn.L1_MEMORY_CONFIG)  # b, d, 1
        ttnn.deallocate(C1)
        ttnn.deallocate(hidden_state1)

        C3 = ttnn.permute(C2, (0, 3, 1, 2))  # b, d
        ttnn.deallocate(C2)

        # x * D
        xD = ttnn.mul(x, self.D, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x)

        # add xD and x
        output = ttnn.add(xD, C3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(xD)
        ttnn.deallocate(C3)

        return output
