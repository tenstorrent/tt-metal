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
        self.num_users = 32
        self.hidden_size = args.d_inner
        self.configs = configs
        self.n = 32
        self.rank = self.args.dt_rank
        
        self.row = 4
        self.col = 8

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
            tm_fn=preprocess_B, postfix="B_proj"
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
        print('****dt_proj_weights', self.dt_proj_weights.shape, self.dt_proj_bias.shape)

        # B_intermediate_tranform_weights = torch.eye(self.n).repeat(1, self.hidden_size).unsqueeze(0).unsqueeze(0)

        # A weight
        A_weight_name = "mixer.A_log"
        def preprocess_A(x):
            x = -torch.exp(x.float())
            # padding with inf
            x = F.pad(x, (0, 16), "constant", float("-inf"))
            x = x.reshape(1, self.hidden_size*32)  # (1, 2en)
            return x.repeat(self.num_users, 1).unsqueeze(0).unsqueeze(0) # b, 2en

        self.A = load_fn(A_weight_name, tm_fn=preprocess_A, postfix=f"A_{self.num_users}")

        # D weight
        D_weight_name = "mixer.D"
        self.D = load_fn(
            D_weight_name,
            lambda x: x.repeat(self.num_users, 1).unsqueeze(0).unsqueeze(0),
            postfix=f"D_{self.num_users}",
        )

        # hidden state
        prev_hidden_states = torch.zeros((1, 1, self.num_users, self.hidden_size*self.n))
        self.tt_hidden_state = load_fn(f"tt_hidden_state_{self.num_users}", torch_tensor=prev_hidden_states)


    def forward(self, x):
        print("**********ssm block", x.shape)

            

        def post_hook_to_print_output(operation, args, kwargs, output):
            print (operation.name)
            for arg in args:
                if type(arg) == ttnn.Tensor:
                    print (arg)
                    ttnn.deallocate(arg)
            for arg in kwargs:
                if type(arg) == ttnn.Tensor:
                    print (arg)
                    ttnn.deallocate(arg)
                
        # delta
                
        delta_t = ttnn.linear(x, self.delta_t_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        delta_t_old = delta_t
        delta_t = ttnn.linear(delta_t, self.dt_proj_weights, bias=self.dt_proj_bias, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=self.row, x=self.col))
        ttnn.deallocate(delta_t_old)
        delta_t_old = delta_t
        delta_t = ttnn.softplus(delta_t, parameter1=1.0, parameter2=20.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(delta_t_old)
        delta_t_old = delta_t
        delta_t = ttnn.repeat_interleave(delta_t, self.n, dim=3)
        ttnn.deallocate(delta_t_old)
        delta_t_old = delta_t

            
        # shard delta and A
        delta_t = ttnn.to_memory_config(delta_t, memory_config=self.configs["sharded_dn"])
        A = ttnn.to_memory_config(self.A, memory_config=self.configs["sharded_dn"])
        
        #abar
        abar = ttnn.mul(delta_t, A, memory_config=self.configs['sharded_dn'])
        ttnn.deallocate(A)
        ttnn.deallocate(delta_t)
        abar_old = abar
        abar = ttnn.exp(abar, memory_config=self.configs['sharded_dn'])
        ttnn.deallocate(abar_old)
            
        # multiply abar and hidden_state
        hidden_state = ttnn.to_memory_config(self.tt_hidden_state, memory_config=self.configs["sharded_dn"])        
        amulh = ttnn.mul(abar, hidden_state, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(hidden_state)
        ttnn.deallocate(abar)

        # B
        B = ttnn.linear(x, self.B_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        B_old = B
        B = ttnn.repeat(B, ttnn.Shape([1, 1, 1, self.hidden_size], [1, 1, 32, self.hidden_size]))
        ttnn.deallocate(B_old)
            
        # shard B
        B_old = B
        B = ttnn.to_memory_config(B, memory_config=self.configs['sharded_dn'])
        ttnn.deallocate(B_old)
        
        # bbar
        delta_t = ttnn.to_memory_config(delta_t_old, memory_config=self.configs['sharded_dn'])
        ttnn.deallocate(delta_t_old)
        bbar = ttnn.mul(delta_t, B, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(delta_t)
        ttnn.deallocate(B)
            
        # multiply bbar and x
        x_bcast = ttnn.repeat_interleave(x, self.n, dim=3)
        x_bcast_old = x_bcast
        x_bcast = ttnn.to_memory_config(x_bcast, memory_config=self.configs['sharded_dn'])
        ttnn.deallocate(x_bcast_old)
        bmulx = ttnn.mul(bbar, x_bcast, memory_config=self.configs["sharded_dn"])
        ttnn.deallocate(bbar)
        ttnn.deallocate(x_bcast)
        
        # add amulh and bmulx
        hidden_state = ttnn.add(amulh, bmulx, memory_config=self.configs["sharded_large"])
        ttnn.deallocate(amulh)
        ttnn.deallocate(bmulx)
        ttnn.deallocate(self.tt_hidden_state)
        self.tt_hidden_state = ttnn.to_memory_config(hidden_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden_state)
    
        
        return x

        # compute C
        #C_proj = ttnn.to_memory_config(self.C_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
        C0 = ttnn.linear(x, self.C_proj_weights, memory_config=ttnn.L1_MEMORY_CONFIG)  # b,n
        #ttnn.deallocate(C_proj)
        C1 = ttnn.permute(C0, (0, 2, 3, 1))  # b,n,1
        ttnn.deallocate(C0)

        # hidden state @ C
        #hidden_state1 = ttnn.to_memory_config(hidden_state1, memory_config=ttnn.L1_MEMORY_CONFIG)
        #ttnn.deallocate(hidden_state1)
        hidden_state3 = ttnn.to_torch(hidden_state1)
        ttnn.deallocate(hidden_state1)
        #hidden_state3 = ttnn.reshape(hidden_state2, (1, self.num_users, self.hidden_size, self.n))  # b, d, 32
        hidden_state3 = hidden_state3.reshape(1, self.num_users, self.hidden_size, self.n)  # b, d, 32
        hidden_state3 = ttnn.from_torch(hidden_state3, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        C2 = ttnn.matmul(hidden_state3, C1, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=self.row, x=self.col))  # b, d, 1
        ttnn.deallocate(C1)
        C3 = ttnn.permute(C2, (0, 3, 1, 2)) # b, d
        ttnn.deallocate(C2)

        # x * D
        xD = ttnn.mul(x, self.D, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x)

        # add xD and x
        output = ttnn.add(xD, C3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(xD)
        ttnn.deallocate(C3)

        return output
