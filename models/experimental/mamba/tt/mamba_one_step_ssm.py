# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import tt_lib

from typing import Callable

from models.utility_functions import torch2tt_tensor
from models.helper_funcs import Linear
from models.experimental.mamba.reference.args import ModelArgs


class TtMambaSSM(torch.nn.Module):
    def __init__(self, args: ModelArgs, device: tt_lib.device, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args

        """
        We need to split up the x_proj weights because in the reference
        implementation they perform the linear operation for dt, B, and C in a
        single step. Here we can't do that because it would involve fallback op
        slicing, so we break up the weights ahead of time and do the linear ops
        separately.
        """
        x_proj_weight_name = "mixer.x_proj.weight"
        self.delta_t_proj_weights = load_fn(x_proj_weight_name, lambda x: x[: self.args.dt_rank, :], postfix="delta_t")
        self.delta_t_proj = Linear(self.args.d_inner, self.args.dt_rank, self.delta_t_proj_weights, bias=None)

        self.B_proj_weights = load_fn(
            x_proj_weight_name,
            lambda x: x[self.args.dt_rank : (self.args.dt_rank + self.args.d_state), :],
            postfix="B_proj",
        )

        self.C_proj_weights = load_fn(
            x_proj_weight_name, lambda x: x[(self.args.dt_rank + self.args.d_state) :, :], postfix="C_proj"
        )

        self.B_proj = Linear(self.args.d_inner, self.args.d_state, self.B_proj_weights, bias=None)
        self.C_proj = Linear(self.args.d_inner, self.args.d_state, self.C_proj_weights, bias=None)

        A_weight_name = "mixer.A_log"

        def preprocess_A(x):
            x = -torch.exp(x.float())  # (2E, N)
            return x.repeat(self.args.batch_size, 1, 1).reshape(
                self.args.batch_size, 1, -1, self.args.d_state
            )  # (BS, 1, 2E, N)

        self.A = load_fn(A_weight_name, tm_fn=preprocess_A, postfix=f"A_{self.args.batch_size}")

        D_weight_name = "mixer.D"
        self.D = load_fn(
            D_weight_name,
            lambda x: x.repeat(self.args.batch_size, 1).reshape(self.args.batch_size, 1, -1, self.args.d_inner),
            postfix=f"D_{self.args.batch_size}",
        )
        self.D = tt_lib.tensor.permute(self.D, [0, 2, 3, 1])

        dt_proj_weight_name = "mixer.dt_proj.weight"
        dt_proj_bias_name = "mixer.dt_proj.bias"
        self.dt_proj_weights = load_fn(dt_proj_weight_name)
        self.dt_proj_bias = load_fn(
            dt_proj_bias_name,
        )
        self.dt_proj = Linear(self.args.dt_rank, self.args.d_inner, self.dt_proj_weights, bias=self.dt_proj_bias)

        prev_hidden_states = torch.zeros((args.batch_size, 1, args.d_inner, args.d_state))
        self.tt_hidden_state = load_fn(f"tt_hidden_state_{args.batch_size}", torch_tensor=prev_hidden_states)
        self.B_intermediate = load_fn(
            f"B_intermediate_{args.batch_size}",
            torch_tensor=torch.ones((args.batch_size, 1, args.d_inner, args.d_state)),
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        delta_t = self.delta_t_proj(x)
        delta_t = self.dt_proj(delta_t)
        delta_t = tt_lib.tensor.softplus(
            delta_t,
            output_mem_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
            ),
        )
        delta_t = tt_lib.tensor.permute(delta_t, [0, 2, 3, 1])

        B = self.B_proj(x)
        C = self.C_proj(x)

        B = tt_lib.tensor.transpose(B, 2, 1)
        C = tt_lib.tensor.permute(C, [0, 2, 3, 1])

        delta_A = tt_lib.tensor.bcast(
            self.A, delta_t, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W
        )
        delta_A = tt_lib.tensor.exp(delta_A)

        delta_B = tt_lib.tensor.bcast(
            self.B_intermediate, B, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.H
        )
        B.deallocate()
        delta_B = tt_lib.tensor.bcast(
            delta_B, delta_t, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W
        )

        delta_A_h = tt_lib.tensor.mul(delta_A, self.tt_hidden_state)
        x = tt_lib.tensor.permute(x, [0, 2, 3, 1])
        delta_B_x = tt_lib.tensor.bcast(
            delta_B, x, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W
        )

        self.tt_hidden_state = tt_lib.tensor.add(delta_A_h, delta_B_x)
        self.output = tt_lib.tensor.bmm(
            self.tt_hidden_state,
            C,
            kernel_config=tt_lib.tensor.WormholeComputeKernelConfig(
                math_fidelity=tt_lib.tensor.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
            ),
        )
        C.deallocate()
        x = tt_lib.tensor.mul(self.D, x)
        self.output = tt_lib.tensor.add(self.output, x)
        x.deallocate()

        self.output = tt_lib.tensor.permute(self.output, [0, 3, 1, 2])

        return self.output
