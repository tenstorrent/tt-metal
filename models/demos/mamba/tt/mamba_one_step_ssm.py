# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
import tt_lib as ttl
from typing import Callable

from models.demos.mamba.reference.args import ModelArgs
from models.demos.mamba.reference.args import ModelMode


class TtMambaSSM(torch.nn.Module):
    def __init__(self, args: ModelArgs, device, configs, load_fn: Callable):
        super().__init__()

        self.device = device
        self.args = args

        # hidden state
        self.batch_size = args.batch_size
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
        self.delta_t_proj_weights = load_fn(
            x_proj_weight_name,
            lambda x: x[: self.args.dt_rank, :].transpose(-1, -2),
            postfix="delta_t",
            tt_dtype=self.configs["dtype"]["weights"],
        )

        # B_proj_weights
        def preprocess_B(x):
            x = x[self.args.dt_rank : (self.args.dt_rank + self.args.d_state), :]
            x = x.transpose(-1, -2)
            x = torch.nn.functional.pad(x, (0, 16), "constant", 0)
            return x

        self.B_proj_weights = load_fn(
            x_proj_weight_name,
            tm_fn=preprocess_B,
            postfix="B_proj",
            tt_dtype=self.configs["dtype"]["weights"],
        )

        # C_proj_weights
        def preprocess_C(x):
            x = x[(self.args.dt_rank + self.args.d_state) :, :].transpose(-1, -2)
            x = torch.nn.functional.pad(x, (0, 16), "constant", 0)
            return x

        self.C_proj_weights = load_fn(
            x_proj_weight_name, preprocess_C, postfix="C_proj", tt_dtype=self.configs["dtype"]["weights"]
        )

        # dt_proj_weights
        dt_proj_weight_name = "mixer.dt_proj.weight"
        dt_proj_bias_name = "mixer.dt_proj.bias"
        self.dt_proj_weights = load_fn(
            dt_proj_weight_name, lambda x: x.transpose(-1, -2), tt_dtype=self.configs["dtype"]["weights"]
        )
        self.dt_proj_bias = load_fn(dt_proj_bias_name, tt_dtype=self.configs["dtype"]["weights"])

        A_weight_name = "mixer.A_log"

        def preprocess_A(x):
            x = -torch.exp(x.float())  # (2E, N) where N=16
            x = torch.nn.functional.pad(x, (0, 16), "constant", float("-inf"))  # (2E, N) where N=32
            x = x.reshape(1, x.shape[0] * x.shape[1])  # (1, 2EN)
            return x.repeat(self.configs["outer_dim"], 1)  # (B, 2EN)

        def preprocess_A_decode(x):
            x = -torch.exp(x.float())  # (2E, N) where N=16
            x = torch.nn.functional.pad(x, (0, 16), "constant", float("-inf"))  # (2E, N) where N=32
            x = x.reshape(1, x.shape[0] * x.shape[1])  # (1, 2EN)
            return x.repeat(self.configs["num_users"], 1)  # (B, 2EN)

        self.A = load_fn(A_weight_name, tm_fn=preprocess_A, postfix=f"A_{self.configs['outer_dim']}")

        self.A_decode = load_fn(A_weight_name, tm_fn=preprocess_A_decode, postfix=f"A_{self.configs['num_users']}")

        # D weight
        D_weight_name = "mixer.D"
        self.D = load_fn(
            D_weight_name,
            lambda x: x.repeat(self.configs["outer_dim"], 1),
            postfix=f"D_{self.configs['outer_dim']}",
        )

        self.D_decode = load_fn(
            D_weight_name,
            lambda x: x.repeat(self.configs["num_users"], 1),
            postfix=f"D_{self.configs['num_users']}",
        )

        # hidden state

        if self.configs["mode"] == ModelMode.DECODE:
            prev_hidden_states = torch.zeros((1, 1, self.batch_size, self.hidden_size * self.n))
            self.tt_hidden_states = load_fn(
                f"tt_hidden_state_{self.batch_size}", torch_tensor=prev_hidden_states, tt_layout=ttnn.TILE_LAYOUT
            )
        else:
            prev_hidden_states = torch.zeros((1, 1, 1, self.hidden_size * self.n))
            self.prev_hidden_state = load_fn(
                f"tt_hidden_state_{1}", torch_tensor=prev_hidden_states, tt_layout=ttnn.ROW_MAJOR_LAYOUT
            )
            self.tt_hidden_states = []

        self.compute_kernel_config = ttl.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttl.tensor.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.eltwise_math_fidelity = ttl.tensor.MathFidelity.HiFi2
        self.core_grid_row = self.configs["core_grid_row"]
        self.core_grid_col = self.configs["core_grid_col"]

    def to_decode(self, decode_config):
        self.configs = decode_config
        # deallocate prefill A and D, need to reinitialize when back in prefill mode
        ttnn.deallocate(self.A)
        ttnn.deallocate(self.D)
        self.A = self.A_decode
        self.D = self.D_decode
        self.tt_hidden_states = torch.cat(self.tt_hidden_states, dim=2)
        self.tt_hidden_states = ttnn.from_torch(
            self.tt_hidden_states,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.configs["dtype"]["activations"],
        )

    def forward(self, x):
        assert len(x.shape) == 4, "SSM block expects inputs to be rank 4"

        # delta
        delta_t0 = ttnn.linear(
            x,
            self.delta_t_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
            dtype=self.configs["dtype"]["activations"],
        )

        delta_t1 = ttnn.linear(
            delta_t0,
            self.dt_proj_weights,
            bias=self.dt_proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
            dtype=self.configs["dtype"]["activations"],
        )
        ttnn.deallocate(delta_t0)

        delta_t2 = ttnn.softplus(
            delta_t1,
            beta=1.0,
            threshold=20.0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(delta_t1)

        # calculate abar
        abar0 = ttnn.to_memory_config(self.A, memory_config=ttnn.L1_MEMORY_CONFIG)
        abar1 = ttnn.experimental.operations.primary.transformers.ssm_eltwise_mul(
            abar0,
            delta_t2,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
            math_fidelity=self.eltwise_math_fidelity,
        )
        ttnn.deallocate(abar0)

        abar2 = ttnn.exp(
            abar1,
            fast_and_approximate_mode=True,
            memory_config=ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        )
        ttnn.deallocate(abar1)

        # B
        B0 = ttnn.linear(
            x,
            self.B_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
            dtype=self.configs["dtype"]["activations"],
        )

        # bbar
        bbar0 = ttnn.experimental.operations.primary.transformers.ssm_eltwise_mul(
            B0,
            delta_t2,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
            math_fidelity=self.eltwise_math_fidelity,
        )
        ttnn.deallocate(delta_t2)
        ttnn.deallocate(B0)

        # bbar * x
        bmulx0 = ttnn.experimental.operations.primary.transformers.ssm_eltwise_mul(
            bbar0,
            x,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
            math_fidelity=self.eltwise_math_fidelity,
        )

        # deallocate bbar
        ttnn.deallocate(bbar0)

        if self.configs["mode"] == ModelMode.DECODE:
            # multiply abar and hidden_state
            hidden_state0 = ttnn.multiply(
                abar2,
                self.tt_hidden_states,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=self.configs["dtype"]["activations"],
                output_tensor=abar2,
            )

            # add amulh and bmulx
            hidden_state0 = ttnn.add(
                hidden_state0,
                bmulx0,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=self.configs["dtype"]["activations"],
                output_tensor=hidden_state0,
            )
            ttnn.deallocate(bmulx0)

            ttnn.deallocate(self.tt_hidden_states)
            self.tt_hidden_states = ttnn.to_memory_config(
                hidden_state0, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=self.configs["dtype"]["activations"]
            )

        elif self.configs["mode"] == ModelMode.PREFILL:
            prev_hidden_state = ttnn.to_memory_config(
                self.prev_hidden_state, memory_config=self.configs["sharded_prev_hidden"]
            )
            abar2_sharded = ttnn.to_memory_config(abar2, self.configs["sharded_scan"])
            ttnn.deallocate(abar2)
            bmulx0_sharded = ttnn.to_memory_config(bmulx0, self.configs["sharded_scan"])
            ttnn.deallocate(bmulx0)
            hidden_states_sharded = ttl.operations.primary.transformers.ssm_prefix_scan(
                abar2_sharded,
                bmulx0_sharded,
                prev_hidden_state,
                output_mem_config=self.configs["sharded_scan"],
                output_dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(abar2_sharded)
            ttnn.deallocate(bmulx0_sharded)

            # TODO Need to update when chunkimg is supported, however, this also need to be reset to zeros for next user
            # self.prev_hidden_state = ttnn.to_memory_config(prev_hidden_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            self.tt_hidden_states.append(ttnn.to_torch(prev_hidden_state))
            ttnn.deallocate(prev_hidden_state)

            hidden_state0 = ttnn.to_memory_config(hidden_states_sharded, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(hidden_states_sharded)

        # compute C
        C0 = ttnn.linear(
            x,
            self.C_proj_weights,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=self.core_grid_row, x=self.core_grid_col),
            dtype=self.configs["dtype"]["activations"],
        )  # b,n

        # c * hidden_state
        C1 = ttnn.experimental.operations.primary.transformers.ssm_eltwise_mul(
            C0,
            hidden_state0,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
            math_fidelity=self.eltwise_math_fidelity,
        )
        ttnn.deallocate(hidden_state0)
        ttnn.deallocate(C0)

        # Reduction matmul
        C2 = ttnn.experimental.operations.primary.transformers.ssm_1d_sum_reduce(
            C1,
            output_mem_config=ttl.tensor.MemoryConfig(
                ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1
            ),
            output_dtype=self.configs["dtype"]["activations"],
            math_fidelity=self.eltwise_math_fidelity,
        )
        ttnn.deallocate(C1)

        # x * D
        x = ttnn.multiply(
            x, self.D, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=self.configs["dtype"]["activations"], output_tensor=x
        )

        x = ttnn.add(
            x, C2, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=self.configs["dtype"]["activations"], output_tensor=x
        )
        ttnn.deallocate(C2)

        return x
