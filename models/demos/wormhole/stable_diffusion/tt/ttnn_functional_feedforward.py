# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_geglu import geglu
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    determine_largest_subblock_size,
)
import torch
import math


def compare(tensor, name, reshape=False):
    from models.utility_functions import comp_pcc

    tensor = ttnn.from_device(tensor)
    tensor = ttnn.to_torch(tensor)

    golden = torch.load(name)
    if reshape:
        golden = golden.reshape(tensor.shape)

    while len(tensor.shape) > len(golden.shape):
        golden = golden.unsqueeze(0)
    while len(golden.shape) > len(tensor.shape):
        tensor = tensor.unsqueeze(0)

    passed, message = comp_pcc(tensor, golden, 0.95)
    print(f"Maches on {name}: {passed} with message {message}, tensor shape: {tensor.shape}")


class feedforward:
    def __init__(self, device, parameters):
        self.device = device
        self.parameters = parameters
        self.parameters.net[2].weight = ttnn.unsqueeze_to_4D(self.parameters.net[2].weight)
        self.parameters.net[2].bias = ttnn.unsqueeze_to_4D(self.parameters.net[2].bias)
        self.geglu = geglu(device, parameters.net[0])
        self.block_sharded_memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
        self.l1_interleaved_memory_config = ttnn.L1_MEMORY_CONFIG
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.grid_sizes = {8192: (5, 8), 2048: (5, 8), 512: (8, 8), 128: (8, 4)}
        self.out_subblock_hs = {8192: 8, 2048: 8, 512: 2, 128: 1}

    def __call__(self, config, hidden_states):
        hidden_states = self.geglu(config, hidden_states)

        interleaved_output = False
        size = hidden_states.shape[-2]
        grid_size = self.grid_sizes[size]
        M, K, N = hidden_states.shape[-2], hidden_states.shape[-1], self.parameters.net[2].weight.shape[-1]
        in0_block_h = M // grid_size[1] // 32
        in0_block_w = K // grid_size[0] // 32
        out_block_h = math.ceil(M / grid_size[1] / 32)
        out_block_w = math.ceil(N / grid_size[0] / 32)
        out_subblock_h, out_subblock_w = determine_largest_subblock_size(out_block_h, out_block_w)
        # TODO: https://github.com/tenstorrent/tt-metal/issues/7560
        if size == 512:
            out_subblock_h = 1
            out_subblock_w = 1
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            transpose_mcast=False,
            fused_activation=None,
        )
        if hidden_states.shape[-2] == 8192:
            hidden_states = ttnn.reallocate(hidden_states)
        hidden_states = ttnn.linear(
            hidden_states,
            self.parameters.net[2].weight,
            bias=self.parameters.net[2].bias,
            program_config=program_config,
            memory_config=self.l1_interleaved_memory_config if interleaved_output else self.block_sharded_memory_config,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_config,
        )
        if interleaved_output:
            hidden_states = ttnn.interleaved_to_sharded(
                hidden_states,
                grid_size,
                [hidden_states.shape[-2] // grid_size[1], hidden_states.shape[-1] // grid_size[0]],
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
        return hidden_states
