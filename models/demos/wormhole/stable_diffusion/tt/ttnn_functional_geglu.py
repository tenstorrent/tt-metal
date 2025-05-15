# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import determine_blocking


def split_linear_params(params):
    dim = -1
    device = params.proj.weight.device()
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    weight = ttnn.to_torch(params.proj.weight)
    bias = ttnn.to_torch(params.proj.bias)

    proj_weight, gate_weight = torch.split(weight, weight.shape[dim] // 2, dim=dim)
    proj_bias, gate_bias = torch.split(bias, bias.shape[dim] // 2, dim=dim)

    while len(proj_weight.shape) < 4:
        proj_weight = proj_weight.unsqueeze(0)
    proj_weight = ttnn.from_torch(proj_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    proj_weight = ttnn.to_device(proj_weight, device, memory_config=memory_config)

    while len(gate_weight.shape) < 4:
        gate_weight = gate_weight.unsqueeze(0)
    gate_weight = ttnn.from_torch(gate_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    gate_weight = ttnn.to_device(gate_weight, device, memory_config=memory_config)

    while len(proj_bias.shape) < 4:
        proj_bias = proj_bias.unsqueeze(0)
    proj_bias = ttnn.from_torch(proj_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    proj_bias = ttnn.to_device(proj_bias, device, memory_config=memory_config)

    while len(gate_bias.shape) < 4:
        gate_bias = gate_bias.unsqueeze(0)
    gate_bias = ttnn.from_torch(gate_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    gate_bias = ttnn.to_device(gate_bias, device, memory_config=memory_config)

    params.proj.proj_weight = proj_weight
    params.proj.gate_weight = gate_weight
    params.proj.proj_bias = proj_bias
    params.proj.gate_bias = gate_bias

    del params.proj.weight
    del params.proj.bias
    return params


class geglu:
    def __init__(self, device, parameters):
        self.device = device
        parameters = split_linear_params(parameters)
        self.parameters = parameters
        self.grid_sizes = {8192: (5, 8), 2048: (5, 8), 512: (8, 8), 128: (8, 4)}
        self.out_subblock_hs = {8192: 8, 2048: 8, 512: 2, 128: 1}

        self.l1_interleaved_memory_config = ttnn.L1_MEMORY_CONFIG
        self.block_sharded_memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, config, hidden_states):
        # TODO: Output sharded once https://github.com/tenstorrent/tt-metal/issues/6775 is fixed
        interleaved_output = False
        size = hidden_states.shape[-2]
        grid_size = self.grid_sizes[size]
        M, K, N = hidden_states.shape[-2], hidden_states.shape[-1], self.parameters.proj.proj_weight.shape[-1]
        if not hidden_states.is_sharded():
            hidden_states = ttnn.interleaved_to_sharded(
                hidden_states,
                grid_size,
                [M // grid_size[1], K // grid_size[0]],
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
        in0_block_h, in0_block_w, out_subblock_h, out_subblock_w, out_block_h, out_block_w = determine_blocking(
            M, K, N, grid_size
        )
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
        proj = ttnn.linear(
            hidden_states,
            self.parameters.proj.proj_weight,
            bias=self.parameters.proj.proj_bias,
            program_config=program_config,
            memory_config=self.l1_interleaved_memory_config if interleaved_output else self.block_sharded_memory_config,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_config,
        )
        if interleaved_output:
            proj = ttnn.interleaved_to_sharded(
                proj,
                grid_size,
                [proj.shape[-2] // grid_size[1], proj.shape[-1] // grid_size[0]],
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
        if hidden_states.shape[-2] == 8192:
            proj = ttnn.reallocate(proj)

        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            transpose_mcast=False,
            fused_activation=[ttnn.UnaryOpType.GELU, True],
        )
        gate = ttnn.linear(
            hidden_states,
            self.parameters.proj.gate_weight,
            bias=self.parameters.proj.gate_bias,
            program_config=program_config,
            memory_config=self.l1_interleaved_memory_config if interleaved_output else self.block_sharded_memory_config,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.compute_kernel_config,
        )
        if interleaved_output:
            gate = ttnn.interleaved_to_sharded(
                gate,
                grid_size,
                [gate.shape[-2] // grid_size[1], gate.shape[-1] // grid_size[0]],
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
        if hidden_states.shape[-2] == 8192:
            gate = ttnn.reallocate(gate)
        ret = ttnn.mul(proj, gate, memory_config=gate.memory_config())
        ttnn.deallocate(proj)
        ttnn.deallocate(gate)
        return ret
