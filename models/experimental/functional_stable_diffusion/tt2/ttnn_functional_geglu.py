# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import tt_lib as ttl


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def split_linear_params(params):
    dim = -1
    device = params.proj.weight.device()
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    weight = ttnn_to_torch(params.proj.weight)
    bias = ttnn_to_torch(params.proj.bias)

    proj_weight, gate_weight = torch.split(weight, weight.shape[dim] // 2, dim=dim)
    proj_bias, gate_bias = torch.split(bias, bias.shape[dim] // 2, dim=dim)

    proj_weight = ttnn.from_torch(proj_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    proj_weight = ttnn.to_device(proj_weight, device, memory_config=memory_config)

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
        self.grid_sizes = {8192: (8, 5), 2048: (8, 5), 512: (8, 8), 128: (4, 8)}
        self.out_subblock_hs = {8192: 8, 2048: 8, 512: 2, 128: 1}

        self.l1_interleaved_memory_config = ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        )
        self.block_sharded_memory_config = ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        )
        self.compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, config, hidden_states):
        # TODO: Output sharded once https://github.com/tenstorrent-metal/tt-metal/issues/6775 is fixed
        interleaved_output = True
        size = hidden_states.shape[-2]
        grid_size = self.grid_sizes[size]
        M, K, N = hidden_states.shape[-2], hidden_states.shape[-1], self.parameters.proj.proj_weight.shape[-1]
        Nt = N // 32
        G = grid_size[1]
        per_core_N = (Nt - 1) // (G - 1) if Nt != 16 else 4
        program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // grid_size[1] // 32,
            out_subblock_h=self.out_subblock_hs[size] if interleaved_output else 1,
            out_subblock_w=1,
            per_core_M=M // grid_size[0] // 32,
            per_core_N=per_core_N,
            fused_activation=None,
            transpose_mcast=True,
        )
        proj = ttnn.experimental.operations.primary.matmul(
            hidden_states,
            self.parameters.proj.proj_weight,
            bias=self.parameters.proj.proj_bias,
            program_config=program_config,
            output_mem_config=self.l1_interleaved_memory_config,
            output_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=self.compute_kernel_config,
        )
        proj = ttnn.experimental.tensor.interleaved_to_sharded(
            proj,
            grid_size,
            [proj.shape[-2] // grid_size[0], proj.shape[-1] // grid_size[1]],
            ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.experimental.tensor.ShardOrientation.COL_MAJOR,
        )
        if hidden_states.shape[-2] == 8192:
            proj = ttnn.reallocate(proj)

        program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // grid_size[1] // 32,
            out_subblock_h=self.out_subblock_hs[size] if interleaved_output else 1,
            out_subblock_w=1,
            per_core_M=M // grid_size[0] // 32,
            per_core_N=per_core_N,
            fused_activation=[ttnn.experimental.tensor.FusibleActivation.GELU, False],
            transpose_mcast=True,
        )
        gate = ttnn.experimental.operations.primary.matmul(
            hidden_states,
            self.parameters.proj.gate_weight,
            bias=self.parameters.proj.gate_bias,
            program_config=program_config,
            output_mem_config=self.l1_interleaved_memory_config,
            output_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=self.compute_kernel_config,
        )
        gate = ttnn.experimental.tensor.interleaved_to_sharded(
            gate,
            grid_size,
            [gate.shape[-2] // grid_size[0], gate.shape[-1] // grid_size[1]],
            ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.experimental.tensor.ShardOrientation.COL_MAJOR,
        )
        if hidden_states.shape[-2] == 8192:
            gate = ttnn.reallocate(gate)
        ret = ttnn.mul(proj, gate, memory_config=gate.memory_config())
        ttnn.deallocate(proj)
        ttnn.deallocate(gate)
        return ret
