# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion.tt2.ttnn_functional_geglu import geglu
import tt_lib as ttl
import torch


def compare(tensor, name, reshape=False):
    from models.utility_functions import comp_pcc

    tensor = ttnn.to_layout(tensor, ttnn.ROW_MAJOR_LAYOUT)
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
        self.parameters.net[2].bias = ttnn.unsqueeze_to_4D(self.parameters.net[2].bias)
        self.geglu = geglu(device, parameters.net[0])
        self.block_sharded_memory_config = ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        )
        self.l1_interleaved_memory_config = ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        )
        self.compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
            math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.grid_sizes = {8192: (8, 5), 2048: (8, 5), 512: (8, 8), 128: (4, 8)}
        self.out_subblock_hs = {8192: 8, 2048: 8, 512: 2, 128: 1}

    def __call__(self, config, hidden_states):
        hidden_states = self.geglu(config, hidden_states)

        # TODO: Output sharded once https://github.com/tenstorrent-metal/tt-metal/issues/6775 is fixed
        interleaved_output = True
        size = hidden_states.shape[-2]
        grid_size = self.grid_sizes[size]
        M, K, N = hidden_states.shape[-2], hidden_states.shape[-1], self.parameters.net[2].weight.shape[-1]
        Nt = N // 32
        G = grid_size[1]
        per_core_N = (Nt - 1) // (G - 1) if Nt != 16 else 4
        program_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=K // grid_size[1] // 32,
            out_subblock_h=1,
            out_subblock_w=per_core_N,
            per_core_M=M // grid_size[0] // 32,
            per_core_N=per_core_N,
            fused_activation=None,
            transpose_mcast=True,
        )
        if hidden_states.shape[-2] == 8192:
            hidden_states = ttnn.reallocate(hidden_states)
        hidden_states = ttnn.experimental.operations.primary.matmul(
            hidden_states,
            self.parameters.net[2].weight,
            bias=self.parameters.net[2].bias,
            program_config=program_config,
            output_mem_config=self.l1_interleaved_memory_config
            if interleaved_output
            else self.block_sharded_memory_config,
            output_dtype=ttnn.experimental.tensor.DataType.BFLOAT8_B,
            compute_kernel_config=self.compute_kernel_config,
        )
        if interleaved_output:
            hidden_states = ttnn.experimental.tensor.interleaved_to_sharded(
                hidden_states,
                grid_size,
                [hidden_states.shape[-2] // grid_size[0], hidden_states.shape[-1] // grid_size[1]],
                ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.experimental.tensor.ShardOrientation.COL_MAJOR,
            )
        return hidden_states
