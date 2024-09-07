# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List

import ttnn

from models.utility_functions import nearest_y


# this function takes activations that are assumed to be non-padded and weights that are assumed to be padded
# (since they are pushed to device at the moment of model initialization);
# it also takes in number of slices since this should be determined at the moment of pushing weights,
# but in general with 512 < seq_len <= 1024 we should use 4 slices, with 1024 < seq_len <= 2048 we should use 8 slices
def falcon_lm_head_matmul_2d(
    hidden_states: ttnn.Tensor,
    weights: List[ttnn.Tensor],
    num_slices: int,
    lm_head_padding: ttnn.Tensor,
    out_mem_config: ttnn.MemoryConfig,
    out_dtype: ttnn.DataType,
):
    assert (
        hidden_states.device().arch() == ttnn.device.Arch.WORMHOLE_B0
    ), "Falcon LM head is only supported for Wormhole BO arch"

    seq_len = hidden_states.get_legacy_shape()[-2]

    assert seq_len % 32 == 0, f"Sequence length must be a multiple of 32, instead it is {seq_len}"
    assert seq_len > 512, f"Falcon lm head 2d is only supported for sequence length > 512, instead it is {seq_len}"
    assert seq_len <= 2048, f"Falcon lm head 2d is only supported for sequence length <= 2048, instead it is {seq_len}"

    assert (
        len(weights) == num_slices
    ), f"Weights are expected to be split into {num_slices} slices, instead there are {len(weights)}"
    weights_inner_dim_in_tiles = weights[0].get_legacy_shape()[-2] // 32
    assert (
        weights_inner_dim_in_tiles == 144
    ), f"Weights are expected to be padded to the inner dim 144 in tiles, instead they are {weights_inner_dim_in_tiles}"

    hidden_states = ttnn.concat([hidden_states, lm_head_padding], -1)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    grid = hidden_states.device().compute_with_storage_grid_size()
    activations_m_in_tiles = seq_len // 32
    weights_n_in_tiles = weights[0].get_legacy_shape()[-1] // 32

    # calculate parameters for the given sequence length
    grid = hidden_states.device().compute_with_storage_grid_size()
    out_subblock_h = 2
    out_subblock_w = 4
    per_core_M = nearest_y(activations_m_in_tiles / grid.y, out_subblock_h)
    per_core_N = nearest_y(weights_n_in_tiles / grid.x, out_subblock_w)
    in0_block_w = 4 if seq_len <= 1024 else 8

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )

    out_slices = []
    for i in range(num_slices):
        out_slices.append(
            ttnn.matmul(
                hidden_states,
                weights[i],
                program_config=program_config,
                memory_config=out_mem_config,
                dtype=out_dtype,
                compute_kernel_config=compute_kernel_config,
            )
        )

    out = ttnn.concat(out_slices, -1)
    for i in range(num_slices):
        out_slices[i].deallocate(True)

    return out
