# SPDX-FileCopyrightText: 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import FeedForward
from models.utility_functions import (
    comp_pcc,
)


@pytest.mark.parametrize(
    "seq_len",
    [
        128,
        170,
        212,
        256,
        341,
        426,
        512,
        682,
        852,
        1024,
        1365,
        1706,
        2048,
        2730,
        3412,
        4096,
        5461,
        6826,
        8192,
        10922,
        13652,
        16384,
    ],
)
@pytest.mark.parametrize("use_matmul", [True, False], ids=["matmul", "linear"])
@pytest.mark.parametrize("split_sequence", [True, False], ids=["split_sequence", "no_split_sequence"])
@pytest.mark.parametrize("l1_activations", [True, False], ids=["l1_activations", "dram_activations"])
def test_ff1(device, seq_len, use_matmul, split_sequence, l1_activations):
    if split_sequence and seq_len <= 1024:
        pytest.skip("Sequence length must be greater than 1024 when split_sequence is True")

    in0_shape = (1, seq_len // 1024, 1024, 2048) if split_sequence else (1, 1, seq_len, 2048)
    in1_shape = (1, 1, 2048, 3584)

    in0_torch = torch.randn(in0_shape)
    in1_torch = torch.randn(in1_shape)

    golden = torch.nn.functional.linear(in0_torch, in1_torch.permute(0, 1, 3, 2).reshape(3584, 2048))

    in0_tt = ttnn.from_torch(
        in0_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG if l1_activations else ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            shard_spec=ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(12, 0),
                        ),
                    }
                ),
                shard_shape=[2048, 320],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        layout=ttnn.TILE_LAYOUT,
    )
    if use_matmul:
        out_tt = ttnn.matmul(
            in0_tt,
            in1_tt,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            ),
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 10),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=4,
                out_block_h=8,
                out_block_w=16,
                per_core_M=8,
                per_core_N=16,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            ),
        )
    else:
        out_tt = ttnn.linear(
            in0_tt,
            in1_tt,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            ),
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 10),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=4,
                out_block_h=8,
                out_block_w=16,
                per_core_M=8,
                per_core_N=16,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            ),
        )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg


@pytest.mark.parametrize(
    "seq_len",
    [
        128,
        170,
        212,
        256,
        341,
        426,
        512,
        682,
        852,
        1024,
        1365,
        1706,
        2048,
        2730,
        3412,
        4096,
        5461,
        6826,
        8192,
        10922,
        13652,
        16384,
    ],
)
@pytest.mark.parametrize("use_matmul", [True, False], ids=["matmul", "linear"])
@pytest.mark.parametrize("split_sequence", [True, False], ids=["split_sequence", "no_split_sequence"])
def test_ff2(device, seq_len, use_matmul, split_sequence):
    if split_sequence and seq_len <= 1024:
        pytest.skip("Sequence length must be greater than 1024 when split_sequence is True")

    in0_shape = (1, seq_len // 1024, 1024, 3584) if split_sequence else (1, 1, seq_len, 3584)
    in1_shape = (1, 1, 3584, 2048)

    in0_torch = torch.randn(in0_shape)
    in1_torch = torch.randn(in1_shape)

    golden = torch.nn.functional.linear(in0_torch, in1_torch.permute(0, 1, 3, 2).reshape(2048, 3584))

    in0_tt = ttnn.from_torch(
        in0_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    in1_tt = ttnn.from_torch(
        in1_torch,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            shard_spec=ttnn.ShardSpec(
                grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(12, 0),
                        ),
                    }
                ),
                shard_shape=[3584, 192],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        layout=ttnn.TILE_LAYOUT,
    )

    if use_matmul:
        out_tt = ttnn.matmul(
            in0_tt,
            in1_tt,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            ),
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 10),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=2,
                out_block_h=8,
                out_block_w=10,
                per_core_M=8,
                per_core_N=10,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            ),
        )
    else:
        out_tt = ttnn.linear(
            in0_tt,
            in1_tt,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            ),
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(7, 10),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=2,
                out_block_h=8,
                out_block_w=10,
                per_core_M=8,
                per_core_N=10,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=False,
            ),
        )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg
