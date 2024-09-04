# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    skip_for_grayskull,
    nearest_32,
)


def run_test_create_head_max_width_shard(device, n_local_heads, n_local_kv_heads, head_dim, batch):
    ## Split Heads
    seq_len = 1
    total_heads = n_local_heads + n_local_kv_heads * 2
    # Prepare input
    proj_output = torch.rand(1, seq_len, batch, head_dim * total_heads)
    proj_output_tt = ttnn.from_torch(
        proj_output,
        layout=ttnn.TILE_LAYOUT,
    )
    # Use ttnn shape to get padding of batch
    padded_batch = proj_output_tt.shape.with_tile_padding()[2]
    shard_spec_1_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(0, 0),
            ),
        }
    )
    CREATE_HEAD_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_1_cores_grid,
            [
                padded_batch,
                head_dim * total_heads,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    proj_output_tt = proj_output_tt.to(device=device, mem_config=CREATE_HEAD_INPUT_MEMCFG)

    # tt operation
    (
        q_heads_tt,  # [seqlen, n_local_heads, bsz, head_dim]
        k_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        v_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads_decode(
        proj_output_tt,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        memory_config=HEIGHT_SHARDED_MEMCFG,
        # unpadded_batch_size=batch if batch != padded_batch else None,
    )
    logger.info(f"q_heads_tt: {q_heads_tt.memory_config()}")
    logger.info(f"k_heads_tt: {k_heads_tt.memory_config()}")
    logger.info(f"v_heads_tt: {v_heads_tt.memory_config()}")

    # torch operation
    q_heads_torch = proj_output[:, :, :batch, : head_dim * n_local_heads].view(seq_len, batch, n_local_heads, head_dim)
    k_heads_torch = proj_output[
        :, :, :batch, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)
    ].view(seq_len, batch, n_local_kv_heads, head_dim)
    v_heads_torch = proj_output[:, :, :batch, head_dim * (n_local_heads + n_local_kv_heads) :].view(
        seq_len, batch, n_local_kv_heads, head_dim
    )

    # compare
    q_heads_tt_cpu = tt2torch_tensor(q_heads_tt)[..., :n_local_heads, :]
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = tt2torch_tensor(k_heads_tt)[..., :n_local_kv_heads, :]
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = tt2torch_tensor(v_heads_tt)[..., :n_local_kv_heads, :]
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_local_heads, n_local_kv_heads, head_dim, batch",
    ((8, 1, 128, 32), (8, 4, 96, 32), (16, 2, 64, 32), (8, 1, 128, 16), (8, 1, 128, 8)),
)
def test_create_head_max_width_shard(
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    batch,
    device,
    use_program_cache,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_create_head_max_width_shard(device, n_local_heads, n_local_kv_heads, head_dim, batch)


def run_test_create_min_width_shard(
    device,
    n_local_heads,
    n_local_kv_heads,
    head_dim,
):
    ## Split Heads
    batch = 32
    seq_len = 1
    total_heads = n_local_heads + n_local_kv_heads * 2
    total_cores = total_heads * head_dim // 32
    core_x = min(total_cores, 8)
    core_y = max(1, total_cores // core_x)
    # Prepare input
    proj_output = torch.rand(1, seq_len, batch, head_dim * total_heads)

    # TT configs
    shard_spec_n_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(core_x - 1, core_y - 1),
            ),
        }
    )
    CREATE_HEAD_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_n_cores_grid,
            [
                32,
                32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    # Prepare tt input
    proj_output_tt = torch2tt_tensor(proj_output, tt_device=None).to(device=device, mem_config=CREATE_HEAD_INPUT_MEMCFG)

    # tt operation
    (
        q_heads_tt,  # [seqlen, n_local_heads, bsz, head_dim]
        k_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        v_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads_decode(
        proj_output_tt,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        memory_config=HEIGHT_SHARDED_MEMCFG,
    )
    logger.info(f"q_heads_tt: {q_heads_tt.memory_config()}")
    logger.info(f"k_heads_tt: {k_heads_tt.memory_config()}")
    logger.info(f"v_heads_tt: {v_heads_tt.memory_config()}")

    # torch operation
    q_heads_torch = proj_output[:, :, :, : head_dim * n_local_heads].view(seq_len, batch, n_local_heads, head_dim)
    k_heads_torch = proj_output[:, :, :, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)].view(
        seq_len, batch, n_local_kv_heads, head_dim
    )
    v_heads_torch = proj_output[:, :, :, head_dim * (n_local_heads + n_local_kv_heads) :].view(
        seq_len, batch, n_local_kv_heads, head_dim
    )

    # compare
    q_heads_tt_cpu = tt2torch_tensor(q_heads_tt)[..., :n_local_heads, :]
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = tt2torch_tensor(k_heads_tt)[..., :n_local_kv_heads, :]
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = tt2torch_tensor(v_heads_tt)[..., :n_local_kv_heads, :]
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_local_heads, n_local_kv_heads, head_dim",
    ((8, 1, 128), (8, 4, 96), (16, 2, 64)),
)
def test_create_min_width_shard(
    n_local_heads,
    n_local_kv_heads,
    head_dim,
    device,
    use_program_cache,
):
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_create_min_width_shard(
            device,
            n_local_heads,
            n_local_kv_heads,
            head_dim,
        )
