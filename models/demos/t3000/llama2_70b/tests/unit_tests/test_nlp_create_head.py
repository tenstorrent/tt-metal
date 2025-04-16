# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import get_devices_for_t3000, skip_for_grayskull, torch2tt_tensor, tt2torch_tensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_test_create_head1(
    devices,
    batch,
    seq_len,
):
    ## Split Heads
    n_local_heads = 8
    n_local_kv_heads = 1
    head_dim = 128
    # Prepare input
    proj_output = torch.rand(1, seq_len, batch, head_dim * 10)

    # TT configs
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
                32,
                1280,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    # Prepare tt input
    proj_output_tt = torch2tt_tensor(proj_output, tt_device=None).to(
        device=devices[0], mem_config=CREATE_HEAD_INPUT_MEMCFG
    )

    # tt operation
    (
        q_heads_tt,  # [seqlen, n_local_heads, bsz, head_dim]
        k_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        v_heads_tt,  # [seqlen, n_local_kv_heads, bsz, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads(
        proj_output_tt,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        transpose_k_heads=False,
        memory_config=HEIGHT_SHARDED_MEMCFG,
    )
    logger.info(f"q_heads_tt: {q_heads_tt.memory_config()}")
    logger.info(f"k_heads_tt: {k_heads_tt.memory_config()}")
    logger.info(f"v_heads_tt: {v_heads_tt.memory_config()}")

    # torch operation
    q_heads_torch = (
        proj_output[:, :, :, : head_dim * n_local_heads]
        .view(seq_len, batch, n_local_heads, head_dim)
        .permute(0, 2, 1, 3)
    )
    k_heads_torch = (
        proj_output[:, :, :, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)]
        .view(seq_len, batch, n_local_kv_heads, head_dim)
        .permute(0, 2, 1, 3)
    )
    v_heads_torch = (
        proj_output[:, :, :, head_dim * (n_local_heads + n_local_kv_heads) :]
        .view(seq_len, batch, n_local_kv_heads, head_dim)
        .permute(0, 2, 1, 3)
    )

    # compare
    q_heads_tt_cpu = tt2torch_tensor(q_heads_tt)
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch)
    logger.info(f"PCC value: {output_pcc_q}")
    assert out_pass_q

    k_heads_tt_cpu = tt2torch_tensor(k_heads_tt)
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch)
    logger.info(f"PCC value: {output_pcc_k}")
    assert out_pass_k

    v_heads_tt_cpu = tt2torch_tensor(v_heads_tt)
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch)
    logger.info(f"PCC value: {output_pcc_v}")
    assert out_pass_v


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_create_head1(
    batch,
    seq_len,
    all_devices,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    run_test_create_head1(
        devices,
        batch,
        seq_len,
    )


def run_test_create_head2(
    devices,
    batch,
    seq_len,
):
    ## Split Heads
    n_local_heads = 8
    n_local_kv_heads = 1
    head_dim = 128
    # Prepare input
    proj_output = torch.rand(1, seq_len, batch, head_dim * 10)

    # TT configs
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
                32,
                1280,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    # Prepare tt input
    proj_output_tt = torch2tt_tensor(proj_output, tt_device=None).to(
        device=devices[0], mem_config=CREATE_HEAD_INPUT_MEMCFG
    )

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
    q_heads_torch = torch.cat(
        [
            proj_output[:, :, :, : head_dim * n_local_heads].view(seq_len, batch, n_local_heads, head_dim),
            torch.zeros(seq_len, batch, 32 - n_local_heads, head_dim),
        ],
        dim=-2,
    )
    k_heads_torch = torch.cat(
        [
            proj_output[:, :, :, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)].view(
                seq_len, batch, n_local_kv_heads, head_dim
            ),
            torch.zeros(seq_len, batch, 32 - n_local_kv_heads, head_dim),
        ],
        dim=-2,
    )
    v_heads_torch = torch.cat(
        [
            proj_output[:, :, :, head_dim * (n_local_heads + n_local_kv_heads) :].view(
                seq_len, batch, n_local_kv_heads, head_dim
            ),
            torch.zeros(seq_len, batch, 32 - n_local_kv_heads, head_dim),
        ],
        dim=-2,
    )

    # compare
    q_heads_tt_cpu = tt2torch_tensor(q_heads_tt)
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = tt2torch_tensor(k_heads_tt)
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = tt2torch_tensor(v_heads_tt)
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_create_head2(
    batch,
    seq_len,
    all_devices,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    run_test_create_head2(
        devices,
        batch,
        seq_len,
    )


def run_test_create_head3(
    devices,
    batch,
    seq_len,
):
    ## Split Heads
    n_local_heads = 8
    n_local_kv_heads = 1
    head_dim = 128
    # Prepare input
    proj_output = torch.rand(1, seq_len, batch, head_dim * 10)

    # TT configs
    shard_spec_40_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 4),
            ),
        }
    )
    CREATE_HEAD_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_40_cores_grid,
            [
                32,
                32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    # Prepare tt input
    proj_output_tt = torch2tt_tensor(proj_output, tt_device=None).to(
        device=devices[0], mem_config=CREATE_HEAD_INPUT_MEMCFG
    )

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
    q_heads_torch = torch.cat(
        [
            proj_output[:, :, :, : head_dim * n_local_heads].view(seq_len, batch, n_local_heads, head_dim),
            torch.zeros(seq_len, batch, 32 - n_local_heads, head_dim),
        ],
        dim=-2,
    )
    k_heads_torch = torch.cat(
        [
            proj_output[:, :, :, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)].view(
                seq_len, batch, n_local_kv_heads, head_dim
            ),
            torch.zeros(seq_len, batch, 32 - n_local_kv_heads, head_dim),
        ],
        dim=-2,
    )
    v_heads_torch = torch.cat(
        [
            proj_output[:, :, :, head_dim * (n_local_heads + n_local_kv_heads) :].view(
                seq_len, batch, n_local_kv_heads, head_dim
            ),
            torch.zeros(seq_len, batch, 32 - n_local_kv_heads, head_dim),
        ],
        dim=-2,
    )

    # compare
    q_heads_tt_cpu = tt2torch_tensor(q_heads_tt)
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_torch)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = tt2torch_tensor(k_heads_tt)
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_torch)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = tt2torch_tensor(v_heads_tt)
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_torch)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "batch, seq_len",
    ((32, 1),),
)
def test_create_head3(
    batch,
    seq_len,
    all_devices,
    use_program_cache,
):
    n_devices = 8
    devices = get_devices_for_t3000(all_devices, num_devices=1)
    torch.manual_seed(0)

    for i in range(3):
        # multiple loops to test program caching
        run_test_create_head3(
            devices,
            batch,
            seq_len,
        )
