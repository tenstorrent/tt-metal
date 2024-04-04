# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import tt_lib as ttl
from models.utility_functions import tt2torch_tensor, comp_pcc
from models.utility_functions import is_grayskull
import torch


def run_create_qkv_heads_test(
    batch,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    device,
    transpose_k=False,
    in_mem_config=None,
    out_mem_config=None,
):
    torch.manual_seed(1234)

    q_shape = [batch, 1, seq_len, num_kv_heads, num_q_heads // num_kv_heads * head_dim]
    k_shape = [batch, 1, seq_len, num_kv_heads, head_dim]
    v_shape = [batch, 1, seq_len, num_kv_heads, head_dim]

    # torch reference vectors
    if dtype == ttl.tensor.DataType.FLOAT32:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)
    QKV = torch.concat([Q.flatten(-2, -1), K.flatten(-2, -1), V.flatten(-2, -1)], -1)
    QKV_interleaved = torch.concat([Q, K, V], -1).flatten(-2, -1)

    in0_shard_spec = ttl.tensor.ShardSpec(
        ttl.tensor.CoreRangeSet(
            {
                ttl.tensor.CoreRange(
                    ttl.tensor.CoreCoord(0, 0),
                    ttl.tensor.CoreCoord(num_kv_heads - 1, batch - 1),
                ),
            }
        ),
        [
            seq_len,
            QKV.shape[-1] // num_kv_heads,
        ],
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )

    in0_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1, in0_shard_spec
    )
    in0_t = ttl.tensor.Tensor(QKV_interleaved, dtype).to(ttl.tensor.Layout.TILE).to(device, in0_mem_config)

    out_shard_spec = in0_shard_spec
    out_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, ttl.tensor.BufferType.L1, out_shard_spec
    )
    q, k, v = ttl.tensor.create_qkv_heads(
        in0_t,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        output_mem_config=out_mem_config,
    )

    assert list(q.get_legacy_shape()) == [batch, num_q_heads, seq_len, head_dim]
    assert list(k.get_legacy_shape()) == [batch, num_kv_heads, seq_len, head_dim]
    assert list(v.get_legacy_shape()) == [batch, num_kv_heads, seq_len, head_dim]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    (ref_q, ref_k, ref_v) = torch.split(
        QKV, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
    )
    # Additional shuffling for Q, K, V heads
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)

    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(ref_q, pyt_got_back_rm_q, pcc)
    logger.debug(f"Q passing={passing_pcc_q}")
    logger.debug(f"Q output pcc={output_pcc_q}")

    passing_pcc_k, output_pcc_k = comp_pcc(ref_k, pyt_got_back_rm_k, pcc)
    logger.debug(f"K passing={passing_pcc_k}")
    logger.debug(f"K output pcc={output_pcc_k}")

    passing_pcc_v, output_pcc_v = comp_pcc(ref_v, pyt_got_back_rm_v, pcc)
    logger.debug(f"V passing={passing_pcc_v}")
    logger.debug(f"V output pcc={output_pcc_v}")

    assert passing_pcc_q
    assert passing_pcc_k
    assert passing_pcc_v


@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT8_B, ttl.tensor.DataType.BFLOAT16, ttl.tensor.DataType.FLOAT32),
    ids=["BFLOAT8_B", "BFLOAT16", "FLOAT32"],
)
@pytest.mark.parametrize(
    "batch, seq_len, num_q_heads, num_kv_heads, head_dim",
    (
        (7, 224, 8, 8, 64),
        (7, 384, 8, 8, 64),
    ),
)
def test_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    device,
):
    if is_grayskull() and dtype == ttl.tensor.DataType.FLOAT32:
        pytest.skip("Skipping float32 tests on Grayskull")

    run_create_qkv_heads_test(batch, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, device)
