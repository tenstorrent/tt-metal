# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.utility_functions import tt2torch_tensor, comp_pcc
from models.utility_functions import is_grayskull, skip_for_blackhole
import torch
import ttnn


def run_create_qkv_heads_test(
    batch,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    cores_h,
    cores_w,
    device,
    transpose_k,
    in_mem_config=None,
    out_mem_config=None,
):
    torch.manual_seed(1234)

    q_shape = [batch, 1, seq_len, num_kv_heads, num_q_heads // num_kv_heads * head_dim]
    k_shape = [batch, 1, seq_len, num_kv_heads, head_dim]
    v_shape = [batch, 1, seq_len, num_kv_heads, head_dim]

    # torch reference vectors
    if dtype == ttnn.float32:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)
    QKV = torch.concat([Q.flatten(-2, -1), K.flatten(-2, -1), V.flatten(-2, -1)], -1)
    QKV_interleaved = torch.concat([Q, K, V], -1).flatten(-2, -1)

    in0_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(cores_w - 1, cores_h - 1),
                ),
            }
        ),
        [
            seq_len,
            QKV.shape[-1] // cores_w,
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )

    in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
    in0_t = ttnn.Tensor(QKV_interleaved, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    out_shard_spec = in0_shard_spec
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    q, k, v = ttnn.experimental.create_qkv_heads(
        in0_t,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k,
        memory_config=out_mem_config,
    )

    assert list(q.get_legacy_shape()) == [batch, num_q_heads, seq_len, head_dim]
    if transpose_k:
        assert list(k.get_legacy_shape()) == [batch, num_kv_heads, head_dim, seq_len]
    else:
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

    if transpose_k:
        ref_k = torch.transpose(ref_k, -2, -1)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    elif (
        dtype == ttnn.float32 and transpose_k
    ):  # conversion from fp32 to tf32 when unpack writes to register for compute will decrease pcc in the transpose case
        pcc = 0.9999999
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


@skip_for_blackhole("L1 and Circular buffers are crashing on BH, see #12349")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32),
    ids=["BFLOAT8_B", "BFLOAT16", "FLOAT32"],
)
@pytest.mark.parametrize(
    "transpose_k",
    (True, False),
)
@pytest.mark.parametrize(
    "batch, seq_len, num_q_heads, num_kv_heads, head_dim, cores_h, cores_w",
    (
        (7, 224, 8, 8, 64, 7, 8),
        (7, 384, 8, 8, 64, 7, 8),
        (7, 224, 16, 16, 64, 7, 8),
        (7, 384, 16, 16, 64, 7, 8),
        (7, 384, 16, 8, 64, 7, 8),
        (7, 384, 12, 6, 64, 7, 6),
        (7, 384, 12, 12, 64, 7, 6),
    ),
)
def test_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    transpose_k,
    cores_h,
    cores_w,
    dtype,
    device,
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    run_create_qkv_heads_test(
        batch, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, cores_h, cores_w, device, transpose_k
    )


def run_create_q_and_kv_heads_test(
    batch,
    q_seq_len,
    kv_seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    cores_h,
    cores_w,
    device,
    transpose_k,
    in_mem_config=None,
    out_mem_config=None,
):
    torch.manual_seed(1234)

    q_shape = [batch, 1, q_seq_len, num_q_heads, head_dim]
    k_shape = [batch, 1, kv_seq_len, num_kv_heads, head_dim]
    v_shape = [batch, 1, kv_seq_len, num_kv_heads, head_dim]

    # torch reference vectors
    if dtype == ttnn.float32:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16

    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)

    KV = torch.concat([K.flatten(-2, -1), V.flatten(-2, -1)], -1)
    KV_interleaved = torch.concat([K, V], -1).flatten(-2, -1)
    Q_flattened = Q.flatten(-2, -1)

    kv_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(cores_w - 1, cores_h - 1),
                ),
            }
        ),
        [
            kv_seq_len,
            KV_interleaved.shape[-1] // cores_w,
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )

    q_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(cores_w - 1, cores_h - 1),
                ),
            }
        ),
        [
            q_seq_len,
            Q_flattened.shape[-1] // cores_w,
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
        False,
    )

    kv_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, kv_shard_spec)
    kv_t = ttnn.Tensor(KV_interleaved, dtype).to(ttnn.TILE_LAYOUT).to(device, kv_mem_config)

    q_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, q_shard_spec)
    q_t = ttnn.Tensor(Q_flattened, dtype).to(ttnn.TILE_LAYOUT).to(device, q_mem_config)

    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    q, k, v = ttnn.experimental.create_qkv_heads_from_separate_tensors(
        q_t,
        kv_t,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k,
        memory_config=out_mem_config,
    )

    assert list(q.get_legacy_shape()) == [batch, num_q_heads, q_seq_len, head_dim]
    if transpose_k:
        assert list(k.get_legacy_shape()) == [batch, num_kv_heads, head_dim, kv_seq_len]
    else:
        assert list(k.get_legacy_shape()) == [batch, num_kv_heads, kv_seq_len, head_dim]

    assert list(v.get_legacy_shape()) == [batch, num_kv_heads, kv_seq_len, head_dim]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    (ref_k, ref_v) = torch.split(KV, [num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)

    # Additional shuffling for Q, K, V heads
    ref_q = torch.reshape(Q_flattened, [batch, q_seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, kv_seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, kv_seq_len, num_kv_heads, head_dim]).transpose(-3, -2)

    if transpose_k:
        ref_k = torch.transpose(ref_k, -2, -1)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    elif (
        dtype == ttnn.float32 and transpose_k
    ):  # conversion from fp32 to tf32 when unpack writes to register for compute will decrease pcc in the transpose case
        pcc = 0.9999999
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
    (ttnn.bfloat8_b, ttnn.bfloat16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "transpose_k",
    (True, False),
)
@pytest.mark.parametrize(
    "batch, q_seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, cores_h, cores_w",
    (
        (2, 4096, 96, 8, 8, 64, 2, 8),
        (2, 1024, 96, 8, 8, 96, 2, 8),
        (2, 256, 96, 8, 8, 160, 2, 8),
        (2, 64, 96, 8, 8, 160, 2, 4),
    ),
)
def test_nlp_create_q_and_kv_heads_separate_test(
    batch,
    q_seq_len,
    kv_seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    transpose_k,
    cores_h,
    cores_w,
    dtype,
    device,
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    elif is_grayskull() and q_seq_len == 4096 and dtype == ttnn.bfloat16:
        pytest.skip("Spec runs out of L1 on on Grayskull")

    run_create_q_and_kv_heads_test(
        batch, q_seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, dtype, cores_h, cores_w, device, transpose_k
    )
