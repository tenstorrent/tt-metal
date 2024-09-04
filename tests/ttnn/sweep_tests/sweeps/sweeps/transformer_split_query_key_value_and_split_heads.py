# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import is_wormhole_b0, is_grayskull

# use combinations of batch_size/core height and q_heads/kv_heads/core width to keep permutations under control
# some failures are known (e.g. batch_size > cores_h, seq_q > seq_kv, num_kv_heads != num_q_heads when transpose = true) though they shouldn't be failures
# try to minimize the number of permutations of known failures that shouldn't fail to keep test quick
# interleaved tests are all expected to fail since the input format is different for sharded and interleaved, and the test mimicks the sharded path
# they need to be changed to match the sharded path

parameters = {
    "batch_size_cores_h": [(2, 2), (7, 7), (4, 2)],  # 3  [batch=1] case also needed
    "seq_len_q_kv": [
        (64, 64),
        (256, 96),
        (64, 96),
    ],  # 3 [seq_q = seq_kv = 224, 384, and seq_q = 1024, 4096, seq_kv = 96] cases needed by BERT, SD, falcon
    "num_q_kv_heads_cores_w": [
        (8, 8, 8),
        (4, 4, 2),
        (16, 8, 8),
    ],  # 3 [q_heads = kv_heads = 12] cases also used in assorted models
    "head_dim": [64, 160],  # 2 [96, 128] also used
    "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],  # 2
    "transpose_k": [True],  # 1
    "separate_tensors": [False, True],  # 2
    "input_memory_config": [ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG],  # 2
}


def skip(
    batch_size_cores_h,
    seq_len_q_kv,
    num_q_kv_heads_cores_w,
    head_dim,
    input_dtype,
    transpose_k,
    separate_tensors,
    input_memory_config,
) -> Tuple[bool, Optional[str]]:
    batch_size = batch_size_cores_h[0]
    cores_h = batch_size_cores_h[1]

    seq_len_q = seq_len_q_kv[0]
    seq_len_kv = seq_len_q_kv[1]

    num_q_heads = num_q_kv_heads_cores_w[0]
    num_kv_heads = num_q_kv_heads_cores_w[1]
    cores_w = num_q_kv_heads_cores_w[2]

    if is_wormhole_b0():
        if cores_h > 7 or cores_w > 8:
            return True, "Wormhole B0 does not support more than 7 cores in height and 8 cores in width"

    if is_grayskull():
        if input_dtype == ttnn.float32:
            return True, "Grayskull does not support FP32 data type"

    if input_memory_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
        if batch_size % cores_h != 0:
            return True, "batch_size should be divisible by cores_h"

        if (num_kv_heads * head_dim) % cores_w != 0:
            return True, "num_kv_heads * head_dim should be divisible by cores_w"

        if (num_q_heads * head_dim) % cores_w != 0:
            return True, "num_q_heads * head_dim should be divisible by cores_w"

        if (num_kv_heads * head_dim) % 32 != 0:
            return True, "num_kv_heads * head_dim should be divisible by Tile Width"

        if (num_q_heads * head_dim) % 32 != 0:
            return True, "num_q_heads * head_dim should be divisible by Tile Width"

    if not separate_tensors:
        if (num_q_heads % num_kv_heads) != 0:
            return True, "num_q_heads should be divisible by num_kv_heads when separate_tensors is False"
        if seq_len_kv != seq_len_q:
            return True, "seq_len_kv should be equal to seq_len_q when separate_tensors is False"

    return False, None


def xfail(**_) -> Tuple[bool, Optional[str]]:
    return False, None


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

    q_shape = [batch, q_seq_len, num_q_heads, head_dim]
    k_shape = [batch, kv_seq_len, num_kv_heads, head_dim]
    v_shape = [batch, kv_seq_len, num_kv_heads, head_dim]
    KV_shape = [batch, kv_seq_len, 2 * num_kv_heads * head_dim]
    Q_shape_flattened = [batch, q_seq_len, num_q_heads * head_dim]

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

    if in_mem_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
        kv_mem_config = ttnn.create_sharded_memory_config(
            KV_shape, core_grid=ttnn.CoreGrid(y=cores_h, x=cores_w), strategy=ttnn.ShardStrategy.BLOCK
        )
        kv_t = ttnn.from_torch(
            KV_interleaved, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=kv_mem_config
        )

        q_mem_config = ttnn.create_sharded_memory_config(
            Q_shape_flattened, core_grid=ttnn.CoreGrid(y=cores_h, x=cores_w), strategy=ttnn.ShardStrategy.BLOCK
        )
        q_t = ttnn.from_torch(
            Q_flattened, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=q_mem_config
        )

        out_mem_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    else:
        kv_t = ttnn.from_torch(
            KV_interleaved, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=in_mem_config
        )
        q_t = ttnn.from_torch(
            Q_flattened, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=in_mem_config
        )
        out_mem_config = in_mem_config

    if num_q_heads == num_kv_heads:
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            q_t, kv_input_tensor=kv_t, num_heads=num_q_heads, transpose_key=transpose_k, memory_config=out_mem_config
        )
    else:
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            q_t,
            kv_input_tensor=kv_t,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            transpose_key=transpose_k,
            memory_config=out_mem_config,
        )

    pyt_got_back_rm_q = ttnn.to_torch(q)
    pyt_got_back_rm_k = ttnn.to_torch(k)
    pyt_got_back_rm_v = ttnn.to_torch(v)

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

    query_matches, query_message = check_with_pcc(ref_q, pyt_got_back_rm_q, pcc)
    key_matches, key_message = check_with_pcc(ref_k, pyt_got_back_rm_k, pcc)
    value_matches, value_message = check_with_pcc(ref_v, pyt_got_back_rm_v, pcc)

    passed = query_matches and key_matches and value_matches
    message = ""
    if not query_matches:
        message += f"query: {query_message}; "
    if not key_matches:
        message += f"key: {key_message}; "
    if not value_matches:
        message += f"value: {value_message}; "

    return passed, message


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

    q_shape = [batch, seq_len, num_kv_heads, num_q_heads // num_kv_heads * head_dim]
    k_shape = [batch, seq_len, num_kv_heads, head_dim]
    v_shape = [batch, seq_len, num_kv_heads, head_dim]
    QKV_shape = [batch, seq_len, (2 * num_kv_heads + num_q_heads) * head_dim]

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

    if in_mem_config == ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG:
        in0_mem_config = ttnn.create_sharded_memory_config(
            QKV_shape, core_grid=ttnn.CoreGrid(y=cores_h, x=cores_w), strategy=ttnn.ShardStrategy.BLOCK
        )
        in0_t = ttnn.from_torch(
            QKV_interleaved, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=in0_mem_config
        )
        out_mem_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    else:
        in0_t = ttnn.from_torch(
            QKV_interleaved, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=in_mem_config
        )
        out_mem_config = in_mem_config

    if num_kv_heads == num_q_heads:
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            in0_t, num_heads=num_q_heads, transpose_key=transpose_k, memory_config=out_mem_config
        )
    else:
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            in0_t,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            transpose_key=transpose_k,
            memory_config=out_mem_config,
        )

    pyt_got_back_rm_q = ttnn.to_torch(q)
    pyt_got_back_rm_k = ttnn.to_torch(k)
    pyt_got_back_rm_v = ttnn.to_torch(v)

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

    query_matches, query_message = check_with_pcc(ref_q, pyt_got_back_rm_q, pcc)
    key_matches, key_message = check_with_pcc(ref_k, pyt_got_back_rm_k, pcc)
    value_matches, value_message = check_with_pcc(ref_v, pyt_got_back_rm_v, pcc)

    passed = query_matches and key_matches and value_matches
    message = ""
    if not query_matches:
        message += f"query: {query_message}; "
    if not key_matches:
        message += f"key: {key_message}; "
    if not value_matches:
        message += f"value: {value_message}; "

    return passed, message


def run(
    batch_size_cores_h,
    seq_len_q_kv,
    num_q_kv_heads_cores_w,
    head_dim,
    input_dtype,
    transpose_k,
    separate_tensors,
    input_memory_config,
    *,
    device,
):
    batch_size = batch_size_cores_h[0]
    cores_h = batch_size_cores_h[1]

    seq_len_q = seq_len_q_kv[0]
    seq_len_kv = seq_len_q_kv[1]

    num_q_heads = num_q_kv_heads_cores_w[0]
    num_kv_heads = num_q_kv_heads_cores_w[1]
    cores_w = num_q_kv_heads_cores_w[2]

    if separate_tensors:
        passed, message = run_create_q_and_kv_heads_test(
            batch_size,
            seq_len_q,
            seq_len_kv,
            num_q_heads,
            num_kv_heads,
            head_dim,
            input_dtype,
            cores_h,
            cores_w,
            device,
            transpose_k,
            input_memory_config,
        )
    else:
        passed, message = run_create_qkv_heads_test(
            batch_size,
            seq_len_q,
            num_q_heads,
            num_kv_heads,
            head_dim,
            input_dtype,
            cores_h,
            cores_w,
            device,
            transpose_k,
            input_memory_config,
        )

    return passed, message
