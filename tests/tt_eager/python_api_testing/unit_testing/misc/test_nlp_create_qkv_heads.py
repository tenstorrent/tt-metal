# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.utility_functions import tt2torch_tensor, comp_pcc
from models.utility_functions import is_grayskull
import torch
import ttnn

"""
Falcon-7B shapes + functionality
"""


def run_nlp_create_qkv_heads_falcon7b_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device):
    torch.manual_seed(1234)

    in0_shape = [batch, 1, seq_len, 4672]

    A = torch.randn(in0_shape)

    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    q, k, v = ttnn.experimental.nlp_create_qkv_heads_falcon7b(in0_t, memory_config=out_mem_config)

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in0_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.get_dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.get_dtype()}")
    logger.debug(f"k: {k.memory_config().buffer_type} and {k.get_dtype()}")
    logger.debug(f"v: {v.memory_config().buffer_type} and {v.get_dtype()}")

    assert list(q.get_legacy_shape()) == [batch, 71, seq_len, 64]
    assert list(k.get_legacy_shape()) == [batch, 1, seq_len, 64]
    assert list(v.get_legacy_shape()) == [batch, 1, seq_len, 64]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    (ref_q, ref_k, ref_v) = torch.split(A, [4544, 64, 64], dim=-1)
    # Additional shuffling for Q head
    ref_q = torch.reshape(ref_q, [batch, seq_len, 71, 64]).transpose(-3, -2)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.debug(f"Q passing={passing_pcc_q}")
    logger.debug(f"Q output pcc={output_pcc_q}")
    assert passing_pcc_q
    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.debug(f"K passing={passing_pcc_k}")
    logger.debug(f"K output pcc={output_pcc_k}")
    assert passing_pcc_k
    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.debug(f"V passing={passing_pcc_v}")
    logger.debug(f"V output pcc={output_pcc_v}")
    assert passing_pcc_v


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32),
    ids=["BFLOAT8_B", "BFLOAT16", "FLOAT32"],
)
@pytest.mark.parametrize(
    "batch, seq_len",
    ((1, 32), (1, 64), (1, 128)),
    ids=[
        "batch1_seq32",
        "batch1_seq64",
        "batch1_seq128",
    ],
)
def test_nlp_create_qkv_heads_falcon7b_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, request, device):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    run_nlp_create_qkv_heads_falcon7b_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device)


def test_nlp_create_qkv_heads_falcon7b_with_program_cache(device, use_program_cache):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.DRAM_MEMORY_CONFIG
    for _ in range(2):
        run_nlp_create_qkv_heads_falcon7b_test(1, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_nlp_create_qkv_heads_falcon7b_test(1, 32, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2


"""
Generic shapes + functionality
"""


def run_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    transpose_k_heads,
    read_from_input_tensor_kv,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
):
    torch.manual_seed(1234)

    if read_from_input_tensor_kv:
        in0_shape = [batch, 1, seq_len, num_q_heads * head_dim]
        in1_shape = [batch, 1, seq_len, 2 * num_kv_heads * head_dim]
        A = torch.randn(in0_shape)
        B = torch.randn(in1_shape)
        in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)
        in1_t = ttnn.Tensor(B, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)
    else:
        in0_shape = [batch, 1, seq_len, (num_q_heads + 2 * num_kv_heads) * head_dim]
        A = torch.randn(in0_shape)
        in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)

    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        in1_t if read_from_input_tensor_kv else None,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k_heads,
        memory_config=out_mem_config,
    )

    # Check memory of inputs and outputs
    assert in0_t.memory_config().buffer_type == in_mem_config.buffer_type
    assert q.memory_config().buffer_type == out_mem_config.buffer_type
    assert k.memory_config().buffer_type == out_mem_config.buffer_type
    assert v.memory_config().buffer_type == out_mem_config.buffer_type
    logger.debug(f"in0: {in0_t.memory_config().buffer_type} and {in0_t.get_dtype()}")
    logger.debug(f"q: {q.memory_config().buffer_type} and {q.get_dtype()}")
    logger.debug(f"k: {k.memory_config().buffer_type} and {k.get_dtype()}")
    logger.debug(f"v: {v.memory_config().buffer_type} and {v.get_dtype()}")

    assert list(q.get_legacy_shape()) == [batch, num_q_heads, seq_len, head_dim]
    if transpose_k_heads:
        assert list(k.get_legacy_shape()) == [batch, num_kv_heads, head_dim, seq_len]
    else:
        assert list(k.get_legacy_shape()) == [batch, num_kv_heads, seq_len, head_dim]
    assert list(v.get_legacy_shape()) == [batch, num_kv_heads, seq_len, head_dim]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    if read_from_input_tensor_kv:
        ref_q = A
        (ref_k, ref_v) = torch.split(B, [num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    else:
        (ref_q, ref_k, ref_v) = torch.split(
            A, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
        )

    # Additional shuffling for Q, K, V heads
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    if transpose_k_heads:
        ref_k = ref_k.transpose(-2, -1)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    elif dtype == ttnn.float32:  # conversion from fp32 to tf32 will decrease pcc
        pcc = 0.9999999
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.debug(f"Q passing={passing_pcc_q}")
    logger.debug(f"Q output pcc={output_pcc_q}")

    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.debug(f"K passing={passing_pcc_k}")
    logger.debug(f"K output pcc={output_pcc_k}")

    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.debug(f"V passing={passing_pcc_v}")
    logger.debug(f"V output pcc={output_pcc_v}")
    assert passing_pcc_q
    assert passing_pcc_k
    assert passing_pcc_v


@pytest.mark.parametrize(
    "out_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["in_DRAM", "in_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32),
    ids=["BFLOAT8_B", "BFLOAT16", "FLOAT32"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_q_heads, num_kv_heads, transpose_k_heads, read_from_input_tensor_kv",
    (
        (1, 128, 64, 71, 1, False, False),
        (111, 64, 96, 5, 3, True, False),
        (5, 1024, 64, 8, 8, True, True),
    ),
)
def test_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    transpose_k_heads,
    read_from_input_tensor_kv,
    dtype,
    in_mem_config,
    out_mem_config,
    request,
    device,
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")
    if dtype == ttnn.float32 and (batch == 111 or batch == 5) and in_mem_config == ttnn.L1_MEMORY_CONFIG:
        logger.warning("fp32 tensor too large to fit L1")
    else:
        run_nlp_create_qkv_heads_test(
            batch,
            seq_len,
            head_dim,
            num_q_heads,
            num_kv_heads,
            transpose_k_heads,
            read_from_input_tensor_kv,
            dtype,
            in_mem_config,
            out_mem_config,
            device,
        )


def test_nlp_create_qkv_heads_with_program_cache(device, use_program_cache):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_nlp_create_qkv_heads_test(5, 1024, 64, 4, 2, True, False, dtype, mem_config, mem_config, device)
        # Same in0_shape to make sure cache misses if we have additional optional tensor works
        run_nlp_create_qkv_heads_test(5, 1024, 64, 8, 8, True, True, dtype, mem_config, mem_config, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2


def run_sharded_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    read_from_input_tensor_kv,
    dtype,
    device,
):
    torch.manual_seed(1234)
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = num_kv_heads
    shard_grid = ttnn.CoreRangeSet(ttnn.num_cores_to_corerange_set(num_cores, compute_grid_size, True))
    q_shape = [seq_len, 1, batch, num_cores, num_q_heads // num_cores * head_dim]
    kv_shape = [seq_len, 1, batch, num_cores, num_kv_heads // num_cores * head_dim]
    Q = torch.randn(q_shape)
    K = torch.randn(kv_shape)
    V = torch.randn(kv_shape)

    if read_from_input_tensor_kv:
        A = torch.concat([Q.flatten(-2, -1)], -1)
        B = torch.concat([K.flatten(-2, -1), V.flatten(-2, -1)], -1)
        A_interleaved = torch.concat([Q], -1).flatten(-2, -1)
        B_interleaved = torch.concat([K, V], -1).flatten(-2, -1)
        in0_shard_spec = ttnn.ShardSpec(
            shard_grid,
            [
                seq_len * batch,
                A_interleaved.shape[-1] // num_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        in1_shard_spec = ttnn.ShardSpec(
            shard_grid,
            [
                seq_len * batch,
                B_interleaved.shape[-1] // num_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
        in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in1_shard_spec)
        in0_t = ttnn.Tensor(A_interleaved, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)
        in1_t = ttnn.Tensor(B_interleaved, dtype).to(ttnn.TILE_LAYOUT).to(device, in1_mem_config)
    else:
        A = torch.concat([Q.flatten(-2, -1), K.flatten(-2, -1), V.flatten(-2, -1)], -1)
        A_interleaved = torch.concat([Q, K, V], -1).flatten(-2, -1)
        in0_shard_spec = ttnn.ShardSpec(
            shard_grid,
            [
                seq_len * batch,
                A_interleaved.shape[-1] // num_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
        in0_t = ttnn.Tensor(A_interleaved, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    out_shard_spec = in0_shard_spec
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        in1_t if read_from_input_tensor_kv else None,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=out_mem_config,
    )

    assert list(q.get_legacy_shape()) == [seq_len, num_q_heads, batch, head_dim]
    assert list(k.get_legacy_shape()) == [seq_len, num_kv_heads, batch, head_dim]
    assert list(v.get_legacy_shape()) == [seq_len, num_kv_heads, batch, head_dim]

    pyt_got_back_rm_q = tt2torch_tensor(q)
    pyt_got_back_rm_k = tt2torch_tensor(k)
    pyt_got_back_rm_v = tt2torch_tensor(v)

    if read_from_input_tensor_kv:
        ref_q = A
        (ref_k, ref_v) = torch.split(B, [num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    else:
        (ref_q, ref_k, ref_v) = torch.split(
            A, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
        )

    # Additional shuffling for Q, K, V heads
    ref_q = torch.reshape(ref_q, [seq_len, batch, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [seq_len, batch, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [seq_len, batch, num_kv_heads, head_dim]).transpose(-3, -2)

    if dtype == ttnn.bfloat8_b:
        pcc = 0.99
    else:
        pcc = 1.0

    passing_pcc_q, output_pcc_q = comp_pcc(pyt_got_back_rm_q, ref_q, pcc)
    logger.debug(f"Q passing={passing_pcc_q}")
    logger.debug(f"Q output pcc={output_pcc_q}")

    passing_pcc_k, output_pcc_k = comp_pcc(pyt_got_back_rm_k, ref_k, pcc)
    logger.debug(f"K passing={passing_pcc_k}")
    logger.debug(f"K output pcc={output_pcc_k}")

    passing_pcc_v, output_pcc_v = comp_pcc(pyt_got_back_rm_v, ref_v, pcc)
    logger.debug(f"V passing={passing_pcc_v}")
    logger.debug(f"V output pcc={output_pcc_v}")
    assert passing_pcc_q
    assert passing_pcc_k
    assert passing_pcc_v


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32),
    ids=["BFLOAT8_B", "BFLOAT16", "FLOAT32"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_q_heads, num_kv_heads, read_from_input_tensor_kv",
    (
        (32, 1, 64, 16, 1, False),
        (32, 1, 64, 16, 1, True),
        (32, 1, 64, 32, 2, False),
        (32, 1, 64, 32, 2, True),
        (32, 1, 64, 32, 32, False),
        (32, 1, 64, 32, 32, True),
    ),
)
def test_sharded_nlp_create_qkv_heads_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    read_from_input_tensor_kv,
    dtype,
    device,
):
    if is_grayskull() and dtype == ttnn.float32:
        pytest.skip("Skipping float32 tests on Grayskull")

    run_sharded_nlp_create_qkv_heads_test(
        batch,
        seq_len,
        head_dim,
        num_q_heads,
        num_kv_heads,
        read_from_input_tensor_kv,
        dtype,
        device,
    )


def test_sharded_nlp_create_qkv_heads_with_program_cache(device, use_program_cache):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_sharded_nlp_create_qkv_heads_test(32, 1, 64, 16, 8, False, dtype, device)
        # Same in0_shape to make sure cache misses if we have additional optional tensor works
        run_sharded_nlp_create_qkv_heads_test(32, 1, 64, 32, 1, True, dtype, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 2
