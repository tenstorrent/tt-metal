# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.common.utility_functions import tt2torch_tensor, comp_pcc
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

    assert list(q.padded_shape) == [batch, 71, seq_len, 64]
    assert list(k.padded_shape) == [batch, 1, seq_len, 64]
    assert list(v.padded_shape) == [batch, 1, seq_len, 64]

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
    run_nlp_create_qkv_heads_falcon7b_test(batch, seq_len, dtype, in0_mem_config, out_mem_config, device)


def test_nlp_create_qkv_heads_falcon7b_with_program_cache(device):
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

    assert list(q.padded_shape) == [batch, num_q_heads, seq_len, head_dim]
    if transpose_k_heads:
        assert list(k.padded_shape) == [batch, num_kv_heads, head_dim, seq_len]
    else:
        assert list(k.padded_shape) == [batch, num_kv_heads, seq_len, head_dim]
    assert list(v.padded_shape) == [batch, num_kv_heads, seq_len, head_dim]

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


@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=[
        "out_DRAM",
    ],
)
@pytest.mark.parametrize(
    "in_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG,),
    ids=[
        "in_DRAM",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat8_b,
        ttnn.bfloat16,
    ),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize("batch", (1,))
# Disabling 131072 seq_len case because of #17309
# @pytest.mark.parametrize("seq_len", (128, 1024, 30720, 131072))
@pytest.mark.parametrize("seq_len", (128, 1024, 30720))
@pytest.mark.parametrize("head_dim", (128,))
@pytest.mark.parametrize("num_q_heads", (32,))
@pytest.mark.parametrize("num_kv_heads", (4,))
@pytest.mark.parametrize("parallel_factor", (1, 2, 4))
@pytest.mark.parametrize("transpose_k_heads", (False,))
@pytest.mark.parametrize("read_from_input_tensor_kv", (False,))
def test_nlp_create_qkv_heads_llama_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    parallel_factor,
    transpose_k_heads,
    read_from_input_tensor_kv,
    dtype,
    in_mem_config,
    out_mem_config,
    request,
    device,
):
    num_q_heads = num_q_heads // parallel_factor
    num_kv_heads = num_kv_heads // parallel_factor
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


def test_nlp_create_qkv_heads_with_program_cache(device):
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
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
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
        )
        in1_shard_spec = ttnn.ShardSpec(
            shard_grid,
            [
                seq_len * batch,
                B_interleaved.shape[-1] // num_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
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

    assert list(q.padded_shape) == [seq_len, num_q_heads, batch, head_dim]
    assert list(k.padded_shape) == [seq_len, num_kv_heads, batch, head_dim]
    assert list(v.padded_shape) == [seq_len, num_kv_heads, batch, head_dim]

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


def test_sharded_nlp_create_qkv_heads_with_program_cache(device):
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


"""
K=V tied shapes + functionality

The fused input carries Q and a single K/V section -- (num_q_heads + num_kv_heads) * head_dim
wide instead of (num_q_heads + 2 * num_kv_heads) -- and V is read from the same columns as K.
Used by models that tie K and V to one projection (Gemma4's global layers), where computing the
duplicate columns in the projection matmul is pure waste.
"""


def run_nlp_create_qkv_heads_kv_tied_test(
    batch,
    seq_len,
    head_dim,
    num_q_heads,
    num_kv_heads,
    dtype,
    in_mem_config,
    out_mem_config,
    device,
    read_from_input_tensor_kv,
):
    torch.manual_seed(1234)

    if read_from_input_tensor_kv:
        # Q on its own, and one K/V section in the second tensor rather than two.
        A = torch.randn([batch, 1, seq_len, num_q_heads * head_dim])
        B = torch.randn([batch, 1, seq_len, num_kv_heads * head_dim])
        in1_t = ttnn.Tensor(B, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)
        src_q, src_kv = A, B
    else:
        A = torch.randn([batch, 1, seq_len, (num_q_heads + num_kv_heads) * head_dim])
        in1_t = None
        src_q, src_kv = torch.split(A, [num_q_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)

    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        in1_t,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=out_mem_config,
        kv_tied=True,
    )

    assert list(q.padded_shape) == [batch, num_q_heads, seq_len, head_dim]
    assert list(k.padded_shape) == [batch, num_kv_heads, seq_len, head_dim]
    assert list(v.padded_shape) == [batch, num_kv_heads, seq_len, head_dim]

    got_q = tt2torch_tensor(q)
    got_k = tt2torch_tensor(k)
    got_v = tt2torch_tensor(v)

    ref_q = torch.reshape(src_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_kv = torch.reshape(src_kv, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)

    pcc = 0.99 if dtype == ttnn.bfloat8_b else 1.0

    passing_q, pcc_q = comp_pcc(got_q, ref_q, pcc)
    logger.debug(f"Q passing={passing_q} pcc={pcc_q}")
    passing_k, pcc_k = comp_pcc(got_k, ref_kv, pcc)
    logger.debug(f"K passing={passing_k} pcc={pcc_k}")
    passing_v, pcc_v = comp_pcc(got_v, ref_kv, pcc)
    logger.debug(f"V passing={passing_v} pcc={pcc_v}")
    assert passing_q
    assert passing_k
    assert passing_v
    # The whole point of the mode: V is K's columns read a second time, so the two outputs are
    # the same values in two tensors. A PCC check alone would pass on an off-by-one column read.
    assert torch.equal(got_k, got_v)


@pytest.mark.parametrize("read_from_input_tensor_kv", (False, True), ids=["fused_qkv", "q_and_kv"])
@pytest.mark.parametrize(
    "out_mem_config",
    (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_q_heads, num_kv_heads",
    (
        (1, 128, 512, 4, 1),  # Gemma4-31B global layer, per device at TP=8
        (1, 128, 64, 8, 2),  # >1 KV head, so a tied read spanning several head slices
        (2, 32, 32, 3, 3),  # batch > 1 and num_kv_heads == num_q_heads
    ),
    ids=["gemma4_global", "multi_kv_head", "batch2"],
)
def test_nlp_create_qkv_heads_kv_tied(
    batch, seq_len, head_dim, num_q_heads, num_kv_heads, dtype, out_mem_config, read_from_input_tensor_kv, device
):
    run_nlp_create_qkv_heads_kv_tied_test(
        batch,
        seq_len,
        head_dim,
        num_q_heads,
        num_kv_heads,
        dtype,
        ttnn.DRAM_MEMORY_CONFIG,
        out_mem_config,
        device,
        read_from_input_tensor_kv,
    )


def test_nlp_create_qkv_heads_kv_tied_rejects_untied_widths(device, expect_error):
    """An input still carrying two K/V sections must fail rather than be read as a tied one.

    This is the mistake the mode invites: turn kv_tied on and forget to narrow the projection.
    Reading half of an untied input would produce plausible-looking K and V from the wrong columns.
    """
    dtype = ttnn.bfloat16
    num_q_heads, num_kv_heads, head_dim, seq_len = 4, 1, 64, 32

    # Fused and still sized for two K/V sections: (4 + 2) * head_dim is not a multiple of (4 + 1).
    A = torch.randn([1, 1, seq_len, (num_q_heads + 2 * num_kv_heads) * head_dim])
    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(RuntimeError, "Unsupported input shape"):
        ttnn.experimental.nlp_create_qkv_heads(
            in0_t, None, num_heads=num_q_heads, num_kv_heads=num_kv_heads, transpose_k_heads=False, kv_tied=True
        )

    # Separate KV tensor still carrying both sections, so its implied head_dim is twice Q's.
    Q = torch.randn([1, 1, seq_len, num_q_heads * head_dim])
    B = torch.randn([1, 1, seq_len, 2 * num_kv_heads * head_dim])
    q_t = ttnn.Tensor(Q, dtype).to(ttnn.TILE_LAYOUT).to(device, ttnn.DRAM_MEMORY_CONFIG)
    kv_t = ttnn.Tensor(B, dtype).to(ttnn.TILE_LAYOUT).to(device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(RuntimeError, "Head dims must be the same"):
        ttnn.experimental.nlp_create_qkv_heads(
            q_t, kv_t, num_heads=num_q_heads, num_kv_heads=num_kv_heads, transpose_k_heads=False, kv_tied=True
        )


def test_nlp_create_qkv_heads_kv_tied_rejects_ambiguous_fused_widths(device, expect_error):
    """A fused shape divisible both tied and untied section counts must be rejected as ambiguous."""
    dtype = ttnn.bfloat16
    num_q_heads, num_kv_heads, head_dim, seq_len = 1, 1, 64, 32

    # Untied fused width (1 + 2) * 64 == 192 is also divisible by tied sections (1 + 1).
    A = torch.randn([1, 1, seq_len, (num_q_heads + 2 * num_kv_heads) * head_dim])
    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(RuntimeError, "Ambiguous kv_tied fused input shape"):
        ttnn.experimental.nlp_create_qkv_heads(
            in0_t, None, num_heads=num_q_heads, num_kv_heads=num_kv_heads, transpose_k_heads=False, kv_tied=True
        )


def test_nlp_create_qkv_heads_kv_tied_transpose_k_heads(device):
    """transpose_k_heads only selects which CB K lands in, so it is orthogonal to the tied rewind.

    K comes back transposed on its last two dims while V does not, so the two outputs are
    transposes of each other rather than equal.
    """
    batch, seq_len, head_dim, num_q_heads, num_kv_heads = 1, 64, 64, 4, 1
    dtype = ttnn.bfloat16
    A = torch.randn([batch, 1, seq_len, (num_q_heads + num_kv_heads) * head_dim])
    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, ttnn.DRAM_MEMORY_CONFIG)

    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        None,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        kv_tied=True,
    )

    assert list(q.padded_shape) == [batch, num_q_heads, seq_len, head_dim]
    assert list(k.padded_shape) == [batch, num_kv_heads, head_dim, seq_len]
    assert list(v.padded_shape) == [batch, num_kv_heads, seq_len, head_dim]

    got_q = tt2torch_tensor(q)
    got_k = tt2torch_tensor(k)
    got_v = tt2torch_tensor(v)

    src_q, src_kv = torch.split(A, [num_q_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    ref_q = torch.reshape(src_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_kv = torch.reshape(src_kv, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)

    passing_q, pcc_q = comp_pcc(got_q, ref_q, 1.0)
    logger.debug(f"Q passing={passing_q} pcc={pcc_q}")
    passing_v, pcc_v = comp_pcc(got_v, ref_kv, 1.0)
    logger.debug(f"V passing={passing_v} pcc={pcc_v}")
    assert passing_q
    assert passing_v
    # Same columns read twice, with K transposed on the way out.
    assert torch.equal(got_k, got_v.transpose(-2, -1))


def run_sharded_nlp_create_qkv_heads_kv_tied_test(
    batch, seq_len, head_dim, num_q_heads, num_kv_heads, read_from_input_tensor_kv, dtype, device
):
    torch.manual_seed(1234)
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = num_kv_heads
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
    q_shape = [seq_len, 1, batch, num_cores, num_q_heads // num_cores * head_dim]
    kv_shape = [seq_len, 1, batch, num_cores, num_kv_heads // num_cores * head_dim]
    Q = torch.randn(q_shape)
    KV = torch.randn(kv_shape)

    in1_t = None
    if read_from_input_tensor_kv:
        A = Q.flatten(-2, -1)
        B = KV.flatten(-2, -1)
        in0_shard_spec = ttnn.ShardSpec(
            shard_grid, [seq_len * batch, A.shape[-1] // num_cores], ttnn.ShardOrientation.ROW_MAJOR
        )
        in1_shard_spec = ttnn.ShardSpec(
            shard_grid, [seq_len * batch, B.shape[-1] // num_cores], ttnn.ShardOrientation.ROW_MAJOR
        )
        in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
        in1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in1_shard_spec)
        in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)
        in1_t = ttnn.Tensor(B, dtype).to(ttnn.TILE_LAYOUT).to(device, in1_mem_config)
        ref_q_src, ref_kv_src = A, B
    else:
        # One K/V section in the fused shard, so the per-core width is
        # (num_q_heads / num_kv_heads + 1) * head_dim rather than + 2.
        A_interleaved = torch.concat([Q, KV], -1).flatten(-2, -1)
        in0_shard_spec = ttnn.ShardSpec(
            shard_grid, [seq_len * batch, A_interleaved.shape[-1] // num_cores], ttnn.ShardOrientation.ROW_MAJOR
        )
        in0_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
        in0_t = ttnn.Tensor(A_interleaved, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)
        ref_q_src, ref_kv_src = torch.split(
            torch.concat([Q.flatten(-2, -1), KV.flatten(-2, -1)], -1),
            [num_q_heads * head_dim, num_kv_heads * head_dim],
            dim=-1,
        )

    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        in1_t,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=out_mem_config,
        kv_tied=True,
    )

    assert list(q.padded_shape) == [seq_len, num_q_heads, batch, head_dim]
    assert list(k.padded_shape) == [seq_len, num_kv_heads, batch, head_dim]
    assert list(v.padded_shape) == [seq_len, num_kv_heads, batch, head_dim]

    got_q = tt2torch_tensor(q)
    got_k = tt2torch_tensor(k)
    got_v = tt2torch_tensor(v)

    ref_q = torch.reshape(ref_q_src, [seq_len, batch, num_q_heads, head_dim]).transpose(-3, -2)
    ref_kv = torch.reshape(ref_kv_src, [seq_len, batch, num_kv_heads, head_dim]).transpose(-3, -2)

    pcc = 0.99 if dtype == ttnn.bfloat8_b else 1.0

    passing_q, pcc_q = comp_pcc(got_q, ref_q, pcc)
    logger.debug(f"Q passing={passing_q} pcc={pcc_q}")
    passing_k, pcc_k = comp_pcc(got_k, ref_kv, pcc)
    logger.debug(f"K passing={passing_k} pcc={pcc_k}")
    passing_v, pcc_v = comp_pcc(got_v, ref_kv, pcc)
    logger.debug(f"V passing={passing_v} pcc={pcc_v}")
    assert passing_q
    assert passing_k
    assert passing_v
    assert torch.equal(got_k, got_v)


@pytest.mark.parametrize("read_from_input_tensor_kv", (False, True), ids=["fused_qkv", "q_and_kv"])
@pytest.mark.parametrize("dtype", (ttnn.bfloat8_b, ttnn.bfloat16), ids=["BFLOAT8_B", "BFLOAT16"])
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_q_heads, num_kv_heads",
    (
        (32, 1, 64, 16, 1),
        (32, 1, 64, 32, 2),
    ),
    ids=["1_kv_head", "2_kv_heads"],
)
def test_sharded_nlp_create_qkv_heads_kv_tied(
    batch, seq_len, head_dim, num_q_heads, num_kv_heads, read_from_input_tensor_kv, dtype, device
):
    run_sharded_nlp_create_qkv_heads_kv_tied_test(
        batch, seq_len, head_dim, num_q_heads, num_kv_heads, read_from_input_tensor_kv, dtype, device
    )


def test_sharded_nlp_create_qkv_heads_kv_tied_with_program_cache(device):
    dtype = ttnn.bfloat8_b
    mem_config = ttnn.L1_MEMORY_CONFIG
    for _ in range(2):
        run_sharded_nlp_create_qkv_heads_kv_tied_test(32, 1, 64, 32, 2, False, dtype, device)
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.Tensor(py_dummy_tensor, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)

    assert device.num_program_cache_entries() == 1
