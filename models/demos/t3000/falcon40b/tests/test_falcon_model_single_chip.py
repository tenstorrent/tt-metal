# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc, torch2tt_tensor, tt2torch_tensor, pad_by_zero, get_devices_for_t3000


@pytest.mark.parametrize(
    "shard_orientation",
    (ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR),
)
@pytest.mark.parametrize(
    "output_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "in1_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "in0_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "batch, K, seq_len, q_heads, kv_heads",
    (
        (32, 64, 512 + 96, 16, 1),  # 8 chip pre-attn matmul shapes
        (32, 1024 + 32, 64, 16, 1),  # 8 chip post-attn matmul shapes
    ),
)
def test_group_attn_matmul(
    batch, K, seq_len, q_heads, kv_heads, in0_sharded, in1_sharded, output_sharded, shard_orientation, device
):
    torch.manual_seed(0)

    compute_grid_size = device.compute_with_storage_grid_size()

    interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG

    # NOTE: Mixed precision is supported as well; but might not have enough space for larger seq_len with BFLOAT16
    in0_dtype = ttnn.bfloat8_b
    in1_dtype = ttnn.bfloat8_b
    output_dtype = ttnn.bfloat8_b

    q_len = 1
    input_shape_a = [q_len, q_heads, batch, K]
    input_shape_b = [batch, kv_heads, K, seq_len]

    input_tensor_a = torch.randn(input_shape_a).bfloat16()
    input_tensor_b = torch.randn(input_shape_b).bfloat16()

    tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
    tt_input_tensor_b = ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)

    if in0_sharded:
        tt_input_tensor_a = ttnn.interleaved_to_sharded(
            tt_input_tensor_a,
            compute_grid_size,
            [q_len * batch, K],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if in1_sharded:
        tt_input_tensor_b = ttnn.interleaved_to_sharded(
            tt_input_tensor_b,
            compute_grid_size,
            [kv_heads * K, seq_len],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if output_sharded:
        output_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    else:
        output_mem_config = interleaved_mem_config

    tt_output_tensor_on_device = ttnn.experimental.group_attn_matmul(
        tt_input_tensor_a,
        tt_input_tensor_b,
        compute_with_storage_grid_size=compute_grid_size,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )
    if output_sharded:
        tt_output_tensor_on_device = ttnn.sharded_to_interleaved(tt_output_tensor_on_device, interleaved_mem_config)

    tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    input_tensor_a = input_tensor_a.to(torch.float)
    input_tensor_b = torch.repeat_interleave(input_tensor_b.to(torch.float), q_heads // kv_heads, dim=1)
    golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

    allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
    assert allclose, f"FAILED: {output}"


@pytest.mark.parametrize("in0_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize(
    "M, K, N, num_cores",
    [
        [32, 8192, 1152, 8],
    ],
)
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat8_b])
def test_sharded_matmul_1d_in0(
    device, in0_sharded, out_sharded, M, K, N, num_cores, activations_dtype, weights_dtype, function_level_defaults
):
    grid_size = (8, 1)

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]
    bias_shape = [1, 1, 1, N]

    interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    bias = torch.randn(bias_shape).bfloat16().float()

    in0_t = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)
    in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
    bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)[0]

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if in0_sharded:
        in0_t = ttnn.interleaved_to_sharded(
            in0_t,
            grid_size,
            [M, K // num_cores],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        in0_block_w=32,
        out_subblock_h=1,
        out_subblock_w=5,
        per_core_M=1,
        per_core_N=5,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        bias=bias_t,
        program_config=program_config,
        memory_config=output_mem_config,
        dtype=activations_dtype,
    )
    if out_sharded:
        output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    pt_out = in0 @ in1 + bias

    tt_out = tt2torch_tensor(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize(
    "M, K, N, num_cores",
    [
        [32, 8192, 65024, 32],
    ],
    ids=["lm_head_shape"],
)
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("in0_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
def test_sharded_matmul_1d_in0_multi_chip(
    pcie_devices,
    num_devices,
    in0_sharded,
    out_sharded,
    M,
    K,
    N,
    num_cores,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    if num_devices == 8:
        pytest.skip("Need tunnelling support to run on 8 devices!")

    grid_size = (8, 4)
    devices = pcie_devices[:num_devices]

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]

    interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in1_slices = torch.chunk(in1, num_devices, dim=-1)

    in0_t = []
    in1_t = []
    for i in range(num_devices):
        logger.info(f"Putting tensors on device: {i}")
        in0_temp = torch2tt_tensor(in0, devices[i], tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

        if in0_sharded:
            in0_temp = ttnn.interleaved_to_sharded(
                in0_temp,
                grid_size,
                [M, K // num_cores],
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
        in0_t.append(in0_temp)

        in1_t.append(
            torch2tt_tensor(in1_slices[i], devices[i], tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
        )

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if num_devices == 4:
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif num_devices == 8:
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    output_t = []
    for i in range(num_devices):
        logger.info(f"Running matmul on device: {i}")
        output_t.append(
            ttnn.matmul(
                in0_t[i],
                in1_t[i],
                program_config=program_config,
                memory_config=output_mem_config,
                dtype=activations_dtype,
            )
        )

    pt_out = in0 @ in1

    tt_out = torch.cat([tt2torch_tensor(out_t) for out_t in output_t], -1)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize("num_devices", [4, 8])
@pytest.mark.parametrize(
    "M, K, N, num_cores",
    [
        [32, 8192, 65024, 32],
    ],
    ids=["lm_head_shape"],
)
@pytest.mark.parametrize("out_sharded", [True], ids=["out_sharded"])
@pytest.mark.parametrize("in0_sharded", [True], ids=["in0_sharded"])
@pytest.mark.parametrize("weights_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("activations_dtype", [ttnn.bfloat8_b])
def test_sharded_matmul_1d_in0_multi_chip(
    all_devices,
    num_devices,
    in0_sharded,
    out_sharded,
    M,
    K,
    N,
    num_cores,
    activations_dtype,
    weights_dtype,
    function_level_defaults,
):
    grid_size = (8, 4)
    devices = get_devices_for_t3000(all_devices, num_devices)

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]

    interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG
    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in1_slices = torch.chunk(in1, num_devices, dim=-1)

    in0_t = []
    in1_t = []
    for i in range(num_devices):
        logger.info(f"Putting tensors on device: {i}")
        in0_temp = torch2tt_tensor(in0, devices[i], tt_memory_config=interleaved_mem_config, tt_dtype=activations_dtype)

        if in0_sharded:
            in0_temp = ttnn.interleaved_to_sharded(
                in0_temp,
                grid_size,
                [M, K // num_cores],
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
        in0_t.append(in0_temp)

        in1_t.append(
            torch2tt_tensor(in1_slices[i], devices[i], tt_memory_config=interleaved_mem_config, tt_dtype=weights_dtype)
        )

    output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config

    if num_devices == 4:
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=16,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif num_devices == 8:
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=8,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    output_t = []
    for i in range(num_devices):
        logger.info(f"Running matmul on device: {i}")
        output_t.append(
            ttnn.matmul(
                in0_t[i],
                in1_t[i],
                program_config=program_config,
                memory_config=output_mem_config,
                dtype=activations_dtype,
            )
        )

    pt_out = in0 @ in1

    tt_out = torch.cat([tt2torch_tensor(out_t) for out_t in output_t], -1)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b, ttnn.bfloat16),
    ids=["BFLOAT8_B", "BFLOAT16"],
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, num_q_heads, num_kv_heads, read_from_input_tensor_kv",
    (
        (32, 1, 64, 16, 1, False),
        (32, 1, 64, 16, 1, True),
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
    torch.manual_seed(1234)
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = num_kv_heads
<<<<<<< HEAD
    shard_grid = ttnn.CoreRangeSet(
        ttnn.experimental.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
=======
    shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        ttnn.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
>>>>>>> #11838: Update files
    )
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
        in0_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            in0_shard_spec,
        )
        in1_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            in1_shard_spec,
        )
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
        in0_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            in0_shard_spec,
        )
        in0_t = ttnn.Tensor(A_interleaved, dtype).to(ttnn.TILE_LAYOUT).to(device, in0_mem_config)

    out_shard_spec = in0_shard_spec
    out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        out_shard_spec,
    )
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
