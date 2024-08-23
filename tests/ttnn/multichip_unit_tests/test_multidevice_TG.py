# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from ttnn import (
    ShardTensorToMesh,
    ShardTensor2dMesh,
    ReplicateTensorToMesh,
    ConcatMeshToTensor,
    ConcatMesh2dToTensor,
    ListMeshToTensor,
    MeshToTensor,
)
from models.utility_functions import nearest_32


@pytest.mark.skip("1D device mesh not supported")
@pytest.mark.parametrize(
    "mesh_device",
    [
        32,
    ],
    indirect=True,
)
def test_galaxy_matmul_1d_fracture(mesh_device):
    act_pt = torch.randn(1, 1, 32, 8192)
    weights_pt = torch.randn(1, 1, 8192, 32768)
    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    weights = ttnn.from_torch(
        weights_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=3),
    )

    gt = act_pt @ weights_pt

    compute_kernel_attn = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out = ttnn.matmul(
        act,
        weights,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=4, x=8),
        compute_kernel_config=compute_kernel_attn,
    )
    out = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N, weights_dtype",
    [
        # llama3_1-70B
        pytest.param(32, 8192, 28 * 1024, ttnn.bfloat4_b, id="Llama3-70B_decode_FF1"),  # same shapes for FF1 and FF3
        pytest.param(32, 28 * 1024, 8192, ttnn.bfloat8_b, id="Llama3-70B_decode_FF2"),
        pytest.param(512, 8192, 28 * 1024, ttnn.bfloat4_b, id="Llama3-70B_prefill_seq512_FF1"),
        pytest.param(512, 28 * 1024, 8192, ttnn.bfloat8_b, id="Llama3-70B_prefill_seq512_FF2"),
        # llama3_1-405B
        pytest.param(
            32, 16 * 1024, 52 * 1024, ttnn.bfloat4_b, id="Llama3-405B_decode_FF1"
        ),  # same shapes for FF1 and FF3
        pytest.param(32, 52 * 1024, 16 * 1024, ttnn.bfloat8_b, id="Llama3-405B_decode_FF2"),
        pytest.param(128, 16 * 1024, 52 * 1024, ttnn.bfloat4_b, id="Llama3-405B_prefill_seq128_FF1"),
        pytest.param(128, 52 * 1024, 16 * 1024, ttnn.bfloat8_b, id="Llama3-405B_prefill_seq128_FF2"),
        pytest.param(256, 16 * 1024, 52 * 1024, ttnn.bfloat4_b, id="Llama3-405B_prefill_seq256_FF1"),
        pytest.param(256, 52 * 1024, 16 * 1024, ttnn.bfloat8_b, id="Llama3-405B_prefill_seq256_FF2"),
        # pytest.param(
        #     512, 16 * 1024, 52 * 1024, ttnn.bfloat4_b, id="Llama3-405B_prefill_seq512_FF1"
        # ),  # PCC check failed, PCC: -0.00014127559109112134, see issue 10936
        pytest.param(512, 52 * 1024, 16 * 1024, ttnn.bfloat8_b, id="Llama3-405B_prefill_seq512_FF2"),
    ],
)
# Llama FF1, FF2, FF3 in MLP with dram interleaved weights
def test_galaxy_matmul_2d_fracture(M, K, N, weights_dtype, mesh_shape, mesh_device):
    act_pt = torch.randn(1, 1, M, K)
    weights_pt = torch.randn(1, 1, K, N)

    # If K < N it's FF1, else FF2
    act_shard_dim = (None, 3) if K < N else (3, None)  # None means to replicate along this dim
    weight_shard_dim = (3, 2) if K < N else (2, 3)
    concat_dim = (3, 1) if K < N else (1, 3)  # dim 1 for reduce, dim 3 for concatenating fractures

    K = K // mesh_shape[1] if K < N else K // mesh_shape[0]
    N = N // mesh_shape[0] if K < N else N // mesh_shape[1]

    core_grid = ttnn.CoreGrid(y=1, x=8)
    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(M // core_grid.y, K // core_grid.x),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=act_mem_config if M == 32 else ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=act_shard_dim),
    )
    weights = ttnn.from_torch(
        weights_pt,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=weight_shard_dim),
    )

    gt = act_pt @ weights_pt

    compute_kernel_lofi = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    out = ttnn.matmul(
        act,
        weights,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_lofi,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if M == 32 else ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.to_torch(out, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=concat_dim))
    out = torch.sum(out, dim=1, keepdim=True)

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.skip("See GH #10673: DRAM-SHARDED Matmuls gives ND PCC on TG")
@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N, weights_dtype",
    [
        pytest.param(32, 8192, 32768, ttnn.bfloat4_b, id="Llama3-70B_decode_FF1"),
        pytest.param(32, 32768, 8192, ttnn.bfloat8_b, id="Llama3-70B_decode_FF2"),
    ],
)
# Llama FF1, FF2, FF3 in MLP with dram sharded weights
def test_galaxy_matmul_2d_fracture_dram_sharded(M, K, N, weights_dtype, mesh_shape, mesh_device):
    act_pt = torch.randn(1, 1, M, K)
    weights_pt = torch.randn(1, 1, K, N)

    gt = act_pt @ weights_pt

    act_shard_dim = (3, None) if K == 8192 else (None, 3)
    weight_shard_dim = (2, 3) if K == 8192 else (3, 2)
    concat_dim = (1, 3) if K == 8192 else (3, 1)

    K = K // mesh_shape[1] if K == 8192 else K // mesh_shape[0]
    N = N // mesh_shape[0] if N == 32768 else N // mesh_shape[1]

    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, K // 8),
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=act_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, dims=act_shard_dim, mesh_shape=mesh_shape),
    )

    compute_kernel_lofi = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(
                    mesh_device.get_device(0).dram_grid_size().x - 1, mesh_device.get_device(0).dram_grid_size().y - 1
                ),
            )
        }
    )
    shard_shape = (K, nearest_32(N // 12))  # padded cols to divide by 12
    shard_spec = ttnn.ShardSpec(weight_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    weight_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    weights = ttnn.from_torch(
        weights_pt,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=weight_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, dims=weight_shard_dim, mesh_shape=mesh_shape),
    )

    DRAM_SHARDED_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=K // 8 // 32,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        per_core_M=M // 32,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=N // 8 // 32,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fused_activation=None,
    )

    out = ttnn.matmul(
        act,
        weights,
        program_config=DRAM_SHARDED_PROGCFG,
        compute_kernel_config=compute_kernel_lofi,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )

    out = ttnn.to_torch(out, mesh_composer=ConcatMesh2dToTensor(mesh_device, dims=concat_dim, mesh_shape=mesh_shape))
    out = torch.sum(out, dim=1, keepdim=True)

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, N",
    [
        pytest.param(32, 28 * 1024, id="Llama3-70B_decode_FF1"),
        pytest.param(512, 28 * 1024, id="Llama3-70B_prefill_seq512_FF1"),
        pytest.param(32, 52 * 1024, id="Llama3-405B_decode_FF1"),
        pytest.param(256, 52 * 1024, id="Llama3-405B_prefill_seq256_FF1"),
        pytest.param(512, 52 * 1024, id="Llama3-405B_prefill_seq512_FF1"),
    ],
)
# Llama FF1 * FF3 in MLP
def test_galaxy_eltwise_mul_2d_fracture(M, N, mesh_shape, mesh_device):
    FF1_pt = torch.randn(1, 1, M, N)
    FF3_pt = torch.randn(1, 1, M, N)

    FF1 = ttnn.from_torch(
        FF1_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=mesh_shape),
    )

    FF3 = ttnn.from_torch(
        FF3_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ShardTensor2dMesh(mesh_device, dims=(3, None), mesh_shape=mesh_shape),
    )

    gt = FF1_pt * FF3_pt

    out = ttnn.mul(
        FF1,
        FF3,
        dtype=ttnn.bfloat16,
    )

    out = ttnn.to_torch(out, mesh_composer=ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=mesh_shape))
    out = out[:, 0:1, :, :]  # select the first column

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99999)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "M, N",
    [
        pytest.param(32, 8192, id="Llama3-70B_decode"),
        pytest.param(512, 8192, id="Llama3-70B_prefill_seq512"),
        pytest.param(32, 16 * 1024, id="Llama3-400B_decode"),
        pytest.param(256, 16 * 1024, id="Llama3-400B_prefill_seq256"),
        # pytest.param(512, 16 * 1024, id="Llama3-400B_prefill_seq512"),  # Skipped, OOM
    ],
)
# Llama residual add
def test_galaxy_eltwise_add(M, N, mesh_device):
    residual_pt = torch.randn(1, 1, M, N)
    attn_output_pt = torch.randn(1, 1, M, N)

    shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 3),
            ),
        }
    )

    LN_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                M,
                N // 32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    residual = ttnn.from_torch(
        residual_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=LN_OUTPUT_MEMCFG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    attn_output = ttnn.from_torch(
        attn_output_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=LN_OUTPUT_MEMCFG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    gt = residual_pt + attn_output_pt

    out = ttnn.add(
        residual,
        attn_output,
        dtype=ttnn.bfloat16,
        memory_config=LN_OUTPUT_MEMCFG,
    )

    out = ttnn.to_torch(out, mesh_composer=ListMeshToTensor(mesh_device))[0]

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99999)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, N, head_dim, num_heads",
    [
        (32, 8192, 128, 80),  # Llama3-70B decode attn fused_qkv
        (32, 8192, 128, 64),  # Llama3-70B decode attn selfout
        # (32, 16 * 1024, 128, 144),  # Llama3-405B decode attn fused_qkv
        # (32, 16 * 1024, 128, 128),  # Llama3-405B decode attn selfout
    ],
    ids=[
        "Llama3-70B-decode-attn-fused_qkv",
        "Llama3-70B-decode-attn-selfout",
        # "Llama3-405B-decode-attn-fused_qkv",  # Skipping because of CI failure
        # "Llama3-405B-decode-attn-selfout",  # Skipping because of CI failure
    ],
)
# Llama attention matmuls
def test_galaxy_attn_matmul(M, N, head_dim, num_heads, mesh_shape, mesh_device):
    act_pt = torch.randn(1, 1, M, N)
    weights_pt = torch.randn(1, 1, N, head_dim * num_heads)

    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    weights = ttnn.from_torch(
        weights_pt,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=mesh_shape),
    )

    compute_kernel_attn = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    if num_heads == 80:
        core_grid = ttnn.CoreGrid(y=5, x=8)
    elif num_heads == 64:
        core_grid = ttnn.CoreGrid(y=4, x=8)
    elif num_heads == 144:
        core_grid = ttnn.CoreGrid(y=6, x=8)
    elif num_heads == 128:
        core_grid = ttnn.CoreGrid(y=8, x=8)
    else:
        assert False

    out = ttnn.matmul(
        act,
        weights,
        dtype=ttnn.bfloat16,
        core_grid=core_grid,
        compute_kernel_config=compute_kernel_attn,
    )

    gt = act_pt @ weights_pt

    out = ttnn.to_torch(out, mesh_composer=ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=mesh_shape))
    out = out[:, 0:1, :, :]  # select the first column

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


def num_to_corerange(total_max_cores):
    if total_max_cores == 1:
        return ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    assert total_max_cores < 8 or (total_max_cores % 8 == 0 and total_max_cores <= 80)  # TG has 10x8 grids
    num_x = min(total_max_cores, 8)
    num_y = total_max_cores // num_x
    assert num_x * num_y == total_max_cores
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((8, 4), id="8x4_grid"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, n_local_heads, n_local_kv_heads, is_multicore",
    [
        (8, 1, 128, 8, 1, True),  # Llama3-70B decode multicore
        # (8, 1, 128, 16, 1, True),  # Llama3-405B decode multicore
        (8, 1, 128, 16, 1, False),  # Llama3-405B decode single core
    ],
    ids=[
        "Llama3-70B-decode",
        # "Llama3-405B-decode" # Not enough cores to run one tile per core: RuntimeError: TT_FATAL @ ../tt_metal/impl/allocator/allocator.cpp:123: num_shards.value() <= num_compute_banks
        "Llama3-405B-decode-singlecore-sharded",
    ],
)
# Llama nlp_create_heads for decode
# 8 attention groups are fractured over 8 devices along the column
# users are fractured over 4 devices along the rows (batch of 8)
# Note: interleaved inputs are not supported
def test_galaxy_nlp_create_heads_decode(
    batch, seq_len, head_dim, n_local_heads, n_local_kv_heads, is_multicore, mesh_device
):
    total_heads = n_local_heads + n_local_kv_heads * 2
    qkv_heads_pt = torch.rand(1, seq_len, batch, head_dim * total_heads)
    total_max_cores = total_heads * head_dim // 32 if is_multicore else 1  # 40 for llama3-70B; 72 for llama3-405B

    shard_spec_n_cores_grid = ttnn.CoreRangeSet(
        {num_to_corerange(total_max_cores)}
    )  # for 40 cores it's 0,0 - 7,4; for 72 cores it's 0,0 - 7,8

    CREATE_HEAD_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_n_cores_grid,
            [
                32,
                32 if is_multicore else total_heads * head_dim,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    qkv_heads = ttnn.from_torch(
        qkv_heads_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=CREATE_HEAD_INPUT_MEMCFG,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    # tt operation
    (
        q_heads_tt,  # [seqlen, bsz, padded_n_local_heads, head_dim]
        k_heads_tt,  # [seqlen, bsz, padded_n_local_kv_heads, head_dim]
        v_heads_tt,  # [seqlen, bsz, padded_n_local_kv_heads, head_dim]
    ) = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv_heads,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    logger.info(f"q_heads_tt: {q_heads_tt.memory_config()}")
    logger.info(f"k_heads_tt: {k_heads_tt.memory_config()}")
    logger.info(f"v_heads_tt: {v_heads_tt.memory_config()}")

    q_heads_pt = qkv_heads_pt[:, :, :, : head_dim * n_local_heads].view(seq_len, batch, n_local_heads, head_dim)
    k_heads_pt = qkv_heads_pt[:, :, :, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)].view(
        seq_len, batch, n_local_kv_heads, head_dim
    )
    v_heads_pt = qkv_heads_pt[:, :, :, head_dim * (n_local_heads + n_local_kv_heads) :].view(
        seq_len, batch, n_local_kv_heads, head_dim
    )

    # compare
    q_heads_tt_cpu = ttnn.to_torch(q_heads_tt, mesh_composer=ListMeshToTensor(mesh_device))[0][..., :n_local_heads, :]
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_pt, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = ttnn.to_torch(k_heads_tt, mesh_composer=ListMeshToTensor(mesh_device))[0][
        ..., :n_local_kv_heads, :
    ]
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_pt, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = ttnn.to_torch(v_heads_tt, mesh_composer=ListMeshToTensor(mesh_device))[0][
        ..., :n_local_kv_heads, :
    ]
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_pt, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, n_local_heads, n_local_kv_heads",
    [
        (8, 1, 128, 8, 1),  # Llama3-70B decode attn
        (8, 1, 128, 16, 1),  # Llama3-405B decode attn
    ],
    ids=["Llama3-70B-decode", "Llama3-405B-decode"],
)
# Llama rotary matmul (decode only)
def test_galaxy_rotary_matmul(batch, seq_len, head_dim, n_local_heads, n_local_kv_heads, mesh_device):
    q_heads_pt = torch.rand(
        seq_len, batch, max(n_local_heads, 32), head_dim
    )  # Unpad batch=32 to 8 for each column group
    k_heads_pt = torch.rand(seq_len, batch, max(n_local_kv_heads, 32), head_dim)
    rot_mat_pt = torch.rand(1, batch, head_dim, head_dim)

    shard_spec_n_cores_grid = ttnn.CoreRangeSet({num_to_corerange(batch)})
    ROTARY_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_n_cores_grid,
            [
                32,
                head_dim,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    ROT_MAT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_n_cores_grid,
            [
                head_dim,
                head_dim,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    query_layer = ttnn.from_torch(
        q_heads_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ROTARY_INPUT_MEMCFG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    key_layer = ttnn.from_torch(
        k_heads_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ROTARY_INPUT_MEMCFG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    rot_mats = ttnn.from_torch(
        rot_mat_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ROT_MAT_MEMCFG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    compute_kernel_rotary = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    ROT_MAT_MM_PROGCFG = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=1,
        per_core_N=4,
    )

    query_layer = ttnn.matmul(
        query_layer,
        rot_mats,
        program_config=ROT_MAT_MM_PROGCFG,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_rotary,
    )

    key_layer = ttnn.matmul(
        key_layer,
        rot_mats,
        program_config=ROT_MAT_MM_PROGCFG,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_rotary,
    )

    query_layer_gt = q_heads_pt @ rot_mat_pt
    key_layer_gt = k_heads_pt @ rot_mat_pt

    query_layer_cpu = ttnn.to_torch(query_layer, mesh_composer=ListMeshToTensor(mesh_device))[0]
    key_layer_cpu = ttnn.to_torch(key_layer, mesh_composer=ListMeshToTensor(mesh_device))[0]

    out_pass_q, out_pcc_q = comp_pcc(query_layer_cpu, query_layer_gt, pcc=0.999)
    logger.info(f"PCC value: {out_pcc_q}")
    out_pass_k, out_pcc_k = comp_pcc(key_layer_cpu, key_layer_gt, pcc=0.999)
    logger.info(f"PCC value: {out_pcc_k}")

    assert out_pass_q and out_pass_k


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [8])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.DataType.BFLOAT16])
class TestUpdateCache:
    @pytest.mark.parametrize("seq_len", [128, 2048])
    def test_fill_cache(
        self, seq_len, head_dim, max_seq_len, num_users, num_heads, input_dtype, mesh_device, use_program_cache
    ):
        cache_dtype = input_dtype
        input_shape = [1, num_heads, seq_len, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()

        cachett = ttnn.from_torch(
            cache,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        for i in range(num_users):
            x = torch.randn(input_shape).bfloat16().float()

            xt = x

            compute_grid_size = mesh_device.get_device(0).compute_with_storage_grid_size()
            num_cores = min(seq_len // 32 * num_heads, 32)  # Always use max 32 cores for testing
            mesh_shape = ttnn.CoreRangeSet(ttnn.num_cores_to_corerange_set(num_cores, compute_grid_size, True))
            input_shard_spec = ttnn.ShardSpec(
                mesh_shape,
                [
                    xt.numel() // xt.shape[-1] // num_cores,
                    xt.shape[-1],
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            )
            input_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
            )

            xt = ttnn.from_torch(
                xt,
                dtype=input_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=input_mem_config,
                mesh_mapper=ReplicateTensorToMesh(mesh_device),
            )

            cachett = ttnn.fill_cache(cachett, xt, i)
            cache[i : i + 1, :, : x.shape[-2], :] = x

        tt_got_back = ttnn.to_torch(cachett, mesh_composer=ListMeshToTensor(mesh_device))[0]
        eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq

    @pytest.mark.parametrize("cache_idx", [0, 2047])
    @pytest.mark.parametrize("cache_dtype", [ttnn.DataType.BFLOAT8_B])
    @pytest.mark.parametrize("batch_offset", [0])  # Only used when num_users < 32 and batch_offset + num_users <= 32
    def test_update_cache_decode(
        self,
        cache_idx,
        head_dim,
        max_seq_len,
        num_users,
        batch_offset,
        num_heads,
        input_dtype,
        cache_dtype,
        mesh_device,
        use_program_cache,
    ):
        if num_users > 32 or (num_users + batch_offset) > 32:
            pytest.skip("Batch offset is only used when num_users < 32 and batch_offset + num_users <= 32")
        input_shape = [num_users, num_heads, 1, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]

        cache = torch.randn(cache_shape).bfloat16().float()

        cachett = ttnn.from_torch(
            cache,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        x = torch.randn(input_shape).bfloat16().float()
        # pad dim0 of x to 32 if batch size is less than 32, make 0-batch_offset elements 0, batch_offset-batch_offset+num_users elements non-zero, and rest 0
        x_new = x.clone()
        if num_users < 32:
            x_new = torch.cat((torch.zeros(batch_offset, num_heads, 1, head_dim), x_new), dim=0)
            x_new = torch.cat((x_new, torch.zeros(32 - num_users - batch_offset, num_heads, 1, head_dim)), dim=0)
            assert x_new.shape[0] == 32, f"Expected x.shape[0] to be 32, got {x_new.shape[0]}"
        xt = x_new.permute(2, 1, 0, 3)
        compute_grid_size = mesh_device.get_device(0).compute_with_storage_grid_size()
        num_cores = min(max(num_users, 32) // 32 * num_heads, compute_grid_size.x * compute_grid_size.y)
        mesh_shape = ttnn.CoreRangeSet(ttnn.num_cores_to_corerange_set(num_cores, compute_grid_size, True))
        input_shard_spec = ttnn.ShardSpec(
            mesh_shape,
            [
                xt.numel() // xt.shape[-1] // num_cores,
                xt.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            input_shard_spec,
        )

        xt = ttnn.from_torch(
            xt,
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=input_mem_config,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        cachett = ttnn.update_cache(cachett, xt, cache_idx, batch_offset=batch_offset)
        cache[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]] = x

        tt_got_back = ttnn.to_torch(cachett, mesh_composer=ListMeshToTensor(mesh_device))[0]

        eq_cache, output_cache = comp_pcc(cache, tt_got_back)  # checks the entire kv cache
        eq_update, output_update = comp_pcc(
            x, tt_got_back[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]]
        )  # checks the updated parts
        logger.info(output_cache)
        logger.info(output_update)
        assert eq_cache and eq_update


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power


def get_chunk_size(s):
    if s <= 32:
        return 32
    if s <= 64:
        return 32
    if s <= 128:
        return 32
    if s <= 256:
        return 256
    if s <= 2048:
        return 512
    return 512


def run_test_sdpa_decode_single_iter(
    mesh_device,
    b,
    nh,
    nkv,
    s,
    d,
    dtype,
    grid_size,
    q_dtype=ttnn.bfloat16,
    mask_dtype=ttnn.bfloat16,
    sharded_in=False,
    sharded_out=False,
):
    compute_grid_size = mesh_device.get_device(0).compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need {grid_size} grid size to run this test but core grid is {compute_grid_size}")

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    torch.manual_seed(1234)

    num_parallel_cores = grid_size[0] * grid_size[1] // b
    if num_parallel_cores == 1:
        min_pcc = 0.90
    else:
        min_pcc = 0.99
        if q_dtype == ttnn.DataType.BFLOAT8_B:
            min_pcc = 0.98
        min_pcc = 0.97 if dtype == ttnn.DataType.BFLOAT4_B else min_pcc

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    mesh_shape = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.ShardSpec(mesh_shape, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR, False)

    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    K = torch.randn(nkv, b, s, d)
    V = torch.randn(nkv, b, s, d)

    tt_K = ttnn.from_torch(
        K,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    tt_V = ttnn.from_torch(
        V,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )
    start_idx = s // 2
    scale = d**-0.5

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=0,  # Unused
        k_chunk_size=0,  # Unused
    )

    k_chunk_size = get_chunk_size(start_idx + 1)
    padded_layer_len = nearest_n(start_idx + 1, n=k_chunk_size)

    # Test various sequence lengths
    logger.debug(f"Testing with sequence length: {start_idx}")
    logger.debug(f"Using chunk size: {k_chunk_size}")
    logger.debug(f"Using padded layer length: {padded_layer_len}")
    logger.debug(f"Using padded num heads: {padded_num_heads}")

    attn_mask = torch.zeros((1, b, padded_num_heads, padded_layer_len))
    # Assume all users are at same position
    attn_mask[:, :, :, start_idx:] = torch.finfo(torch.float32).min

    Q = torch.randn(1, b, padded_num_heads, d)

    tt_Q = ttnn.from_torch(
        Q,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    tt_back = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        [start_idx for _ in range(b)],
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=height_sharded_memcfg if sharded_out else dram_memcfg,
    )

    tt_back = ttnn.to_torch(tt_back, mesh_composer=ListMeshToTensor(mesh_device))[0]
    tt_back = tt_back[:, :, :nh, :]

    Q_slice = Q[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, d
    K_slice = K[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
    V_slice = V[:, :, :padded_layer_len, :].permute(1, 0, 2, 3)  # nh, b, S, d
    attn_mask_slice = attn_mask[:, :, :nh, :].permute(1, 2, 0, 3)  # b, nh, 1, S
    expect = torch.nn.functional.scaled_dot_product_attention(
        Q_slice, K_slice, V_slice, attn_mask_slice, scale=scale, is_causal=False
    )  # b, nh, 1, d
    expect = expect.squeeze().unsqueeze(0)

    out_pass, out_pcc = comp_pcc(expect, tt_back, min_pcc)

    logger.debug(f"python vs pytorch: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "dtype, q_dtype, mask_dtype",
    [
        [
            ttnn.DataType.BFLOAT16,
            ttnn.DataType.BFLOAT16,
            ttnn.DataType.BFLOAT16,
        ],
    ],
    ids=[
        "all_bfp16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    (
        [8, 8, 1, 32768, 128, (8, 8)],  # Llama3-70B
        [8, 16, 1, 32768, 128, (8, 8)],  # Llama3-405B
    ),
    ids=["Llama3-70B-decode", "Llama3-405B-decode"],
)
def test_sdpa_decode_sharded(mesh_device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, mask_dtype):
    run_test_sdpa_decode_single_iter(
        mesh_device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, mask_dtype, sharded_in=True, sharded_out=False
    )
    run_test_sdpa_decode_single_iter(
        mesh_device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, mask_dtype, sharded_in=True, sharded_out=True
    )


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, n_local_heads, n_local_kv_heads, padded_local_heads",
    [
        (8, 1, 128, 8, 1, 32),  # Llama3-70B decode
        (8, 1, 128, 16, 1, 32),  # Llama3-405B decode
    ],
    ids=["Llama3-70B-decode", "Llama3-405B-decode"],
)
def test_galaxy_nlp_concat_heads_decode(
    batch, seq_len, head_dim, n_local_heads, n_local_kv_heads, padded_local_heads, mesh_device
):
    concat_head_input = torch.rand(seq_len, batch, padded_local_heads, head_dim)

    mesh_shape = ttnn.CoreRangeSet({num_to_corerange(batch)})
    CONCAT_HEADS_INPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            mesh_shape,
            [
                padded_local_heads,  # Each core has padded_local_heads
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    concat_head_input_tt = ttnn.from_torch(
        concat_head_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=CONCAT_HEADS_INPUT_MEMCFG,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    concat_head_output = ttnn.experimental.nlp_concat_heads_decode(
        concat_head_input_tt,
        num_heads=n_local_heads,
    )  # (seqlen, 1, batch, hidden_size)

    logger.info(f"concat_head_output: {concat_head_output.memory_config()}")

    # Input: (1, 8, 32(8), 128)
    # Output: (1, 1, 8, 1024)
    concat_head_output_pt = concat_head_input[:, :, :n_local_heads].reshape(1, 1, batch, head_dim * n_local_heads)

    # Compare
    concat_head_output_tt_cpu = ttnn.to_torch(concat_head_output, mesh_composer=ListMeshToTensor(mesh_device))[0]
    concat_head_output_tt_unpadded = concat_head_output_tt_cpu[:, :, :batch, :]
    out_pass, output_pcc = comp_pcc(concat_head_output_tt_unpadded, concat_head_output_pt, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc}")

    assert out_pass


def rmsnorm(x, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma + beta


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "M, N",
    [
        (32, 8192),  # Llama3-70B decode
        (32, 16 * 1024),  # Llama3-405B decode
    ],
    ids=["Llama3-70B-decode", "Llama3-405B-decode"],
)
def test_galaxy_layernorm(M, N, mesh_device):
    layernorm_input = torch.rand(1, 1, M, N) * 2 - 0.95
    norm_weights = torch.rand(1, 1, N // 32, 32) * 2 - 1
    norm_eps = 1e-05

    num_cores = 32
    shard_spec_32_cores_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 3),
            ),
        }
    )

    LN_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_spec_32_cores_grid,
            [
                M,
                N // 32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    LN_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    LN_PROGCFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=M // 32,
        block_w=N // num_cores // 32,
        inplace=True,
    )

    layernorm_input_tt = ttnn.from_torch(
        layernorm_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=LN_OUTPUT_MEMCFG,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    norm_weights_tt = ttnn.from_torch(
        norm_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    norm_output = ttnn.rms_norm(
        layernorm_input_tt,
        epsilon=norm_eps,
        weight=norm_weights_tt,
        program_config=LN_PROGCFG,
        memory_config=LN_OUTPUT_MEMCFG,
        compute_kernel_config=LN_COMPUTE_KERNEL_CONFIG,
    )

    # Compare
    beta = torch.zeros(1, 1, N // 32, 32)
    norm_output_tt_cpu = ttnn.to_torch(norm_output, mesh_composer=ListMeshToTensor(mesh_device))[0]
    ref_rmsnorm = rmsnorm(layernorm_input, norm_weights.flatten(), beta.flatten(), norm_eps)

    out_pass, output_pcc = comp_pcc(norm_output_tt_cpu, ref_rmsnorm, pcc=0.999)
    logger.info(f"PCC value: {output_pcc}")

    assert out_pass


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_device_submesh(mesh_device):
    rows, cols, tile_size = 8, 4, 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)
    for index, device_tensor in enumerate(device_tensors):
        row_idx = index // cols
        col_idx = index % cols

        device_tensor_torch = ttnn.to_torch(device_tensor)
        row_start, row_end = row_idx * tile_size, (row_idx + 1) * tile_size
        col_start, col_end = col_idx * tile_size, (col_idx + 1) * tile_size
        assert torch.all(device_tensor_torch == full_tensor[0, 0, row_start:row_end, col_start:col_end])


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 4), id="1x4_grid")], indirect=True)
def test_device_line_all_gather_1x4(mesh_device):
    rows, cols, tile_size = 1, 4, 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn_tensor = ttnn.line_all_gather(ttnn_tensor, dim=3, num_links=1)

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)
    for index, device_tensor in enumerate(device_tensors):
        device_tensor_torch = ttnn.to_torch(device_tensor)
        print(device_tensor_torch)
        assert torch.all(device_tensor_torch == full_tensor)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 1), id="8x1_grid")], indirect=True)
def test_device_line_all_gather_8x1(mesh_device):
    rows, cols, tile_size = 8, 1, 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn_tensor = ttnn.line_all_gather(ttnn_tensor, dim=2, cluster_axis=0, mesh_device=mesh_device, num_links=1)

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)
    for index, device_tensor in enumerate(device_tensors):
        device_tensor_torch = ttnn.to_torch(device_tensor)
        assert torch.all(device_tensor_torch == full_tensor)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", (0, 1))
@pytest.mark.parametrize("dim", (2, 3))
def test_device_line_all_gather_8x4_data(mesh_device, cluster_axis: int, dim: int):
    """
    Test the line-all-gather operation on a 8x4 mesh.

    Data Pattern for initial sharding [TILE_SIZE*mesh_device_ROWS, TILE_SIZE*mesh_device_COLS]:
    [1, 1, 1, 1, 1, 1, 1, 1]
    [2, 2, 2, 2, 2, 2, 2, 2]
    [3, 3, 3, 3, 3, 3, 3, 3]
    [...]
    [8, 8, 8, 8, 8, 8, 8, 8]

    Data-pattern per-device before line-all-gather:
    - Each device receives a shard of the tensor with shape: [1, 1, 32, 32]

    Expected data-pattern per-device after line-all-gather:
    - Every device along the column contains the whole column tensor
    - output: [[1],[2],[3],[4],[5],[6],[7],[8]], shape: [1, 1, TILE_SIZE * mesh_device_ROWS, 32]
    """

    (rows, cols), tile_size = mesh_device.shape, 32
    full_tensor = torch.zeros((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)

    for i in range(rows):
        full_tensor[0, 0, i * tile_size : (i + 1) * tile_size, :] = torch.full((tile_size, tile_size * cols), i + 1.0)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn_tensor = ttnn.line_all_gather(
        ttnn_tensor, dim=dim, cluster_axis=cluster_axis, mesh_device=mesh_device, num_links=1
    )

    device_tensors: typing.List[ttnn.Tensor] = ttnn.get_device_tensors(ttnn_tensor)

    # device iteration happens in logical row-major
    for index, device_tensor in enumerate(device_tensors):
        row_index = index // cols
        col_index = index % cols
        device_tensor_torch = ttnn.to_torch(device_tensor)

        if dim == 2:
            if cluster_axis == 0:  # cluster along rows
                expected = full_tensor[..., :tile_size]
            else:  # cluster along columns
                expected = full_tensor[..., row_index * tile_size : (row_index + 1) * tile_size, :tile_size].repeat(
                    1, 1, cols, 1
                )
        elif dim == 3:
            if cluster_axis == 0:  # cluster along rows
                expected = full_tensor[..., :tile_size]
                expected = torch.permute(expected, (0, 1, 3, 2))
            else:  # cluster along columns
                expected = full_tensor[..., row_index * tile_size : (row_index + 1) * tile_size, :]

        assert torch.allclose(device_tensor_torch, expected, atol=1e-3)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_visualize_mesh_device(mesh_device):
    ttnn.visualize_mesh_device(mesh_device)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_concat_mesh_device_2d(mesh_device):
    (rows, cols), tile_size = mesh_device.shape, 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    read_back_tensor = ttnn.to_torch(
        ttnn_tensor, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    )
    assert torch.allclose(read_back_tensor, full_tensor, atol=1e-3)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "dims",
    [
        pytest.param((0, 1), id="shard_batch_and_channel"),
        pytest.param((2, 3), id="shard_height_and_width"),  # TODO(jchu):per device shards can be less than 32?
    ],
)
def test_shard_and_concat_2d_various_dims(mesh_device, dims):
    rows, cols = mesh_device.shape
    batch, channels, height, width = 16, 64, 128, 128
    full_tensor = torch.rand((batch, channels, height, width), dtype=torch.float32)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims)
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)

    read_back_tensor = ttnn.to_torch(
        ttnn_tensor, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=dims)
    )

    assert torch.allclose(read_back_tensor, full_tensor, atol=1e-6)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "tensor_shape",
    [
        pytest.param((32, 64, 128, 128), id="standard_shape"),
        pytest.param((1, 1, 256, 256), id="single_batch_channel"),
        pytest.param((64, 3, 224, 224), id="imagenet_like"),
    ],
)
def test_shard_and_concat_2d_various_shapes(mesh_device, tensor_shape):
    rows, cols = mesh_device.shape
    full_tensor = torch.rand(tensor_shape, dtype=torch.float32)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, 3))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)

    read_back_tensor = ttnn.to_torch(
        ttnn_tensor, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(2, 3))
    )

    assert torch.allclose(read_back_tensor, full_tensor, atol=1e-6)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_shard_and_concat_2d_non_divisible(mesh_device):
    rows, cols = mesh_device.shape
    # Create a tensor with dimensions not perfectly divisible by the mesh shape
    full_tensor = torch.rand((30, 62, 130, 126), dtype=torch.float32)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, 3))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)

    read_back_tensor = ttnn.to_torch(
        ttnn_tensor, mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(2, 3))
    )

    assert torch.allclose(read_back_tensor, full_tensor, atol=1e-6)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 1), id="8x1_grid")], indirect=True)
def test_line_all_gather_column_major(mesh_device):
    """
    The input tensor is size [1, 1, 32, 32*8] and it will get sharded onto an 8-row (8x1) device-mesh as follows:

    [SHARD0]
    [SHARD1]
    [SHARD2]
    ...
    [SHARD7]

    This exercises the sharding onto a column of devices (note the reversed `dims` argument) whereas the default
    behaviour is to map onto the device-mesh in a row-major fashion. We will issue a line-all-gather along the vertical
    axis (cluster_axis=0) and then gather on the width dimension of the tensor `dim=3`.

    Each of the 8 devices will have a result output tensor of size [1, 1, 32, 32*8]
    """
    rows, cols, tile_size = 8, 1, 32
    full_tensor = torch.rand((1, 1, tile_size, tile_size * rows), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(-1, -2))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)
    ttnn_tensor = ttnn.line_all_gather(ttnn_tensor, dim=3, cluster_axis=0, mesh_device=mesh_device, num_links=1)
    tt_outputs = ttnn.to_torch(ttnn_tensor, mesh_composer=ListMeshToTensor(mesh_device))
    for output in tt_outputs[1:]:
        assert output.shape == (1, 1, 32, 32 * 8)
        assert torch.allclose(output, tt_outputs[0])


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("cluster_axis", (1,))
@pytest.mark.parametrize("dim", (0,))
@pytest.mark.parametrize("async_mode", (False, True))
def test_device_line_all_gather_8x4_data(mesh_device, cluster_axis: int, dim: int, async_mode: bool):
    """
    Test the line-all-gather operation on a 8x4 mesh.
    Data Pattern for initial sharding [TILE_SIZE*mesh_device_ROWS, TILE_SIZE*mesh_device_COLS]:
    Input:
        [1, 1, 1, 1]
        [2, 2, 2, 2]
        [3, 3, 3, 3]
        [...]
        [8, 8, 8, 8]
    Data-pattern per-device before line-all-gather:
    - Each device receives a shard of the tensor with shape: [1, 1, 32, 32]
    Expected data-pattern per-device after line-all-gather:
    - Every device along the column contains the whole column tensor stacked on `dim` dimension
    - Every device will have the shape: [4, 1, 32, 32]
    """
    if async_mode:
        for i in mesh_device.get_device_ids():
            device = mesh_device.get_device(i)
            device.enable_async(True)

    (rows, cols), tile_size = mesh_device.shape, 32
    full_tensor = torch.zeros((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)
    for i in range(rows):
        full_tensor[0, 0, i * tile_size : (i + 1) * tile_size, :] = torch.full((tile_size, tile_size * cols), i + 1.0)
    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn_tensor = ttnn.line_all_gather(
        ttnn_tensor, dim=dim, cluster_axis=cluster_axis, mesh_device=mesh_device, num_links=1
    )


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_visualize_mesh_device_with_tensor_row_major(mesh_device):
    rows, cols, tile_size = 4, 4, 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)


@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_visualize_mesh_device_with_tensor_col_major(mesh_device):
    rows, cols, tile_size = 8, 2, 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        full_tensor, mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device)
    ttnn.visualize_mesh_device(mesh_device, tensor=ttnn_tensor)
