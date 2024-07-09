# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
import ttnn.experimental
import ttnn.experimental.operations
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from ttnn import (
    ShardTensorToMesh,
    ReplicateTensorToMesh,
    ConcatMeshToTensor,
    ListMeshToTensor,
    TensorToMesh,
    MeshToTensor,
)


@pytest.mark.parametrize(
    "device_mesh",
    [
        32,
    ],
    indirect=True,
)
def test_galaxy_matmul_1d_fracture(device_mesh):
    act_pt = torch.randn(1, 1, 32, 8192)
    weights_pt = torch.randn(1, 1, 8192, 32768)
    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    weights = ttnn.from_torch(
        weights_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensorToMesh(device_mesh, dim=3),
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
    out = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(device_mesh, dim=3))

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
    print(out_pcc)
    assert out_pass


class ShardTensor2dMesh(TensorToMesh):
    def __init__(self, device_mesh, dims, cluster_shape):
        super().__init__(device_mesh)
        self.dims = dims
        self.cluster_shape = cluster_shape

    def map(self, tensor: torch.tensor):
        # Returns list of tensors to map to row-major ordering of chips in cluster
        tensors_grid_y = None
        if self.dims[1] == None:
            tensors_grid_y = [tensor.clone() for _ in range(self.cluster_shape[1])]
        else:
            tensors_grid_y = torch.chunk(tensor, self.cluster_shape[1], dim=self.dims[1])

        tensors_grid_all = None
        if self.dims[0] == None:
            tensors_grid_all = [t.clone() for t in tensors_grid_y for _ in range(self.cluster_shape[0])]
        else:
            tensors_grid_all = [
                tt for t in tensors_grid_y for tt in torch.chunk(t, self.cluster_shape[0], dim=self.dims[0])
            ]

        return list(tensors_grid_all)

    def config(self):
        return {
            "strategy": "shard",
            "shard_dim": f"{self.dims[0] if self.dims[0] else self.dims[1]}",
        }


class ConcatMesh2DToTensor(MeshToTensor):
    def __init__(self, device_mesh, dims, cluster_shape):
        self.dims = dims
        self.cluster_shape = cluster_shape
        self.device_mesh = device_mesh

    def compose(self, tensor: ttnn.Tensor) -> torch.Tensor:
        tt_shards = [ttnn.to_torch(tt_input_tensor) for tt_input_tensor in ttnn.get_device_tensors(tensor)]

        row_concat = []
        for cluster_row in range(self.cluster_shape[1]):
            start = cluster_row * self.cluster_shape[0]
            end = start + self.cluster_shape[0]
            row_concat.append(torch.cat(tt_shards[start:end], dim=self.dims[0]))
        all_concat = torch.cat(row_concat, dim=self.dims[1])
        return all_concat


@pytest.mark.parametrize(
    "cluster_shape",
    [
        (4, 8),
        # (8, 4), # cluster shape should always be the same as the device mesh grid shape
    ],
)
@pytest.mark.parametrize(
    "device_mesh",
    [
        (4, 8),
        # (8, 4), # cluster shape should always be the same as the device mesh grid shape
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M,K,N",
    [
        (32, 8192, 32768),  # Llama3-70B decode FF1
        (32, 32768, 8192),  # Llama3-70B decode FF2
        (512, 8192, 32768),  # Llama3-70B prefill FF1
        (512, 32768, 8192),  # Llama3-70B prefill FF2
        (32, 16 * 1024, 64 * 1024),  # Llama3-400B decode FF1
        (32, 64 * 1024, 16 * 1024),  # Llama3-400B decode FF2
        # (512, 16*1024, 64*1024),# Llama3-400B prefill FF1 # Skipped, OOM
        (512, 64 * 1024, 16 * 1024),  # Llama3-400B prefill FF2
        (32, 8192, 1280),  # Llama3-70B decode QKV
    ],
)
def test_galaxy_matmul_2d_fracture(M, K, N, cluster_shape, device_mesh):
    act_pt = torch.randn(1, 1, M, K)
    weights_pt = torch.randn(1, 1, K, N)

    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensor2dMesh(device_mesh, dims=(3, None), cluster_shape=cluster_shape),
    )
    weights = ttnn.from_torch(
        weights_pt,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensor2dMesh(device_mesh, dims=(2, 3), cluster_shape=cluster_shape),
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
        use_1d_systolic_array=True if M == 32 else False,
        compute_kernel_config=compute_kernel_attn,
    )

    out = ttnn.to_torch(out, mesh_composer=ConcatMesh2DToTensor(device_mesh, dims=(1, 3), cluster_shape=cluster_shape))
    out = torch.sum(out, dim=1, keepdim=True)

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


@pytest.mark.parametrize(
    "cluster_shape",
    [
        (4, 8),
        # (8, 4), # cluster shape should always be the same as the device mesh grid shape
    ],
)
@pytest.mark.parametrize(
    "device_mesh",
    [
        (4, 8),
        # (8, 4), # cluster shape should always be the same as the device mesh grid shape
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, N",
    [
        (32, 32768),  # Llama3-70B decode FF1
        (512, 32768),  # Llama3-70B prefill FF1
        # (32, 64 * 1024),  # Llama3-400B decode FF1
        # (512, 64*1024),# Llama3-400B prefill FF1 # Skipped, OOM
    ],
)
def test_galaxy_eltwise_mul_2d_fracture(M, N, cluster_shape, device_mesh):
    FF1_pt = torch.randn(1, 1, M, N)
    FF3_pt = torch.randn(1, 1, M, N)

    FF1 = ttnn.from_torch(
        FF1_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensor2dMesh(device_mesh, dims=(None, 3), cluster_shape=cluster_shape),
    )

    FF3 = ttnn.from_torch(
        FF3_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ShardTensor2dMesh(device_mesh, dims=(None, 3), cluster_shape=cluster_shape),
    )

    gt = FF1_pt * FF3_pt

    out = ttnn.mul(
        FF1,
        FF3,
        dtype=ttnn.bfloat16,
    )

    out = ttnn.to_torch(out, mesh_composer=ConcatMesh2DToTensor(device_mesh, dims=(1, 3), cluster_shape=cluster_shape))
    out = out[:, 0:1, :, :]  # select the first column

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99999)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


class ReplicateShardTensor2dMesh(TensorToMesh):
    def __init__(self, device_mesh, dims, cluster_shape):
        super().__init__(device_mesh)
        self.dims = dims
        self.cluster_shape = cluster_shape

    def map(self, tensor: torch.Tensor):
        result = []

        if self.dims[0] is None and self.dims[1] is not None:
            # Replicate along rows, shard along columns
            sharded_tensors = list(torch.chunk(tensor, self.cluster_shape[1], dim=self.dims[1]))
            for shard in sharded_tensors:
                result.extend([shard.clone() for _ in range(self.cluster_shape[0])])
        elif self.dims[0] is not None and self.dims[1] is None:
            # Replicate along columns, shard along rows
            sharded_tensors = list(torch.chunk(tensor, self.cluster_shape[0], dim=self.dims[0]))
            for _ in range(self.cluster_shape[1]):
                result.extend([shard.clone() for shard in sharded_tensors])
        else:
            raise ValueError(
                "One dimension must be None (for replication) and the other must be specified (for sharding)"
            )

        return result

    def config(self):
        return {
            "strategy": "replicate",
        }


@pytest.mark.parametrize(
    "cluster_shape",
    [
        (4, 8),
    ],
)
@pytest.mark.parametrize(
    "device_mesh",
    [
        (4, 8),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, N, head_dim, num_heads",
    [
        (32, 8192, 128, 80),  # Llama3-70B decode attn fused_qkv
        (32, 8192, 128, 64),  # Llama3-70B decode attn selfout
    ],
    ids=["Llama3-70B-decode-attn-fused_qkv", "Llama3-70B-decode-attn-selfout"],
)
def test_galaxy_attn_matmul(M, N, head_dim, num_heads, cluster_shape, device_mesh):
    act_pt = torch.randn(1, 1, M, N)
    qkv_weights_pt = torch.randn(1, 1, N, head_dim * num_heads)

    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    qkv_weights = ttnn.from_torch(
        qkv_weights_pt,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateShardTensor2dMesh(device_mesh, dims=(None, 3), cluster_shape=cluster_shape),
    )

    compute_kernel_attn = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    out = ttnn.matmul(
        act,
        qkv_weights,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=5, x=8) if num_heads == 80 else ttnn.CoreGrid(y=4, x=8),
        use_1d_systolic_array=True,
        compute_kernel_config=compute_kernel_attn,
    )

    gt = act_pt @ qkv_weights_pt

    out = ttnn.to_torch(out, mesh_composer=ConcatMesh2DToTensor(device_mesh, dims=(1, 3), cluster_shape=cluster_shape))
    out = out[:, 0:1, :, :]  # select the first column

    out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
    logger.info(f"PCC value: {out_pcc}")
    assert out_pass


def pad_and_reshape_heads(tensor, seq_len, batch, n_local_heads, head_dim, padded_heads=32):
    reshaped = tensor.view(seq_len, batch, n_local_heads, head_dim)
    padding = torch.zeros(seq_len, batch, padded_heads - n_local_heads, head_dim)
    return torch.cat([reshaped, padding], dim=-2)


@pytest.mark.parametrize(
    "device_mesh",
    [
        (4, 8),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, n_local_heads, n_local_kv_heads",
    [
        (8, 1, 128, 8, 1),  # Llama3-70B decode attn
    ],
    ids=["Llama3-70B-decode-attn"],
)
def test_galaxy_nlp_create_heads_decode(batch, seq_len, head_dim, n_local_heads, n_local_kv_heads, device_mesh):
    total_heads = n_local_heads + n_local_kv_heads * 2
    qkv_heads_pt = torch.rand(1, seq_len, batch, head_dim * total_heads)

    total_cores = total_heads * head_dim // 32
    core_x = min(total_cores, 8)
    core_y = max(1, total_cores // core_x)

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
    qkv_heads = ttnn.from_torch(
        qkv_heads_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=CREATE_HEAD_INPUT_MEMCFG,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    # tt operation
    (
        q_heads_tt,  # [seqlen, bsz, padded_n_local_heads, head_dim]
        k_heads_tt,  # [seqlen, bsz, padded_n_local_kv_heads, head_dim]
        v_heads_tt,  # [seqlen, bsz, padded_n_local_kv_heads, head_dim]
    ) = ttnn.experimental.tensor.nlp_create_qkv_heads_decode(
        qkv_heads,
        num_heads=n_local_heads,
        num_kv_heads=n_local_kv_heads,
        output_mem_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )
    logger.info(f"q_heads_tt: {q_heads_tt.memory_config()}")
    logger.info(f"k_heads_tt: {k_heads_tt.memory_config()}")
    logger.info(f"v_heads_tt: {v_heads_tt.memory_config()}")

    q_heads_pt = pad_and_reshape_heads(
        qkv_heads_pt[:, :, :batch, : head_dim * n_local_heads], seq_len, batch, n_local_heads, head_dim
    )

    k_heads_pt = pad_and_reshape_heads(
        qkv_heads_pt[:, :, :batch, head_dim * n_local_heads : head_dim * (n_local_heads + n_local_kv_heads)],
        seq_len,
        batch,
        n_local_kv_heads,
        head_dim,
    )

    v_heads_pt = pad_and_reshape_heads(
        qkv_heads_pt[:, :, :batch, head_dim * (n_local_heads + n_local_kv_heads) :],
        seq_len,
        batch,
        n_local_kv_heads,
        head_dim,
    )

    # compare
    q_heads_tt_cpu = ttnn.to_torch(q_heads_tt, mesh_composer=ListMeshToTensor(device_mesh))[0]
    out_pass_q, output_pcc_q = comp_pcc(q_heads_tt_cpu, q_heads_pt, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_q}")

    k_heads_tt_cpu = ttnn.to_torch(k_heads_tt, mesh_composer=ListMeshToTensor(device_mesh))[0]
    out_pass_k, output_pcc_k = comp_pcc(k_heads_tt_cpu, k_heads_pt, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_k}")

    v_heads_tt_cpu = ttnn.to_torch(v_heads_tt, mesh_composer=ListMeshToTensor(device_mesh))[0]
    out_pass_v, output_pcc_v = comp_pcc(v_heads_tt_cpu, v_heads_pt, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc_v}")

    assert out_pass_q and out_pass_k and out_pass_v


@pytest.mark.parametrize(
    "device_mesh",
    [
        (4, 8),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, n_local_heads, n_local_kv_heads",
    [
        (8, 1, 128, 8, 1),  # Llama3-70B decode attn
    ],
    ids=["Llama3-70B-decode-attn"],
)
def test_galaxy_rotary_matmul(batch, seq_len, head_dim, n_local_heads, n_local_kv_heads, device_mesh):
    q_heads_pt = torch.rand(seq_len, batch, max(n_local_heads, 32), head_dim)  # Unpad batch=32 to 8
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
        device=device_mesh,
        memory_config=ROTARY_INPUT_MEMCFG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    key_layer = ttnn.from_torch(
        k_heads_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        memory_config=ROTARY_INPUT_MEMCFG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    rot_mats = ttnn.from_torch(
        rot_mat_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_mesh,
        memory_config=ROT_MAT_MEMCFG,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
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

    query_layer_cpu = ttnn.to_torch(query_layer, mesh_composer=ListMeshToTensor(device_mesh))[0]
    key_layer_cpu = ttnn.to_torch(key_layer, mesh_composer=ListMeshToTensor(device_mesh))[0]

    out_pass_q, out_pcc_q = comp_pcc(query_layer_cpu, query_layer_gt, pcc=0.999)
    logger.info(f"PCC value: {out_pcc_q}")
    out_pass_k, out_pcc_k = comp_pcc(key_layer_cpu, key_layer_gt, pcc=0.999)
    logger.info(f"PCC value: {out_pcc_k}")

    assert out_pass_q and out_pass_k


@pytest.mark.parametrize(
    "device_mesh",
    [
        (4, 8),
    ],
    indirect=True,
)
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("num_users", [8])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("input_dtype", [ttnn.experimental.tensor.DataType.BFLOAT16])
class TestUpdateCache:
    @pytest.mark.parametrize("seq_len", [128, 2048])
    def test_fill_cache(
        self, seq_len, head_dim, max_seq_len, num_users, num_heads, input_dtype, device_mesh, use_program_cache
    ):
        cache_dtype = input_dtype
        input_shape = [1, num_heads, seq_len, head_dim]
        cache_shape = [num_users, num_heads, max_seq_len, head_dim]
        cache = torch.randn(cache_shape).bfloat16().float()

        cachett = ttnn.from_torch(
            cache,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        for i in range(num_users):
            x = torch.randn(input_shape).bfloat16().float()

            xt = x

            compute_grid_size = device_mesh.get_device(0).compute_with_storage_grid_size()
            num_cores = min(seq_len // 32 * num_heads, 32)  # Always use max 32 cores for testing
            shard_grid = ttnn.CoreRangeSet(
                ttnn.experimental.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
            )
            input_shard_spec = ttnn.ShardSpec(
                shard_grid,
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
                device=device_mesh,
                memory_config=input_mem_config,
                mesh_mapper=ReplicateTensorToMesh(device_mesh),
            )

            cachett = ttnn.experimental.tensor.fill_cache(cachett, xt, i)
            cache[i : i + 1, :, : x.shape[-2], :] = x

        tt_got_back = ttnn.to_torch(cachett, mesh_composer=ListMeshToTensor(device_mesh))[0]
        eq, output = comp_pcc(cache, tt_got_back)
        logger.info(output)
        assert eq

    @pytest.mark.parametrize("cache_idx", [0, 127, 2047])
    @pytest.mark.parametrize("cache_dtype", [ttnn.experimental.tensor.DataType.BFLOAT8_B])
    @pytest.mark.parametrize(
        "batch_offset", [0, 16]
    )  # Only used when num_users < 32 and batch_offset + num_users <= 32
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
        device_mesh,
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
            device=device_mesh,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        x = torch.randn(input_shape).bfloat16().float()
        # pad dim0 of x to 32 if batch size is less than 32, make 0-batch_offset elements 0, batch_offset-batch_offset+num_users elements non-zero, and rest 0
        x_new = x.clone()
        if num_users < 32:
            x_new = torch.cat((torch.zeros(batch_offset, num_heads, 1, head_dim), x_new), dim=0)
            x_new = torch.cat((x_new, torch.zeros(32 - num_users - batch_offset, num_heads, 1, head_dim)), dim=0)
            assert x_new.shape[0] == 32, f"Expected x.shape[0] to be 32, got {x_new.shape[0]}"
        xt = x_new.permute(2, 1, 0, 3)
        compute_grid_size = device_mesh.get_device(0).compute_with_storage_grid_size()
        num_cores = min(max(num_users, 32) // 32 * num_heads, compute_grid_size.x * compute_grid_size.y)
        shard_grid = ttnn.CoreRangeSet(
            ttnn.experimental.tensor.num_cores_to_corerange_set(num_cores, compute_grid_size, True)
        )
        input_shard_spec = ttnn.ShardSpec(
            shard_grid,
            [
                xt.numel() // xt.shape[-1] // num_cores,
                xt.shape[-1],
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        input_mem_config = ttnn.MemoryConfig(
            ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.experimental.tensor.BufferType.L1,
            input_shard_spec,
        )

        xt = ttnn.from_torch(
            xt,
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            memory_config=input_mem_config,
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )

        cachett = ttnn.experimental.tensor.update_cache(cachett, xt, cache_idx, batch_offset=batch_offset)
        cache[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]] = x

        tt_got_back = ttnn.to_torch(cachett, mesh_composer=ListMeshToTensor(device_mesh))[0]

        eq_cache, output_cache = comp_pcc(cache, tt_got_back)  # checks the entire kv cache
        eq_update, output_update = comp_pcc(
            x, tt_got_back[0:num_users, 0:num_heads, cache_idx : cache_idx + x.shape[-2], 0 : x.shape[-1]]
        )  # checks the updated parts
        logger.info(output_cache)
        logger.info(output_update)
        assert eq_cache and eq_update


# TODO: Import from SDPA pytests
def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power
    # if (2**math.log2(x) == x):
    #     return x
    # return 2**(int(x).bit_length())


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.experimental.tensor.CoreRange(
        ttnn.experimental.tensor.CoreCoord(0, 0),
        ttnn.experimental.tensor.CoreCoord(num_x - 1, num_y - 1),
    )


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
    device_mesh,
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
    compute_grid_size = device_mesh.get_device(0).compute_with_storage_grid_size()
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
    dram_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR, False)

    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    K = torch.randn(nkv, b, s, d)
    V = torch.randn(nkv, b, s, d)

    tt_K = ttnn.from_torch(
        K,
        device=device_mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    tt_V = ttnn.from_torch(
        V,
        device=device_mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )
    start_idx = s // 2
    scale = d**-0.5

    k_chunk_size = get_chunk_size(start_idx)
    program_config = ttnn.experimental.operations.primary.transformers.SDPAMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
    )

    padded_layer_len = nearest_n(start_idx, n=k_chunk_size)

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
        device=device_mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    tt_attn_mask = ttnn.from_torch(
        attn_mask,
        device=device_mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_memcfg,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    tt_back = ttnn.experimental.operations.primary.transformers.scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        tt_attn_mask,
        scale=scale,
        program_config=program_config,
        valid_seq_len=padded_layer_len,
        compute_kernel_config=compute_kernel_config,
        output_mem_config=height_sharded_memcfg if sharded_out else dram_memcfg,
    )

    tt_back = ttnn.to_torch(tt_back, mesh_composer=ListMeshToTensor(device_mesh))[0]
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


@pytest.mark.parametrize(
    "device_mesh",
    [
        (4, 8),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "dtype, q_dtype, mask_dtype",
    [
        [
            ttnn.experimental.tensor.DataType.BFLOAT16,
            ttnn.experimental.tensor.DataType.BFLOAT16,
            ttnn.experimental.tensor.DataType.BFLOAT16,
        ],
    ],
    ids=[
        "all_bfp16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    ([8, 8, 1, 32768, 128, (8, 8)],),  # Llama3-70B
)
def test_sdpa_decode_sharded(device_mesh, b, nh, nkv, s, d, dtype, grid_size, q_dtype, mask_dtype):
    run_test_sdpa_decode_single_iter(
        device_mesh, b, nh, nkv, s, d, dtype, grid_size, q_dtype, mask_dtype, sharded_in=True, sharded_out=False
    )
    run_test_sdpa_decode_single_iter(
        device_mesh, b, nh, nkv, s, d, dtype, grid_size, q_dtype, mask_dtype, sharded_in=True, sharded_out=True
    )


@pytest.mark.parametrize(
    "device_mesh",
    [
        (4, 8),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch, seq_len, head_dim, n_local_heads, n_local_kv_heads, padded_local_heads",
    [
        (8, 1, 128, 8, 1, 32),  # Llama3-70B decode attn
    ],
    ids=["Llama3-70B-decode-attn"],
)
def test_galaxy_nlp_concat_heads_decode(
    batch, seq_len, head_dim, n_local_heads, n_local_kv_heads, padded_local_heads, device_mesh
):
    concat_head_input = torch.rand(seq_len, batch, padded_local_heads, head_dim)

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(batch)})
    SCORES_BATCHED_MM_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_grid,
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
        memory_config=SCORES_BATCHED_MM_OUTPUT_MEMCFG,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    concat_head_output = ttnn.experimental.tensor.nlp_concat_heads_decode(
        concat_head_input_tt,
        num_heads=n_local_heads,
    )  # (seqlen, 1, batch, hidden_size)

    logger.info(f"concat_head_output: {concat_head_output.memory_config()}")

    # Input: (1, 8, 32(8), 128)
    # Output: (1, 1, 8, 1024)
    concat_head_output_pt = concat_head_input[:, :, :n_local_heads].reshape(1, 1, batch, head_dim * n_local_heads)

    # Compare
    concat_head_output_tt_cpu = ttnn.to_torch(concat_head_output, mesh_composer=ListMeshToTensor(device_mesh))[0]
    concat_head_output_tt_unpadded = concat_head_output_tt_cpu[:, :, :batch, :]
    out_pass, output_pcc = comp_pcc(concat_head_output_tt_unpadded, concat_head_output_pt, pcc=0.9999)
    logger.info(f"PCC value: {output_pcc}")

    assert out_pass


def rmsnorm(x, gamma, beta, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma + beta


@pytest.mark.parametrize(
    "device_mesh",
    [
        (4, 8),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, N",
    [
        (32, 8192),  # Llama3-70B decode
    ],
    ids=["Llama3-70B-decode"],
)
def test_galaxy_layernorm(M, N, device_mesh):
    layernorm_input = torch.rand(1, 1, M, N) * 2 - 0.95
    norm_weights = torch.rand(1, 1, 256, 32) * 2 - 1
    norm_eps = 1e-05

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

    layernorm_input_tt = ttnn.from_torch(
        layernorm_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=LN_OUTPUT_MEMCFG,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    norm_weights_tt = ttnn.from_torch(
        norm_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device_mesh,
        mesh_mapper=ReplicateTensorToMesh(device_mesh),
    )

    LN_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    LN_PROGCFG = ttnn.experimental.operations.primary.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=M // 32,
        block_w=8,
        inplace=True,
    )

    norm_output = ttnn.experimental.operations.primary.rmsnorm(
        layernorm_input_tt,
        norm_eps,
        norm_weights_tt,
        program_config=LN_PROGCFG,
        output_mem_config=LN_OUTPUT_MEMCFG,
        compute_kernel_config=LN_COMPUTE_KERNEL_CONFIG,
    )

    # Compare
    beta = torch.zeros(1, 1, 256, 32)
    norm_output_tt_cpu = ttnn.to_torch(norm_output, mesh_composer=ListMeshToTensor(device_mesh))[0]
    ref_rmsnorm = rmsnorm(layernorm_input, norm_weights.flatten(), beta.flatten(), norm_eps)

    out_pass, output_pcc = comp_pcc(norm_output_tt_cpu, ref_rmsnorm, pcc=0.999)
    logger.info(f"PCC value: {output_pcc}")

    assert out_pass
