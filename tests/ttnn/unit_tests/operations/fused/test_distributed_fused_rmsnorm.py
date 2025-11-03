# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

import ttnn

from models.common.utility_functions import tt2torch_tensor

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from tests.tests_common.skip_reasons import LEGACY_CCL_SKIP
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

from tests.ttnn.unit_tests.operations.fused.test_distributed_layernorm_ulp import setup_ccl_semaphores


def get_rot_transformation_mat():
    # ROPE op uses a single tile
    TILE_SIZE = 32
    rot_emb_matrix = torch.zeros(1, 1, TILE_SIZE, TILE_SIZE)
    rot_emb_matrix[..., torch.arange(0, TILE_SIZE, 2), torch.arange(1, TILE_SIZE, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, TILE_SIZE, 2), torch.arange(0, TILE_SIZE, 2)] = -1
    return rot_emb_matrix


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def run_distributed_fused_rmsnorm(
    mesh_device,
    tp_mesh_axis,
    inp_shape,
    dtype,
    stats_dtype,
    topology,
    num_heads_per_device=1,
    use_weight=True,
    use_rope=False,
):
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    prog_cfg = ttnn.LayerNormDistributedDefaultProgramConfig(
        legacy_reduction=False,
        legacy_rsqrt=False,
    )

    torch.manual_seed(1234)

    num_heads = num_heads_per_device * mesh_device.shape[tp_mesh_axis]

    torch_input = torch.randn(inp_shape) * 4 - 1

    if use_weight:
        torch_weight = torch.rand(inp_shape[-1:]).unsqueeze(0)
        assert torch_weight.shape == (1, inp_shape[-1])

    tt_rope_cos = None
    tt_rope_sin = None
    tt_transformation_mat = None
    if use_rope:
        head_dim = inp_shape[-1] // num_heads
        rope_cos = torch.randn(1, 1, inp_shape[2], head_dim // 2)
        rope_cos = torch.stack([rope_cos, rope_cos], dim=-1).flatten(-2)
        rope_sin = torch.randn(1, 1, inp_shape[2], head_dim // 2)
        rope_sin = torch.stack([rope_sin, rope_sin], dim=-1).flatten(-2)

        transformation_mat = get_rot_transformation_mat()

        rope_shard_dims = [None, None]
        rope_shard_dims[tp_mesh_axis - 1] = 2  # Sharding on sequence dimension
        tt_rope_cos = ttnn.from_torch(
            rope_cos,
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=rope_shard_dims, mesh_shape=list(mesh_device.shape)),
        )
        tt_rope_sin = ttnn.from_torch(
            rope_sin,
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=rope_shard_dims, mesh_shape=list(mesh_device.shape)),
        )

        tt_transformation_mat = ttnn.from_torch(
            transformation_mat,
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    epsilon = 1e-5

    tensor_shard_dims = [None, None]
    tensor_shard_dims[tp_mesh_axis] = -1
    tensor_shard_dims[1 - tp_mesh_axis] = -2
    tt_inp = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tensor_shard_dims, mesh_shape=list(mesh_device.shape)),
    )

    tt_weight = None
    if use_weight:
        weight_shard_dims = [None, None]
        weight_shard_dims[tp_mesh_axis] = -1
        tt_weight = ttnn.from_torch(
            torch_weight,
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=weight_shard_dims, mesh_shape=list(mesh_device.shape)),
        )

    ccl_semaphore_handles = setup_ccl_semaphores(mesh_device)
    ttnn.synchronize_device(mesh_device)
    tt_stats = ttnn.experimental.wan_fused_rmsnorm_pre_allgather(
        tt_inp, compute_kernel_config=compute_kernel_config, dtype=stats_dtype
    )
    # tt_stats = ttnn.rms_norm_pre_all_gather(
    #     tt_inp, compute_kernel_config=compute_kernel_config, dtype=stats_dtype, distributed_program_config=prog_cfg,
    # )

    ttnn.synchronize_device(tt_inp.device())
    tt_stats_gathered = ttnn.experimental.all_gather_async(
        tt_stats,
        dim=3,
        multi_device_global_semaphore=ccl_semaphore_handles,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_device=tt_inp.device(),
        topology=topology,
        cluster_axis=tp_mesh_axis,
    )

    tt_out = ttnn.experimental.wan_fused_rmsnorm_post_allgather(
        tt_inp,
        tt_stats_gathered,
        epsilon=epsilon,
        num_heads_per_device=num_heads_per_device,
        weight=tt_weight,
        transformation_mat=tt_transformation_mat,
        rope_cos=tt_rope_cos,
        rope_sin=tt_rope_sin,
        compute_kernel_config=compute_kernel_config,
    )
    # tt_out = ttnn.rms_norm_post_all_gather(
    #     tt_inp, tt_stats_gathered, epsilon=epsilon, compute_kernel_config=compute_kernel_config, distributed_program_config=prog_cfg,
    # )
    # if use_rope:
    #     tt_out = ttnn.experimental.rotary_embedding_llama(tt_out, tt_rope_cos, tt_rope_sin, tt_transformation_mat, compute_kernel_config=compute_kernel_config)

    tensor_cat_dims = [None, None]
    tensor_cat_dims[tp_mesh_axis] = -3 if num_heads > 1 else -1
    tensor_cat_dims[1 - tp_mesh_axis] = -2
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tensor_cat_dims, mesh_shape=list(mesh_device.shape)),
    )

    # reference impl
    out_torch = torch.nn.functional.rms_norm(torch_input, normalized_shape=(inp_shape[-1],), eps=epsilon)
    if use_weight:
        out_torch = out_torch * torch_weight

    # create heads fusion
    out_torch = out_torch.reshape(inp_shape[0], inp_shape[2], num_heads, -1).permute(0, 2, 1, 3)

    if use_rope:
        out_torch = apply_rotary_emb(out_torch, rope_cos, rope_sin)

    passing, output_str = comp_allclose(tt_out, out_torch, rtol=1e-1, atol=1e-01)
    logger.debug(f"torch vs tt distributed fused rmsnorm = {output_str}")
    assert passing


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [128, 256, 2048, 18944],
    ids=["seqlen128", "seqlen256", "seqlen2048", "seqlen18944"],
)
@pytest.mark.parametrize("hidden_dim", [256, 5120, 8192], ids=["hidden_dim256", "hidden_dim5120", "hidden_dim8192"])
@pytest.mark.parametrize("num_heads_per_device", [1, 2, 10], ids=["num_heads1_", "num_heads2", "num_heads10"])
@pytest.mark.parametrize("use_weight", [True, False], ids=["has_weight", "no_weight"])
@pytest.mark.parametrize("use_rope", [True, False], ids=["has_rope", "no_rope"])
@pytest.mark.parametrize(
    "mesh_device, tp_mesh_axis",
    [
        [(1, 8), 1],
        [(2, 4), 1],
        [(2, 4), 0],
    ],
    ids=["1x8_tp1", "2x4_tp1", "2x4_tp0"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear),
    ],
    ids=["fabric_linear"],
    indirect=["device_params"],
)
def test_distributed_fused_rmsnorm(
    mesh_device,
    tp_mesh_axis,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    topology,
    num_heads_per_device,
    use_weight,
    use_rope,
    reset_seeds,
):
    num_heads = num_heads_per_device * mesh_device.shape[tp_mesh_axis]
    if hidden_dim // 32 % num_heads != 0:
        pytest.skip("hidden_dim must be divisible by 32 * num_heads")
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_fused_rmsnorm(
        mesh_device, tp_mesh_axis, inp_shape, dtype, stats_dtype, topology, num_heads_per_device, use_weight, use_rope
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize("seqlen", [2048])
@pytest.mark.parametrize("hidden_dim", [5120, 8192], ids=["hidden_dim5120", "hidden_dim8192"])
@pytest.mark.parametrize("num_heads_per_device", [1, 2, 10], ids=["num_heads1_", "num_heads2", "num_heads10"])
@pytest.mark.parametrize("use_weight", [True, False], ids=["has_weight", "no_weight"])
@pytest.mark.parametrize(
    "mesh_device, tp_mesh_axis",
    [
        [(1, 8), 1],
        [(2, 4), 1],
        [(2, 4), 0],
    ],
    ids=["1x8_tp1", "2x4_tp1", "2x4_tp0"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params, topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear),
    ],
    ids=["fabric_linear"],
    indirect=["device_params"],
)
def test_distributed_fused_rmsnorm_program_cache(
    mesh_device,
    tp_mesh_axis,
    seqlen,
    hidden_dim,
    dtype,
    stats_dtype,
    topology,
    num_heads_per_device,
    use_weight,
    reset_seeds,
):
    num_heads = num_heads_per_device * mesh_device.shape[tp_mesh_axis]
    if hidden_dim // 32 % num_heads != 0:
        pytest.skip("hidden_dim must be divisible by 32 * num_heads")
    inp_shape = (1, 1, seqlen, hidden_dim)
    dummy_tensors = []
    for i in range(2):
        run_distributed_fused_rmsnorm(
            mesh_device, tp_mesh_axis, inp_shape, dtype, stats_dtype, topology, num_heads_per_device, use_weight
        )
        dummy_tensors.append(
            ttnn.from_torch(
                torch.zeros(32, 32),
                dtype=dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
