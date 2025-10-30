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


def run_distributed_fused_rmsnorm(
    mesh_device, tp_mesh_axis, inp_shape, dtype, stats_dtype, topology, num_heads_per_device=1
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

    canon_inp = torch.randn(inp_shape) * 4 - 1
    epsilon = 1e-5

    tensor_shard_dims = [None, None]
    tensor_shard_dims[tp_mesh_axis] = -1
    tensor_shard_dims[1 - tp_mesh_axis] = -2
    tt_inp = ttnn.from_torch(
        canon_inp,
        dtype=dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tensor_shard_dims, mesh_shape=list(mesh_device.shape)),
    )

    ccl_semaphore_handles = setup_ccl_semaphores(mesh_device)
    ttnn.synchronize_device(mesh_device)
    tt_stats = ttnn.experimental.fused_rmsnorm_pre_allgather(
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

    tt_out = ttnn.experimental.fused_rmsnorm_post_allgather(
        tt_inp,
        tt_stats_gathered,
        epsilon=epsilon,
        num_heads_per_device=num_heads_per_device,
        compute_kernel_config=compute_kernel_config,
    )
    # tt_out = ttnn.rms_norm_post_all_gather(
    #     tt_inp, tt_stats_gathered, epsilon=epsilon, compute_kernel_config=compute_kernel_config, distributed_program_config=prog_cfg,
    # )

    tensor_cat_dims = [None, None]
    tensor_cat_dims[tp_mesh_axis] = -3 if num_heads > 1 else -1
    tensor_cat_dims[1 - tp_mesh_axis] = -2
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tensor_cat_dims, mesh_shape=list(mesh_device.shape)),
    )

    # reference impl
    out_torch = torch.nn.functional.rms_norm(canon_inp, normalized_shape=(inp_shape[-1],), eps=epsilon)

    # create heads fusion
    out_torch = out_torch.reshape(inp_shape[0], inp_shape[2], num_heads, -1).permute(0, 2, 1, 3)
    passing, output_str = comp_allclose(tt_out, out_torch, rtol=1e-1, atol=1e-01)
    logger.debug(f"torch vs tt distributed fused rmsnorm = {output_str}")
    assert passing


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16_in"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16], ids=["BFLOAT16_stats"])
@pytest.mark.parametrize(
    "seqlen",
    [128, 256, 2048, 8192, 9472, 18944],
    ids=["seqlen128", "seqlen256", "seqlen2048", "seqlen8192", "seqlen9472", "seqlen18944"],
)
@pytest.mark.parametrize(
    "hidden_dim", [256, 2048, 5120, 8192], ids=["hidden_dim256", "hidden_dim2048", "hidden_dim5120", "hidden_dim8192"]
)
@pytest.mark.parametrize("num_heads_per_device", [1, 2, 10], ids=["num_heads1_", "num_heads2", "num_heads10"])
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
    mesh_device, tp_mesh_axis, seqlen, hidden_dim, dtype, stats_dtype, topology, num_heads_per_device, reset_seeds
):
    num_heads = num_heads_per_device * mesh_device.shape[tp_mesh_axis]
    if hidden_dim // 32 % num_heads != 0:
        pytest.skip("hidden_dim must be divisible by 32 * num_heads")
    inp_shape = (1, 1, seqlen, hidden_dim)
    run_distributed_fused_rmsnorm(
        mesh_device, tp_mesh_axis, inp_shape, dtype, stats_dtype, topology, num_heads_per_device
    )
