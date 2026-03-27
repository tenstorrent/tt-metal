# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_blackhole
from tests.nightly.t3000.ccl.test_minimal_matmul_strided_reduce_scatter_async import (
    MinimalMatmulStridedReduceScatterTestConfig,
    create_global_semaphores,
    run_minimal_matmul_strided_reduce_scatter_impl,
)
from tests.nightly.t3000.ccl.test_strided_reduce_scatter_async import run_reduce_scatter_impl
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# ──────────────────────────────────────────────────────────────────────────────
# Strided reduce-scatter: t3000 (1×8)
# Representative: 128×16 tile slice, 8-core MM grid, 2 N-blocks, 6 RS workers
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["1x8"])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_strided_reduce_scatter_async_t3000(mesh_device, num_links, topology):
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_reduce_scatter_impl(
        mesh_device,
        mesh_device.get_num_devices(),
        [4, 1, 4096, 4096],
        3,
        num_links,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mem_config,
        mem_config,
        rs_topology=topology,
        enable_trace=False,
        num_iters=1,
        small_random_ints=True,
        use_barrier=True,
        use_persistent_buffers=True,
        use_strided=True,
        verify_output_shape=True,
        verify_output_pcc=True,
        mm_cores_y=8,
        mm_block_ht=4,
        mm_block_wt=4,
        mm_N_full_block_wt=8,
        chunk_width_in_mm_blocks=1,
        num_workers_per_link=6,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Minimal matmul + strided reduce-scatter: t3000 (1×8)
# Representative: M=3072, N=4096, 8×6 core grid, 6 RS workers (Wan FF shape)
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["1x8"])
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize("cluster_axis", [1], ids=["axis_1"])
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
def test_minimal_matmul_strided_reduce_scatter_async_t3000(mesh_device, num_links, cluster_axis, topology):
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    run_minimal_matmul_strided_reduce_scatter_impl(
        mesh_device,
        M=3072,
        K=512,
        N=4096,
        dim=3,
        num_links=num_links,
        input_dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mem_config_input=mem_config,
        mem_config_mm=mem_config,
        mem_config_rs=mem_config,
        topology=topology,
        mm_block_m=256,
        mm_block_k=256,
        mm_block_n=256,
        subblock_h=1,
        subblock_w=1,
        mm_core_grid=ttnn.CoreCoord(8, 6),
        chunk_width_in_mm_blocks=2,
        num_workers_per_link=6,
        rs_mode="fused",
        cluster_axis=cluster_axis,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Optional-feature tests: fused_activation_relu, bias, addcmul
# ──────────────────────────────────────────────────────────────────────────────

_FEATURE_BASE_CFG = MinimalMatmulStridedReduceScatterTestConfig(
    M=128,
    K=256,
    N=512,
    dim=3,
    mm_block_m=64,
    mm_block_k=64,
    mm_block_n=64,
    mm_core_grid=ttnn.CoreCoord(8, 2),
    chunk_width_in_mm_blocks=1,
)


def _run_optional_feature_test(mesh_device, topology, cluster_axis, feature):
    """Exercise one optional feature of minimal_matmul_strided_reduce_scatter_async.

    Shares setup with the main test helper but wires up the optional inputs and
    adjusts the golden computation accordingly.
    """
    torch.manual_seed(42)
    TILE_SIZE = 32

    cfg = _FEATURE_BASE_CFG
    num_links = 1
    num_devices = mesh_device.shape[cluster_axis]
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    mesh_device.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    ccl_semaphores = create_global_semaphores(mesh_device, all_cores, 0)
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, all_cores, 0)
    rs_core_grid_offset = ttnn.CoreCoord(0, cfg.mm_core_grid.y)

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    matmul_config = ttnn.MinimalMatmulConfig(
        M_block_size=cfg.mm_block_m // TILE_SIZE,
        K_block_size=cfg.mm_block_k // TILE_SIZE,
        N_block_size=cfg.mm_block_n // TILE_SIZE,
        subblock_h=1,
        subblock_w=1,
        compute_with_storage_grid_size=cfg.mm_core_grid,
    )

    # Input: replicated; weight: sharded so each device has different weights
    shard_dims = [None, None]
    shard_dims[cluster_axis] = 0
    torch_input = torch.randn([1, 1, cfg.M, cfg.K])
    torch_weight_global = torch.randn([num_devices, 1, cfg.K, cfg.N])
    torch_weight_chunks = torch.chunk(torch_weight_global, num_devices, dim=0)

    input_mesh = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    weight_mesh = ttnn.from_torch(
        torch_weight_global,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )

    # Per-device MM golden outputs
    mm_golds = [torch.matmul(torch_input, torch_weight_chunks[d]).float() for d in range(num_devices)]

    # Feature-specific setup: build extra_kwargs and adjust rs_input_per_device
    extra_kwargs = {}
    rs_input_per_device = mm_golds  # default: RS on raw MM output

    if feature == "fused_activation_relu":
        # Applied to MM output before RS
        rs_input_per_device = [torch.relu(mm_out) for mm_out in mm_golds]
        extra_kwargs = {"fused_activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)}

    elif feature == "bias":
        # Bias added to each device's MM output before RS
        torch_bias = torch.randn([1, 1, 1, cfg.N])
        bias_mesh = ttnn.from_torch(
            torch_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rs_input_per_device = [mm_out + torch_bias for mm_out in mm_golds]
        extra_kwargs = {"bias": bias_mesh}

    elif feature == "addcmul":
        # The reader reads only the first N/num_devices columns from each tensor (b=0 in the kernel
        # batch loop). ternary_a must be full [M, N] and ternary_b must be [1, N] to pass validation
        # in minimal_matmul_device_operation.cpp (N is derived from the weight shape). Replicate the
        # full tensors; every device reads the same first-N/num_devices slice from its local copy.
        scalar = 0.5
        torch_addcmul_a = torch.randn([1, 1, cfg.M, cfg.N])
        torch_addcmul_b = torch.randn([1, 1, 1, cfg.N])
        addcmul_a_mesh = ttnn.from_torch(
            torch_addcmul_a,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        addcmul_b_mesh = ttnn.from_torch(
            torch_addcmul_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        extra_kwargs = {
            "fused_ternary_scalar": scalar,
            "addcmul_input_tensor1": addcmul_a_mesh,
            "addcmul_input_tensor2": addcmul_b_mesh,
        }

    # Golden: reduce-scatter over rs_input_per_device
    torch_rs_reduced = torch.sum(torch.stack(rs_input_per_device), dim=0)
    torch_rs_scattered = list(torch.chunk(torch_rs_reduced, num_devices, dim=cfg.dim))
    if feature == "addcmul":
        # Each device reads the first N/num_devices columns of the replicated tensors.
        slice_n = cfg.N // num_devices
        addcmul_a_slice = torch_addcmul_a[..., :slice_n]  # [1, 1, M, N/num_devices]
        addcmul_b_slice = torch_addcmul_b[..., :slice_n]  # [1, 1, 1, N/num_devices] broadcast
        torch_rs_scattered = [addcmul_a_slice + scalar * chunk * addcmul_b_slice for chunk in torch_rs_scattered]

    # Run the op
    mm_out, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        input_mesh,
        weight_mesh,
        cfg.dim,
        ccl_semaphores,
        rs_core_grid_offset,
        num_links=num_links,
        memory_config_mm=mem_config,
        rs_output_mem_config=mem_config,
        topology=topology,
        cluster_axis=cluster_axis,
        config=matmul_config,
        compute_kernel_config=compute_config,
        barrier_semaphore=barrier_semaphore,
        chunk_width_in_mm_blocks=cfg.chunk_width_in_mm_blocks,
        **extra_kwargs,
    )
    ttnn.synchronize_device(mesh_device)

    # Verify RS output on each device
    concat_dims = [cfg.dim, cfg.dim]
    concat_dims[1 - cluster_axis] = 0 if cfg.dim != 0 else 1
    concat_mesh_shape = list(mesh_device.shape)
    concat_mesh_shape[1 - cluster_axis] = 1
    tt_rs_torch = ttnn.to_torch(
        rs_out,
        mesh_composer=ttnn.create_mesh_composer(
            mesh_device, ttnn.MeshComposerConfig(concat_dims, ttnn.MeshShape(concat_mesh_shape))
        ),
    )
    tt_rs_chunks = torch.chunk(tt_rs_torch, num_devices, dim=cfg.dim)
    for d in range(num_devices):
        eq, msg = comp_pcc(tt_rs_chunks[d], torch_rs_scattered[d], 0.99)
        logger.info(f"feature={feature} device {d}: {msg}")
        assert eq, f"feature={feature} device {d} FAILED: {msg}"

    logger.info(f"All checks passed for feature={feature}!")


@skip_for_blackhole("t3000 tests are wormhole_b0 only")
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True, ids=["1x8"])
@pytest.mark.parametrize("cluster_axis", [1], ids=["axis_1"])
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 1531456}, ttnn.Topology.Ring)],
    indirect=["device_params"],
    ids=["fabric_ring"],
)
@pytest.mark.parametrize(
    "feature",
    ["fused_activation_relu", "bias", "addcmul"],
)
def test_minimal_matmul_strided_reduce_scatter_async_optional_features(mesh_device, topology, cluster_axis, feature):
    _run_optional_feature_test(mesh_device, topology, cluster_axis, feature)
