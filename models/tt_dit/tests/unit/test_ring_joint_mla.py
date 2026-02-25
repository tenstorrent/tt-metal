# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.padding import get_padded_vision_seq_len
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand


def create_global_semaphores(mesh_device, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor):
    submesh_shape = [0, 0]
    submesh_shape[rp_axis] = rp_factor
    submesh_shape[up_axis] = up_factor
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape[0], submesh_shape[1]))
    return submesh_device


def run_ring_joint_sdpa(
    submesh,
    b,
    nhq,
    nhk,
    nhv,
    base_seq_len,
    padded_seq_len,
    joint_seq_len,
    head_dim_q,
    head_dim_k,
    head_dim_v,
    q_chunk_size,
    k_chunk_size,
    dtype,
    n_iters,
    trace_enabled,
    num_links,
    rp_axis,
    up_axis,
    all_gather_topology,
    skip_check,
    pcc_threshold,
    max_mse=None,
    is_causal=False,
):
    full_compute_grid = submesh.compute_with_storage_grid_size()
    sdpa_compute_grid = (full_compute_grid.x, full_compute_grid.y - 1)
    ccl_core_grid_offset = (0, full_compute_grid.y - 1)

    # Basic CCL setup
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [create_global_semaphores(submesh, ccl_sub_device_crs, 0) for _ in range(n_iters)]

    kv_shard_dims = [None, None]
    kv_shard_dims[rp_axis] = None  # Output of AllGather is not sharded on RP axis
    kv_shard_dims[up_axis] = 1  # UP shards on heads dim1

    # Create persistent output buffers
    # Check sharding on these
    ag_output_shape_k = (b, nhk, padded_seq_len, head_dim_k)
    ag_output_shape_v = (b, nhv, padded_seq_len, head_dim_v)
    print("ag_output_shape_k = ", ag_output_shape_k)
    print("ag_output_shape_v = ", ag_output_shape_v)

    persistent_k_output_shard_dims = [None, None]
    persistent_k_output_shard_dims[up_axis] == 1 if nhk != 1 else None

    persistent_output_buffers = [
        [
            ttnn.from_torch(
                torch.zeros(ag_output_shape_k),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    submesh, mesh_shape=tuple(submesh.shape), dims=persistent_k_output_shard_dims
                ),
            ),
            ttnn.from_torch(
                torch.zeros(ag_output_shape_v),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=kv_shard_dims),
            ),
        ]
        for _ in range(n_iters)
    ]

    print("Persistent output buffer[0] shape = ", persistent_output_buffers[0][0].shape)
    print("Persistent output buffer[1] shape = ", persistent_output_buffers[0][1].shape)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nhq, base_seq_len, head_dim_q)
    K = fa_rand(b, nhk, base_seq_len, head_dim_k)
    V = fa_rand(b, nhv, base_seq_len, head_dim_v)

    padded_Q = torch.cat([Q, torch.zeros(b, nhq, padded_seq_len - base_seq_len, head_dim_q)], dim=2)
    padded_K = torch.cat([K, torch.zeros(b, nhk, padded_seq_len - base_seq_len, head_dim_k)], dim=2)
    padded_V = torch.cat([V, torch.zeros(b, nhv, padded_seq_len - base_seq_len, head_dim_v)], dim=2)

    joint_Q = fa_rand(b, nhq, joint_seq_len, head_dim_q)
    joint_K = fa_rand(b, nhk, joint_seq_len, head_dim_k)
    joint_V = fa_rand(b, nhv, joint_seq_len, head_dim_v)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")
    logger.debug(f"padded_Q: {padded_Q.shape}")
    logger.debug(f"padded_K: {padded_K.shape}")
    logger.debug(f"padded_V: {padded_V.shape}")

    sdpa_input_shard_dims = [None, None]
    sdpa_input_shard_dims[rp_axis] = 2  # sequence dim
    sdpa_input_shard_dims[up_axis] = 1  # head dim

    # Joint input only sharded on head dim
    sdpa_joint_shard_dims = [None, None]
    sdpa_joint_shard_dims[up_axis] = 1  # head dim

    sdpa_k_input_shard_dims = [None, None]
    sdpa_k_input_shard_dims[rp_axis] = 2  # sequence dim
    if nhk == 1:
        sdpa_k_input_shard_dims[up_axis] = None  # Do not shard on head_dim, as there is 1 k head for all q heads in MLA
    else:
        sdpa_k_input_shard_dims[up_axis] = 1  # head dim

    tt_Q = ttnn.from_torch(
        padded_Q,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_K = ttnn.from_torch(
        padded_K,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_k_input_shard_dims),
    )
    tt_V = ttnn.from_torch(
        padded_V,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_joint_Q = ttnn.from_torch(
        joint_Q,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_joint_shard_dims),
    )
    # split on head if there is nh > 1, else replicate
    joint_k_mesh_mapper = (
        ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_k_input_shard_dims)
        if nhk > 1
        else ttnn.ReplicateTensorToMesh(submesh)
    )
    tt_joint_K = ttnn.from_torch(
        joint_K,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=joint_k_mesh_mapper,
    )
    tt_joint_V = ttnn.from_torch(
        joint_V,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_joint_shard_dims),
    )

    logger.debug(f"tt_Q: {tt_Q.shape}")
    logger.debug(f"tt_joint_Q: {tt_joint_Q.shape}")

    tt_out_list = []
    tt_joint_out_list = []

    def run_iters(tt_out_list, tt_joint_out_list):
        for i in range(n_iters):
            print("Running ring-joint sdpa with the following shapes:")
            print("tt_Q: ", tt_Q.shape)
            print("tt_K: ", tt_K.shape)
            print("tt_V: ", tt_V.shape)
            print("tt_joint_Q: ", tt_joint_Q.shape)
            print("tt_joint_K: ", tt_joint_K.shape)
            print("tt_joint_V: ", tt_joint_V.shape)
            tt_out, tt_joint_out, tt_lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_joint_Q,
                tt_joint_K,
                tt_joint_V,
                persistent_output_buffer_k=persistent_output_buffers[i][0],
                persistent_output_buffer_v=persistent_output_buffers[i][1],
                joint_strategy="rear",
                logical_n=base_seq_len,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=ccl_semaphore_handles[i],
                num_links=num_links,
                cluster_axis=rp_axis,
                mesh_device=submesh,
                topology=all_gather_topology,
                subdevice_id=worker_sub_device_id,
                ccl_core_grid_offset=ccl_core_grid_offset,
                is_causal=is_causal,
            )
            tt_out_list.append(tt_out)
            tt_joint_out_list.append(tt_joint_out)

    if trace_enabled:
        logger.info("Compile run")
        run_iters([], [])
        logger.info("Capture trace")
        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        run_iters(tt_out_list, tt_joint_out_list)
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        ttnn.synchronize_device(submesh)
        logger.info("Execute trace")
        ttnn.execute_trace(submesh, trace_id, blocking=False)
        ttnn.release_trace(submesh, trace_id)
        ttnn.synchronize_device(submesh)

    else:
        logger.info("Run without trace")
        run_iters(tt_out_list, tt_joint_out_list)

    if not skip_check:
        pt_Q = torch.cat([Q, joint_Q], dim=2)
        pt_K = torch.cat([K, joint_K], dim=2)
        pt_V = torch.cat([V, joint_V], dim=2)
        print("Running on host...")
        gt = torch.nn.functional.scaled_dot_product_attention(pt_Q, pt_K, pt_V, is_causal=is_causal)
        print("Done running on host...")
        print("Host output shape: ", gt.shape)
        gt_out = gt[:, :, :base_seq_len, :]
        gt_joint_out = gt[:, :, base_seq_len:, :]

        for i in range(n_iters):
            print("Synchronize call...")
            ttnn.synchronize_device(submesh)
            print("Done synchronizing...")
            tt_out = ttnn.to_torch(
                tt_out_list[i],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims
                ),
            )
            print("Started doing to_torch stuff...")
            joint_shard_dims = [None, None]
            joint_shard_dims[up_axis] = 1
            joint_shard_dims[rp_axis] = 0  # Concat replicas on sequence length into batch
            tt_joint_out = ttnn.to_torch(
                tt_joint_out_list[i],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    submesh, mesh_shape=tuple(submesh.shape), dims=joint_shard_dims
                ),
            )
            print("Done to torch stuff...")
            # Slice out any tile-padding
            tt_out = tt_out[:, :, :base_seq_len, :]
            tt_joint_out = tt_joint_out[:, :, :joint_seq_len, :]
            logger.debug(f"tt_out: {tt_out.shape}")
            logger.debug(f"tt_joint_out: {tt_joint_out.shape}")

            passing = True
            out_pass, out_pcc = comp_pcc(tt_out, gt_out, pcc_threshold)
            logger.debug("spatial")
            logger.debug(f"{out_pcc}")
            mse = ((gt_out - tt_out) ** 2).mean()
            logger.debug(f"mse: {mse}")
            if max_mse is not None and mse > max_mse:
                passing = False
            passing = passing and out_pass

            if joint_seq_len > 0:
                logger.debug("prompt")
                for joint_replica_id in range(tt_joint_out.shape[0]):
                    joint_replica_out = tt_joint_out[joint_replica_id, :, :, :]
                    out_pass, out_pcc = comp_pcc(joint_replica_out, gt_joint_out, pcc_threshold)
                    logger.debug(f"{out_pcc}")
                    mse = ((gt_joint_out - joint_replica_out) ** 2).mean()
                    logger.debug(f"mse: {mse}")
                    if max_mse is not None and mse > max_mse:
                        passing = False
                    passing = passing and out_pass

            assert passing


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nhq, nhk, nhv, base_seq_len, head_dim_q, head_dim_k, head_dim_v",
    [
        (1, 64, 1, 64, 4 * 4 * 1024, 576, 576, 128),
        (1, 64, 1, 64, 4 * 32, 64, 64, 32),
        # base case
        (1, 2, 1, 2, 4 * 32, 32, 32, 32),
    ],
)
@pytest.mark.parametrize("q_chunk_size", [32], ids=["q32"])
@pytest.mark.parametrize("k_chunk_size", [32], ids=["k32"])
@pytest.mark.parametrize(
    "n_iters, trace_enabled, skip_check",
    [
        (1, False, False),
    ],
    ids=["no_trace"],
)
@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"worker_l1_size": 1344544, "trace_region_size": 1000000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        "line",
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    ids=["2x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [1, 4, 0, 2],
    ],
    ids=[
        "4rpx2p",
    ],
)
def test_mla_sdpa(
    mesh_device,
    b,
    nhq,
    nhk,
    nhv,
    base_seq_len,
    head_dim_q,
    head_dim_k,
    head_dim_v,
    q_chunk_size,
    k_chunk_size,
    dtype,
    n_iters,
    trace_enabled,
    num_links,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    all_gather_topology,
    skip_check,
    reset_seeds,
):
    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor

    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    padded_seq_len = get_padded_vision_seq_len(base_seq_len, mesh_device_shape[rp_axis])

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    joint_seq_len = 0  # causality is enabled only for non-joint cases

    run_ring_joint_sdpa(
        submesh,
        b,
        nhq,
        nhk,
        nhv,
        base_seq_len,
        padded_seq_len,
        joint_seq_len,
        head_dim_q,
        head_dim_k,
        head_dim_v,
        q_chunk_size,
        k_chunk_size,
        dtype,
        n_iters,
        trace_enabled,
        num_links,
        rp_axis,
        up_axis,
        all_gather_topology,
        skip_check,
        0.999,
        is_causal=False,
    )
