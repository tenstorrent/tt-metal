# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.tests.unit.test_ring_joint_attention import create_ring_joint_sdpa_submesh
from models.tt_dit.utils.padding import get_padded_vision_seq_len
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand


def create_fabric_router_config(max_payload_size=8192):
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def run_exp_ring_joint_sdpa(
    submesh,
    b,
    nh,
    base_seq_len,
    padded_seq_len,
    joint_seq_len,
    d,
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
    num_workers_per_link=5,
    num_buffers_per_channel=32,
):
    full_compute_grid = submesh.compute_with_storage_grid_size()
    sdpa_compute_grid = (full_compute_grid.x - 1, full_compute_grid.y)

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

    # create global semaphore handles: one per link for per-chunk sync
    ccl_semaphore_handles = [
        [ttnn.create_global_semaphore(submesh, ccl_sub_device_crs, 0) for _ in range(num_links)] for _ in range(n_iters)
    ]

    kv_shard_dims = [None, None]
    kv_shard_dims[rp_axis] = None  # Output of AllGather is not sharded on RP axis
    kv_shard_dims[up_axis] = 1  # UP shards on heads dim1

    # Create persistent output buffers
    ag_output_shape = (b, nh, padded_seq_len, d)

    persistent_output_buffers = [
        [
            ttnn.from_torch(
                torch.zeros(ag_output_shape),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=kv_shard_dims),
            )
            for _ in range(2)  # Num inputs K, V
        ]
        for _ in range(n_iters)
    ]

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        submesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, base_seq_len, d).bfloat16().float()
    K = fa_rand(b, nh, base_seq_len, d).bfloat16().float()
    V = fa_rand(b, nh, base_seq_len, d).bfloat16().float()

    padded_Q = torch.cat([Q, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)
    padded_K = torch.cat([K, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)
    padded_V = torch.cat([V, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)

    joint_Q = fa_rand(b, nh, joint_seq_len, d)
    joint_K = fa_rand(b, nh, joint_seq_len, d)
    joint_V = fa_rand(b, nh, joint_seq_len, d)

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
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
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
    tt_joint_K = ttnn.from_torch(
        joint_K,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_joint_shard_dims),
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
            if not trace_enabled:
                ttnn.synchronize_device(submesh)
            tt_out, tt_joint_out, tt_lse = ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
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
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
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
        gt = torch.nn.functional.scaled_dot_product_attention(pt_Q, pt_K, pt_V, is_causal=False)
        gt_out = gt[:, :, :base_seq_len, :]
        gt_joint_out = gt[:, :, base_seq_len:, :]

        for i in range(n_iters):
            tt_out = ttnn.to_torch(
                tt_out_list[i],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims
                ),
            )
            joint_shard_dims = [None, None]
            joint_shard_dims[up_axis] = 1
            joint_shard_dims[rp_axis] = 0  # Concat replicas on sequence length into batch
            tt_joint_out = ttnn.to_torch(
                tt_joint_out_list[i],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    submesh, mesh_shape=tuple(submesh.shape), dims=joint_shard_dims
                ),
            )
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


def run_test_exp_ring_joint_sdpa(
    mesh_device,
    model_input_shape,
    parallel_config,
    q_chunk_size,
    k_chunk_size,
    n_iters,
    trace_enabled,
    num_links,
    all_gather_topology,
    skip_check,
    dtype,
    pcc_threshold=0.994,
    max_mse=None,
    num_workers_per_link=5,
    num_buffers_per_channel=48,
):
    b, nh, base_seq_len, joint_seq_len, d = model_input_shape
    rp_axis, rp_factor, up_axis, up_factor = parallel_config

    if nh % up_factor != 0:
        orig_nh = nh
        nh = math.ceil(nh / up_factor) * up_factor
        logger.info(f"Rounding up nh from {orig_nh} to {nh} so that it divides evenly by up_factor={up_factor}.")
    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor

    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    padded_seq_len = get_padded_vision_seq_len(base_seq_len, mesh_device_shape[rp_axis])

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    run_exp_ring_joint_sdpa(
        submesh,
        b,
        nh,
        base_seq_len,
        padded_seq_len,
        joint_seq_len,
        d,
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
        max_mse=max_mse,
        num_workers_per_link=num_workers_per_link,
        num_buffers_per_channel=num_buffers_per_channel,
    )


@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "worker_l1_size": 1344544,
                "trace_region_size": 1000000,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(8192),
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["ring"],
)
@pytest.mark.parametrize(
    "mesh_device, num_links, nh, base_seq_len, rp_axis, rp_factor, up_axis, up_factor",
    [
        ((4, 32), 2, 40, 75600, 1, 32, 0, 4),
        ((4, 8), 2, 40, 18944, 1, 8, 0, 4),
        ((1, 4), 2, 10, 8960, 1, 4, 0, 1),
    ],
    ids=["4x32", "4x8", "1x4"],
    indirect=["mesh_device"],
)
def test_exp_ring_joint_sdpa_dit_bh_glx_custom(
    mesh_device,
    num_links,
    nh,
    base_seq_len,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    b, joint_seq_len, d = 1, 0, 128
    q_chunk_size = 224
    k_chunk_size = 512
    n_iters = 5
    trace_enabled = False
    skip_check = False
    pcc_threshold = 0.9993
    max_mse = 8e-5

    if nh % up_factor != 0:
        nh = math.ceil(nh / up_factor) * up_factor
    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)
    padded_seq_len = get_padded_vision_seq_len(base_seq_len, list(mesh_device.shape)[rp_axis])

    run_exp_ring_joint_sdpa(
        submesh,
        b,
        nh,
        base_seq_len,
        padded_seq_len,
        joint_seq_len,
        d,
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
        max_mse=max_mse,
    )
