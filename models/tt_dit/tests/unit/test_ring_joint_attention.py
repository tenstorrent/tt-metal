# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from tracy.process_model_log import post_process_ops_log, run_device_profiler

import ttnn
from models.common.utility_functions import is_blackhole
from models.tt_dit.utils.padding import get_padded_vision_seq_len
from tests.tests_common.cache_entries_counter import CacheEntriesCounter
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand


def torch_sdpa(q, k, v, joint_q, joint_k, joint_v, num_devices):
    scale = k.size(-1) ** -0.5
    seq_len = k.size(2)
    slice_seq_len = seq_len // num_devices
    out = None
    lse = None
    lse_list = []
    Q = torch.cat([q, joint_q], dim=2)
    for ring_id in range(num_devices):
        k_slice = k[:, :, ring_id * slice_seq_len : (ring_id + 1) * slice_seq_len, :]
        v_slice = v[:, :, ring_id * slice_seq_len : (ring_id + 1) * slice_seq_len, :]
        if ring_id == num_devices - 1:
            k_slice = torch.cat([k_slice, joint_k], dim=2)
            v_slice = torch.cat([v_slice, joint_v], dim=2)
        attn_weights = torch.matmul(Q, k_slice.transpose(-2, -1)) * scale
        cur_max, _ = torch.max(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - cur_max)
        cur_sum = torch.sum(attn_weights, dim=-1, keepdim=True)
        cur_out = torch.matmul(attn_weights, v_slice)
        cur_out = cur_out / cur_sum
        cur_lse = cur_max + torch.log(cur_sum)
        if ring_id == 0:
            out = cur_out
            lse = cur_lse
        else:
            sig = F.sigmoid(cur_lse - lse)
            out = out - sig * (out - cur_out)
            lse = lse - F.logsigmoid(lse - cur_lse)
        lse_list.append(lse)

    return out, lse_list


def create_global_semaphores(mesh_device, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def logical_length_tensor(mesh_device, value, *, on_device=True):
    """Single-valued uint32 tensor for the logical_n / logical_l transport, replicated to every device.
    on_device=False returns the spec-matching host twin for ttnn.copy_host_to_device_tensor."""
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        **({"device": mesh_device} if on_device else {}),
    )


def create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor):
    submesh_shape = [0, 0]
    submesh_shape[rp_axis] = rp_factor
    submesh_shape[up_axis] = up_factor
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape[0], submesh_shape[1]))
    submesh_device.cache_entries_counter = CacheEntriesCounter(submesh_device)
    return submesh_device


def run_ring_joint_sdpa_model_config(
    mesh_device,
    *,
    b,
    nh,
    base_seq_len,
    joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    num_links,
    ccl_reserve_last_column,
    use_column_major_ccl,
    use_wormhole_compute_kernel_config,
    pcc_threshold=0.999,
    fp32_dest_acc_en: bool = False,
):
    """
    Run ring_joint_scaled_dot_product_attention matching all model-specific
    details: grid layout, CCL placement, compute kernel config, dtype, etc.
    """
    # Pad heads to be divisible by up_factor (tensor-parallel factor)
    if nh % up_factor != 0:
        nh = math.ceil(nh / up_factor) * up_factor

    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)
    padded_seq_len = get_padded_vision_seq_len(base_seq_len, rp_factor)

    full_grid = submesh.compute_with_storage_grid_size()

    # --- Grid layout: match model's CCL core placement exactly ---
    if ccl_reserve_last_column:
        # Wan: reserve last column for CCL
        sdpa_compute_grid = (full_grid.x - 1, full_grid.y)
        ccl_core_grid_offset = (full_grid.x - 1, 0)
    else:
        # SD3.5 / Mochi: reserve last row for CCL
        sdpa_compute_grid = (full_grid.x, full_grid.y - 1)
        ccl_core_grid_offset = (0, full_grid.y - 1)

    # --- Sub-device setup ---
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group(sub_device_stall_group)

    # --- Global semaphores ---
    ccl_semaphore_handles = [ttnn.create_global_semaphore(submesh, ccl_sub_device_crs, 0) for _ in range(2)]

    # --- Persistent output buffers for all-gather K/V ---
    kv_shard_dims = [None, None]
    kv_shard_dims[up_axis] = 1  # Sharded on heads (dim 1)

    ag_output_shape = (b, nh, padded_seq_len, d)
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(ag_output_shape),
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=kv_shard_dims),
        )
        for _ in range(2)  # K, V
    ]

    # --- Program config: exact grid from model ---
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    # --- Compute kernel config: match model exactly ---
    if use_wormhole_compute_kernel_config:
        # SD3.5 uses WormholeComputeKernelConfig directly
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )
    else:
        # Wan / Mochi use init_device_compute_kernel_config
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            submesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=False,
        )

    # --- Create input tensors ---
    Q = fa_rand(b, nh, base_seq_len, d)
    K = fa_rand(b, nh, base_seq_len, d)
    V = fa_rand(b, nh, base_seq_len, d)

    padded_Q = torch.cat([Q, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)
    padded_K = torch.cat([K, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)
    padded_V = torch.cat([V, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)

    joint_Q = fa_rand(b, nh, joint_seq_len, d)
    joint_K = fa_rand(b, nh, joint_seq_len, d)
    joint_V = fa_rand(b, nh, joint_seq_len, d)

    logger.debug(f"Q: {Q.shape}, padded_Q: {padded_Q.shape}, joint_Q: {joint_Q.shape}")

    # Shard dims: RP on sequence (dim 2), UP on heads (dim 1)
    sdpa_input_shard_dims = [None, None]
    sdpa_input_shard_dims[rp_axis] = 2
    sdpa_input_shard_dims[up_axis] = 1

    # Joint only sharded on heads (replicated across RP axis)
    sdpa_joint_shard_dims = [None, None]
    sdpa_joint_shard_dims[up_axis] = 1

    dtype = ttnn.bfloat16

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

    logger.debug(f"tt_Q: {tt_Q.shape}, tt_joint_Q: {tt_joint_Q.shape}")
    logger.debug(f"sdpa_compute_grid: {sdpa_compute_grid}, ccl_core_grid_offset: {ccl_core_grid_offset}")
    logger.debug(f"use_column_major_ccl: {use_column_major_ccl}")

    # --- Run the op ---
    sdpa_kwargs = dict(
        persistent_output_buffer_k=persistent_output_buffers[0],
        persistent_output_buffer_v=persistent_output_buffers[1],
        joint_strategy="rear",
        logical_n=base_seq_len,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_semaphore_handles,
        num_links=num_links,
        cluster_axis=rp_axis,
        mesh_device=submesh,
        topology=ttnn.Topology.Linear,
        subdevice_id=worker_sub_device_id,
        ccl_core_grid_offset=ccl_core_grid_offset,
    )
    if use_column_major_ccl:
        sdpa_kwargs["use_column_major_ccl"] = True

    tt_out, tt_joint_out, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        tt_joint_Q,
        tt_joint_K,
        tt_joint_V,
        **sdpa_kwargs,
    )

    # --- Verify correctness ---
    pt_Q = torch.cat([Q, joint_Q], dim=2)
    pt_K = torch.cat([K, joint_K], dim=2)
    pt_V = torch.cat([V, joint_V], dim=2)
    gt = torch.nn.functional.scaled_dot_product_attention(pt_Q, pt_K, pt_V, is_causal=False)
    gt_out = gt[:, :, :base_seq_len, :]
    gt_joint_out = gt[:, :, base_seq_len:, :]

    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_out = tt_out[:, :, :base_seq_len, :]

    passing = True
    out_pass, out_pcc = comp_pcc(tt_out, gt_out, pcc_threshold)
    mse = ((gt_out - tt_out) ** 2).mean()
    logger.info(f"spatial PCC: {out_pcc}, MSE: {mse}")
    passing = passing and out_pass

    if joint_seq_len > 0:
        joint_shard_dims = [None, None]
        joint_shard_dims[up_axis] = 1
        joint_shard_dims[rp_axis] = 0  # Concat replicas into batch
        tt_joint_out = ttnn.to_torch(
            tt_joint_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=joint_shard_dims),
        )
        tt_joint_out = tt_joint_out[:, :, :joint_seq_len, :]
        for replica_id in range(tt_joint_out.shape[0]):
            replica_out = tt_joint_out[replica_id, :, :, :]
            jout_pass, jout_pcc = comp_pcc(replica_out, gt_joint_out, pcc_threshold)
            jmse = ((gt_joint_out - replica_out) ** 2).mean()
            logger.info(f"joint replica {replica_id} PCC: {jout_pcc}, MSE: {jmse}")
            passing = passing and jout_pass

    assert passing


def run_ring_joint_sdpa(
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
    fp32_dest_acc_en: bool = False,
    # logical_n as a device tensor instead of a host int (logical_l stays defaulted). Pure transport change.
    logical_as_tensor: bool = False,
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
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, base_seq_len, d)
    K = fa_rand(b, nh, base_seq_len, d)
    V = fa_rand(b, nh, base_seq_len, d)

    padded_Q = torch.cat([Q, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)
    padded_K = torch.cat([K, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)
    padded_V = torch.cat([V, torch.zeros(b, nh, padded_seq_len - base_seq_len, d)], dim=2)

    joint_Q = fa_rand(b, nh, joint_seq_len, d)
    joint_K = fa_rand(b, nh, joint_seq_len, d)
    joint_V = fa_rand(b, nh, joint_seq_len, d)

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
    # Allocated once, outside run_iters, so a traced capture bakes a stable address.
    tt_logical_n = logical_length_tensor(submesh, base_seq_len) if logical_as_tensor else base_seq_len

    def run_iters(tt_out_list, tt_joint_out_list):
        with submesh.cache_entries_counter.measure():
            for i in range(n_iters):
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
                    logical_n=tt_logical_n,
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


def run_test_ring_joint_sdpa(
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
    fp32_dest_acc_en: bool = False,
):
    b, nh, base_seq_len, joint_seq_len, d = model_input_shape
    rp_axis, rp_factor, up_axis, up_factor = parallel_config
    import math

    if nh % up_factor != 0:
        orig_nh = nh
        nh = math.ceil(nh / up_factor) * up_factor
        logger.info(f"Rounding up nh from {orig_nh} to {nh} so that it divides evenly by up_factor={up_factor}.")
    mesh_device_shape = list(mesh_device.shape)
    if not (mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor):
        pytest.skip(
            f"Mesh shape {mesh_device.shape} cannot satisfy parallel config "
            f"rp_axis={rp_axis} rp_factor={rp_factor}, up_axis={up_axis} up_factor={up_factor}"
        )

    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    padded_seq_len = get_padded_vision_seq_len(base_seq_len, mesh_device_shape[rp_axis])

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    run_ring_joint_sdpa(
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
        fp32_dest_acc_en=fp32_dest_acc_en,
    )


benchmark_model_input_shapes = {
    "wan_14b_720p": (1, 40, 75600, 0, 128),
    "wan_14b_480p": (1, 40, 32760, 0, 128),
    "mochi": (1, 24, 44520, 118, 128),
    "flux": (1, 24, 4096, 512, 128),
    "sd35": (1, 38, 4096, 333, 64),
}

parallel_config_map = {
    "wh_glx": {
        "wan_14b_720p": (0, 8, 1, 4),
        "wan_14b_480p": (0, 8, 1, 4),
        "mochi": (0, 8, 1, 4),
        "flux": (0, 8, 1, 4),
        "sd35": (0, 4, 1, 4),
    },
    "wh_t3k": {
        "wan_14b_720p": (0, 2, 1, 4),
        "wan_14b_480p": (0, 2, 1, 4),
        "mochi": (0, 2, 1, 4),
        "flux": (0, 2, 1, 4),
        "sd35": (0, 2, 1, 2),
    },
    "bh_glx": {
        "wan_14b_720p": (0, 8, 1, 4),
        "wan_14b_480p": (0, 8, 1, 4),
        "mochi": (0, 8, 1, 4),
        "flux": (0, 8, 1, 4),
        "sd35": (0, 4, 1, 4),
    },
    "bh_qb_ge": {
        "wan_14b_720p": (0, 2, 1, 2),
        "wan_14b_480p": (0, 2, 1, 2),
        "mochi": (0, 2, 1, 2),
        "flux": (0, 2, 1, 2),
        "sd35": (0, 2, 1, 2),
    },
}

mesh_device_map = {
    "wh_glx": [(8, 4), 4],
    "wh_t3k": [(2, 4), 1],
    "bh_glx": [(8, 4), 2],
    "bh_qb_ge": [(2, 2), 2],
}


@pytest.fixture(scope="function")
def mesh_shape_or_skip(request):
    """Skip test when requested mesh shape cannot be satisfied, without opening a mesh device."""
    param = request.param

    assert isinstance(param, tuple)
    num_devices_requested = param[0] * param[1]

    if not ttnn.using_distributed_env() and num_devices_requested > ttnn.get_num_devices():
        pytest.skip(
            f"Requested more devices {num_devices_requested} than available {ttnn.get_num_devices()}. Test not applicable for machine"
        )

    return param


all_parallel_configs = list(set(config for configs in parallel_config_map.values() for config in configs.values()))


def get_parallel_config_id(rp_factor, up_factor):
    return f"{rp_factor}rpx{up_factor}up"


all_parallel_config_ids = [
    get_parallel_config_id(rp_factor, up_factor) for rp_axis, rp_factor, up_axis, up_factor in all_parallel_configs
]


@pytest.mark.parametrize(
    "model_input_shape",
    benchmark_model_input_shapes.values(),
    ids=benchmark_model_input_shapes.keys(),
)
@pytest.mark.parametrize("parallel_config", all_parallel_configs, ids=all_parallel_config_ids)
@pytest.mark.parametrize("q_chunk_size", [64, 128, 256], ids=["q64", "q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [64, 128, 256, 512], ids=["k64", "k128", "k256", "k512"])
@pytest.mark.parametrize(
    "n_iters, trace_enabled, skip_check",
    [(1, False, False), (1, False, True)],
    ids=["no_trace_check", "no_trace_no_check"],
)
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
    "mesh_device, num_links",
    mesh_device_map.values(),
    ids=mesh_device_map.keys(),
    indirect=["mesh_device"],
)
def test_ring_joint_sdpa(
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
    reset_seeds,
):
    dtype = ttnn.bfloat16

    run_test_ring_joint_sdpa(
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
    )


@pytest.mark.parametrize(
    "mesh_device_id, mesh_shape_or_skip",
    [(k, v[0]) for k, v in mesh_device_map.items()],
    ids=mesh_device_map.keys(),
    indirect=["mesh_shape_or_skip"],
)
@pytest.mark.skip(
    reason=(
        "Calling pytest within pytest in ttnn is problematic right now. "
        "The parent process maintains an open handle to the device which prevents the child process "
        "from using the device, leading to deadlock. "
        "TODO: This test should be re-enabled when functionality for releasing handles is exposed in ttnn "
        "(currently this exists in C++ as release_ownership but does not exist in python at the moment). "
        "Also, this test doesn't actually test anything so maybe we need to actually do some assertions that might make sense here."
    )
)
def test_ring_joint_sdpa_perf_table(mesh_device_id, mesh_shape_or_skip):
    results = []
    for model_input_id, model_input_shape in benchmark_model_input_shapes.items():
        parallel_config = parallel_config_map[mesh_device_id][model_input_id]
        rp_axis, rp_factor, up_axis, up_factor = parallel_config
        parallel_name = get_parallel_config_id(rp_factor, up_factor)
        k_expr = f"{model_input_id} and {parallel_name} and {mesh_device_id} and no_trace_no_check"
        command = (
            f"-m 'pytest models/tt_dit/tests/unit/test_ring_joint_attention.py::test_ring_joint_sdpa -k \"{k_expr}\"'"
        )

        run_device_profiler(
            command,
            "ring_joint_sdpa",
            check_test_return_code=False,
            device_analysis_types=["device_kernel_duration"],
            is_command_binary_exe=True,
        )
        r = post_process_ops_log("ring_joint_sdpa", sum_vals=False, has_signposts=False)
        attrs = r["ATTRIBUTES"].tolist()
        durations = r["DEVICE KERNEL DURATION [ns]"].tolist()
        result = sorted(zip(durations, attrs), key=lambda x: x[0])[0]
        best_duration, best_attrs = result
        results.append([model_input_shape, model_input_id, parallel_name, best_duration, best_attrs])

    header = "| model_input_id | model_input_shape | parallel_name | padded seq | qchunk, kchunk | measured perf (ms) |"
    sep = "|---:|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)
    for result in results:
        model_input_shape, model_input_id, parallel_name, duration, attrs = result
        q_chunk = attrs.split("q_chunk_size=")[1].split(";")[0]
        k_chunk = attrs.split("k_chunk_size=")[1].split(";")[0]
        new_seqlen = get_padded_vision_seq_len(int(model_input_shape[2]), rp_factor)
        print(
            f"| {model_input_id} | {model_input_shape} | {parallel_name} | {new_seqlen} | {q_chunk}, {k_chunk} | {duration / 1e6:.3f} |"
        )


model_input_shapes = [
    # original smoke cases
    (1, 24, 4096, 512, 128),  # padded-divisible spatial, joint > 0
    (1, 38, 4096, 333, 64),  # many heads, smaller head dim, uneven joint
    (1, 24, 4224, 128, 128),  # N not divisible by chunk, moderate joint
    (1, 2, 3072, 0, 128),  # small head count, no joint
    (1, 2, 4000, 2, 128),  # tiny joint, near-multiple-of-chunk
    # additional stress cases
    (1, 24, 8192, 0, 128),  # long sequence, no joint
    (1, 24, 8200, 64, 128),  # long, non-multiple N, small joint
    (1, 16, 1024, 256, 128),  # mid length, significant joint
    (1, 16, 1056, 128, 64),  # mid length, smaller head dim
    (1, 8, 2048, 0, 256),  # wider head dim
    (1, 8, 2176, 128, 128),  # mid length, non-multiple, modest joint
    (1, 4, 512, 64, 128),  # short length with joint
    (1, 4, 4096, 128, 128),
    (1, 2, 256, 16, 64),  # minimal heads/dim
]

model_input_ids = [
    "wan_14b_720p",
    "wan_14b_480p",
    "wan_5b_720p",
    "mochi",
    "flux",
    "long_no_joint",
    "long_unaligned_joint",
    "mid_joint",
    "mid_small_d",
    "wide_d",
    "mid_unaligned_joint",
    "short_joint",
    "batch2",
    "tiny_head",
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, base_seq_len, joint_seq_len, d",
    model_input_shapes,
    ids=model_input_ids,
)
@pytest.mark.parametrize("q_chunk_size", [32, 64, 128, 256], ids=["q32", "q64", "q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [32, 64, 128, 256], ids=["k32", "k64", "k128", "k256"])
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
        "4rpx2up",
    ],
)
# logical_n-only transport (replicated joint, logical_l defaulted): the shape family the DiT and
# minimax_h3 callers use, checked against the host-int path over this whole matrix.
@pytest.mark.parametrize("logical_as_tensor", [False, True], ids=["scalar", "tensor"])
def test_ring_joint_sdpa_shapes(
    mesh_device,
    b,
    nh,
    base_seq_len,
    joint_seq_len,
    d,
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
    logical_as_tensor,
    all_gather_topology,
    skip_check,
    reset_seeds,
):
    mesh_device_shape = list(mesh_device.shape)
    if not (mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor):
        pytest.skip(
            f"Mesh shape {mesh_device.shape} cannot satisfy parallel config "
            f"rp_axis={rp_axis} rp_factor={rp_factor}, up_axis={up_axis} up_factor={up_factor}"
        )

    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)

    padded_seq_len = get_padded_vision_seq_len(base_seq_len, mesh_device_shape[rp_axis])

    logger.debug(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    logger.debug(f"submesh: {submesh.shape}")

    run_ring_joint_sdpa(
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
        0.999,
        logical_as_tensor=logical_as_tensor,
    )


wh_t3k_unit_test_params = pytest.mark.parametrize(
    "input_shape, parallel_config, chunk_sizes, expected_correctness",
    [
        [
            benchmark_model_input_shapes["wan_14b_720p"],
            parallel_config_map["wh_t3k"]["wan_14b_720p"],
            (256, 256),
            (0.9994, 7.5e-5),
        ],
        [
            benchmark_model_input_shapes["wan_14b_480p"],
            parallel_config_map["wh_t3k"]["wan_14b_480p"],
            (256, 256),
            (0.9996, 5e-5),
        ],
        [benchmark_model_input_shapes["mochi"], parallel_config_map["wh_t3k"]["mochi"], (128, 512), (0.9995, 6e-5)],
        [benchmark_model_input_shapes["flux"], parallel_config_map["wh_t3k"]["flux"], (128, 512), (0.9997, 2.2e-5)],
        [benchmark_model_input_shapes["sd35"], parallel_config_map["wh_t3k"]["sd35"], (256, 512), (0.9997, 3.5e-5)],
    ],
    ids=[
        "wan_14b_720p",
        "wan_14b_480p",
        "mochi",
        "flux",
        "sd35",
    ],
)


@wh_t3k_unit_test_params
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
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["wh_t3k"]], ids=["2x4"], indirect=["mesh_device"])
def test_ring_joint_sdpa_dit_wh_t3k(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    pcc_threshold, max_mse = expected_correctness
    q_chunk_size, k_chunk_size = chunk_sizes

    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
        pcc_threshold=pcc_threshold,
        max_mse=max_mse,
    )


bh_qb_ge_unit_test_params = pytest.mark.parametrize(
    "input_shape, parallel_config, chunk_sizes, expected_correctness",
    [
        [
            benchmark_model_input_shapes["wan_14b_720p"],
            parallel_config_map["bh_qb_ge"]["wan_14b_720p"],
            (128, 512),
            (0.9994, 7e-5),
        ],
        [
            benchmark_model_input_shapes["wan_14b_480p"],
            parallel_config_map["bh_qb_ge"]["wan_14b_480p"],
            (128, 512),
            (0.9996, 5e-5),
        ],
        [benchmark_model_input_shapes["mochi"], parallel_config_map["bh_qb_ge"]["mochi"], (128, 512), (0.9995, 6e-5)],
        [benchmark_model_input_shapes["flux"], parallel_config_map["bh_qb_ge"]["flux"], (128, 512), (0.9997, 2.2e-5)],
        [benchmark_model_input_shapes["sd35"], parallel_config_map["bh_qb_ge"]["sd35"], (256, 512), (0.9997, 3.5e-5)],
    ],
    ids=[
        "wan_14b_720p",
        "wan_14b_480p",
        "mochi",
        "flux",
        "sd35",
    ],
)


@bh_qb_ge_unit_test_params
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
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["bh_qb_ge"]], ids=["2x2"], indirect=["mesh_device"])
def test_ring_joint_sdpa_dit_bh_qb_ge(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    pcc_threshold, max_mse = expected_correctness
    q_chunk_size, k_chunk_size = chunk_sizes

    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
        pcc_threshold=pcc_threshold,
        max_mse=max_mse,
    )


wh_glx_unit_test_params = pytest.mark.parametrize(
    "input_shape, parallel_config, chunk_sizes, expected_correctness",
    [
        [
            benchmark_model_input_shapes["wan_14b_720p"],
            parallel_config_map["wh_glx"]["wan_14b_720p"],
            (256, 256),
            (0.9993, 8e-5),
        ],
        [
            benchmark_model_input_shapes["wan_14b_480p"],
            parallel_config_map["wh_glx"]["wan_14b_480p"],
            (128, 512),
            (0.9995, 6e-5),
        ],
        [benchmark_model_input_shapes["mochi"], parallel_config_map["wh_glx"]["mochi"], (128, 512), (0.9994, 7e-5)],
        [benchmark_model_input_shapes["flux"], parallel_config_map["wh_glx"]["flux"], (128, 256), (0.9997, 3e-5)],
        [benchmark_model_input_shapes["sd35"], parallel_config_map["wh_glx"]["sd35"], (256, 512), (0.9997, 4e-5)],
    ],
    ids=[
        "wan_14b_720p",
        "wan_14b_480p",
        "mochi",
        "flux",
        "sd35",
    ],
)


@wh_glx_unit_test_params
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
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["wh_glx"]], ids=["8x4"], indirect=["mesh_device"])
@pytest.mark.skipif(
    ttnn.cluster.get_cluster_type() not in (ttnn.cluster.ClusterType.GALAXY, ttnn.cluster.ClusterType.TG),
    reason="test_ring_joint_sdpa_dit_wh_glx requires a Wormhole Galaxy (6U/TG) cluster",
)
def test_ring_joint_sdpa_dit_wh_glx(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    pcc_threshold, max_mse = expected_correctness
    q_chunk_size, k_chunk_size = chunk_sizes

    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
        pcc_threshold=pcc_threshold,
        max_mse=max_mse,
    )


bh_glx_unit_test_params = pytest.mark.parametrize(
    "input_shape, parallel_config, chunk_sizes, expected_correctness",
    [
        [
            benchmark_model_input_shapes["wan_14b_720p"],
            parallel_config_map["bh_glx"]["wan_14b_720p"],
            (128, 512),
            (0.9993, 8e-5),
        ],
        [
            benchmark_model_input_shapes["wan_14b_480p"],
            parallel_config_map["bh_glx"]["wan_14b_480p"],
            (256, 256),
            (0.9995, 6e-5),
        ],
        [benchmark_model_input_shapes["mochi"], parallel_config_map["bh_glx"]["mochi"], (128, 512), (0.9994, 7e-5)],
        [benchmark_model_input_shapes["flux"], parallel_config_map["bh_glx"]["flux"], (64, 512), (0.9997, 3e-5)],
        [benchmark_model_input_shapes["sd35"], parallel_config_map["bh_glx"]["sd35"], (128, 512), (0.9997, 4e-5)],
    ],
    ids=[
        "wan_14b_720p",
        "wan_14b_480p",
        "mochi",
        "flux",
        "sd35",
    ],
)


@bh_glx_unit_test_params
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
@pytest.mark.parametrize("mesh_device, num_links", [mesh_device_map["bh_glx"]], ids=["8x4"], indirect=["mesh_device"])
@pytest.mark.skipif(
    ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
    reason="test_ring_joint_sdpa_dit_bh_glx requires a Blackhole Galaxy cluster",
)
def test_ring_joint_sdpa_dit_bh_glx(
    mesh_device,
    input_shape,
    parallel_config,
    chunk_sizes,
    expected_correctness,
    num_links,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    n_iters = 1
    trace_enabled = False
    skip_check = False
    pcc_threshold, max_mse = expected_correctness
    q_chunk_size, k_chunk_size = chunk_sizes

    run_test_ring_joint_sdpa(
        mesh_device,
        input_shape,
        parallel_config,
        q_chunk_size,
        k_chunk_size,
        n_iters,
        trace_enabled,
        num_links,
        all_gather_topology,
        skip_check,
        dtype,
        pcc_threshold=pcc_threshold,
        max_mse=max_mse,
    )


def run_ring_joint_sdpa_sharded_prompt(
    submesh,
    *,
    b,
    nh,
    base_seq_len,
    padded_seq_len,
    padded_joint_seq_len,
    d,
    rp_axis,
    rp_factor,
    up_axis,
    q_chunk_size,
    k_chunk_size,
    num_links,
    logical_l=None,
    logical_as_tensor=False,
    topology=ttnn.Topology.Linear,
    pcc_threshold=0.999,
):
    """
    joint_tensor_q/k/v are sharded L/P per device on rp_axis dim=2; logical_l activates the
    internal gather and the output joint tensor is also sharded L/P per device.

    logical_l (defaulting to padded_joint_seq_len) is the real joint length;

    logical_as_tensor selects which lengths travel as device tensors: False/"none", True/"both", "n", "l".
    Pure transport change, so the result must be identical; the garbage-filled pad tails make any missed
    mask collapse PCC.
    """
    tensor_modes = {False: "none", True: "both"}
    tensor_mode = tensor_modes.get(logical_as_tensor, logical_as_tensor)
    assert tensor_mode in ("none", "both", "n", "l"), f"bad logical_as_tensor={logical_as_tensor}"
    dtype = ttnn.bfloat16

    full_compute_grid = submesh.compute_with_storage_grid_size()
    sdpa_compute_grid = (full_compute_grid.x, full_compute_grid.y - 1)
    ccl_core_grid_offset = (0, full_compute_grid.y - 1)

    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group([worker_sub_device_id])

    ccl_semaphore_handles = [ttnn.create_global_semaphore(submesh, ccl_sub_device_crs, 0) for _ in range(2)]

    # ---- Joint-shard input dims: both spatial and prompt sharded on rp_axis seq dim ----
    sdpa_input_shard_dims = [None, None]
    sdpa_input_shard_dims[rp_axis] = 2
    sdpa_input_shard_dims[up_axis] = 1

    logical_l = padded_joint_seq_len if logical_l is None else logical_l
    assert (
        0 < logical_l <= padded_joint_seq_len
    ), f"logical_l={logical_l} must be in (0, padded_joint_seq_len={padded_joint_seq_len}]"

    # ---- PyTorch reference data ----
    Q = fa_rand(b, nh, base_seq_len, d)
    K = fa_rand(b, nh, base_seq_len, d)
    V = fa_rand(b, nh, base_seq_len, d)
    # Real joint tokens; the padded tail is zero-filled below.
    joint_Q_real = fa_rand(b, nh, logical_l, d)
    joint_K_real = fa_rand(b, nh, logical_l, d)
    joint_V_real = fa_rand(b, nh, logical_l, d)

    joint_pad = padded_joint_seq_len - logical_l
    joint_Q = torch.cat([joint_Q_real, torch.zeros(b, nh, joint_pad, d)], dim=2) if joint_pad else joint_Q_real
    joint_K = torch.cat([joint_K_real, torch.zeros(b, nh, joint_pad, d)], dim=2) if joint_pad else joint_K_real
    joint_V = torch.cat([joint_V_real, torch.zeros(b, nh, joint_pad, d)], dim=2) if joint_pad else joint_V_real

    # When base_seq_len (logical_n) is NOT tile-aligned, the last real spatial tile is a chunk-final
    # sub-tile partial. Fill the spatial K/V pad tail with large garbage instead of zeros so a missing
    # global_n partial-column mask is OBSERVABLE: a leaked zero key barely moves PCC, but leaked garbage
    # collapses it. Tile-aligned cases keep zero pad (no partial column to expose) -> unchanged behavior.
    spatial_pad = padded_seq_len - base_seq_len
    if spatial_pad > 0 and base_seq_len % ttnn.TILE_SIZE != 0:
        spatial_pad_K = 8.0 * fa_rand(b, nh, spatial_pad, d)
        spatial_pad_V = 8.0 * fa_rand(b, nh, spatial_pad, d)
    else:
        spatial_pad_K = torch.zeros(b, nh, spatial_pad, d)
        spatial_pad_V = torch.zeros(b, nh, spatial_pad, d)
    padded_Q = torch.cat([Q, torch.zeros(b, nh, spatial_pad, d)], dim=2)
    padded_K = torch.cat([K, spatial_pad_K], dim=2)
    padded_V = torch.cat([V, spatial_pad_V], dim=2)

    # Ground truth attends only real (unpadded) spatial + joint keys; the kernel masks both tails.
    pt_Q_full = torch.cat([Q, joint_Q_real], dim=2)
    pt_K_full = torch.cat([K, joint_K_real], dim=2)
    pt_V_full = torch.cat([V, joint_V_real], dim=2)
    gt_full = torch.nn.functional.scaled_dot_product_attention(pt_Q_full, pt_K_full, pt_V_full, is_causal=False)
    gt_spatial = gt_full[:, :, :base_seq_len, :]
    gt_joint = gt_full[:, :, base_seq_len : base_seq_len + logical_l, :]

    # ---- TT persistent buffers for spatial K/V ----
    kv_shard_dims = [None, None]
    kv_shard_dims[up_axis] = 1
    ag_kv_shape = (b, nh, padded_seq_len, d)
    persistent_kv_bufs = [
        ttnn.from_torch(
            torch.zeros(ag_kv_shape),
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=kv_shard_dims),
        )
        for _ in range(2)
    ]

    # ---- TT persistent buffers for gathered joint K/V (full L, replicated on rp_axis) ----
    joint_kv_shard_dims = [None, None]
    joint_kv_shard_dims[up_axis] = 1
    ag_joint_shape = (b, nh, padded_joint_seq_len, d)
    persistent_joint_kv_bufs = [
        ttnn.from_torch(
            torch.zeros(ag_joint_shape),
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=joint_kv_shard_dims),
        )
        for _ in range(2)
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

    # ---- joint tensors sharded on rp_axis seq (L/P per device) ----
    joint_shard_dims_rp = [None, None]
    joint_shard_dims_rp[rp_axis] = 2
    joint_shard_dims_rp[up_axis] = 1
    tt_joint_Q = ttnn.from_torch(
        joint_Q,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=joint_shard_dims_rp),
    )
    tt_joint_K = ttnn.from_torch(
        joint_K,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=joint_shard_dims_rp),
    )
    tt_joint_V = ttnn.from_torch(
        joint_V,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=joint_shard_dims_rp),
    )

    logger.debug(
        f"Sharded-joint test: Q={tt_Q.shape}, joint_Q={tt_joint_Q.shape}, "
        f"joint_padded_per_device={padded_joint_seq_len // rp_factor}, "
        f"padded_joint_seq_len={padded_joint_seq_len}, logical_l={logical_l}"
    )

    tt_out, tt_joint_out, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        tt_joint_Q,
        tt_joint_K,
        tt_joint_V,
        persistent_output_buffer_k=persistent_kv_bufs[0],
        persistent_output_buffer_v=persistent_kv_bufs[1],
        joint_strategy="rear",
        logical_n=logical_length_tensor(submesh, base_seq_len) if tensor_mode in ("both", "n") else base_seq_len,
        logical_l=logical_length_tensor(submesh, logical_l) if tensor_mode in ("both", "l") else logical_l,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dim=2,
        multi_device_global_semaphore=ccl_semaphore_handles,
        num_links=num_links,
        cluster_axis=rp_axis,
        mesh_device=submesh,
        topology=topology,
        subdevice_id=worker_sub_device_id,
        ccl_core_grid_offset=ccl_core_grid_offset,
        persistent_output_buffer_joint_k=persistent_joint_kv_bufs[0],
        persistent_output_buffer_joint_v=persistent_joint_kv_bufs[1],
    )
    logger.info(f"Done processing...")

    ttnn.synchronize_device(submesh)

    logger.info(f"Done synchronizing...")

    # ---- Spatial output: concat along rp seq, trim padding ----
    tt_out_pt = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_out_pt = tt_out_pt[:, :, :base_seq_len, :]

    out_pass, out_pcc = comp_pcc(tt_out_pt, gt_spatial, pcc_threshold)
    logger.info(f"[sharded-joint] spatial PCC={out_pcc}")
    assert out_pass, f"Spatial PCC {out_pcc} below threshold {pcc_threshold}"

    # ---- Joint output: each device holds its L/P shard; concat to full L ----
    tt_joint_out_pt = ttnn.to_torch(
        tt_joint_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=joint_shard_dims_rp),
    )
    tt_joint_out_pt = tt_joint_out_pt[:, :, :logical_l, :]

    jout_pass, jout_pcc = comp_pcc(tt_joint_out_pt, gt_joint, pcc_threshold)
    logger.info(f"[sharded-joint] joint PCC={jout_pcc}")
    assert jout_pass, f"Joint PCC {jout_pcc} below threshold {pcc_threshold}"


# The full matrix above covers "neither" and "both"; this covers the two mixed modes on a shape with a
# sub-tile partial column on BOTH tails, where a crossed-over transport shows immediately.
@pytest.mark.parametrize(
    "mesh_device, sp_axis, b, nh, base_seq_len, padded_joint_seq_len, d, q_chunk_size, k_chunk_size, logical_l",
    [
        ((2, 4), 0, 1, 24, 63, 64, 64, 32, 32, 63),
        ((4, 8), 0, 1, 24, 127, 128, 64, 32, 32, 127),
    ],
    ids=["m2x4", "m4x8"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("logical_as_tensor", ["n", "l"], ids=["n_only", "l_only"])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["line"],
)
def test_ring_joint_sdpa_sharded_prompt_mixed_logical_transport(
    mesh_device,
    sp_axis,
    b,
    nh,
    base_seq_len,
    padded_joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    logical_l,
    logical_as_tensor,
    all_gather_topology,
    reset_seeds,
):
    """One logical length as a device tensor, the other as a host int."""
    sp_factor = mesh_device.shape[sp_axis]
    up_axis = 1 - sp_axis
    num_links = sharded_prompt_num_links(mesh_device.shape)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*mesh_device.shape))
    submesh.cache_entries_counter = CacheEntriesCounter(submesh)
    padded_seq_len = get_padded_vision_seq_len(base_seq_len, sp_factor)
    run_ring_joint_sdpa_sharded_prompt(
        submesh,
        b=b,
        nh=nh,
        base_seq_len=base_seq_len,
        padded_seq_len=padded_seq_len,
        padded_joint_seq_len=padded_joint_seq_len,
        d=d,
        rp_axis=sp_axis,
        rp_factor=sp_factor,
        up_axis=up_axis,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        num_links=num_links,
        logical_l=logical_l,
        logical_as_tensor=logical_as_tensor,
        topology=all_gather_topology,
    )


def sharded_prompt_num_links(mesh_shape):
    shape = tuple(mesh_shape)
    # (wh_links, bh_links)
    ground_truth = {
        (2, 2): (2, 2),
        (2, 4): (1, 2),
        (4, 8): (4, 2),
    }
    assert shape in ground_truth, f"No num_links ground truth for mesh shape {shape}"
    wh_links, bh_links = ground_truth[shape]
    return bh_links if is_blackhole() else wh_links


# The same set of tail-masking use cases is run on each mesh shape (all joint-tail but the last).
#   - packed:      logical_l == padded joint, chunk == per-shard tiles (Sk_chunk_t=1).
#   - straddle:    last shard carries a chunk-final sub-tile partial (Sk_chunk_t=1).
#   - emptyshards: trailing shard(s) fully empty + one straddle shard.
#   - sk2partial:  Sk_chunk_t=2 chunk-final sub-tile partial (tile-count aligned yet 31-col partial).
#   - packed_sk2:  logical_l == padded joint, chunk == per-shard tiles (Sk_chunk_t=2).
#   - localn:      chunk wider than per-shard joint tiles, tile-aligned, packed (pure active_Sk
#                  narrowing over zero-filled pad tiles; the (4,8) case is the FLUX.2 prod shape).
#   - spatial_straddle: SPATIAL tail instead — base_seq_len (logical_n) chunk-final sub-tile partial
#                  on the last spatial shard, joint packed (reproduces the spatial global_n gate hole).
@pytest.mark.parametrize(
    "mesh_device, sp_axis, b, nh, base_seq_len, padded_joint_seq_len, d, q_chunk_size, k_chunk_size, logical_l",
    [
        # ---- mesh (2,4) ----
        ((2, 4), 0, 1, 24, 64, 64, 64, 32, 32, None),  # packed: 2 full shards
        ((2, 4), 0, 1, 24, 64, 64, 64, 32, 32, 63),  # straddle: shard0 full, shard1 31+1 pad
        ((2, 4), 0, 1, 24, 64, 64, 64, 32, 32, 20),  # emptyshards: shard0 20+12 pad, shard1 empty
        ((2, 4), 0, 1, 24, 128, 128, 64, 64, 64, 63),  # sk2partial: 64/shard=2t, shard0 partial, shard1 empty
        ((2, 4), 0, 1, 24, 64, 128, 64, 64, 64, None),  # packed_sk2: 64/shard=2t, chunk=2t
        ((2, 4), 0, 1, 24, 64, 128, 64, 64, 128, None),  # localn: 64/shard=2t inside a 4t chunk
        ((2, 4), 0, 1, 24, 63, 64, 64, 32, 32, None),  # spatial_straddle: spatial shard1 31+1 pad, joint packed
        # ---- mesh (4,8) ----
        ((4, 8), 0, 1, 24, 64, 128, 64, 32, 32, None),  # packed: 4 full shards
        ((4, 8), 0, 1, 24, 64, 128, 64, 32, 32, 127),  # straddle: shards0-2 full, shard3 31+1 pad
        ((4, 8), 0, 1, 24, 64, 128, 64, 32, 32, 40),  # emptyshards: shard0 full, shard1 8+24, shards2-3 empty
        ((4, 8), 0, 1, 24, 256, 256, 64, 64, 64, 191),  # sk2partial: 64/shard=2t, shard2 partial, shard3 empty
        ((4, 8), 0, 1, 24, 64, 256, 64, 64, 64, None),  # packed_sk2: 64/shard=2t, chunk=2t
        # localn: sp_axis=1 -> sp_factor=8, so 512/8=64=2t inside a 16t chunk (FLUX.2 prod shape)
        ((4, 8), 1, 1, 24, 4096, 512, 128, 256, 512, 512),
        ((4, 8), 0, 1, 24, 127, 128, 64, 32, 32, None),  # spatial_straddle: spatial shard3 31+1 pad, joint packed
    ],
    ids=[
        "m2x4_packed",
        "m2x4_straddle",
        "m2x4_emptyshards",
        "m2x4_sk2partial",
        "m2x4_packed_sk2",
        "m2x4_localn",
        "m2x4_spatial_straddle",
        "m4x8_packed",
        "m4x8_straddle",
        "m4x8_emptyshards",
        "m4x8_sk2partial",
        "m4x8_packed_sk2",
        "m4x8_localn",
        "m4x8_spatial_straddle",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["line"],
)
# Both ways of supplying logical_n / logical_l must agree over the whole matrix above.
@pytest.mark.parametrize("logical_as_tensor", [False, True], ids=["scalar", "tensor"])
def test_ring_joint_sdpa_sharded_prompt(
    mesh_device,
    sp_axis,
    b,
    nh,
    base_seq_len,
    padded_joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    logical_l,
    logical_as_tensor,
    all_gather_topology,
    reset_seeds,
):
    """
    Functional correctness test for the B1-a sharded-joint path.
    joint_tensor_q/k/v are sharded L/P per device; logical_l activates the internal gather.
    padded_joint_seq_len is the device tensor length; logical_l (or padded_joint_seq_len when None)
    is the real joint length driving the tail mask.
    """
    sp_factor = mesh_device.shape[sp_axis]
    up_axis = 1 - sp_axis
    num_links = sharded_prompt_num_links(mesh_device.shape)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*mesh_device.shape))
    submesh.cache_entries_counter = CacheEntriesCounter(submesh)
    padded_seq_len = get_padded_vision_seq_len(base_seq_len, sp_factor)
    assert padded_joint_seq_len % sp_factor == 0, "padded_joint_seq_len must be divisible by sp_factor"
    run_ring_joint_sdpa_sharded_prompt(
        submesh,
        b=b,
        nh=nh,
        base_seq_len=base_seq_len,
        padded_seq_len=padded_seq_len,
        padded_joint_seq_len=padded_joint_seq_len,
        d=d,
        rp_axis=sp_axis,
        rp_factor=sp_factor,
        up_axis=up_axis,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        num_links=num_links,
        logical_l=logical_l,
        logical_as_tensor=logical_as_tensor,
        topology=all_gather_topology,
    )


# One captured ring-joint SDPA segment plus its runtime args; 32 MB is ample headroom.
LOGICAL_TENSOR_TRACE_REGION_SIZE = 32 * 1024 * 1024


@pytest.mark.parametrize(
    "mesh_device, sp_axis, b, nh, padded_seq_len, padded_joint_seq_len, d, q_chunk_size, k_chunk_size, logical_pairs",
    [
        # (logical_n, logical_l) pairs moving every live-driven geometry: full, sub-tile tails, emptied
        # iterations, mixed. logical_l must exceed the per-device joint shard to select the sharded path;
        # a fully-empty iteration needs ring_size >= 3, so the (4,8) row carries that case.
        ((2, 4), 0, 1, 24, 128, 128, 64, 32, 32, [(128, 128), (95, 127), (64, 65), (127, 100)]),
        ((4, 8), 0, 1, 24, 256, 256, 64, 32, 32, [(256, 256), (191, 127), (64, 65), (255, 200)]),
    ],
    ids=["m2x4", "m4x8"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": LOGICAL_TENSOR_TRACE_REGION_SIZE,
            },
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=["line"],
)
# Both joint layouts, because they reach the transport differently: the sharded joint varies logical_n AND
# logical_l, while the replicated joint (the DiT / minimax_h3 usage) pins logical_l by shape and varies
# logical_n alone, leaving logical_lt compile-time.
@pytest.mark.parametrize("joint_sharded", [True, False], ids=["sharded_joint", "replicated_joint"])
def test_ring_joint_sdpa_logical_tensor_trace_replay(
    mesh_device,
    sp_axis,
    b,
    nh,
    padded_seq_len,
    padded_joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    logical_pairs,
    joint_sharded,
    all_gather_topology,
    reset_seeds,
):
    """ONE captured trace, replayed once per (logical_n, logical_l) pair with only the length tensors
    refreshed in place, must match the host-scalar path bit-for-bit.

    Inputs are uploaded once, so the lengths are the only thing distinguishing one replay from the next.
    Catches the two multi-dispatch failure modes single-dispatch tests cannot: a stale read of the
    fixed-address tensors (missing invalidate_l1_cache), and an iteration emptied by the live length but
    marked active by the placeholder-derived mask (compute waits on work that never arrives).
    """
    sp_factor = mesh_device.shape[sp_axis]
    up_axis = 1 - sp_axis
    num_links = sharded_prompt_num_links(mesh_device.shape)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*mesh_device.shape))
    submesh.cache_entries_counter = CacheEntriesCounter(submesh)
    assert padded_seq_len % sp_factor == 0, "padded_seq_len must be divisible by sp_factor"
    assert padded_joint_seq_len % sp_factor == 0, "padded_joint_seq_len must be divisible by sp_factor"
    submesh.enable_program_cache()  # a trace replays cached programs; capture cannot JIT-compile

    dtype = ttnn.bfloat16
    full_compute_grid = submesh.compute_with_storage_grid_size()
    sdpa_compute_grid = (full_compute_grid.x, full_compute_grid.y - 1)
    ccl_core_grid_offset = (0, full_compute_grid.y - 1)

    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group([worker_sub_device_id])
    ccl_semaphore_handles = [ttnn.create_global_semaphore(submesh, ccl_sub_device_crs, 0) for _ in range(2)]

    spatial_shard_dims = [None, None]
    spatial_shard_dims[sp_axis] = 2
    spatial_shard_dims[up_axis] = 1
    kv_buf_shard_dims = [None, None]
    kv_buf_shard_dims[up_axis] = 1
    # Sharded joint: L/P per device (seq sharded), needing the internal gather. Replicated: full L per
    # device, head-sharded only.
    joint_shard_dims = list(spatial_shard_dims) if joint_sharded else list(kv_buf_shard_dims)
    # A replicated joint output is a full copy per ring device; concat the replicas into batch so the
    # comparison covers every one of them.
    joint_composer_dims = list(joint_shard_dims)
    if not joint_sharded:
        joint_composer_dims[sp_axis] = 0

    # Garbage (not zero) everywhere: rows past logical_n / logical_l must be masked out, and leaked
    # garbage collapses the comparison where a leaked zero would barely move it.
    padded_Q = 8.0 * fa_rand(b, nh, padded_seq_len, d)
    padded_K = 8.0 * fa_rand(b, nh, padded_seq_len, d)
    padded_V = 8.0 * fa_rand(b, nh, padded_seq_len, d)
    joint_Q = 8.0 * fa_rand(b, nh, padded_joint_seq_len, d)
    joint_K = 8.0 * fa_rand(b, nh, padded_joint_seq_len, d)
    joint_V = 8.0 * fa_rand(b, nh, padded_joint_seq_len, d)

    def upload(host_tensor, dims):
        return ttnn.from_torch(
            host_tensor,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=dims),
        )

    # Allocated ONCE: under trace these addresses are baked into the capture.
    tt_Q = upload(padded_Q, spatial_shard_dims)
    tt_K = upload(padded_K, spatial_shard_dims)
    tt_V = upload(padded_V, spatial_shard_dims)
    tt_joint_Q = upload(joint_Q, joint_shard_dims)
    tt_joint_K = upload(joint_K, joint_shard_dims)
    tt_joint_V = upload(joint_V, joint_shard_dims)
    persistent_kv_bufs = [
        upload(torch.zeros(b, nh, padded_seq_len, d), kv_buf_shard_dims),
        upload(torch.zeros(b, nh, padded_seq_len, d), kv_buf_shard_dims),
    ]
    persistent_joint_kv_bufs = [
        upload(torch.zeros(b, nh, padded_joint_seq_len, d), kv_buf_shard_dims),
        upload(torch.zeros(b, nh, padded_joint_seq_len, d), kv_buf_shard_dims),
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

    tt_logical_n = logical_length_tensor(submesh, logical_pairs[0][0])
    tt_logical_l = logical_length_tensor(submesh, logical_pairs[0][1]) if joint_sharded else None

    def load_lengths(logical_n, logical_l):
        """Deliver a pair the way a traced runner does: in-place refresh, no reallocation."""
        ttnn.copy_host_to_device_tensor(logical_length_tensor(submesh, logical_n, on_device=False), tt_logical_n)
        if joint_sharded:
            ttnn.copy_host_to_device_tensor(logical_length_tensor(submesh, logical_l, on_device=False), tt_logical_l)

    def call(logical_n, logical_l):
        joint_kwargs = (
            {
                "logical_l": logical_l,
                "persistent_output_buffer_joint_k": persistent_joint_kv_bufs[0],
                "persistent_output_buffer_joint_v": persistent_joint_kv_bufs[1],
            }
            if joint_sharded
            else {}
        )
        return ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            tt_joint_Q,
            tt_joint_K,
            tt_joint_V,
            persistent_output_buffer_k=persistent_kv_bufs[0],
            persistent_output_buffer_v=persistent_kv_bufs[1],
            joint_strategy="rear",
            logical_n=logical_n,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=ccl_semaphore_handles,
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=submesh,
            topology=all_gather_topology,
            subdevice_id=worker_sub_device_id,
            ccl_core_grid_offset=ccl_core_grid_offset,
            **joint_kwargs,
        )

    spatial_composer = ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=spatial_shard_dims)
    joint_composer = ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=joint_composer_dims)

    def valid_rows(tt_out, tt_joint_out, logical_n, logical_l):
        # A replicated joint has no logical_l to vary: every row of it is real.
        joint_rows = logical_l if joint_sharded else padded_joint_seq_len
        spatial = ttnn.to_torch(tt_out, mesh_composer=spatial_composer)[:, :, :logical_n, :]
        joint = ttnn.to_torch(tt_joint_out, mesh_composer=joint_composer)[:, :, :joint_rows, :]
        return spatial, joint

    trace_id = None
    try:
        # 1) Host-scalar references, eager, one program per pair. Same device inputs as the replays.
        references = []
        for logical_n, logical_l in logical_pairs:
            tt_out, tt_joint_out, tt_stats = call(logical_n, logical_l)
            ttnn.synchronize_device(submesh)
            references.append(valid_rows(tt_out, tt_joint_out, logical_n, logical_l))
            for t in (tt_out, tt_joint_out, tt_stats):
                ttnn.deallocate(t)

        # Guard against a vacuous pass: if every pair produced the same numbers, a replay reading a stale
        # length would still match. Compare each pair's spatial output over the rows they share.
        for i in range(1, len(logical_pairs)):
            shared = min(logical_pairs[i][0], logical_pairs[0][0])
            assert not torch.equal(references[i][0][:, :, :shared, :], references[0][0][:, :, :shared, :]), (
                f"pairs {logical_pairs[0]} and {logical_pairs[i]} give identical output over their shared "
                f"{shared} rows, so replays could not distinguish them; pick different lengths"
            )

        # 2) Compile the tensor-path program, then capture it ONCE. Its host scalars are the placeholders,
        #    identical for every pair, so the capture holds no per-pair information at all.
        load_lengths(*logical_pairs[0])
        warm = call(tt_logical_n, tt_logical_l)
        ttnn.synchronize_device(submesh)
        for t in warm:
            ttnn.deallocate(t)

        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        tt_out_traced, tt_joint_out_traced, _ = call(tt_logical_n, tt_logical_l)
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        ttnn.synchronize_device(submesh)

        # 3) Replay the single capture. The order is deliberately not ascending: a forward pass, then a
        #    descending pass (where a stale length is too LARGE and over-attends), then out of order.
        n_pairs = len(logical_pairs)
        replay_order = list(range(n_pairs)) + list(reversed(range(n_pairs))) + [0, n_pairs - 1, 1]
        for replay_idx, i in enumerate(replay_order):
            logical_n, logical_l = logical_pairs[i]
            load_lengths(logical_n, logical_l)
            ttnn.execute_trace(submesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(submesh)
            got_spatial, got_joint = valid_rows(tt_out_traced, tt_joint_out_traced, logical_n, logical_l)
            ref_spatial, ref_joint = references[i]
            for name, got, ref in (("spatial", got_spatial, ref_spatial), ("joint", got_joint, ref_joint)):
                assert torch.equal(got, ref), (
                    f"replay {replay_idx} of order {replay_order}: pair {i} "
                    f"(logical_n={logical_n}, logical_l={logical_l}): traced {name} output differs from the "
                    f"host-scalar path (max abs diff {(got - ref).abs().max().item()}). Matching pair "
                    f"{i - 1} instead points at a stale length read."
                )
            logger.info(f"logical-tensor trace replay {replay_idx} (pair {i}={logical_pairs[i]}): bit-exact")

        logger.success(
            f"logical-tensor trace: 1 capture, {len(replay_order)} replays over {n_pairs} "
            f"(logical_n, logical_l) pairs (order {replay_order}), all bit-exact vs the host-scalar path"
        )
    finally:
        if trace_id is not None:
            ttnn.release_trace(submesh, trace_id)
        submesh.reset_sub_device_stall_group()
        submesh.clear_loaded_sub_device_manager()
        submesh.remove_sub_device_manager(sub_device_manager)
