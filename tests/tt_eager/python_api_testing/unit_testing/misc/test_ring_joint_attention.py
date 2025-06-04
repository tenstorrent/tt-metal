# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import torch.nn.functional as F
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0, skip_for_blackhole
from .test_scaled_dot_product_attention import fa_rand


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


def run_ring_joint_sdpa(
    device,
    b,
    nh,
    seq_len,
    joint_seq_len,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    use_high_precision_compute=False,
    grid_size=None,
    # topology=ttnn.MeshTopology.RING,
):
    num_devices = 8
    local_seq_len = seq_len // num_devices
    torch.manual_seed(1234)

    compute_grid = grid_size or device.compute_with_storage_grid_size()

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=compute_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    if use_high_precision_compute:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    Q = fa_rand(b, nh, local_seq_len, d)
    K = fa_rand(b, nh, seq_len, d)
    V = fa_rand(b, nh, seq_len, d)

    joint_Q = fa_rand(b, nh, joint_seq_len, d)
    joint_K = fa_rand(b, nh, joint_seq_len, d)
    joint_V = fa_rand(b, nh, joint_seq_len, d)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_joint_Q = ttnn.from_torch(joint_Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_joint_K = ttnn.from_torch(joint_K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_joint_V = ttnn.from_torch(joint_V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out, tt_joint_out, tt_lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        tt_joint_Q,
        tt_joint_K,
        tt_joint_V,
        joint_strategy="rear",
        logical_n=seq_len,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    print(f"tt_lse.shape: {tt_lse.shape}")
    tt_out = ttnn.to_torch(tt_out)
    tt_joint_out = ttnn.to_torch(tt_joint_out)
    tt_lse = ttnn.to_torch(tt_lse)
    tt_lse_list = torch.chunk(tt_lse, num_devices, dim=0)
    # Slice out any tile-padding
    tt_out = tt_out[:, :, :local_seq_len, :]
    tt_joint_out = tt_joint_out[:, :, :joint_seq_len, :]
    logger.debug(f"tt_out: {tt_out.shape}")
    logger.debug(f"tt_joint_out: {tt_joint_out.shape}")

    gt, gt_lse_list = torch_sdpa(Q, K, V, joint_Q, joint_K, joint_V, num_devices)
    gt_out = gt[:, :, :local_seq_len, :]
    gt_joint_out = gt[:, :, local_seq_len:, :]

    passing = True
    for out, gt in [(tt_out, gt_out), (tt_joint_out, gt_joint_out)]:
        out_pass, out_pcc = comp_pcc(gt, out, 0.994)
        logger.debug(f"python vs pytorch: {out_pcc}")
        logger.debug(f"mse: {((gt - out) ** 2).mean()}")
        passing = passing and out_pass

    for i, (lse, gt) in enumerate(zip(tt_lse_list, gt_lse_list)):
        lse = lse[:, :, : local_seq_len + joint_seq_len]
        lse_pass, lse_pcc = comp_pcc(gt, lse, 0.98)
        logger.debug(f"python vs pytorch LSE {i}: {lse_pcc}")
        logger.debug(f"mse: {((gt - lse) ** 2).mean()}")
        passing = passing and lse_pass
        print(f"tt: max: {lse.max()}, min: {lse.min()}, mean: {lse.mean()}")
        print(f"gt: max: {gt.max()}, min: {gt.min()}, mean: {gt.mean()}")

    assert passing


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [64], ids=["q256"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k256"])
@pytest.mark.parametrize("b", [1], ids=["b1"])
@pytest.mark.parametrize("nh", [5], ids=["nh5"])
@pytest.mark.parametrize("d", [64], ids=["d64"])
@pytest.mark.parametrize(
    "seq_len, joint_seq_len",
    [
        (4096, 333),
    ],
)
def test_ring_joint_sdpa(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype, reset_seeds):
    if q_chunk_size == 512 and k_chunk_size == 512:
        pytest.skip("OOM config.")
    ttnn.device.DisablePersistentKernelCache()
    run_ring_joint_sdpa(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [64, 128], ids=["q64", "q128"])
@pytest.mark.parametrize("k_chunk_size", [64, 128], ids=["k64", "k128"])
@pytest.mark.parametrize(
    "b, nh, seq_len, joint_seq_len, d",
    [
        (1, 5, 4096, 333, 64),  # SD3.5 large
        (1, 5, 1024, 118, 64),
        (1, 3, 44 * 1024, 118, 128),  # Mochi
        (1, 2, 8192, 666, 64),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "worker_l1_size": 1344544,  # This increases space for ring-buffer, required for large kernel binary
        }
    ],
    indirect=True,
)
def test_ring_joint_sdpa_stress(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype):
    if q_chunk_size == 512 and k_chunk_size == 512:
        pytest.skip("OOM config.")
    ttnn.device.DisablePersistentKernelCache()
    run_ring_joint_sdpa(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype)


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [64], ids=["q256"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k256"])
@pytest.mark.parametrize("b", [1], ids=["b1"])
@pytest.mark.parametrize("nh", [40], ids=["nh40"])
@pytest.mark.parametrize("d", [64], ids=["d64"])
@pytest.mark.parametrize(
    "seq_len, joint_seq_len",
    [
        (4096, 333),
    ],
)
@pytest.mark.parametrize("n_iters, trace_enabled", [(1, False), (10, True)], ids=["no_trace", "yes_trace"])
@pytest.mark.parametrize("num_links", [1])
@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        # (
        #     {"worker_l1_size": 1344544, "trace_region_size": 200000, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
        #     ttnn.Topology.Ring,
        # ),
        (
            {"worker_l1_size": 1344544, "trace_region_size": 200000, "fabric_config": ttnn.FabricConfig.FABRIC_1D},
            ttnn.Topology.Linear,
        ),
    ],
    indirect=["device_params"],
    ids=[
        # "ring",
        "line",
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 2, 1, 4],  # 2x4 RP x UP
        [0, 2, 1, 2],  # 2x2 RP x UP
        [0, 2, 1, 1],  # 2x1 RP x UP
        [1, 2, 0, 2],  # 2x2 UP x RP
        [1, 2, 0, 1],  # 1x2 UP x RP
        [1, 4, 0, 1],  # 1x4 UP x RP
        [1, 8, 0, 1],  # 1x8 UP x RP
    ],
    ids=[
        "2rpx4up",
        "2rpx2up",
        "2rpx1up",
        "2upx2rp",
        "1upx2rp",
        "1upx4rp",
        "1upx8rp",
    ],
)
def test_ring_joint_sdpa_perf(
    mesh_device,
    use_program_cache,
    b,
    nh,
    seq_len,
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
    all_gather_topology,
):
    if rp_factor == 8 and rp_axis == 1:
        mesh_device.reshape(ttnn.MeshShape(1, 8))

    mesh_device_shape = list(mesh_device.shape)
    assert mesh_device_shape[rp_axis] >= rp_factor and mesh_device_shape[up_axis] >= up_factor
    submesh_shape = [0, 0]
    submesh_shape[rp_axis] = rp_factor
    submesh_shape[up_axis] = up_factor
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    print(f"RP axis: {rp_axis} factor: {rp_factor}, UP axis: {up_axis} factor: {up_factor}")
    print(f"submesh: {submesh.shape}")
    num_devices = rp_factor * up_factor

    full_compute_grid = submesh.compute_with_storage_grid_size()
    sdpa_compute_grid = (8, 7)
    ccl_core_grid_offset = (0, 7)

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
    ccl_semaphore_handles = [
        create_global_semaphores(submesh, num_devices, ccl_sub_device_crs, 0) for _ in range(n_iters)
    ]

    kv_shard_dims = [None, None]
    kv_shard_dims[rp_axis] = None  # Output of AllGather is not sharded on RP axis
    kv_shard_dims[up_axis] = 1  # UP shards on heads dim1

    # Create persistent output buffers
    ag_output_shape = (b, nh, seq_len, d)
    persistent_intermediate_buffers = [
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

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, seq_len, d)
    K = fa_rand(b, nh, seq_len, d)
    V = fa_rand(b, nh, seq_len, d)

    joint_Q = fa_rand(b, nh, joint_seq_len, d)
    joint_K = fa_rand(b, nh, joint_seq_len, d)
    joint_V = fa_rand(b, nh, joint_seq_len, d)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    sdpa_input_shard_dims = [None, None]
    sdpa_input_shard_dims[rp_axis] = 2  # sequence dim
    sdpa_input_shard_dims[up_axis] = 1  # head dim

    # Joint input only sharded on head dim
    sdpa_joint_shard_dims = [None, None]
    sdpa_joint_shard_dims[up_axis] = 1  # head dim

    tt_Q = ttnn.from_torch(
        Q,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_K = ttnn.from_torch(
        K,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_V = ttnn.from_torch(
        V,
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

    print(f"tt_Q: {tt_Q.shape}")
    print(f"tt_joint_Q: {tt_joint_Q.shape}")

    tt_out_list = []
    tt_joint_out_list = []

    def run_iters(tt_out_list, tt_joint_out_list):
        for i in range(n_iters):
            # K_gathered = ttnn.experimental.all_gather_async(
            #     tt_K,
            #     dim=2,
            #     multi_device_global_semaphore=[ccl_semaphore_handles[i][0]],
            #     num_links=1,
            #     topology=ttnn.Topology.Ring,
            #     subdevice_id=worker_sub_device_id,
            # )
            # V_gathered = ttnn.experimental.all_gather_async(
            #     tt_V,
            #     dim=2,
            #     multi_device_global_semaphore=[ccl_semaphore_handles[i][1]],
            #     num_links=1,
            #     topology=ttnn.Topology.Ring,
            #     subdevice_id=worker_sub_device_id,
            # )
            tt_out, tt_joint_out, tt_lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_joint_Q,
                tt_joint_K,
                tt_joint_V,
                persistent_intermediate_buffer_k=persistent_intermediate_buffers[i][0],
                persistent_intermediate_buffer_v=persistent_intermediate_buffers[i][1],
                persistent_output_buffer_k=persistent_output_buffers[i][0],
                persistent_output_buffer_v=persistent_output_buffers[i][1],
                joint_strategy="rear",
                logical_n=seq_len,
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
        print("Compile run")
        run_iters([], [])
        print("Capture trace")
        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        run_iters(tt_out_list, tt_joint_out_list)
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        ttnn.synchronize_device(submesh)
        print("Execute trace")
        ttnn.execute_trace(submesh, trace_id, blocking=False)
        ttnn.release_trace(submesh, trace_id)
        ttnn.synchronize_device(submesh)

    else:
        print("Run without trace")
        run_iters(tt_out_list, tt_joint_out_list)

    gt, gt_lse_list = torch_sdpa(Q, K, V, joint_Q, joint_K, joint_V, num_devices)
    gt_out = gt[:, :, :seq_len, :]
    gt_joint_out = gt[:, :, seq_len:, :]

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
            mesh_composer=ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=joint_shard_dims),
        )[:1]
        # Slice out any tile-padding
        tt_out = tt_out[:, :, :seq_len, :]
        tt_joint_out = tt_joint_out[:, :, :joint_seq_len, :]
        logger.debug(f"tt_out: {tt_out.shape}")
        logger.debug(f"tt_joint_out: {tt_joint_out.shape}")

        passing = True
        for out, gt in [(tt_out, gt_out), (tt_joint_out, gt_joint_out)]:
            out_pass, out_pcc = comp_pcc(gt, out, 0.994)
            logger.debug(f"python vs pytorch: {out_pcc}")
            logger.debug(f"mse: {((gt - out) ** 2).mean()}")
            passing = passing and out_pass

        assert passing

    ttnn.close_mesh_device(submesh)


@skip_for_grayskull("Unsupported in GS since L1 runs OOM with most configs")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [128], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize("b", [1], ids=["b1"])
@pytest.mark.parametrize("nh", [1], ids=["nh1"])
@pytest.mark.parametrize(
    "seq_len, joint_seq_len",
    [
        (4096, 100),
    ],
)
@pytest.mark.parametrize(
    "d",
    [128],
    ids=[
        "d128",
    ],
)
def test_ring_joint_sdpa_program_cache(
    device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype, use_program_cache
):
    dummy_tensors = []
    for _ in range(3):
        dummy_tensors.append(
            ttnn.from_torch(fa_rand(b, nh, seq_len, d), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        )
        run_ring_joint_sdpa(device, b, nh, seq_len, joint_seq_len, d, q_chunk_size, k_chunk_size, dtype, dummy_tensors)
