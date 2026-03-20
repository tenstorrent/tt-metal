# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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


def create_balanced_chunk_order(rp_factor):
    """Create balanced chunk order for sequence reordering.

    For rp_factor=4, creates 2*4=8 chunks with order: 0,7,1,6,2,5,3,4
    This interleaves chunks from start and end to balance workload.
    """
    num_chunks = 2 * rp_factor
    balanced_order = []

    left = 0
    right = num_chunks - 1

    for i in range(num_chunks):
        if i % 2 == 0:
            balanced_order.append(left)
            left += 1
        else:
            balanced_order.append(right)
            right -= 1

    return balanced_order


def reorder_tensor_chunks(tensor, chunk_order, seq_dim=2):
    """Reorder tensor chunks along sequence dimension according to chunk_order."""
    seq_len = tensor.shape[seq_dim]
    num_chunks = len(chunk_order)
    chunk_size = seq_len // num_chunks

    # Split into chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        if seq_dim == 2:
            chunks.append(tensor[:, :, start:end, :])
        else:
            raise NotImplementedError(f"Reordering for seq_dim={seq_dim} not implemented")

    # Reorder chunks according to chunk_order
    reordered_chunks = [chunks[i] for i in chunk_order]

    # Concatenate reordered chunks
    return torch.cat(reordered_chunks, dim=seq_dim)


def reverse_reorder_tensor_chunks(tensor, chunk_order, seq_dim=2):
    """Reverse the chunk reordering to restore original order."""
    # Create inverse permutation
    inverse_order = [0] * len(chunk_order)
    for new_pos, orig_pos in enumerate(chunk_order):
        inverse_order[orig_pos] = new_pos

    logger.debug(f"inverse order: {inverse_order}")
    return reorder_tensor_chunks(tensor, inverse_order, seq_dim)


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
    q_dtype,
    kv_dtype,
    n_iters,
    trace_enabled,
    num_links,
    rp_axis,
    up_axis,
    all_gather_topology,
    skip_check,
    pcc_threshold,
    max_mse=None,
    use_column_major_ccl=True,
    is_causal=False,
    is_balanced=False,
):
    full_compute_grid = submesh.compute_with_storage_grid_size()
    logger.info(f"Full grid: x {full_compute_grid.x} * y {full_compute_grid.y}")
    available_compute_grid = (12, 10)  # Set a fixed available compute grid for testing
    logger.warning(f"Using grid: x {available_compute_grid[0]} * y {available_compute_grid[1]}")
    sdpa_compute_grid = (available_compute_grid[0] - 1, available_compute_grid[1])
    ccl_core_grid_offset = (available_compute_grid[0] - 1, 0)

    # Basic CCL setup
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0), ttnn.CoreCoord(available_compute_grid[0] - 1, available_compute_grid[1] - 1)
            )
        }
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

    persistent_k_output_shard_dims = [None, None]
    if nhk != 1:
        persistent_k_output_shard_dims[up_axis] = 1

    persistent_output_buffers = [
        [
            ttnn.from_torch(
                torch.zeros(ag_output_shape_k),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=kv_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    submesh, mesh_shape=tuple(submesh.shape), dims=persistent_k_output_shard_dims
                ),
            ),
            ttnn.from_torch(
                torch.zeros(ag_output_shape_v),
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=kv_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=kv_shard_dims),
            ),
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

    Q = fa_rand(b, nhq, base_seq_len, head_dim_q)
    K = fa_rand(b, nhk, base_seq_len, head_dim_k)
    V = fa_rand(b, nhv, base_seq_len, head_dim_v)

    padded_Q = torch.cat([Q, torch.zeros(b, nhq, padded_seq_len - base_seq_len, head_dim_q)], dim=2)
    padded_K = torch.cat([K, torch.zeros(b, nhk, padded_seq_len - base_seq_len, head_dim_k)], dim=2)
    padded_V = torch.cat([V, torch.zeros(b, nhv, padded_seq_len - base_seq_len, head_dim_v)], dim=2)

    # Apply balanced reordering if requested
    chunk_order = None
    if is_balanced:
        rp_factor = submesh.shape[rp_axis]
        chunk_order = create_balanced_chunk_order(rp_factor)
        logger.info(f"Balanced reordering: rp_factor={rp_factor}, num_chunks={2*rp_factor}, order={chunk_order}")

        padded_Q = reorder_tensor_chunks(padded_Q, chunk_order, seq_dim=2)
        padded_K = reorder_tensor_chunks(padded_K, chunk_order, seq_dim=2)
        padded_V = reorder_tensor_chunks(padded_V, chunk_order, seq_dim=2)

    # Always create joint tensors (use dummy tensors when joint_seq_len = 0)
    joint_Q = fa_rand(b, nhq, joint_seq_len, head_dim_q)
    joint_K = fa_rand(b, nhk, joint_seq_len, head_dim_k)
    joint_V = fa_rand(b, nhv, joint_seq_len, head_dim_v)
    logger.debug(f"jointQ: {joint_Q.shape}")
    logger.debug(f"jointK: {joint_K.shape}")
    logger.debug(f"jointV: {joint_V.shape}")

    # Print shapes of all inputs along with input names
    logger.debug(f"padded_Q: {padded_Q.shape}")
    logger.debug(f"padded_K: {padded_K.shape}")
    logger.debug(f"padded_V: {padded_V.shape}")
    if is_balanced:
        logger.debug(f"Balanced reordering applied with chunk order: {chunk_order}")

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
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    tt_K = ttnn.from_torch(
        padded_K,
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_k_input_shard_dims),
    )
    tt_V = ttnn.from_torch(
        padded_V,
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims),
    )
    # Always convert joint tensors to ttnn (including dummy tensors)
    tt_joint_Q = ttnn.from_torch(
        joint_Q,
        dtype=q_dtype,
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
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=joint_k_mesh_mapper,
    )
    tt_joint_V = ttnn.from_torch(
        joint_V,
        dtype=kv_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_joint_shard_dims),
    )

    tt_out_list = []

    def run_iters(tt_out_list):
        for i in range(n_iters):
            logger.debug("Running ring-joint sdpa with the following shapes:")
            logger.debug(f"tt_Q: {tt_Q.shape}")
            logger.debug(f"tt_K: {tt_K.shape}")
            logger.debug(f"tt_V: {tt_V.shape}")
            logger.debug(f"tt_joint_Q: {tt_joint_Q.shape}")
            logger.debug(f"tt_joint_K: {tt_joint_K.shape}")
            logger.debug(f"tt_joint_V: {tt_joint_V.shape}")
            if joint_seq_len == 0:
                logger.debug("Using dummy joint tensors (joint_seq_len = 0)")

            # Always call with joint tensors and strategy (use dummy tensors when joint_seq_len = 0)
            tt_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
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
                use_column_major_ccl=use_column_major_ccl,
                is_causal=is_causal,
                is_balanced=is_balanced,
            )
            tt_out_list.append(tt_out)

    if trace_enabled:
        logger.info("Compile run")
        run_iters([])
        logger.info("Capture trace")
        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        run_iters(tt_out_list)
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        ttnn.synchronize_device(submesh)
        logger.info("Execute trace")
        ttnn.execute_trace(submesh, trace_id, blocking=False)
        ttnn.release_trace(submesh, trace_id)
        ttnn.synchronize_device(submesh)

    else:
        logger.info("Run without trace")
        run_iters(tt_out_list)

    if not skip_check:
        # Only use main tensors for host reference (no joint tensors since joint_seq_len=0)
        logger.debug("Running on host...")
        gt_out = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)
        logger.debug("Done running on host...")
        logger.debug("Host output shape: ", gt_out.shape)

        for i in range(n_iters):
            logger.debug("Synchronize call...")
            ttnn.synchronize_device(submesh)
            logger.debug("Done synchronizing...")
            tt_out = ttnn.to_torch(
                tt_out_list[i],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    submesh, mesh_shape=tuple(submesh.shape), dims=sdpa_input_shard_dims
                ),
            )

            # Reverse reordering for TT output if balanced reordering was applied
            if is_balanced and chunk_order is not None:
                logger.debug("Reversing balanced reordering for TT output")
                # First slice to padded sequence length, then reverse reorder
                tt_out_padded = tt_out[:, :, :padded_seq_len, :]
                tt_out_reordered = reverse_reorder_tensor_chunks(tt_out_padded, chunk_order, seq_dim=2)
                tt_out = tt_out_reordered[:, :, :base_seq_len, :]
            else:
                # Slice out any tile-padding
                tt_out = tt_out[:, :, :base_seq_len, :]

            logger.debug(f"tt_out: {tt_out.shape}")

            passing = True
            out_pass, out_pcc = comp_pcc(tt_out, gt_out, pcc_threshold)
            logger.debug(f"{out_pcc}")
            mse = ((gt_out - tt_out) ** 2).mean()
            logger.debug(f"mse: {mse}")
            if max_mse is not None and mse > max_mse:
                passing = False
            passing = passing and out_pass

            assert passing


@pytest.mark.parametrize("q_dtype, kv_dtype", [(ttnn.bfloat16, ttnn.bfloat8_b)], ids=["q_bf16_kv_bf8"])
@pytest.mark.parametrize(
    "b, nhq, nhk, nhv, base_seq_len, head_dim_q, head_dim_k, head_dim_v",
    [
        (1, 1, 1, 1, 4 * 128, 32, 32, 32),  # 1280: 320 tokens per device on 4x1 ring
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
    [(4, 1)],
    ids=["4x1"],
    indirect=True,
)
@pytest.mark.parametrize(
    "rp_axis, rp_factor, up_axis, up_factor",
    [
        [0, 4, 1, 1],
    ],
    ids=[
        "4rpx1p",
    ],
)
@pytest.mark.parametrize("is_balanced", [False, True], ids=["no_balancing", "balanced"])
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
    q_dtype,
    kv_dtype,
    n_iters,
    trace_enabled,
    num_links,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    all_gather_topology,
    skip_check,
    is_balanced,
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
        q_dtype,
        kv_dtype,
        n_iters,
        trace_enabled,
        num_links,
        rp_axis,
        up_axis,
        all_gather_topology,
        skip_check,
        0.999,
        use_column_major_ccl=True,
        is_causal=True,
        is_balanced=is_balanced,
    )
