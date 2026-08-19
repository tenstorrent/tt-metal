# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.tests.unit.test_ring_joint_attention import create_ring_joint_sdpa_submesh, logical_length_tensor
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
    # The op reserves the last column for the fabric MUX (sdpa_grid.x = x - 1) and needs one Q
    # chunk per SDPA column, so size the grid from the chunk count.
    local_padded_N = padded_seq_len // tuple(submesh.shape)[rp_axis]
    sdpa_compute_grid = (math.ceil(local_padded_N / q_chunk_size) + 1, full_compute_grid.y)

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
    "mesh_device, num_links, nh, base_seq_len, rp_axis, rp_factor, up_axis, up_factor, q_chunk_size, k_chunk_size, pad_to",
    [
        ((4, 32), 2, 40, 75600, 1, 32, 0, 4, 224, 512, None),
        # Head-serial passes: nh/up_factor heads land on each device and the op walks
        # ceil(heads_per_device / grid_rows) of them per core row as serial passes. With 10 grid
        # rows, 40 heads -> 10 per device -> 1 pass; 80 heads -> 20 per device -> 2 passes.
        ((4, 32), 2, 80, 75600, 1, 32, 0, 4, 224, 512, None),
        # Minimal spillover: 44 heads -> 11 per device -> row 0 runs 2 passes (heads 0 and 10),
        # rows 1-9 run 1 pass (heads 1-9) on the same P=2 build. Isolates the multi-pass row.
        ((4, 32), 2, 44, 75600, 1, 32, 0, 4, 224, 512, None),
        # H3 15s: 108544 = 106 * 1024 -> 3392 local tiles -> q=320 (11 columns), k=384. Resident Q
        # does not fit L1 at P=2, so this is the one config that exercises the factory's streamed-Q
        # fallback (stream_q). 56 heads -> 14/device: rows 0-3 run 2 passes, rows 4-9 run 1.
        ((4, 32), 2, 56, 108544, 1, 32, 0, 4, 320, 384, None),
        ((4, 8), 2, 40, 18944, 1, 8, 0, 4, 224, 512, None),
        # Whole-chunk skip: the pad tail on the LAST ring device covers an entire K chunk, so the
        # "KV chunk beyond logical_n" skip fires and one ring iteration processes fewer chunks than
        # the rest (here 1 instead of 2). Mirrors the fl2va 4x32 pipeline hang geometry
        # (base 31,930 -> padded 32,768: N_local 32 tiles, logical_nt one tile short of the last
        # chunk) scaled to sp=8: padded 8,192 -> logical_nt 240, last shard chunks at tile 224
        # (processed) and 240 (skipped). pad_to overrides get_padded_vision_seq_len because its
        # 32*sp alignment cannot produce a >= one-chunk tail at sp=8.
        ((4, 8), 2, 56, 7680, 1, 8, 0, 4, 96, 512, 8192),
        ((1, 4), 2, 10, 8960, 1, 4, 0, 1, 224, 512, None),
    ],
    ids=["4x32", "4x32_2pass", "4x32_1spill", "4x32_2pass_streamq", "4x8", "4x8_chunkskip", "1x4"],
    indirect=["mesh_device"],
)
@pytest.mark.skipif(
    ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
    reason="test_ring_joint_sdpa_dit_bh_glx requires a Blackhole Galaxy cluster",
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
    q_chunk_size,
    k_chunk_size,
    pad_to,
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    b, joint_seq_len, d = 1, 0, 128
    n_iters = 5
    trace_enabled = False
    skip_check = False
    pcc_threshold = 0.9993
    max_mse = 8e-5

    if nh % up_factor != 0:
        nh = math.ceil(nh / up_factor) * up_factor
    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)
    padded_seq_len = (
        pad_to if pad_to is not None else get_padded_vision_seq_len(base_seq_len, list(mesh_device.shape)[rp_axis])
    )

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


# One captured exp-ring-joint SDPA segment plus its runtime args; 32 MB is ample headroom.
LOGICAL_TENSOR_TRACE_REGION_SIZE = 32 * 1024 * 1024

# This op is not bit-reproducible: two identical host-scalar calls on identical inputs differ (verified
# on the unmodified op, scalar API only). So the logical_n tensor path is scored against the torch
# golden and required to match the scalar path's PCC, rather than asserted bit-equal to it.
PCC_THRESHOLD = 0.99
PCC_TOLERANCE = 0.01


@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "worker_l1_size": 1344544,
                "trace_region_size": LOGICAL_TENSOR_TRACE_REGION_SIZE,
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
    "mesh_device, num_links, b, nh, joint_seq_len, d, padded_seq_len, "
    "rp_axis, rp_factor, up_axis, up_factor, q_chunk_size, k_chunk_size, logical_ns",
    [
        # logical_n values chosen so each SKIPS A DIFFERENT NUMBER OF KV CHUNKS, which is what makes the
        # replay meaningful: the skipped count drives this op's credit caps, the injector's per-link gate
        # demand and the writer's forwarded-chunk count, so a length baked at capture time desynchronizes
        # them and hangs. ring_size=8 over padded 8192 gives 32 tiles per shard, and k_chunk=256 gives 8
        # tiles per chunk, so the last shard holds 4 chunks starting at global tiles 224/232/240/248:
        #   8192 -> nt 256, 0 skips, tile- and chunk-aligned (mask tile present but never applied)
        #   8191 -> nt 256, 0 skips, sub-tile tail (partial column 31 stamped)
        #   7936 -> nt 248, 1 skip,  tile-aligned tail (partial column 0, so unapplied)
        #   7650 -> nt 240, 2 skips, sub-tile tail (partial column 2)
        #   7200 -> nt 225, 3 skips, tile-aligned tail, near the largest padding the op allows
        # The op's (padded - logical) < per-device rule puts the boundary in the last shard, which is
        # also the joint shard -- so joint+boundary interaction is always exercised.
        # joint_seq_len 1024 makes num_q_chunks = 4 local + 4 joint = 8, a multiple of the 4 SDPA grid
        # columns this geometry yields, which the op requires. k_chunk=256 (8 tiles) is also what keeps
        # the op's streaming-compute path enabled, which its compute kernel static_asserts on.
        # nh=20 is chosen so the grid comes out 10 rows deep (5 heads/device x 2 segments/head), matching
        # the row count and backward/forward worker split of the op's own passing 4x8 config. Shallower
        # grids (4 rows / 2 workers per link) are not bit-reproducible on this op even on the host-scalar
        # path.
        ((4, 8), 2, 1, 20, 1024, 64, 8192, 1, 8, 0, 4, 256, 256, [8191, 8192, 7936, 7650, 7200]),
    ],
    ids=["m4x8"],
    indirect=["mesh_device"],
)
def test_exp_ring_joint_sdpa_logical_n_tensor_trace_replay(
    mesh_device,
    num_links,
    b,
    nh,
    joint_seq_len,
    d,
    padded_seq_len,
    rp_axis,
    rp_factor,
    up_axis,
    up_factor,
    q_chunk_size,
    k_chunk_size,
    logical_ns,
    all_gather_topology,
    reset_seeds,
):
    """ONE captured trace, replayed once per logical_n with only the length tensor refreshed in place,
    must score against the torch golden as well as the host-scalar path (the op is not bit-reproducible;
    see PCC_THRESHOLD).

    Inputs are uploaded once, so logical_n is the only thing distinguishing one replay from the next.
    Catches the failure modes a single-dispatch test cannot: a stale read of the fixed-address tensor
    (missing invalidate_l1_cache), and a chunk-skip count baked at capture time, which would leave the
    reader's credit/gate demand out of step with the writer's forwarded-chunk count.
    """
    dtype = ttnn.bfloat16
    submesh = create_ring_joint_sdpa_submesh(mesh_device, rp_axis, rp_factor, up_axis, up_factor)
    submesh.enable_program_cache()  # a trace replays cached programs; capture cannot JIT-compile

    local_padded_N = padded_seq_len // rp_factor
    for logical_n in logical_ns:
        assert padded_seq_len - logical_n < local_padded_N, (
            f"logical_n={logical_n} violates the op's own validation: (padded {padded_seq_len} - logical) "
            f"must be < per-device {local_padded_N}"
        )

    full_compute_grid = submesh.compute_with_storage_grid_size()
    # Grid derived from the op's own two shape rules rather than the full device grid: it reserves the
    # last column for the fabric MUX and needs one Q chunk per SDPA column (so x = local chunks + 1),
    # and every grid ROW must be covered by a head-segment (so y = B * NQH * segments-per-head).
    num_local_q_chunks = math.ceil(local_padded_N / q_chunk_size)
    num_joint_q_chunks = math.ceil(joint_seq_len / q_chunk_size)
    num_q_chunks = num_local_q_chunks + num_joint_q_chunks
    sdpa_grid_x = num_local_q_chunks
    assert (
        num_q_chunks % sdpa_grid_x == 0
    ), f"num_q_chunks ({num_q_chunks}) must be a multiple of the {sdpa_grid_x} SDPA grid columns"
    grid_rows = b * (nh // up_factor) * (num_q_chunks // sdpa_grid_x)
    assert grid_rows <= full_compute_grid.y, f"needed {grid_rows} rows, device grid has {full_compute_grid.y}"
    sdpa_compute_grid = (sdpa_grid_x + 1, grid_rows)

    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group([worker_sub_device_id])
    ccl_semaphore_handles = [ttnn.create_global_semaphore(submesh, ccl_sub_device_crs, 0) for _ in range(num_links)]

    spatial_shard_dims = [None, None]
    spatial_shard_dims[rp_axis] = 2
    spatial_shard_dims[up_axis] = 1
    kv_buf_shard_dims = [None, None]
    kv_buf_shard_dims[up_axis] = 1
    # The exp op's joint is replicated: full L per device, head-sharded only.
    joint_shard_dims = list(kv_buf_shard_dims)
    joint_composer_dims = list(joint_shard_dims)
    joint_composer_dims[rp_axis] = 0  # concat the per-device replicas into batch so all are compared

    # Garbage (not zero) past logical_n: those rows must be masked out, and leaked garbage collapses the
    # comparison where a leaked zero would barely move it.
    padded_Q = 8.0 * fa_rand(b, nh, padded_seq_len, d)
    padded_K = 8.0 * fa_rand(b, nh, padded_seq_len, d)
    padded_V = 8.0 * fa_rand(b, nh, padded_seq_len, d)
    joint_Q = 8.0 * fa_rand(b, nh, joint_seq_len, d)
    joint_K = 8.0 * fa_rand(b, nh, joint_seq_len, d)
    joint_V = 8.0 * fa_rand(b, nh, joint_seq_len, d)

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

    tt_logical_n = logical_length_tensor(submesh, logical_ns[0])

    def load_length(logical_n):
        """Deliver the length the way a traced runner does: in-place refresh, no reallocation."""
        ttnn.copy_host_to_device_tensor(logical_length_tensor(submesh, logical_n, on_device=False), tt_logical_n)

    def call(logical_n):
        return ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
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
            cluster_axis=rp_axis,
            mesh_device=submesh,
            topology=all_gather_topology,
            subdevice_id=worker_sub_device_id,
            # The op splits the grid rows into a backward and a forward fabric-direction half.
            num_workers_per_link=grid_rows // 2,
        )

    spatial_composer = ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=spatial_shard_dims)
    joint_composer = ttnn.ConcatMesh2dToTensor(submesh, mesh_shape=tuple(submesh.shape), dims=joint_composer_dims)

    def valid_rows(tt_out, tt_joint_out, logical_n):
        spatial = ttnn.to_torch(tt_out, mesh_composer=spatial_composer)[:, :, :logical_n, :]
        joint = ttnn.to_torch(tt_joint_out, mesh_composer=joint_composer)[:, :, :joint_seq_len, :]
        return spatial, joint

    def golden(logical_n):
        """Torch reference for this length: real spatial rows [0, logical_n) plus the joint."""
        q = torch.cat([padded_Q[:, :, :logical_n, :], joint_Q], dim=2)
        k = torch.cat([padded_K[:, :, :logical_n, :], joint_K], dim=2)
        v = torch.cat([padded_V[:, :, :logical_n, :], joint_V], dim=2)
        gt = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        return gt[:, :, :logical_n, :], gt[:, :, logical_n:, :]

    def score(tt_out, tt_joint_out, logical_n):
        """PCC of the spatial and joint outputs against the torch golden."""
        got_spatial, got_joint = valid_rows(tt_out, tt_joint_out, logical_n)
        ref_spatial, ref_joint = golden(logical_n)
        _, spatial_pcc = comp_pcc(ref_spatial, got_spatial, PCC_THRESHOLD)
        joint_pcc = None
        if joint_seq_len > 0:
            # A replicated joint output is one copy per ring device; score the first replica.
            replica = got_joint[0:1] if got_joint.shape[0] > 1 else got_joint
            _, joint_pcc = comp_pcc(ref_joint, replica, PCC_THRESHOLD)
        return spatial_pcc, joint_pcc

    def pcc_value(pcc_str):
        # comp_pcc returns a message like "PCC: 0.9994..."; pull the number out.
        return float(str(pcc_str).split(":")[-1].strip())

    trace_id = None
    try:
        # 1) Scalar mode (logical_n as a host int) -- the pre-existing path.
        scalar_pcc = {}
        for logical_n in logical_ns:
            tt_out, tt_joint_out, tt_stats = call(logical_n)
            ttnn.synchronize_device(submesh)
            scalar_pcc[logical_n] = score(tt_out, tt_joint_out, logical_n)
            for t in (tt_out, tt_joint_out, tt_stats):
                ttnn.deallocate(t)
            logger.info(f"scalar  logical_n={logical_n}: spatial {scalar_pcc[logical_n][0]}")

        # 2) Tensor mode (logical_n as a single-valued device tensor), eager.
        tensor_pcc = {}
        for logical_n in logical_ns:
            load_length(logical_n)
            tt_out, tt_joint_out, tt_stats = call(tt_logical_n)
            ttnn.synchronize_device(submesh)
            tensor_pcc[logical_n] = score(tt_out, tt_joint_out, logical_n)
            for t in (tt_out, tt_joint_out, tt_stats):
                ttnn.deallocate(t)
            logger.info(f"tensor  logical_n={logical_n}: spatial {tensor_pcc[logical_n][0]}")

        # 3) ONE captured trace, replayed per length with only the length tensor refreshed in place.
        #    The capture's host scalar is the placeholder, identical for every length, so a replay that
        #    scores well can only have picked the length up from the tensor.
        load_length(logical_ns[0])
        warm = call(tt_logical_n)
        ttnn.synchronize_device(submesh)
        for t in warm:
            ttnn.deallocate(t)

        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        tt_out_traced, tt_joint_out_traced, _ = call(tt_logical_n)
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        ttnn.synchronize_device(submesh)

        # Replay order is deliberately not ascending: forward, then descending (where a stale length
        # would be too LARGE and over-attend), then out of order.
        n = len(logical_ns)
        replay_order = list(range(n)) + list(reversed(range(n))) + [0, n - 1, 1]
        replay_pcc = {}
        for replay_idx, i in enumerate(replay_order):
            logical_n = logical_ns[i]
            load_length(logical_n)
            ttnn.execute_trace(submesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(submesh)
            got = score(tt_out_traced, tt_joint_out_traced, logical_n)
            replay_pcc.setdefault(logical_n, got)
            logger.info(f"replay {replay_idx} logical_n={logical_n}: spatial {got[0]}")

        # ---- Verdict -------------------------------------------------------------------------
        failures = []
        for logical_n in logical_ns:
            s = pcc_value(scalar_pcc[logical_n][0])
            t = pcc_value(tensor_pcc[logical_n][0])
            r = pcc_value(replay_pcc[logical_n][0])
            logger.info(f"logical_n={logical_n}: scalar={s:.6f} tensor={t:.6f} replay={r:.6f}")
            if s < PCC_THRESHOLD:
                failures.append(f"logical_n={logical_n}: SCALAR path itself scores {s:.6f} < {PCC_THRESHOLD}")
            # The tensor path must be as good as the scalar path, within the run-to-run spread this
            # op already shows on identical inputs.
            if t < s - PCC_TOLERANCE:
                failures.append(f"logical_n={logical_n}: tensor {t:.6f} vs scalar {s:.6f} (tol {PCC_TOLERANCE})")
            if r < s - PCC_TOLERANCE:
                failures.append(f"logical_n={logical_n}: replay {r:.6f} vs scalar {s:.6f} (tol {PCC_TOLERANCE})")
        assert not failures, "logical_n tensor transport regressed:\n  " + "\n  ".join(failures)

        logger.success(
            f"logical_n tensor: {n} lengths scored eager + {len(replay_order)} trace replays from ONE "
            f"capture, all matching the host-scalar path's PCC within {PCC_TOLERANCE}"
        )
    finally:
        if trace_id is not None:
            ttnn.release_trace(submesh, trace_id)
        submesh.reset_sub_device_stall_group()
        submesh.clear_loaded_sub_device_manager()
        submesh.remove_sub_device_manager(sub_device_manager)
