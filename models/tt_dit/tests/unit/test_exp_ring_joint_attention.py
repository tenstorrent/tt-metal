# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
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
    num_workers_per_link=None,
    num_buffers_per_channel=32,
):
    full_compute_grid = submesh.compute_with_storage_grid_size()
    # The op reserves the last column for the fabric MUX (sdpa_grid.x = x - 1) and needs one Q
    # chunk per SDPA column, so size the grid from the chunk count. Rows are split into equal
    # backward/forward MUX-client halves, so the row count must be even: Blackhole's 10 rows are,
    # Wormhole's 9 rows drop to 8.
    # With TT_EXP_SDPA_MUX_BOTTOM_ROW the MUX kernels take the bottom row instead, so SDPA keeps every
    # column and the rows above the MUX row (rounded down to even): 8x8 = 64 cores on Wormhole.
    mux_on_bottom_row = os.environ.get("TT_EXP_SDPA_MUX_BOTTOM_ROW") is not None
    local_padded_N = padded_seq_len // tuple(submesh.shape)[rp_axis]
    if mux_on_bottom_row:
        sdpa_rows = (full_compute_grid.y - 1) - ((full_compute_grid.y - 1) % 2)
        max_sdpa_cols = full_compute_grid.x
    else:
        sdpa_rows = full_compute_grid.y - (full_compute_grid.y % 2)
        max_sdpa_cols = full_compute_grid.x - 1
    # A head's Q chunks must fill whole rows (num_q_chunks % columns == 0); more chunks than columns
    # run as head-segments (segs_per_head = chunks / columns). Take the widest column count that
    # fits the device and divides the chunk count.
    num_q_chunks = math.ceil(local_padded_N / q_chunk_size)
    sdpa_cols = max(c for c in range(min(num_q_chunks, max_sdpa_cols), 0, -1) if num_q_chunks % c == 0)
    if mux_on_bottom_row:
        assert sdpa_cols >= 2 * num_links, (
            f"bottom-row MUX placement needs 2 MUX cores per link on the {sdpa_cols}-wide MUX row; "
            f"{num_links} links do not fit"
        )
        sdpa_compute_grid = (sdpa_cols, sdpa_rows + 1)
    else:
        sdpa_compute_grid = (sdpa_cols + 1, sdpa_rows)
    logger.info(
        f"exp ring SDPA grid: user {sdpa_compute_grid}, SDPA {sdpa_cols}x{sdpa_rows} "
        f"({sdpa_cols * sdpa_rows} cores), {num_q_chunks} Q chunks, mux_on_bottom_row={mux_on_bottom_row}"
    )
    if num_workers_per_link is None:
        num_workers_per_link = sdpa_rows // 2  # one MUX client per SDPA row per direction

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

    # TT_EXP_SDPA_TEST_EXP_APPROX=1: approximate SFPU exp in the softmax. The model runs exact exp
    # ("False is more correct"); the phase-zone split shows exp is ~18% of the pack thread's step, so
    # this is the A/B for that lever. Default exact.
    exp_approx_mode = os.environ.get("TT_EXP_SDPA_TEST_EXP_APPROX") is not None
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=sdpa_compute_grid,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=exp_approx_mode,
    )
    logger.info(f"exp ring SDPA exp_approx_mode={exp_approx_mode}")

    # TT_EXP_SDPA_TEST_DST_FULL_SYNC=1: 16-tile DST (full sync) instead of the 8-tile half-sync
    # default, which lets the factory's subblock search pick (4,4)/(2,8) QK subblocks. A/B knob.
    dst_full_sync_en = os.environ.get("TT_EXP_SDPA_TEST_DST_FULL_SYNC") is not None
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        submesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=dst_full_sync_en,
    )
    logger.info(f"exp ring SDPA compute config: HiFi2, dst_full_sync_en={dst_full_sync_en}")

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
    num_workers_per_link=None,
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


_BH_GLX_ONLY = pytest.mark.skipif(not is_blackhole(), reason="Blackhole Galaxy shape")
_WH_GLX_ONLY = pytest.mark.skipif(is_blackhole(), reason="Wormhole Galaxy shape")


@pytest.mark.parametrize(
    "device_params, all_gather_topology",
    [
        (
            {
                "worker_l1_size": 1344544,
                "trace_region_size": 1000000,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                # Wormhole's fabric caps packets at 7616 B (erisc_datamover_builder.hpp); the H3
                # Wormhole mesh runs the 4 KB router payload, so match it. Blackhole keeps 8 KB.
                "fabric_router_config": create_fabric_router_config(8192 if is_blackhole() else 4096),
            },
            ttnn.Topology.Ring,
        ),
    ],
    indirect=["device_params"],
    ids=["ring"],
)
@pytest.mark.parametrize(
    "mesh_device, num_links, nh, base_seq_len, rp_axis, rp_factor, up_axis, up_factor, q_chunk_size, k_chunk_size",
    [
        pytest.param((4, 32), 2, 40, 75600, 1, 32, 0, 4, 224, 512, id="4x32", marks=_BH_GLX_ONLY),
        # Head-serial passes: nh/up_factor heads land on each device and the op walks
        # ceil(heads_per_device / grid_rows) of them per core row as serial passes. With 10 grid
        # rows, 40 heads -> 10 per device -> 1 pass; 80 heads -> 20 per device -> 2 passes.
        pytest.param((4, 32), 2, 80, 75600, 1, 32, 0, 4, 224, 512, id="4x32_2pass", marks=_BH_GLX_ONLY),
        # Minimal spillover: 44 heads -> 11 per device -> row 0 runs 2 passes (heads 0 and 10),
        # rows 1-9 run 1 pass (heads 1-9) on the same P=2 build. Isolates the multi-pass row.
        pytest.param((4, 32), 2, 44, 75600, 1, 32, 0, 4, 224, 512, id="4x32_1spill", marks=_BH_GLX_ONLY),
        # H3 15s: 108544 = 106 * 1024 -> 3392 local tiles -> q=320 (11 columns), k=384. Resident Q
        # does not fit L1 at P=2, so this is the one config that exercises the factory's streamed-Q
        # fallback (stream_q). 56 heads -> 14/device: rows 0-3 run 2 passes, rows 4-9 run 1.
        pytest.param((4, 32), 2, 56, 108544, 1, 32, 0, 4, 320, 384, id="4x32_2pass_streamq", marks=_BH_GLX_ONLY),
        pytest.param((4, 8), 2, 40, 18944, 1, 8, 0, 4, 224, 512, id="4x8", marks=_BH_GLX_ONLY),
        pytest.param((1, 4), 2, 10, 8960, 1, 4, 0, 1, 224, 512, id="1x4", marks=_BH_GLX_ONLY),
        # Wormhole 4x8 Galaxy, H3 at the SP=32-equivalent shard [1, 14, 3424, 128] (what the exp op
        # measured 21% on Blackhole with): 7 SDPA columns x 8 rows. 3424 rows -> q=512 fills 7 columns;
        # 14 heads on 8 rows -> 2 passes; the L1 budget then only admits k=128 with streamed Q
        # (`_exp_sdpa_l1_bytes` in attention_minimax_h3.py).
        pytest.param((4, 8), 2, 56, 27392, 1, 8, 0, 4, 512, 128, id="4x8_wh_h3_sim32", marks=_WH_GLX_ONLY),
        # Same shard on all 4 links of the Wormhole row: 4 MUX-client columns, 8 MUX kernels.
        pytest.param((4, 8), 4, 56, 27392, 1, 8, 0, 4, 512, 128, id="4x8_wh_h3_sim32_nl4", marks=_WH_GLX_ONLY),
        # Same shard at 14 columns of q=256 -> segs=2, 4 passes (needs kMaxPasses >= 4), k=256 streamed Q.
        pytest.param((4, 8), 2, 56, 27392, 1, 8, 0, 4, 256, 256, id="4x8_wh_h3_sim32_p4", marks=_WH_GLX_ONLY),
        # Sequential passes (run with TT_EXP_SDPA_Q_GROUPS=2): the same shard as pass-outer / ring-inner,
        # 14 chunks of q=256 = 7 columns x 2 groups -> one segment per head, 2 passes of 2 groups. PCC-checked.
        pytest.param((4, 8), 2, 56, 27392, 1, 8, 0, 4, 256, 256, id="4x8_wh_h3_sim32_seq", marks=_WH_GLX_ONLY),
        # The 15 s shard padded to 14336 rows/device (56 chunks of q=256 = 7 columns x 4 groups x 2
        # segments), run with TT_EXP_SDPA_Q_GROUPS=4. Timing only: the torch reference at 114688 tokens
        # is infeasible, so the check is skipped (see _torch_reference_feasible).
        pytest.param((4, 8), 2, 56, 114688, 1, 8, 0, 4, 256, 512, id="4x8_wh_h3_15s_seq", marks=_WH_GLX_ONLY),
        # Same shard on 4 links (run with TT_EXP_SDPA_Q_GROUPS=2 for segs=4: 7 balanced passes per row,
        # no pair dedup so every row forwards; or Q_GROUPS=4 for the segs=2 layout above on 4 links).
        pytest.param((4, 8), 4, 56, 114688, 1, 8, 0, 4, 256, 512, id="4x8_wh_h3_15s_seq_nl4", marks=_WH_GLX_ONLY),
        # Bottom-row MUX layout (run with TT_EXP_SDPA_MUX_BOTTOM_ROW=1 TT_EXP_SDPA_Q_GROUPS=1): SDPA keeps
        # all 8 columns and rows 0-7, the 8 MUX kernels of 4 links fill row 8 -> 64 cores. 4096 rows/device
        # = 16 chunks of q=256 = 8 columns x 2 segments; PCC-checked stand-in for the 15 s shard's 56
        # chunks (8 columns x 7 segments, G=1), which the 15s_seq_nl4 case above times under the same env.
        pytest.param((4, 8), 4, 56, 32768, 1, 8, 0, 4, 256, 256, id="4x8_wh_h3_4096_bot_nl4", marks=_WH_GLX_ONLY),
        # 15 s shard at q=128 on the bottom-row layout (G=1): 112 chunks = 8 columns x 14 segments, 196
        # segments on 8 rows -> 25 half-chunk passes = 12.5 chunk-equivalents per core (q=256: 13). Timing only.
        pytest.param((4, 8), 4, 56, 114688, 1, 8, 0, 4, 128, 512, id="4x8_wh_h3_15s_q128_nl4", marks=_WH_GLX_ONLY),
        # 15 s shard padded to 13824 rows/device (the 13664 real rows + 1.2%) at q=192: 72 chunks = 8 columns
        # x 9 segments, 126 segments on 8 rows -> 16 passes of 6 tile-rows = 96 tile-rows per core against
        # 104 for q=256 at 14336 rows, and 3.6% fewer K/V rows. Timing only.
        pytest.param((4, 8), 4, 56, 110592, 1, 8, 0, 4, 192, 512, id="4x8_wh_h3_15s_q192_nl4", marks=_WH_GLX_ONLY),
        # 15 s shard at q=448 on the bottom-row layout (G=1): 32 chunks = 8 columns x 4 segments, 56 segments
        # on 8 rows -> 7 passes of 14 tile-rows = 98 tile-rows per core. k=256 is the largest K chunk that
        # fits the single-pass L1 budget at q=448 (k=512 needs 1.71 MB of 1.34 MB). Timing only.
        pytest.param((4, 8), 4, 56, 114688, 1, 8, 0, 4, 448, 256, id="4x8_wh_h3_15s_q448_nl4", marks=_WH_GLX_ONLY),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.skipif(
    ttnn.cluster.get_cluster_type() not in (ttnn.cluster.ClusterType.BLACKHOLE_GALAXY, ttnn.cluster.ClusterType.GALAXY),
    reason="exp ring joint SDPA DiT cases need a 32-chip Galaxy (Blackhole or Wormhole)",
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
    all_gather_topology,
    reset_seeds,
):
    dtype = ttnn.bfloat16
    b, joint_seq_len, d = 1, 0, 128
    n_iters = 5
    trace_enabled = False
    # torch's fp32 reference materializes the full score matrix; past ~40k tokens it does not fit
    # host memory, so such cases are timing-only.
    skip_check = base_seq_len > 40000
    if skip_check:
        logger.warning(f"base_seq_len={base_seq_len}: torch reference infeasible, running without the PCC check")
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
