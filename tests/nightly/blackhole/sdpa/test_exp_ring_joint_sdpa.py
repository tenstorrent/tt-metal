# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Exp Ring Joint Attention SDPA Tests for Blackhole

Tests the experimental ring joint attention op accuracy and determinism using
shapes tuned to the op's core constraint: ceil(N_local / q_chunk_size) == sdpa_cols,
where sdpa_cols = full_grid.x - 1 (last column reserved for CCL MUX).

BH hardware constants are hardcoded to handle firmware differences across versions.
Perf tests are included but skipped on CI.
"""

import math
import os
from unittest import mock

import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
import ttnn
from ttnn.operations.ccl import Topology
from loguru import logger
import pytest

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand
from tests.nightly.sdpa_perf_utils import (
    post_process_ops_log,
    compute_cores_used,
    compute_sdpa_flops,
    compute_math_utilization,
)


def create_fabric_router_config(max_payload_size):
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


# ============================================================================
# HARDWARE CONFIGURATION CONSTANTS
# ============================================================================

# Grid dimensions (hardcoded to handle firmware differences across versions)
GALAXY_GRID_COLS = 12
GALAXY_GRID_ROWS = 10
NON_GALAXY_GRID_COLS = 11
NON_GALAXY_GRID_ROWS = 10

# SDPA core columns: last column reserved for CCL MUX
GALAXY_SDPA_COLS = GALAXY_GRID_COLS - 1  # 11
NON_GALAXY_SDPA_COLS = NON_GALAXY_GRID_COLS - 1  # 10

# Galaxy mesh configuration
GALAXY_DEVICE_COUNT = 32
GALAXY_TP_SIZE = 4
GALAXY_SP_SIZE = 8

# Shape constraint: ceil(N_local / Q_CHUNK_SIZE) == sdpa_cols (one Q chunk per SDPA core column).
# Galaxy N_local=2368 matches the real WAN 2.2 shape (75600 total / 32 devices → padded to 2368).
# Non-galaxy N_local=2240 is exact (10 * 224).
GALAXY_N_LOCAL = 2368
NON_GALAXY_N_LOCAL = NON_GALAXY_SDPA_COLS * 224  # 2240

Q_CHUNK_SIZE = 224
K_CHUNK_SIZE = 512

# Accuracy thresholds (matching the unit test)
DEFAULT_PCC_THRESHOLD = 0.9993
DEFAULT_MAX_MSE = 8e-5

BATCH_SIZE = 1
HEAD_DIM = 128
HEADS_PER_DEVICE = 10  # After TP split


# ============================================================================
# HARDWARE DETECTION
# ============================================================================


def detect_devices_without_opening():
    """Count available TT devices without opening them."""
    import glob

    return len(glob.glob("/dev/tenstorrent/*"))


def calculate_mesh_config(num_devices):
    """
    Calculate mesh configuration based on available devices.

    Returns:
        sp_size: Sequence parallel size (devices per ring)
        tp_size: Tensor parallel size (number of rings)
        arch_type: Architecture type string
    """
    if num_devices == GALAXY_DEVICE_COUNT:
        sp_size = GALAXY_SP_SIZE
        tp_size = GALAXY_TP_SIZE
        arch_type = "galaxy_4x8"
    else:
        sp_size = num_devices
        tp_size = 1
        arch_type = f"single_ring_{num_devices}x1"
    return sp_size, tp_size, arch_type


# ============================================================================
# TEST CONFIGURATION GENERATION
# ============================================================================


def generate_test_configs():
    """
    Generate (b, nh, total_seq, d, q_chunk, k_chunk) tuples tuned for available hardware.

    Shapes satisfy: ceil(N_local / Q_CHUNK_SIZE) == sdpa_cols
    - Non-galaxy: N_local = 2240 (10 * 224, exact), total_seq = 2240 * sp_size
    - Galaxy:     N_local = 2368 (ceil = 11 = sdpa_cols), total_seq = 18944 for sp=8

    NOTE: Uses detect_devices_without_opening() to avoid holding device locks
    during pytest collection, which would block subprocess profiling.
    """
    num_devices = detect_devices_without_opening()
    if num_devices < 4:
        return [], []

    sp_size, tp_size, arch_type = calculate_mesh_config(num_devices)
    is_galaxy = arch_type.startswith("galaxy")

    N_local = GALAXY_N_LOCAL if is_galaxy else NON_GALAXY_N_LOCAL
    total_seq = N_local * sp_size  # already aligned to 32*sp_size
    total_heads = HEADS_PER_DEVICE * tp_size

    config = (BATCH_SIZE, total_heads, total_seq, HEAD_DIM, Q_CHUNK_SIZE, K_CHUNK_SIZE)
    config_id = f"{arch_type}-seq{total_seq}-h{total_heads}-q{Q_CHUNK_SIZE}-k{K_CHUNK_SIZE}"
    return [config], [config_id]


# ============================================================================
# CORE EXECUTION FUNCTION
# ============================================================================


def run_exp_ring_joint_sdpa_nightly(
    b,
    nh,
    total_seq,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    pcc_threshold=DEFAULT_PCC_THRESHOLD,
    max_mse=DEFAULT_MAX_MSE,
    do_check=True,
    num_iterations=1,
    num_links=2,
    num_workers_per_link=5,
    num_buffers_per_channel=32,
    max_payload_size=8192,
):
    """
    Run exp_ring_joint_scaled_dot_product_attention and verify accuracy or determinism.

    `total_seq` must already satisfy the shape constraint:
        ceil((total_seq / sp_size) / q_chunk_size) == sdpa_cols

    When `num_iterations > 1`, checks that all outputs are bitwise equal (determinism).
    When `num_iterations == 1`, checks accuracy against PyTorch SDPA reference.
    """
    num_devices = detect_devices_without_opening()
    sp_size, tp_size, arch_type = calculate_mesh_config(num_devices)

    if sp_size < 4:
        pytest.skip(f"Only testing with ≥4 devices in ring, got sp_size={sp_size}")

    is_galaxy = arch_type.startswith("galaxy")

    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING
    topology = Topology.Ring

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router_config=create_fabric_router_config(max_payload_size),
    )

    sp_axis = 1  # column axis for sequence parallel (ring axis)
    tp_axis = 0  # row axis for tensor parallel

    mesh_shape = ttnn.MeshShape(tp_size, sp_size)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    try:
        if tp_size > 1 and nh % tp_size != 0:
            pytest.skip(f"num_heads ({nh}) must be divisible by TP size ({tp_size})")

        # Hardcoded grid sizes to handle firmware differences
        # Add column for CCL MUX
        if is_galaxy:
            sdpa_compute_grid = (GALAXY_SDPA_COLS + 1, GALAXY_GRID_ROWS)
        else:
            sdpa_compute_grid = (NON_GALAXY_SDPA_COLS + 1, NON_GALAXY_GRID_ROWS)

        full_compute_grid = mesh_device.compute_with_storage_grid_size()

        # Sub-device covering all cores (needed for CCL)
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
        worker_sub_device_id = ttnn.SubDeviceId(0)

        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        mesh_device.set_sub_device_stall_group([worker_sub_device_id])

        TILE_SIZE = 32

        def round_up_to_multiple(val, multiple):
            return ((val + multiple - 1) // multiple) * multiple

        chunk_size = TILE_SIZE * sp_size
        padded_total_seq = round_up_to_multiple(total_seq, chunk_size)

        # Input tensors — bfloat16-rounded so reference matches hardware precision
        Q = fa_rand(b, nh, total_seq, d).bfloat16().float()
        K = fa_rand(b, nh, total_seq, d).bfloat16().float()
        V = fa_rand(b, nh, total_seq, d).bfloat16().float()

        padded_Q = torch.cat([Q, torch.zeros(b, nh, padded_total_seq - total_seq, d)], dim=2)
        padded_K = torch.cat([K, torch.zeros(b, nh, padded_total_seq - total_seq, d)], dim=2)
        padded_V = torch.cat([V, torch.zeros(b, nh, padded_total_seq - total_seq, d)], dim=2)

        # Sharding: SP axis → sequence dim (2), UP axis → heads dim (1)
        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        if tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        tt_Q = ttnn.from_torch(
            padded_Q,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
        tt_K = ttnn.from_torch(
            padded_K,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
        tt_V = ttnn.from_torch(
            padded_V,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )

        # Joint inputs: zero-length (WAN 2.2 style — pure self-attention via ring infra)
        joint_seq_len = 0
        sdpa_joint_shard_dims = [None, None]
        if tp_size > 1:
            sdpa_joint_shard_dims[tp_axis] = 1

        tt_joint_Q = ttnn.from_torch(
            torch.zeros(b, nh, joint_seq_len, d, dtype=torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
            ),
        )
        tt_joint_K = ttnn.from_torch(
            torch.zeros(b, nh, joint_seq_len, d, dtype=torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
            ),
        )
        tt_joint_V = ttnn.from_torch(
            torch.zeros(b, nh, joint_seq_len, d, dtype=torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
            ),
        )

        # Persistent K/V output buffers: after all-gather, each device holds the full sequence.
        # Not sharded on the ring (sp) axis; sharded on heads (tp) axis if tp_size > 1.
        kv_shard_dims = [None, None]
        if tp_size > 1:
            kv_shard_dims[tp_axis] = 1

        persistent_buffer_k = [
            ttnn.from_torch(
                torch.zeros(b, nh, padded_total_seq, d),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims
                ),
            )
            for _ in range(num_iterations)
        ]
        persistent_buffer_v = [
            ttnn.from_torch(
                torch.zeros(b, nh, padded_total_seq, d),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims
                ),
            )
            for _ in range(num_iterations)
        ]

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Mesh composer dims for converting output back to torch
        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1

        # Pre-create all semaphore sets before the loop to avoid device writes between iterations
        ccl_semaphores_list = [
            [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(num_links)]
            for _ in range(num_iterations)
        ]

        tt_out_list = []

        for i in range(num_iterations):
            ttnn.synchronize_device(mesh_device)

            tt_out, _tt_joint_out, _tt_lse = ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_joint_Q,
                tt_joint_K,
                tt_joint_V,
                persistent_output_buffer_k=persistent_buffer_k[i],
                persistent_output_buffer_v=persistent_buffer_v[i],
                joint_strategy="rear",
                logical_n=total_seq,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=ccl_semaphores_list[i],
                num_links=num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh_device,
                topology=topology,
                subdevice_id=worker_sub_device_id,
                num_workers_per_link=num_workers_per_link,
                num_buffers_per_channel=num_buffers_per_channel,
            )

            tt_out_list.append(tt_out)

        # to_torch only after all iterations (avoids PCIe readback between launches)
        def to_torch_out(tt_tensor):
            out = ttnn.to_torch(
                tt_tensor,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)
                ),
            )
            return out[:, :, :total_seq, :]

        if num_iterations > 1:
            N_local = total_seq // sp_size
            pass_determinism = True
            reference_output = to_torch_out(tt_out_list[0])
            for i in range(1, num_iterations):
                tt_out_torch = to_torch_out(tt_out_list[i])
                if not torch.equal(reference_output, tt_out_torch):
                    diff_mask = reference_output != tt_out_torch
                    num_diffs = diff_mask.sum().item()
                    max_diff = (reference_output - tt_out_torch).abs().max().item()
                    logger.error(
                        f"Exp ring joint SDPA output at iteration {i} differs from iteration 0: "
                        f"{num_diffs} differing elements, max_diff={max_diff}"
                    )
                    pass_determinism = False

                    # Per-tile mismatch analysis
                    TILE = 32
                    seq_tiles = total_seq // TILE
                    d_tiles = d // TILE

                    tile_diffs = []
                    for bx in range(b):
                        for h in range(nh):
                            for st in range(seq_tiles):
                                for dt in range(d_tiles):
                                    s0, s1 = st * TILE, (st + 1) * TILE
                                    d0, d1 = dt * TILE, (dt + 1) * TILE
                                    ref_tile = reference_output[bx, h, s0:s1, d0:d1]
                                    iter_tile = tt_out_torch[bx, h, s0:s1, d0:d1]
                                    if not torch.equal(ref_tile, iter_tile):
                                        seq_slice = s0 // N_local
                                        seq_tile_in_slice = (s0 - seq_slice * N_local) // TILE
                                        tile_max_diff = (ref_tile - iter_tile).abs().max().item()
                                        tile_diffs.append((bx, h, st, seq_slice, seq_tile_in_slice, dt, tile_max_diff))

                    logger.error(f"  {len(tile_diffs)} mismatching tiles out of {b * nh * seq_tiles * d_tiles}")

                    from collections import Counter

                    slice_counts = Counter(td[3] for td in tile_diffs)
                    for sl, cnt in sorted(slice_counts.items()):
                        logger.error(f"  seq_slice={sl}: {cnt} mismatching tiles")

                    in_slice_counts = Counter(td[4] for td in tile_diffs)
                    logger.error(f"  All unique seq_tile_in_slice values: {sorted(in_slice_counts.keys())}")
                    for st_in, cnt in sorted(in_slice_counts.items()):
                        logger.error(f"  seq_tile_in_slice={st_in}: {cnt} mismatching tiles")

                    for bx, h, st, seq_slice, st_in_slice, dt, mdiff in tile_diffs[:50]:
                        logger.error(
                            f"  tile: batch={bx} head={h} seq_tile={st} seq_slice={seq_slice} "
                            f"seq_tile_in_slice={st_in_slice} d_tile={dt} max_diff={mdiff:.6f}"
                        )
                    if len(tile_diffs) > 50:
                        logger.error(f"  ... and {len(tile_diffs) - 50} more")
                else:
                    logger.info(f"Output iter {i} matches iter 0 (bitwise equal)")

            assert pass_determinism, "Exp ring joint SDPA determinism failed"
            return

        if not do_check:
            return

        # Accuracy check against PyTorch SDPA reference.
        # joint_seq_len=0 so this is pure non-causal SDPA over the full sequence.
        gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
        gt_out = gt[:, :, :total_seq, :]

        tt_out_torch = to_torch_out(tt_out_list[-1])
        out_pass, out_pcc = comp_pcc(gt_out, tt_out_torch, pcc_threshold)
        mse = ((gt_out - tt_out_torch) ** 2).mean().item()
        logger.info(f"PCC: {out_pcc}, MSE: {mse:.2e}")

        assert out_pass, f"PCC {out_pcc} below threshold {pcc_threshold}"
        assert mse <= max_mse, f"MSE {mse:.2e} exceeds threshold {max_mse:.2e}"

    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ============================================================================
# TEST PARAMETERS
# ============================================================================

TEST_CONFIGS, TEST_CONFIG_IDS = generate_test_configs()


# ============================================================================
# TESTS
# ============================================================================


# === TEST 1: PERFORMANCE SWEEP (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize("max_payload_size", [4096, 8192], ids=["4k", "8k"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("b, nh, total_seq, d, q_chunk_size, k_chunk_size", TEST_CONFIGS, ids=TEST_CONFIG_IDS)
def test_exp_ring_joint_attention_sdpa_sweep_perf_impl(
    b, nh, total_seq, d, q_chunk_size, k_chunk_size, dtype, max_payload_size
):
    """Performance sweep test — run locally, skipped on CI."""
    run_exp_ring_joint_sdpa_nightly(
        b,
        nh,
        total_seq,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        num_iterations=5,
        do_check=False,
        max_payload_size=max_payload_size,
    )


# === TEST 2: ACCURACY VERIFICATION ===
@pytest.mark.skipif(len(TEST_CONFIGS) == 0, reason="No valid device configuration detected")
@pytest.mark.parametrize("max_payload_size", [4096, 8192], ids=["4k", "8k"])
@pytest.mark.parametrize("test_global_mask", [False, True], ids=["aligned", "global_mask"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("b, nh, total_seq, d, q_chunk_size, k_chunk_size", TEST_CONFIGS, ids=TEST_CONFIG_IDS)
def test_exp_ring_joint_attention_sdpa_accuracy(
    b, nh, total_seq, d, q_chunk_size, k_chunk_size, dtype, test_global_mask, max_payload_size, reset_seeds
):
    """
    Accuracy verification: 1 iteration, compare against PyTorch SDPA reference.

    When test_global_mask=True, reduces total_seq by 31 to force logical_n to fall
    mid-tile within a K-chunk, exercising the global N lightweight mask path.

    Thresholds (matching exp ring joint unit tests):
    - PCC >= 0.9993
    - MSE <= 8e-5
    """
    logger.info(f"test_global_mask: {test_global_mask}")
    if test_global_mask:
        total_seq = total_seq - 31
    logger.info(f"total_seq: {total_seq}")
    run_exp_ring_joint_sdpa_nightly(
        b,
        nh,
        total_seq,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        pcc_threshold=DEFAULT_PCC_THRESHOLD,
        max_mse=DEFAULT_MAX_MSE,
        max_payload_size=max_payload_size,
    )


# === TEST 3: DETERMINISM VERIFICATION ===
@pytest.mark.skipif(len(TEST_CONFIGS) == 0, reason="No valid device configuration detected")
@pytest.mark.parametrize("max_payload_size", [4096, 8192], ids=["4k", "8k"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("b, nh, total_seq, d, q_chunk_size, k_chunk_size", TEST_CONFIGS, ids=TEST_CONFIG_IDS)
def test_exp_ring_joint_attention_sdpa_determinism(
    b, nh, total_seq, d, q_chunk_size, k_chunk_size, dtype, max_payload_size, reset_seeds
):
    """
    Determinism verification: run 10 times with same inputs, verify all outputs are bitwise equal.
    """
    run_exp_ring_joint_sdpa_nightly(
        b,
        nh,
        total_seq,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        num_iterations=4,
        max_payload_size=max_payload_size,
    )


# === TEST 4: PERFORMANCE TABLE GENERATOR (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.skipif(len(TEST_CONFIGS) == 0, reason="No valid device configuration detected")
@pytest.mark.timeout(1000)
@pytest.mark.parametrize("b, nh, total_seq, d, q_chunk_size, k_chunk_size", TEST_CONFIGS, ids=TEST_CONFIG_IDS)
def test_exp_ring_joint_attention_create_perf_table(b, nh, total_seq, d, q_chunk_size, k_chunk_size):
    """
    Sweep max_payload_size variants for exp ring joint attention SDPA and print a performance table.
    Skipped on CI - run locally with tracy profiler.
    """
    from tracy.process_model_log import run_device_profiler

    num_devices = detect_devices_without_opening()
    sp_size, tp_size, arch_type = calculate_mesh_config(num_devices)
    is_galaxy = arch_type.startswith("galaxy")

    if sp_size < 4:
        pytest.skip(f"Exp ring joint attention requires >=4 devices, got sp_size={sp_size}")

    grid_cols = GALAXY_GRID_COLS if is_galaxy else NON_GALAXY_GRID_COLS
    grid_rows = GALAXY_GRID_ROWS if is_galaxy else NON_GALAXY_GRID_ROWS
    sdpa_cols = GALAXY_SDPA_COLS if is_galaxy else NON_GALAXY_SDPA_COLS
    total_cores = grid_cols * grid_rows
    total_compute_cores = sdpa_cols * grid_rows
    ccl_cores = grid_rows  # CCL MUX column height
    ccl_overhead_pct = (ccl_cores * 100.0) / total_cores

    ring_size = sp_size
    local_nh = nh // tp_size
    local_seq_len = total_seq // sp_size

    # Reconstruct the parametrize id for this shape — must match TEST_CONFIG_IDS format in generate_test_configs()
    config_id = f"{arch_type}-seq{total_seq}-h{nh}-q{q_chunk_size}-k{k_chunk_size}"

    # Sweep dimension: max_payload_size (must match ids in test_exp_ring_joint_attention_sdpa_sweep_perf_impl)
    payload_variants = [(4096, "4k"), (8192, "8k")]

    subdir = "ttnn_exp_ring_joint_sdpa_performance"
    perf_results = []

    for max_payload, payload_id in payload_variants:
        float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]", "PM FPU UTIL (%)"]
        cols = ["ATTRIBUTES"]

        # Parametrize id order is inner-to-outer: config_id-dtype-payload
        command = (
            f"pytest tests/nightly/blackhole/sdpa/"
            f"test_exp_ring_joint_sdpa.py::"
            f"test_exp_ring_joint_attention_sdpa_sweep_perf_impl"
            f"[{config_id}-bf16-{payload_id}]"
        )

        try:
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
            r = post_process_ops_log(
                subdir, float_columns=float_cols, columns=cols, op_name="", sum_vals=False, has_signposts=False
            )

            measured_core_count = int(r["CORE COUNT"][0]) if len(r["CORE COUNT"]) > 0 else 0
            duration_ns = (
                int(r["DEVICE KERNEL DURATION [ns]"].max()) if len(r["DEVICE KERNEL DURATION [ns]"]) > 0 else 0
            )
            fpu_util_col = r.get("PM FPU UTIL (%)", [])
            fpu_util_min = float(fpu_util_col.min()) if len(fpu_util_col) > 0 else 0.0
            fpu_util_max = float(fpu_util_col.max()) if len(fpu_util_col) > 0 else 0.0

            B = b
            batch_parallel = min(B, total_compute_cores)
            nh_parallel = min(total_compute_cores // batch_parallel, local_nh)
            max_q_parallel = total_compute_cores // (batch_parallel * nh_parallel)

            cores_used = compute_cores_used(total_seq, q_chunk_size, total_compute_cores, local_nh, ring_size, b)
            cores_idle = total_compute_cores - cores_used
            compute_util_pct = (cores_used * 100.0) / total_compute_cores

            k_num_chunks = math.ceil(total_seq / k_chunk_size)
            local_q_num_chunks = math.ceil(local_seq_len / q_chunk_size)
            q_per_core = math.ceil(local_q_num_chunks / max_q_parallel) if max_q_parallel > 0 else local_q_num_chunks
            iters_per_core = q_per_core * k_num_chunks

            # Padding waste
            local_q_padded = local_q_num_chunks * q_chunk_size
            global_q_padded = local_q_padded * ring_size
            local_k_num_chunks = math.ceil(local_seq_len / k_chunk_size)
            local_k_padded = local_k_num_chunks * k_chunk_size
            global_k_padded = local_k_padded * ring_size
            actual_work = total_seq * total_seq
            padded_work = global_q_padded * global_k_padded
            total_waste_pct = ((padded_work - actual_work) / padded_work) * 100 if padded_work > 0 else 0

            # Slot waste
            total_q_slots = max_q_parallel * q_per_core if max_q_parallel > 0 else local_q_num_chunks
            wasted_q_slots = max(0, total_q_slots - local_q_num_chunks)
            slot_waste_pct = (wasted_q_slots / total_q_slots) * 100 if total_q_slots > 0 else 0

            # Math utilization — round down to per-column-multiple for consistency with ring_joint_sdpa table
            effective_cores = measured_core_count - measured_core_count % 10
            utilization = compute_math_utilization(
                local_seq_len, total_seq, d, d, local_nh, duration_ns, effective_cores, is_causal=False
            )

            ring_efficiency = (cores_used * 100.0) / total_cores

            perf_results.append(
                {
                    "max_payload_size": max_payload,
                    "payload_id": payload_id,
                    "measured_core_count": measured_core_count,
                    "cores_used": cores_used,
                    "cores_idle": cores_idle,
                    "compute_util_pct": compute_util_pct,
                    "ccl_cores": ccl_cores,
                    "ccl_overhead_pct": ccl_overhead_pct,
                    "ring_efficiency": ring_efficiency,
                    "iters_per_core": iters_per_core,
                    "total_waste_pct": total_waste_pct,
                    "slot_waste_pct": slot_waste_pct,
                    "duration_ns": duration_ns,
                    "duration_ms": duration_ns / 1e6,
                    "utilization": utilization,
                    "fpu_util_min": fpu_util_min,
                    "fpu_util_max": fpu_util_max,
                }
            )
            logger.info(
                f"payload={payload_id}: {duration_ns/1e6:.3f} ms, "
                f"util={utilization:.1f}%, cores={cores_used}/{total_compute_cores} ({compute_util_pct:.0f}%), "
                f"iters/core={iters_per_core}"
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(f"Error running exp ring joint SDPA with max_payload_size={max_payload}: {e}")
            perf_results.append(
                {
                    "max_payload_size": max_payload,
                    "payload_id": payload_id,
                    "duration_ns": None,
                }
            )

    # Sort by duration (best first)
    valid_results = [r for r in perf_results if r["duration_ns"] is not None]
    valid_results.sort(key=lambda x: x["duration_ns"])

    mm_flops = compute_sdpa_flops(total_seq, total_seq, d, d, nh, is_causal=False)

    # Print summary table
    print(f"\n{'='*190}")
    print(
        f"Exp Ring Joint Attention Performance Sweep: b={b}, nh={nh} (global), s={total_seq}, d={d}, "
        f"q_chunk={q_chunk_size}, k_chunk={k_chunk_size}"
    )
    print(f"Architecture: {arch_type}, Ring size: {ring_size} devices, TP size: {tp_size}")
    print(f"Total MM FLOPs (all devices): {mm_flops:,} ({mm_flops/1e9:.2f} GFLOPs)")
    print(f"Per-device workload: Q={local_seq_len} tokens, K/V={total_seq} tokens (via ring), {local_nh} heads")
    print(f"Core Allocation: {total_compute_cores} compute + {ccl_cores} CCL = {total_cores} total cores")
    print(f"{'='*190}")
    header = "| Rank | Payload | Duration (ms) | Compute Used | Compute Idle | Compute Util | CCL Cores | Ring Eff | Iters/Core | Pad Waste | Slot Waste | FPU Util (%)  | Math Util |"
    sep = "|------|---------|---------------|--------------|--------------|--------------|-----------|----------|------------|-----------|------------|---------------|-----------|"
    print(header)
    print(sep)

    for rank, result in enumerate(valid_results, 1):
        fpu_range = f"{result['fpu_util_min']:.1f}-{result['fpu_util_max']:.1f}"
        print(
            f"| {rank:4d} | {result['payload_id']:>7} | {result['duration_ms']:13.3f} | "
            f"{result['cores_used']:12d} | {result['cores_idle']:12d} | {result['compute_util_pct']:11.0f}% | "
            f"{result['ccl_cores']:9d} | {result['ring_efficiency']:7.0f}% | {result['iters_per_core']:10d} | "
            f"{result['total_waste_pct']:8.1f}% | {result['slot_waste_pct']:9.1f}% | {fpu_range:>13} | {result['utilization']:8.1f}% |"
        )

    failed_results = [r for r in perf_results if r["duration_ns"] is None]
    if failed_results:
        print(f"\nFailed configurations:")
        for result in failed_results:
            print(f"  max_payload_size={result['max_payload_size']}")

    if valid_results:
        best = valid_results[0]
        print(
            f"\nBest configuration: max_payload_size={best['max_payload_size']} "
            f"({best['duration_ms']:.3f} ms, {best['utilization']:.1f}% math util, "
            f"{best['cores_used']}/{total_compute_cores} compute cores, {best['ccl_cores']} CCL cores, "
            f"{best['ring_efficiency']:.1f}% ring eff, {best['iters_per_core']} iters/core, "
            f"{best['total_waste_pct']:.1f}% pad waste, {best['slot_waste_pct']:.1f}% slot waste)"
        )

        print(f"\nExp Ring Joint Attention Analysis:")
        print(f"  Ring size: {ring_size} devices")
        print(f"  CCL overhead: {best['ccl_cores']} cores ({best['ccl_overhead_pct']:.1f}% of total)")
        print(f"  Per-device sequence: {local_seq_len} tokens")
        print(f"  Total coordination: {ring_size} devices x {best['ccl_cores']} CCL cores each")

    print(f"{'='*190}\n")


# === TEST 5: PERFORMANCE CHECK (CI-gated by SDPA_PERF_CHECKS=1) ===
# Symmetric +/- band — catches both regressions and unexpected speedups.
EXP_RING_JOINT_PERF_MARGIN = 0.005

EXP_RING_JOINT_PERF_CHECK_CONFIGS = [
    # (ring_size_expected, max_payload_size, payload_id, expected_util)
    # 4-device ring (QuietBox, 4xGalaxy analog)
    (4, 4096, "4k", 65.1),
    (4, 8192, "8k", 65.0),
]


@pytest.mark.skipif(
    os.environ.get("SDPA_PERF_CHECKS") != "1",
    reason="Set SDPA_PERF_CHECKS=1 to run (CI: sdpa perf tests job)",
)
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "ring_size_expected, max_payload_size, payload_id, expected_util",
    EXP_RING_JOINT_PERF_CHECK_CONFIGS,
    ids=[f"ring{cfg[0]}-{cfg[2]}" for cfg in EXP_RING_JOINT_PERF_CHECK_CONFIGS],
)
def test_exp_ring_joint_attention_perf_check(ring_size_expected, max_payload_size, payload_id, expected_util):
    """Measure exp ring joint SDPA math utilization via tracy and assert within +/- EXP_RING_JOINT_PERF_MARGIN."""
    from tracy.process_model_log import run_device_profiler

    num_devices = detect_devices_without_opening()
    sp_size, tp_size, arch_type = calculate_mesh_config(num_devices)

    if sp_size != ring_size_expected:
        pytest.skip(f"Expected ring size {ring_size_expected}, current topology has ring size {sp_size}")

    b, nh, total_seq, d, q_chunk_size, k_chunk_size = TEST_CONFIGS[0]
    config_id = TEST_CONFIG_IDS[0]
    local_nh = nh // tp_size
    local_seq_len = total_seq // sp_size

    subdir = "ttnn_exp_ring_joint_sdpa_perf_check"
    command = (
        f"pytest tests/nightly/blackhole/sdpa/test_exp_ring_joint_sdpa.py::"
        f"test_exp_ring_joint_attention_sdpa_sweep_perf_impl"
        f"[{config_id}-bf16-{payload_id}]"
    )

    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
    cols = ["ATTRIBUTES"]

    with mock.patch.dict(os.environ, {"CI": "false"}):
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        subdir, float_columns=float_cols, columns=cols, op_name="", sum_vals=False, has_signposts=False
    )

    assert (
        len(r["CORE COUNT"]) > 0 and len(r["DEVICE KERNEL DURATION [ns]"]) > 0
    ), "profiler returned no SDPA ops — inner test was skipped or did not produce a kernel run"

    measured_core_count = int(r["CORE COUNT"][0])
    duration_ns = int(r["DEVICE KERNEL DURATION [ns]"].max())

    # Match perf-table effective_cores rounding (ignore non-multiple-of-10 strays)
    effective_cores = measured_core_count - measured_core_count % 10
    assert (
        effective_cores > 0
    ), f"effective_cores=0 (measured_core_count={measured_core_count}) — profiler output incomplete"

    utilization = compute_math_utilization(
        local_seq_len, total_seq, d, d, local_nh, duration_ns, effective_cores, is_causal=False
    )

    lower = expected_util * (1 - EXP_RING_JOINT_PERF_MARGIN)
    upper = expected_util * (1 + EXP_RING_JOINT_PERF_MARGIN)

    logger.info(
        f"Exp ring joint SDPA perf check ring{ring_size_expected}-{payload_id}: "
        f"duration={duration_ns/1e6:.3f} ms, math_util={utilization:.2f}% "
        f"(expected {expected_util:.2f}%, band [{lower:.2f}, {upper:.2f}])"
    )

    assert lower <= utilization <= upper, (
        f"Math utilization {utilization:.2f}% outside band [{lower:.2f}, {upper:.2f}] "
        f"(expected {expected_util:.2f}%, margin +/- {EXP_RING_JOINT_PERF_MARGIN*100:.1f}%)"
    )
