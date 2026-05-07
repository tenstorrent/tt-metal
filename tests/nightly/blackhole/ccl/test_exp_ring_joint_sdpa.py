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

import os
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
import ttnn
from ttnn.operations.ccl import Topology
from loguru import logger
import pytest

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand


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
