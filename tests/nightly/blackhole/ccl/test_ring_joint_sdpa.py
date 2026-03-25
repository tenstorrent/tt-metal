# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Ring Joint Attention SDPA Tests for Video Generation Models on Blackhole

Tests Ring Joint Attention accuracy and determinism using video generation model shapes
on BH multi-chip setups (single ring 1xN or Galaxy 4x8 mesh).
Perf tests are included but skipped on CI.

BH adaptation: uses init_device_compute_kernel_config instead of WormholeComputeKernelConfig.
"""
import os
import math
import torch
from dataclasses import dataclass, field
from itertools import product
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
import ttnn
from ttnn.operations.ccl import Topology
from loguru import logger
import pytest


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Hardware-specific constants
# These are hardcoded to handle firmware differences across versions
GALAXY_GRID_COLS = 12
GALAXY_GRID_ROWS = 10
NON_GALAXY_GRID_COLS = 11
NON_GALAXY_GRID_ROWS = 10

# Derived hardware constants
GALAXY_TOTAL_CORES = GALAXY_GRID_COLS * GALAXY_GRID_ROWS  # 120 cores
NON_GALAXY_TOTAL_CORES = NON_GALAXY_GRID_COLS * NON_GALAXY_GRID_ROWS  # 110 cores

# CCL allocation: last column reserved for CCL operations
GALAXY_CCL_COLUMN = GALAXY_GRID_COLS - 1  # Column 11
GALAXY_SDPA_COLS = GALAXY_CCL_COLUMN  # Columns 0-10
GALAXY_SDPA_CORES = GALAXY_SDPA_COLS * GALAXY_GRID_ROWS  # 110 cores

NON_GALAXY_CCL_COLUMN = NON_GALAXY_GRID_COLS - 1  # Column 10
NON_GALAXY_SDPA_COLS = NON_GALAXY_CCL_COLUMN  # Columns 0-9
NON_GALAXY_SDPA_CORES = NON_GALAXY_SDPA_COLS * NON_GALAXY_GRID_ROWS  # 100 cores

# Galaxy mesh configuration constants
GALAXY_DEVICE_COUNT = 32
GALAXY_TP_SIZE = 4
GALAXY_SP_SIZE = 8

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
#
# Each model defines per-device sequence lengths for Galaxy and Quiet Box,
# tuned so per-core work matches between platforms (11 vs 10 cores/head).
# See test_instructions.md for full parallelization context.

BATCH_SIZE = 1


@dataclass
class ModelConfig:
    """Benchmark configuration for a video generation model."""

    galaxy_seq_per_device: int
    non_galaxy_seq_per_device: int
    heads_per_device: int  # After TP split (same for both platforms)
    head_dim: int
    q_chunk_sizes: list = field(default_factory=list)
    k_chunk_sizes: list = field(default_factory=list)


MODEL_CONFIGS = {
    # WAN 2.2 — 1×Galaxy deployment (32 chips, ~75K total tokens at 720p)
    "wan2_2_1xGLX": ModelConfig(
        galaxy_seq_per_device=9472,
        non_galaxy_seq_per_device=8544,
        heads_per_device=10,
        head_dim=128,
        q_chunk_sizes=[224, 256, 288],
        k_chunk_sizes=[128, 256, 512],
    ),
    # WAN 2.2 — 4×Galaxy deployment (128 chips, 720p split across 4 Galaxies)
    "wan2_2_4xGLX": ModelConfig(
        galaxy_seq_per_device=2368,
        non_galaxy_seq_per_device=2240,
        heads_per_device=10,
        head_dim=128,
        q_chunk_sizes=[224, 256, 288],
        k_chunk_sizes=[128, 256, 512],
    ),
    # VideogenModel1 — 1×Galaxy deployment (115,200 total tokens)
    # Single benchmark config: Sq_chunk_t=7 (q=224), k=512
    "videogen_model1": ModelConfig(
        galaxy_seq_per_device=14400,
        non_galaxy_seq_per_device=13440,
        heads_per_device=10,
        head_dim=128,
        q_chunk_sizes=[224],
        k_chunk_sizes=[512],
    ),
}

# Accuracy threshold constants
DEFAULT_PCC_THRESHOLD = 0.994
DEFAULT_RMSE_THRESHOLD = 0.05

# Performance calculation constants
BLACKHOLE_CLOCK_GHZ = 1.35  # Blackhole clock frequency in GHz
MM_FLOPS_PER_CYCLE_PER_CORE = 2048  # Matrix multiply FLOPs per cycle per core


from tests.nightly.sdpa_perf_utils import (
    post_process_ops_log,
    compute_cores_used as compute_ring_joint_cores_used,
    compute_math_utilization as compute_ring_joint_utilization,
)


def fa_rand(*shape):
    """
    Generate random tensors with Flash Attention-style distribution.
    """
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def torch_joint_sdpa_reference(q, k, v, joint_q, joint_k, joint_v):
    """
    PyTorch reference implementation for ring joint attention with dummy joint tensors.

    Simulates the ring joint attention computation (WAN 2.2 compatible):
    1. Each device processes local Q attending to all K/V (via ring rotation)
    2. Joint tensors are dummy/empty (seq_len=0) like WAN 2.2
    """
    local_seq_len = q.size(2)

    combined_q = torch.cat([q, joint_q], dim=2)

    # Combine K, V with joint_K, joint_V (full distributed sequence + joint)
    combined_k = torch.cat([k, joint_k], dim=2)
    combined_v = torch.cat([v, joint_v], dim=2)

    # Compute attention for local portion (simulating one device)
    attn_out = torch.nn.functional.scaled_dot_product_attention(combined_q, combined_k, combined_v, is_causal=False)

    # Split outputs back into main and joint parts
    main_out = attn_out[:, :, :local_seq_len, :]
    joint_out = attn_out[:, :, local_seq_len:, :]

    return main_out, joint_out


def detect_devices_without_opening():
    """
    Detect the number of available TT devices WITHOUT opening them.
    Uses /dev/tenstorrent/* device files to avoid holding device locks.
    This is required for performance tests that use run_device_profiler().
    """
    import glob

    device_files = glob.glob("/dev/tenstorrent/*")
    return len(device_files)


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


def generate_test_configs():
    """
    Generate (b, nh, s, d, q_chunk, k_chunk) tuples for all model configs.

    Each model defines its own Q/K chunk sizes, so the cross-product is per-model.

    NOTE: Uses detect_devices_without_opening() to avoid holding device locks
    during pytest collection, which would block subprocess profiling.
    """
    num_devices = detect_devices_without_opening()
    if num_devices < 2:
        return [], []

    sp_size, tp_size, arch_type = calculate_mesh_config(num_devices)
    is_galaxy = arch_type.startswith("galaxy")

    configs = []
    config_ids = []

    for model_name, model in MODEL_CONFIGS.items():
        seq_per_device = model.galaxy_seq_per_device if is_galaxy else model.non_galaxy_seq_per_device
        total_seq = seq_per_device * sp_size
        total_heads = model.heads_per_device * tp_size

        for q_chunk, k_chunk in product(model.q_chunk_sizes, model.k_chunk_sizes):
            configs.append((BATCH_SIZE, total_heads, total_seq, model.head_dim, q_chunk, k_chunk))
            config_ids.append(f"{model_name}-{seq_per_device}x{sp_size}_h{total_heads}-k{k_chunk}-q{q_chunk}")

    return configs, config_ids


def generate_perf_table_configs():
    """
    Generate (model_name, b, nh, s, d) tuples for perf table tests.

    One entry per model — the perf table sweeps Q/K chunk sizes internally.
    """
    num_devices = detect_devices_without_opening()
    if num_devices < 2:
        return [], []

    sp_size, tp_size, arch_type = calculate_mesh_config(num_devices)
    is_galaxy = arch_type.startswith("galaxy")

    configs = []
    config_ids = []

    for model_name, model in MODEL_CONFIGS.items():
        seq_per_device = model.galaxy_seq_per_device if is_galaxy else model.non_galaxy_seq_per_device
        total_seq = seq_per_device * sp_size
        total_heads = model.heads_per_device * tp_size

        configs.append((model_name, BATCH_SIZE, total_heads, total_seq, model.head_dim))
        config_ids.append(f"{model_name}_{seq_per_device}x{sp_size}_h{total_heads}")

    return configs, config_ids


def create_global_semaphores(mesh_device, cores, initial_value):
    """Create global semaphore handles for CCL coordination."""
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]


def run_ring_joint_sdpa(
    b,
    nh,
    nkv,
    sq,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    pcc_threshold=DEFAULT_PCC_THRESHOLD,
    rmse_threshold=None,
    do_check=True,
    num_iterations=1,
):
    """
    Run Ring Joint Attention SDPA using direct ttnn operations with auto-detected devices.

    Args:
        b: Batch size (typically 1)
        nh: Number of attention heads
        nkv: Number of key/value heads (must equal nh for joint attention)
        sq: Base sequence length (will be distributed across ring)
        d: Head dimension
        q_chunk_size: Query chunk size for tiling
        k_chunk_size: Key chunk size for tiling
        dtype: Data type (ttnn.bfloat16)
        sk: Key sequence length (defaults to sq if None)
        pcc_threshold: Pearson correlation threshold for accuracy
        rmse_threshold: Root mean square error threshold
        do_check: Whether to verify accuracy against PyTorch reference
        num_iterations: Number of times to run the op (>1 for determinism testing)
    """
    # Ensure reproducible results
    torch.manual_seed(1234)
    if nh != nkv:
        pytest.skip(f"Ring joint attention currently requires nh == nkv, got nh={nh}, nkv={nkv}")

    # Auto-detect mesh configuration based on available devices
    num_devices = detect_devices_without_opening()
    sp_size, tp_size, arch_type = calculate_mesh_config(num_devices)

    # Ring topology requires >2 devices; fall back to linear for <=2
    use_ring = sp_size > 2
    fabric_config = ttnn.FabricConfig.FABRIC_1D_RING if use_ring else ttnn.FabricConfig.FABRIC_1D
    topology = Topology.Ring if use_ring else Topology.Linear

    # Configure fabric for ring joint attention
    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )

    # Mesh axis configuration
    sp_axis = 1  # Column axis for sequence parallel (ring axis)
    tp_axis = 0  # Row axis for tensor parallel (head axis)

    joint_seq_len = 0  # Use empty joint sequence (WAN 2.2 compatible)

    if sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={sp_size}")

    # Open mesh device based on calculated configuration
    mesh_shape = ttnn.MeshShape(tp_size, sp_size)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    num_links = 2

    try:
        if tp_size > 1 and nh % tp_size != 0:
            pytest.skip(f"num_heads ({nh}) must be divisible by TP size ({tp_size}) for multi-ring architecture")

        # Configure compute grid and CCL coordination - USING HARDCODED GRIDS
        # Use hardcoded grid sizes to handle firmware differences across versions
        if arch_type.startswith("galaxy"):
            sdpa_compute_grid = (GALAXY_SDPA_COLS, GALAXY_GRID_ROWS)  # 11x10 for SDPA
            ccl_column = GALAXY_CCL_COLUMN  # Column 11 for CCL
        else:
            sdpa_compute_grid = (NON_GALAXY_SDPA_COLS, NON_GALAXY_GRID_ROWS)  # 10x10 for SDPA
            ccl_column = NON_GALAXY_CCL_COLUMN  # Column 10 for CCL

        # Get actual device grid for sub-device creation
        full_compute_grid = mesh_device.compute_with_storage_grid_size()

        # Create sub-device for CCL operations - Must include ALL cores that operations will use
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_compute_grid.x - 1, full_compute_grid.y - 1))}
        )
        worker_sub_device = ttnn.SubDevice(
            [
                ccl_sub_device_crs,
            ]
        )
        worker_sub_device_id = ttnn.SubDeviceId(0)

        # Set up sub-device manager with stall group
        sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager)
        sub_device_stall_group = [worker_sub_device_id]
        mesh_device.set_sub_device_stall_group(sub_device_stall_group)

        ccl_semaphore_handles = create_global_semaphores(mesh_device, ccl_sub_device_crs, 0)

        # Create tensors with same full sequence length (following DIT model pattern)
        Q = fa_rand(b, nh, sq, d)
        K = fa_rand(b, nh, sq, d)
        V = fa_rand(b, nh, sq, d)

        # Joint tensors - Use dummy tensors like wan2.2 (empty sequence, zero-filled)
        joint_Q = torch.zeros((b, nh, joint_seq_len, d), dtype=torch.bfloat16)
        joint_K = torch.zeros((b, nh, joint_seq_len, d), dtype=torch.bfloat16)
        joint_V = torch.zeros((b, nh, joint_seq_len, d), dtype=torch.bfloat16)

        # Create persistent output buffers
        kv_shard_dims = [None, None]
        kv_shard_dims[sp_axis] = None
        if tp_size > 1:
            kv_shard_dims[tp_axis] = 1

        expected_output_seq_len = sq
        persistent_output_buffer_k = ttnn.from_torch(
            torch.zeros(b, nh, expected_output_seq_len, d),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims),
        )
        persistent_output_buffer_v = ttnn.from_torch(
            torch.zeros(b, nh, expected_output_seq_len, d),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=kv_shard_dims),
        )

        # Create program config
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        # BH adaptation: use init_device_compute_kernel_config instead of WormholeComputeKernelConfig
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # Convert to TT tensors with appropriate mesh sharding
        sdpa_input_shard_dims = [None, None]
        sdpa_input_shard_dims[sp_axis] = 2
        if tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        sdpa_joint_shard_dims = [None, None]
        if tp_size > 1:
            sdpa_joint_shard_dims[tp_axis] = 1

        tt_Q = ttnn.from_torch(
            Q,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
        tt_K = ttnn.from_torch(
            K,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
        tt_V = ttnn.from_torch(
            V,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
        tt_joint_Q = ttnn.from_torch(
            joint_Q,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
            ),
        )
        tt_joint_K = ttnn.from_torch(
            joint_K,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
            ),
        )
        tt_joint_V = ttnn.from_torch(
            joint_V,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
            ),
        )

        # Set logical_n to the original full sequence length
        corrected_logical_n = sq

        # Precompute mesh composer dims
        main_row_dim = sdpa_input_shard_dims[0] if sdpa_input_shard_dims[0] is not None else -1
        main_col_dim = sdpa_input_shard_dims[1] if sdpa_input_shard_dims[1] is not None else -1

        # Run ring joint attention
        reference_output = None
        for i in range(num_iterations):
            tt_out, tt_joint_out, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                tt_Q,
                tt_K,
                tt_V,
                tt_joint_Q,
                tt_joint_K,
                tt_joint_V,
                persistent_output_buffer_k=persistent_output_buffer_k,
                persistent_output_buffer_v=persistent_output_buffer_v,
                joint_strategy="rear",
                logical_n=corrected_logical_n,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=ccl_semaphore_handles,
                num_links=num_links,
                cluster_axis=sp_axis,
                mesh_device=mesh_device,
                topology=topology,
                subdevice_id=worker_sub_device_id,
                ccl_core_grid_offset=(ccl_column, 0),  # Point to CCL column
                use_column_major_ccl=True,
            )

            # Convert main output to torch and slice out tile-padding
            tt_out_torch = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.create_mesh_composer(
                    mesh_device, ttnn.MeshComposerConfig(main_row_dim, main_col_dim)
                ),
            )
            tt_out_torch = tt_out_torch[:, :, :sq, :]

            # Determinism mode: compare each output to the first
            if num_iterations > 1:
                if reference_output is None:
                    reference_output = tt_out_torch
                elif not torch.equal(reference_output, tt_out_torch):
                    diff_mask = reference_output != tt_out_torch
                    num_diffs = diff_mask.sum().item()
                    max_diff = (reference_output - tt_out_torch).abs().max().item()
                    pytest.fail(
                        f"Ring joint SDPA output at iteration {i} differs from iteration 0: "
                        f"{num_diffs} differing elements, max diff = {max_diff}"
                    )

        if num_iterations > 1:
            logger.info(f"Ring joint SDPA determinism verified: all {num_iterations} outputs are exactly equal")
            return

        if not do_check:
            return

        # Convert and verify joint output (only if joint_seq_len > 0)
        if joint_seq_len > 0:
            if arch_type.startswith("galaxy"):
                joint_row_dim = sdpa_joint_shard_dims[0] if sdpa_joint_shard_dims[0] is not None else -1
                joint_col_dim = sdpa_joint_shard_dims[1] if sdpa_joint_shard_dims[1] is not None else -1
                tt_joint_out_torch = ttnn.to_torch(
                    tt_joint_out,
                    mesh_composer=ttnn.create_mesh_composer(
                        mesh_device, ttnn.MeshComposerConfig(joint_row_dim, joint_col_dim)
                    ),
                )
            else:
                tt_joint_out_torch = ttnn.to_torch(
                    tt_joint_out,
                    mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(1, -1)),
                )

            if tt_joint_out_torch.shape[3] != d:
                tt_joint_out_torch = tt_joint_out_torch[:, :, :, :d]
            if tt_joint_out_torch.shape[0] > 1:
                tt_joint_out_torch = tt_joint_out_torch[0:1, :, :, :]
            tt_joint_out_torch = tt_joint_out_torch[:, :, :joint_seq_len, :]
        else:
            logger.info("Joint output - Dummy tensors (seq_len=0), skipping accuracy check (wan2.2 compatible)")

        # Compute PyTorch reference using ring size (SP dimension)
        gt_main, gt_joint = torch_joint_sdpa_reference(Q, K, V, joint_Q, joint_K, joint_V)

        # Verify accuracy for main output
        out_pass_main, out_pcc_main = comp_pcc(gt_main, tt_out_torch, pcc_threshold)
        rmse_main = torch.sqrt(((gt_main - tt_out_torch) ** 2).mean()).item()
        logger.info(f"Main output - PCC: {out_pcc_main}, RMSE: {rmse_main:.6f}")

        if rmse_threshold is not None:
            assert rmse_main < rmse_threshold, f"Main RMSE {rmse_main:.6f} exceeds threshold {rmse_threshold}"
        assert out_pass_main, f"Main PCC {out_pcc_main} below threshold {pcc_threshold}"

        # Verify accuracy for joint output
        if joint_seq_len > 0:
            out_pass_joint, out_pcc_joint = comp_pcc(gt_joint, tt_joint_out_torch, pcc_threshold)
            rmse_joint = torch.sqrt(((gt_joint - tt_joint_out_torch) ** 2).mean()).item()
            logger.info(f"Joint output - PCC: {out_pcc_joint}, RMSE: {rmse_joint:.6f}")
            if rmse_threshold is not None:
                assert rmse_joint < rmse_threshold, f"Joint RMSE {rmse_joint:.6f} exceeds threshold {rmse_threshold}"
            assert out_pass_joint, f"Joint PCC {out_pcc_joint} below threshold {pcc_threshold}"

    finally:
        # Clean up mesh device
        ttnn.close_mesh_device(mesh_device)

        # Restore fabric to disabled state
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# Generate test parameters dynamically based on detected hardware
TEST_CONFIGS, TEST_CONFIG_IDS = generate_test_configs()
PERF_TABLE_CONFIGS, PERF_TABLE_CONFIG_IDS = generate_perf_table_configs()


# === TEST 1: PERFORMANCE SWEEP (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, s, d, q_chunk_size, k_chunk_size",
    TEST_CONFIGS,
    ids=TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_sweep_perf_impl(b, nh, s, d, q_chunk_size, k_chunk_size, dtype):
    """
    Performance sweep test for ring joint attention SDPA.
    Skipped on CI - run locally for performance measurement.
    """
    run_ring_joint_sdpa(b, nh, nh, s, d, q_chunk_size, k_chunk_size, dtype, do_check=False)


# === TEST 2: ACCURACY VERIFICATION ===
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, s, d, q_chunk_size, k_chunk_size",
    TEST_CONFIGS,
    ids=TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_accuracy(b, nh, s, d, q_chunk_size, k_chunk_size, dtype):
    """
    Accuracy verification test for ring joint attention SDPA.

    ACCURACY METRICS:
    - PCC (Pearson Correlation Coefficient): Measures linear correlation
    - RMSE (Root Mean Square Error): Measures absolute error magnitude

    THRESHOLD RATIONALE:
    - PCC = 0.994: Relaxed for joint attention complexity
    """
    pcc_threshold = DEFAULT_PCC_THRESHOLD
    rmse_threshold = DEFAULT_RMSE_THRESHOLD
    run_ring_joint_sdpa(
        b,
        nh,
        nh,
        s,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        pcc_threshold=pcc_threshold,
        rmse_threshold=rmse_threshold,
    )


# === TEST 3: DETERMINISM VERIFICATION ===
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize(
    "b, nh, s, d, q_chunk_size, k_chunk_size",
    TEST_CONFIGS,
    ids=TEST_CONFIG_IDS,
)
def test_ring_joint_attention_sdpa_determinism(b, nh, s, d, q_chunk_size, k_chunk_size, dtype):
    """
    Test ring joint attention SDPA determinism: run 10 times with same inputs and verify outputs match exactly.
    """
    num_iterations = 10
    run_ring_joint_sdpa(b, nh, nh, s, d, q_chunk_size, k_chunk_size, dtype, num_iterations=num_iterations)


# === TEST 4: PERFORMANCE TABLE GENERATOR (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(1000)
@pytest.mark.parametrize(
    "model_name, b, nh, s, d",
    PERF_TABLE_CONFIGS,
    ids=PERF_TABLE_CONFIG_IDS,
)
def test_ring_joint_attention_create_perf_table(model_name, b, nh, s, d):
    """
    Sweep chunk sizes for ring joint attention SDPA and print a performance table.
    Skipped on CI - run locally with tracy profiler.
    """
    from tracy.process_model_log import run_device_profiler

    model = MODEL_CONFIGS[model_name]

    num_devices = detect_devices_without_opening()
    sp_size, tp_size, arch_type = calculate_mesh_config(num_devices)
    ring_size = sp_size
    is_galaxy = arch_type.startswith("galaxy")
    seq_per_device = model.galaxy_seq_per_device if is_galaxy else model.non_galaxy_seq_per_device
    total_heads = model.heads_per_device * tp_size

    if ring_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices, got {ring_size}")

    # Use hardcoded grid constants (cannot query device due to TLB conflicts with subprocess tests)
    if is_galaxy:
        full_grid_rows = GALAXY_GRID_ROWS
        total_compute_cores = GALAXY_SDPA_CORES
        total_cores = GALAXY_TOTAL_CORES
    else:
        full_grid_rows = NON_GALAXY_GRID_ROWS
        total_compute_cores = NON_GALAXY_SDPA_CORES
        total_cores = NON_GALAXY_TOTAL_CORES

    ccl_cores = full_grid_rows  # Full column height for CCL
    ccl_overhead_pct = (ccl_cores * 100.0) / total_cores

    subdir = "ttnn_ring_joint_sdpa_performance"
    perf_results = []

    for q_chunk_size, k_chunk_size in product(model.q_chunk_sizes, model.k_chunk_sizes):
        float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]"]
        cols = ["ATTRIBUTES"]

        config_id = f"{model_name}-{seq_per_device}x{sp_size}_h{total_heads}-k{k_chunk_size}-q{q_chunk_size}"
        command = (
            f"pytest tests/nightly/blackhole/ccl/"
            f"test_ring_joint_sdpa.py::"
            f"test_ring_joint_attention_sdpa_sweep_perf_impl"
            f"[{config_id}-bf16]"
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

            local_seq_len = s // ring_size

            B = BATCH_SIZE
            batch_parallel = min(B, total_compute_cores)
            nh_parallel = min(total_compute_cores // batch_parallel, nh)
            max_q_parallel = total_compute_cores // (batch_parallel * nh_parallel)

            cores_used = compute_ring_joint_cores_used(s, q_chunk_size, total_compute_cores, nh, ring_size)
            cores_idle = total_compute_cores - cores_used
            compute_util_pct = (cores_used * 100.0) / total_compute_cores

            k_num_chunks = math.ceil(s / k_chunk_size)
            local_q_num_chunks = math.ceil(local_seq_len / q_chunk_size)
            q_per_core = math.ceil(local_q_num_chunks / max_q_parallel) if max_q_parallel > 0 else local_q_num_chunks
            iters_per_core = q_per_core * k_num_chunks

            # Padding waste
            local_q_padded = local_q_num_chunks * q_chunk_size
            global_q_padded = local_q_padded * ring_size
            local_k_num_chunks = math.ceil(local_seq_len / k_chunk_size)
            local_k_padded = local_k_num_chunks * k_chunk_size
            global_k_padded = local_k_padded * ring_size
            actual_work = s * s
            padded_work = global_q_padded * global_k_padded
            total_waste_pct = ((padded_work - actual_work) / padded_work) * 100 if padded_work > 0 else 0

            # Slot waste
            total_q_slots = max_q_parallel * q_per_core if max_q_parallel > 0 else local_q_num_chunks
            wasted_q_slots = max(0, total_q_slots - local_q_num_chunks)
            slot_waste_pct = (wasted_q_slots / total_q_slots) * 100 if total_q_slots > 0 else 0

            # Math utilization
            effective_cores = measured_core_count - measured_core_count % 10
            heads_per_device = nh / tp_size
            utilization = compute_ring_joint_utilization(
                local_seq_len, s, d, heads_per_device, duration_ns, effective_cores
            )

            ring_efficiency = (cores_used * 100.0) / total_cores

            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
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
                }
            )
            logger.info(
                f"q={q_chunk_size}, k={k_chunk_size}: {duration_ns/1e6:.3f} ms, "
                f"util={utilization:.1f}%, cores={cores_used}/{total_compute_cores} ({compute_util_pct:.0f}%), "
                f"iters/core={iters_per_core}"
            )

        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            logger.error(
                f"Error running ring joint SDPA with q_chunk_size={q_chunk_size}, k_chunk_size={k_chunk_size}: {e}"
            )
            perf_results.append(
                {
                    "q_chunk_size": q_chunk_size,
                    "k_chunk_size": k_chunk_size,
                    "duration_ns": None,
                }
            )

    # Sort by duration (best first)
    valid_results = [r for r in perf_results if r["duration_ns"] is not None]
    valid_results.sort(key=lambda x: x["duration_ns"])

    mm_flops = 4 * s * s * d * nh

    # Print summary table
    print(f"\n{'='*190}")
    print(f"Ring Joint Attention Performance Sweep: {model_name} — b={b}, nh={nh}, s={s}, d={d}")
    print(f"Architecture: {arch_type}, Ring size: {ring_size} devices")
    print(f"Total MM FLOPs (all devices): {mm_flops:,} ({mm_flops/1e9:.2f} GFLOPs)")
    print(f"Per-device workload: Q={s // ring_size} tokens, K/V={s} tokens (via ring), {nh} heads")
    print(f"Core Allocation: {total_compute_cores} compute + {ccl_cores} CCL = {total_cores} total cores")
    print(f"{'='*190}")
    header = "| Rank | q_chunk | k_chunk | Duration (ms) | Compute Used | Compute Idle | Compute Util | CCL Cores | Ring Eff | Iters/Core | Pad Waste | Slot Waste | Math Util |"
    sep = "|------|---------|---------|---------------|--------------|--------------|--------------|-----------|----------|------------|-----------|------------|-----------|"
    print(header)
    print(sep)

    for rank, result in enumerate(valid_results, 1):
        print(
            f"| {rank:4d} | {result['q_chunk_size']:7d} | {result['k_chunk_size']:7d} | {result['duration_ms']:13.3f} | "
            f"{result['cores_used']:12d} | {result['cores_idle']:12d} | {result['compute_util_pct']:11.0f}% | "
            f"{result['ccl_cores']:9d} | {result['ring_efficiency']:7.0f}% | {result['iters_per_core']:10d} | "
            f"{result['total_waste_pct']:8.1f}% | {result['slot_waste_pct']:9.1f}% | {result['utilization']:8.1f}% |"
        )

    failed_results = [r for r in perf_results if r["duration_ns"] is None]
    if failed_results:
        print(f"\nFailed configurations:")
        for result in failed_results:
            print(f"  q_chunk_size={result['q_chunk_size']}, k_chunk_size={result['k_chunk_size']}")

    if valid_results:
        best = valid_results[0]
        print(
            f"\nBest configuration: q_chunk_size={best['q_chunk_size']}, "
            f"k_chunk_size={best['k_chunk_size']} "
            f"({best['duration_ms']:.3f} ms, {best['utilization']:.1f}% math util, "
            f"{best['cores_used']}/{total_compute_cores} compute cores, {best['ccl_cores']} CCL cores, "
            f"{best['ring_efficiency']:.1f}% ring eff, {best['iters_per_core']} iters/core, "
            f"{best['total_waste_pct']:.1f}% pad waste, {best['slot_waste_pct']:.1f}% slot waste)"
        )

        print(f"\nRing Joint Attention Analysis:")
        print(f"  Ring size: {ring_size} devices")
        print(f"  CCL overhead: {best['ccl_cores']} cores ({best['ccl_overhead_pct']:.1f}% of total)")
        print(f"  Per-device sequence: {s // ring_size} tokens")
        print(f"  Total coordination: {ring_size} devices x {best['ccl_cores']} CCL cores each")

    print(f"{'='*190}\n")
