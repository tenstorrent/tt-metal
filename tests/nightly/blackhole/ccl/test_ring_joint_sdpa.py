# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Ring Joint Attention SDPA Tests for WAN and MLA Models on Blackhole

Tests Ring Joint Attention accuracy and determinism using:
- WAN 2.2 model shapes: standard attention with non-causal, non-balanced mode
- DeepSeek MLA (Multi-Latent Attention): causal attention with balanced zigzag work distribution

Runs on BH multi-chip setups (single ring 1xN or Galaxy 4x8 mesh).
Perf tests are included but skipped on CI.

Model Configurations:
- WAN: nhq == nhk == nhv, d_q == d_k == d_v == 128, bfloat16 for all tensors
- MLA: nhk == 1, d_q == d_k == 576, d_v == 128, bfloat16 for Q, bfloat8_b for K/V

BH adaptation: uses init_device_compute_kernel_config instead of WormholeComputeKernelConfig.
"""
import os
import math
import torch
from dataclasses import dataclass, field
from itertools import product
from typing import ClassVar, List, Tuple
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
import ttnn
from ttnn.operations.ccl import Topology
from loguru import logger
import pytest


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================


@dataclass
class HardwareConfig:
    """Hardware configuration detected at runtime.

    Use HardwareConfig.detect() to create an instance with all values resolved.
    """

    # Galaxy hardware constants
    GALAXY_DEVICE_COUNT: ClassVar[int] = 32
    GALAXY_GRID: ClassVar[Tuple[int, int]] = (12, 10)  # (cols, rows)
    GALAXY_TP_SIZE: ClassVar[int] = 4
    GALAXY_SP_SIZE: ClassVar[int] = 8

    # Non-Galaxy hardware constants
    NON_GALAXY_GRID: ClassVar[Tuple[int, int]] = (11, 10)  # (cols, rows)

    # Instance fields (set by detect())
    is_galaxy: bool
    num_devices: int
    sp_size: int  # Sequence parallel size (devices per ring)
    tp_size: int  # Tensor parallel size (number of rings)
    grid_cols: int
    grid_rows: int

    # Computed fields (set by __post_init__)
    arch_type: str = field(init=False)
    total_cores: int = field(init=False)
    ccl_column: int = field(init=False)
    sdpa_cols: int = field(init=False)
    sdpa_cores: int = field(init=False)

    def __post_init__(self):
        self.arch_type = "galaxy_4x8" if self.is_galaxy else f"single_ring_{self.num_devices}x1"
        self.total_cores = self.grid_cols * self.grid_rows
        self.ccl_column = self.grid_cols - 1
        self.sdpa_cols = self.ccl_column
        self.sdpa_cores = self.sdpa_cols * self.grid_rows

    @classmethod
    def detect(cls) -> "HardwareConfig":
        """Detect hardware and create config with all values resolved."""
        num_devices = _detect_devices_without_opening()
        is_galaxy = num_devices == cls.GALAXY_DEVICE_COUNT
        grid = cls.GALAXY_GRID if is_galaxy else cls.NON_GALAXY_GRID
        return cls(
            is_galaxy=is_galaxy,
            num_devices=num_devices,
            sp_size=cls.GALAXY_SP_SIZE if is_galaxy else num_devices,
            tp_size=cls.GALAXY_TP_SIZE if is_galaxy else 1,
            grid_cols=grid[0],
            grid_rows=grid[1],
        )


@dataclass
class WANConfig:
    """WAN 2.2 model configuration for Ring Joint SDPA.

    WAN uses standard attention with:
    - All head counts equal (nhq == nhk == nhv)
    - Same dimension for Q, K, V (128)
    - Non-causal, non-balanced attention
    - bfloat16 for all tensors

    Use WANConfig.from_hardware(hw) to get a config with hardware-specific values.
    """

    # Hardware-specific constants (per-device sequence lengths)
    GALAXY_SEQ_LENS: ClassVar[List[int]] = [2368, 9472]
    NON_GALAXY_SEQ_LENS: ClassVar[List[int]] = [2240, 8544]

    # Fixed model parameters
    nhq: int = 10
    nhk: int = 10
    nhv: int = 10
    d_q: int = 128
    d_k: int = 128
    d_v: int = 128
    is_causal: bool = False
    is_balanced: bool = False
    pcc_threshold: float = 0.994
    rmse_threshold: float = 0.05

    # Dtypes
    q_dtype: ttnn.DataType = ttnn.bfloat16
    kv_dtype: ttnn.DataType = ttnn.bfloat16

    # Sweep parameters (chunk sizes to test)
    q_chunk_sizes: List[int] = field(default_factory=lambda: [224, 256, 288])
    k_chunk_sizes: List[int] = field(default_factory=lambda: [128, 256, 512])

    # Hardware-dependent (set by from_hardware)
    seq_lens: List[int] = field(default_factory=list)

    @classmethod
    def from_hardware(cls, hw: "HardwareConfig") -> "WANConfig":
        """Create WANConfig with hardware-specific values resolved."""
        return cls(seq_lens=cls.GALAXY_SEQ_LENS if hw.is_galaxy else cls.NON_GALAXY_SEQ_LENS)


@dataclass
class MLAConfig:
    """DeepSeek MLA (Multi-Latent Attention) configuration for Ring Joint SDPA.

    MLA uses:
    - Single K head (nhk=1), multiple Q/V heads
    - Different dimensions for Q/K (576) vs V (128)
    - Causal attention with balanced zigzag workload distribution
    - bfloat16 for Q, bfloat8_b for K/V

    Use MLAConfig.from_hardware(hw) to get a config with hardware-specific values.
    """

    # Hardware-specific constants (heads per device)
    GALAXY_NHQ: ClassVar[int] = 32
    NON_GALAXY_NHQ: ClassVar[int] = 29

    # Fixed model parameters
    nhk: int = 1  # Always 1 for MLA
    d_q: int = 576
    d_k: int = 576
    d_v: int = 128
    is_causal: bool = True
    is_balanced: bool = True
    pcc_threshold: float = 0.999  # Higher threshold for MLA
    rmse_threshold: float = 0.05

    # Dtypes: Q is bfloat16, KV is bfloat8_b
    q_dtype: ttnn.DataType = ttnn.bfloat16
    kv_dtype: ttnn.DataType = ttnn.bfloat8_b

    # Fixed workloads: (seq_len_per_device, q_chunk_size, k_chunk_size)
    workloads: List[Tuple[int, int, int]] = field(default_factory=lambda: [(3200, 160, 160), (4096, 128, 128)])

    # Hardware-dependent (set by from_hardware)
    nhq: int = 0
    nhv: int = 0

    @property
    def seq_lens(self) -> List[int]:
        """Extract unique sequence lengths from workloads."""
        return list(dict.fromkeys(w[0] for w in self.workloads))

    def get_chunk_sizes(self, seq_len_per_device: int) -> Tuple[int, int]:
        """Get (q_chunk_size, k_chunk_size) for a given sequence length."""
        for seq_len, q_chunk, k_chunk in self.workloads:
            if seq_len == seq_len_per_device:
                return (q_chunk, k_chunk)
        raise ValueError(f"No workload found for seq_len_per_device={seq_len_per_device}")

    @classmethod
    def from_hardware(cls, hw: "HardwareConfig") -> "MLAConfig":
        """Create MLAConfig with hardware-specific values resolved."""
        nhq = cls.GALAXY_NHQ if hw.is_galaxy else cls.NON_GALAXY_NHQ
        return cls(nhq=nhq, nhv=nhq)


# ============================================================================
# CONSTANTS
# ============================================================================

BATCH_SIZE = 1

# Performance calculation constants
BLACKHOLE_CLOCK_GHZ = 1.35  # Blackhole clock frequency in GHz
MM_FLOPS_PER_CYCLE_PER_CORE = 2048  # Matrix multiply FLOPs per cycle per core


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def create_balanced_chunk_order(sp_size: int) -> List[int]:
    """Create balanced chunk order for sequence reordering.

    For sp_size=4, creates 2*4=8 chunks with order: [0, 7, 1, 6, 2, 5, 3, 4]
    This interleaves chunks from start and end to balance causal attention workload
    across devices (early chunks have less work, late chunks have more).
    """
    num_chunks = 2 * sp_size
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


def reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: List[int], seq_dim: int = 2) -> torch.Tensor:
    """Reorder tensor chunks along sequence dimension according to chunk_order.

    Used to prepare Q/K/V tensors for balanced causal attention before sending to device.
    """
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


def reverse_reorder_tensor_chunks(tensor: torch.Tensor, chunk_order: List[int], seq_dim: int = 2) -> torch.Tensor:
    """Reverse the chunk reordering to restore original order.

    Used to convert device output back to original sequence order for comparison.
    """
    # Create inverse permutation
    inverse_order = [0] * len(chunk_order)
    for new_pos, orig_pos in enumerate(chunk_order):
        inverse_order[orig_pos] = new_pos

    return reorder_tensor_chunks(tensor, inverse_order, seq_dim)


def post_process_ops_log(
    output_logs_subdir, float_columns=None, columns=None, sum_vals=True, op_name="", has_signposts=False
):
    """Process the ops log CSV and extract performance data."""
    from tracy.process_model_log import get_latest_ops_log_filename
    import pandas as pd

    filename = get_latest_ops_log_filename(output_logs_subdir)
    df = pd.read_csv(filename)

    if has_signposts:
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        start = markers[markers == "start"].index[0]
        stop = markers[markers == "stop"].index[0]
        df = df.iloc[start + 1 : stop]
    if op_name != "":
        df = df[df["OP CODE"] == op_name]

    results = {}
    if float_columns:
        for col in float_columns:
            df_filtered = df[df[col] != "-"]
            if sum_vals:
                results[col] = df_filtered[col].astype(float).sum()
            else:
                results[col] = df_filtered[col].astype(float).to_numpy()
    if columns:
        for col in columns:
            df_filtered = df[df[col] != "-"]
            results[col] = df_filtered[col]
    else:
        results = df
    return results


def compute_ring_joint_cores_used(seqlen, q_chunk_size, compute_cores, num_heads, ring_size):
    """
    Compute number of cores actually used for ring joint attention based on parallelization scheme.
    """
    B = BATCH_SIZE
    local_seq_len = seqlen // ring_size
    q_num_chunks = math.ceil(local_seq_len / q_chunk_size)

    batch_parallel = min(B, compute_cores)
    nh_parallel = min(compute_cores // batch_parallel, num_heads)
    q_parallel = min(compute_cores // (batch_parallel * nh_parallel), q_num_chunks)

    cores_used = batch_parallel * nh_parallel * q_parallel
    return cores_used


def compute_sdpa_flops(sq, sk, d_q, d_v, num_heads, is_causal=False):
    """
    Compute FLOPs for SDPA operation.

    Matches sdpa_perf_model.cpp calculation:
    - Q*K matmul: 2 * d_q * Sq * Sk FLOPs
    - attn*V matmul: 2 * d_v * Sq * Sk FLOPs
    - Causal: divide by 2 (only half the attention matrix computed)
    """
    flops = 2 * sq * sk * (d_q + d_v) * num_heads
    if is_causal:
        flops //= 2
    return flops


def compute_ring_joint_utilization(
    local_seqlen, total_seqlen, d_q, d_v, num_heads_per_device, duration_ns, core_count, is_causal=False
):
    """
    Compute math utilization for ring joint attention.
    """
    mm_flops = compute_sdpa_flops(local_seqlen, total_seqlen, d_q, d_v, num_heads_per_device, is_causal)

    cycles = duration_ns * BLACKHOLE_CLOCK_GHZ
    theoretical_flops = core_count * cycles * MM_FLOPS_PER_CYCLE_PER_CORE
    utilization = (mm_flops / theoretical_flops) * 100
    return utilization


def fa_rand(*shape):
    """
    Generate random tensors with Flash Attention-style distribution.
    """
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def torch_joint_sdpa_reference(q, k, v, joint_q, joint_k, joint_v, is_causal=False):
    """
    PyTorch reference implementation for ring joint attention with dummy joint tensors.

    Simulates the ring joint attention computation:
    1. Each device processes local Q attending to all K/V (via ring rotation)
    2. Joint tensors are dummy/empty (seq_len=0) for WAN 2.2

    Args:
        q, k, v: Main attention tensors
        joint_q, joint_k, joint_v: Joint tensors (can be empty)
        is_causal: Whether to use causal attention mask
    """
    local_seq_len = q.size(2)

    combined_q = torch.cat([q, joint_q], dim=2)

    # Combine K, V with joint_K, joint_V (full distributed sequence + joint)
    combined_k = torch.cat([k, joint_k], dim=2)
    combined_v = torch.cat([v, joint_v], dim=2)

    # Compute attention for local portion (simulating one device)
    attn_out = torch.nn.functional.scaled_dot_product_attention(combined_q, combined_k, combined_v, is_causal=is_causal)

    # Split outputs back into main and joint parts
    main_out = attn_out[:, :, :local_seq_len, :]
    joint_out = attn_out[:, :, local_seq_len:, :]

    return main_out, joint_out


def _detect_devices_without_opening():
    """
    Detect the number of available TT devices WITHOUT opening them.
    Uses /dev/tenstorrent/* device files to avoid holding device locks.
    This is required for performance tests that use run_device_profiler().
    """
    import glob

    device_files = glob.glob("/dev/tenstorrent/*")
    return len(device_files)


def create_global_semaphores(mesh_device, cores, initial_value):
    """Create global semaphore handles for CCL coordination."""
    return [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]


def run_ring_joint_sdpa(
    b,
    nhq,
    nhk,
    sq,
    d_q,
    q_chunk_size,
    k_chunk_size,
    q_dtype,
    *,
    nhv=None,
    d_k=None,
    d_v=None,
    kv_dtype=None,
    is_causal=False,
    is_balanced=False,
    pcc_threshold=0.994,
    rmse_threshold=None,
    do_check=True,
    num_iterations=1,
):
    """
    Run Ring Joint Attention SDPA using direct ttnn operations with auto-detected devices.

    Args:
        b: Batch size (typically 1)
        nhq: Number of query heads
        nhk: Number of key heads (can be 1 for MLA)
        sq: Base sequence length (will be distributed across ring)
        d_q: Query head dimension
        q_chunk_size: Query chunk size for tiling
        k_chunk_size: Key chunk size for tiling
        q_dtype: Data type for Q tensor (e.g., ttnn.bfloat16)
        nhv: Number of value heads (defaults to nhq)
        d_k: Key head dimension (defaults to d_q)
        d_v: Value head dimension (defaults to d_q)
        kv_dtype: Data type for K/V tensors (defaults to q_dtype, can be ttnn.bfloat8_b for MLA)
        is_causal: Whether to use causal attention mask
        is_balanced: Whether to use balanced zigzag work distribution (for causal attention)
        pcc_threshold: Pearson correlation threshold for accuracy
        rmse_threshold: Root mean square error threshold
        do_check: Whether to verify accuracy against PyTorch reference
        num_iterations: Number of times to run the op (>1 for determinism testing)
    """
    # Apply defaults for optional parameters
    if nhv is None:
        nhv = nhq
    if d_k is None:
        d_k = d_q
    if d_v is None:
        d_v = d_q
    if kv_dtype is None:
        kv_dtype = q_dtype

    logger.debug(
        f"run_ring_joint_sdpa params: b={b}, nhq={nhq}, nhk={nhk}, nhv={nhv}, "
        f"sq={sq}, d_q={d_q}, d_k={d_k}, d_v={d_v}, "
        f"q_chunk={q_chunk_size}, k_chunk={k_chunk_size}, "
        f"q_dtype={q_dtype}, kv_dtype={kv_dtype}, "
        f"is_causal={is_causal}, is_balanced={is_balanced}, "
        f"pcc_threshold={pcc_threshold}, rmse_threshold={rmse_threshold}, "
        f"do_check={do_check}, num_iterations={num_iterations}"
    )

    # Ensure reproducible results
    torch.manual_seed(1234)

    # Validate head count constraints
    # For WAN: nhq == nhk == nhv (standard attention)
    # For MLA: nhk == 1, nhq == nhv (multi-latent attention with single K head)
    if nhk != 1 and nhq != nhk:
        pytest.skip(f"Ring joint attention requires nhq == nhk or nhk == 1, got nhq={nhq}, nhk={nhk}")

    # Auto-detect mesh configuration based on available devices
    hw = HardwareConfig.detect()

    # Ring topology requires >2 devices; fall back to linear for <=2
    use_ring = hw.sp_size > 2
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

    if hw.sp_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices in ring, got SP={hw.sp_size}")

    # Open mesh device based on calculated configuration
    mesh_shape = ttnn.MeshShape(hw.tp_size, hw.sp_size)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    num_links = 2

    try:
        if hw.tp_size > 1 and nhq % hw.tp_size != 0:
            pytest.skip(f"num_heads ({nhq}) must be divisible by TP size ({hw.tp_size}) for multi-ring architecture")

        # Configure compute grid and CCL coordination
        sdpa_compute_grid = (hw.sdpa_cols, hw.grid_rows)
        ccl_column = hw.ccl_column

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

        # Create tensors with appropriate shapes
        # Q: [b, nhq, sq, d_q]
        # K: [b, nhk, sq, d_k] - nhk can be 1 for MLA (broadcast to all Q heads)
        # V: [b, nhv, sq, d_v] - nhv typically equals nhq
        Q = fa_rand(b, nhq, sq, d_q)
        K = fa_rand(b, nhk, sq, d_k)
        V = fa_rand(b, nhv, sq, d_v)

        # Joint tensors - Use dummy tensors like WAN 2.2 (empty sequence, zero-filled)
        joint_Q = torch.zeros((b, nhq, joint_seq_len, d_q), dtype=torch.bfloat16)
        joint_K = torch.zeros((b, nhk, joint_seq_len, d_k), dtype=torch.bfloat16)
        joint_V = torch.zeros((b, nhv, joint_seq_len, d_v), dtype=torch.bfloat16)

        # Keep original tensors for reference comparison (before any reordering)
        Q_original, K_original, V_original = Q, K, V

        # Apply balanced reordering if enabled (for causal attention workload balancing)
        chunk_order = None
        if is_balanced:
            chunk_order = create_balanced_chunk_order(hw.sp_size)
            Q = reorder_tensor_chunks(Q, chunk_order)
            K = reorder_tensor_chunks(K, chunk_order)
            V = reorder_tensor_chunks(V, chunk_order)

        # Create persistent output buffers
        kv_shard_dims = [None, None]
        kv_shard_dims[sp_axis] = None
        if hw.tp_size > 1:
            kv_shard_dims[tp_axis] = 1

        # Persistent K output buffer uses nhk and d_k dimensions
        # Persistent V output buffer uses nhv and d_v dimensions
        expected_output_seq_len = sq

        # For K buffer: handle nhk=1 case (MLA) - may need different sharding
        persistent_k_shard_dims = [None, None]
        persistent_k_shard_dims[sp_axis] = None
        if hw.tp_size > 1 and nhk != 1:
            persistent_k_shard_dims[tp_axis] = 1

        persistent_output_buffer_k = ttnn.from_torch(
            torch.zeros(b, nhk, expected_output_seq_len, d_k),
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=persistent_k_shard_dims
            ),
        )
        persistent_output_buffer_v = ttnn.from_torch(
            torch.zeros(b, nhv, expected_output_seq_len, d_v),
            dtype=kv_dtype,
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
        if hw.tp_size > 1:
            sdpa_input_shard_dims[tp_axis] = 1

        # K tensor may have nhk=1 (MLA), different sharding
        sdpa_k_shard_dims = [None, None]
        sdpa_k_shard_dims[sp_axis] = 2
        if hw.tp_size > 1 and nhk != 1:
            sdpa_k_shard_dims[tp_axis] = 1

        sdpa_joint_shard_dims = [None, None]
        if hw.tp_size > 1:
            sdpa_joint_shard_dims[tp_axis] = 1

        sdpa_joint_k_shard_dims = [None, None]
        if hw.tp_size > 1 and nhk != 1:
            sdpa_joint_k_shard_dims[tp_axis] = 1

        # Q tensor uses q_dtype
        tt_Q = ttnn.from_torch(
            Q,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
        # K, V tensors use kv_dtype (can be bfloat8_b for MLA)
        tt_K = ttnn.from_torch(
            K,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_k_shard_dims
            ),
        )
        tt_V = ttnn.from_torch(
            V,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_input_shard_dims
            ),
        )
        tt_joint_Q = ttnn.from_torch(
            joint_Q,
            dtype=q_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_shard_dims
            ),
        )
        tt_joint_K = ttnn.from_torch(
            joint_K,
            dtype=kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=sdpa_joint_k_shard_dims
            ),
        )
        tt_joint_V = ttnn.from_torch(
            joint_V,
            dtype=kv_dtype,
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
                is_causal=is_causal,
                is_balanced=is_balanced,
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

            # Reverse balanced reordering if enabled (restore original sequence order)
            if is_balanced and chunk_order is not None:
                tt_out_torch = reverse_reorder_tensor_chunks(tt_out_torch, chunk_order)

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
            if hw.arch_type.startswith("galaxy"):
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

            if tt_joint_out_torch.shape[3] != d_v:
                tt_joint_out_torch = tt_joint_out_torch[:, :, :, :d_v]
            if tt_joint_out_torch.shape[0] > 1:
                tt_joint_out_torch = tt_joint_out_torch[0:1, :, :, :]
            tt_joint_out_torch = tt_joint_out_torch[:, :, :joint_seq_len, :]
        else:
            logger.info("Joint output - Dummy tensors (seq_len=0), skipping accuracy check (wan2.2 compatible)")

        # Compute PyTorch reference on ORIGINAL data (before balanced reordering)
        gt_main, gt_joint = torch_joint_sdpa_reference(
            Q_original, K_original, V_original, joint_Q, joint_K, joint_V, is_causal=is_causal
        )

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


# ============================================================================
# TEST CASE GENERATION
# ============================================================================


def generate_test_cases():
    """Generate test cases for both WAN and MLA configurations.

    Returns:
        cases: List of tuples with all parameters for run_ring_joint_sdpa
        ids: List of test ID strings for pytest
    """
    hw = HardwareConfig.detect()
    if hw.num_devices < 2:
        return [], []

    cases = []
    ids = []

    # -------------------------------------------------------------------------
    # WAN test cases: standard attention, non-causal, non-balanced
    # -------------------------------------------------------------------------
    wan = WANConfig.from_hardware(hw)

    for seq_len_per_device in wan.seq_lens:
        total_seq_len = seq_len_per_device * hw.sp_size
        total_heads = wan.nhq * hw.tp_size

        for q_chunk in wan.q_chunk_sizes:
            for k_chunk in wan.k_chunk_sizes:
                cases.append(
                    (
                        BATCH_SIZE,  # b
                        total_seq_len,  # sq (total sequence length)
                        total_heads,  # nhq
                        total_heads,  # nhk (same as nhq for WAN)
                        total_heads,  # nhv (same as nhq for WAN)
                        wan.d_q,  # d_q
                        wan.d_k,  # d_k
                        wan.d_v,  # d_v
                        q_chunk,  # q_chunk_size
                        k_chunk,  # k_chunk_size
                        wan.is_causal,  # is_causal
                        wan.is_balanced,  # is_balanced
                        wan.q_dtype,  # q_dtype
                        wan.kv_dtype,  # kv_dtype
                        wan.pcc_threshold,  # pcc_threshold
                        wan.rmse_threshold,  # rmse_threshold
                    )
                )
                ids.append(f"wan-{seq_len_per_device}-q{q_chunk}-k{k_chunk}")

    # -------------------------------------------------------------------------
    # MLA test cases: multi-latent attention, causal, balanced
    # -------------------------------------------------------------------------
    mla = MLAConfig.from_hardware(hw)

    for seq_len_per_device, q_chunk, k_chunk in mla.workloads:
        total_seq_len = seq_len_per_device * hw.sp_size
        total_nhq = mla.nhq * hw.tp_size

        cases.append(
            (
                BATCH_SIZE,  # b
                total_seq_len,  # sq (total sequence length)
                total_nhq,  # nhq
                mla.nhk,  # nhk (always 1 for MLA)
                total_nhq,  # nhv (same as nhq for MLA)
                mla.d_q,  # d_q
                mla.d_k,  # d_k
                mla.d_v,  # d_v
                q_chunk,  # q_chunk_size
                k_chunk,  # k_chunk_size
                mla.is_causal,  # is_causal
                mla.is_balanced,  # is_balanced
                mla.q_dtype,  # q_dtype
                mla.kv_dtype,  # kv_dtype
                mla.pcc_threshold,  # pcc_threshold
                mla.rmse_threshold,  # rmse_threshold
            )
        )
        ids.append(f"mla-{seq_len_per_device}-q{q_chunk}-k{k_chunk}")

    return cases, ids


# Generate test cases for both WAN and MLA
TEST_CASES, TEST_IDS = generate_test_cases()


# === TEST 1: PERFORMANCE SWEEP (skipped on CI) ===
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.parametrize(
    "b, sq, nhq, nhk, nhv, d_q, d_k, d_v, q_chunk_size, k_chunk_size, is_causal, is_balanced, q_dtype, kv_dtype, pcc_threshold, rmse_threshold",
    TEST_CASES,
    ids=TEST_IDS,
)
def test_ring_joint_attention_sdpa_sweep_perf_impl(
    b,
    sq,
    nhq,
    nhk,
    nhv,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_causal,
    is_balanced,
    q_dtype,
    kv_dtype,
    pcc_threshold,
    rmse_threshold,
):
    """
    Performance sweep test for ring joint attention SDPA.
    Skipped on CI - run locally for performance measurement.
    Supports both WAN and MLA configurations.
    """
    run_ring_joint_sdpa(
        b,
        nhq,
        nhk,
        sq,
        d_q,
        q_chunk_size,
        k_chunk_size,
        q_dtype,
        nhv=nhv,
        d_k=d_k,
        d_v=d_v,
        kv_dtype=kv_dtype,
        is_causal=is_causal,
        is_balanced=is_balanced,
        do_check=False,
    )


# === TEST 2: ACCURACY VERIFICATION ===
@pytest.mark.parametrize(
    "b, sq, nhq, nhk, nhv, d_q, d_k, d_v, q_chunk_size, k_chunk_size, is_causal, is_balanced, q_dtype, kv_dtype, pcc_threshold, rmse_threshold",
    TEST_CASES,
    ids=TEST_IDS,
)
def test_ring_joint_attention_sdpa_accuracy(
    b,
    sq,
    nhq,
    nhk,
    nhv,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_causal,
    is_balanced,
    q_dtype,
    kv_dtype,
    pcc_threshold,
    rmse_threshold,
):
    """
    Accuracy verification test for ring joint attention SDPA.
    Supports both WAN and MLA configurations.

    ACCURACY METRICS:
    - PCC (Pearson Correlation Coefficient): Measures linear correlation
    - RMSE (Root Mean Square Error): Measures absolute error magnitude
    """
    run_ring_joint_sdpa(
        b,
        nhq,
        nhk,
        sq,
        d_q,
        q_chunk_size,
        k_chunk_size,
        q_dtype,
        nhv=nhv,
        d_k=d_k,
        d_v=d_v,
        kv_dtype=kv_dtype,
        is_causal=is_causal,
        is_balanced=is_balanced,
        pcc_threshold=pcc_threshold,
        rmse_threshold=rmse_threshold,
    )


# === TEST 3: DETERMINISM VERIFICATION ===
@pytest.mark.parametrize(
    "b, sq, nhq, nhk, nhv, d_q, d_k, d_v, q_chunk_size, k_chunk_size, is_causal, is_balanced, q_dtype, kv_dtype, pcc_threshold, rmse_threshold",
    TEST_CASES,
    ids=TEST_IDS,
)
def test_ring_joint_attention_sdpa_determinism(
    b,
    sq,
    nhq,
    nhk,
    nhv,
    d_q,
    d_k,
    d_v,
    q_chunk_size,
    k_chunk_size,
    is_causal,
    is_balanced,
    q_dtype,
    kv_dtype,
    pcc_threshold,
    rmse_threshold,
):
    """
    Test ring joint attention SDPA determinism: run 10 times with same inputs and verify outputs match exactly.
    Supports both WAN and MLA configurations.
    """
    num_iterations = 10
    run_ring_joint_sdpa(
        b,
        nhq,
        nhk,
        sq,
        d_q,
        q_chunk_size,
        k_chunk_size,
        q_dtype,
        nhv=nhv,
        d_k=d_k,
        d_v=d_v,
        kv_dtype=kv_dtype,
        is_causal=is_causal,
        is_balanced=is_balanced,
        num_iterations=num_iterations,
    )


# === TEST 4: PERFORMANCE TABLE GENERATOR (skipped on CI) ===
def _get_perf_table_params():
    """Generate performance table parameters for both WAN and MLA models."""
    hw = HardwareConfig.detect()
    if hw.num_devices < 2:
        return [], []

    params = []
    ids = []

    # WAN parameters
    wan = WANConfig.from_hardware(hw)
    for seq_len_per_device in wan.seq_lens:
        params.append(("wan", seq_len_per_device))
        ids.append(f"wan-{seq_len_per_device}")

    # MLA parameters
    mla = MLAConfig.from_hardware(hw)
    for seq_len_per_device in mla.seq_lens:
        params.append(("mla", seq_len_per_device))
        ids.append(f"mla-{seq_len_per_device}")

    return params, ids


_PERF_PARAMS, _PERF_IDS = _get_perf_table_params()


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")
@pytest.mark.timeout(1000)
@pytest.mark.parametrize(
    "model_name, seq_len_per_device",
    _PERF_PARAMS,
    ids=_PERF_IDS,
)
def test_ring_joint_attention_create_perf_table(model_name, seq_len_per_device):
    """
    Sweep chunk sizes for ring joint attention SDPA and print a performance table.
    Supports both WAN and MLA configurations.
    Skipped on CI - run locally with tracy profiler.
    """
    from tracy.process_model_log import run_device_profiler

    hw = HardwareConfig.detect()
    ring_size = hw.sp_size

    if ring_size < 2:
        pytest.skip(f"Ring joint attention requires at least 2 devices, got {ring_size}")

    # Select config based on model name
    if model_name == "wan":
        config = WANConfig.from_hardware(hw)
    else:
        config = MLAConfig.from_hardware(hw)

    # Derive values from config
    b = BATCH_SIZE
    s = seq_len_per_device * hw.sp_size  # Total sequence length
    nh = config.nhq * hw.tp_size  # Total heads
    d_q = config.d_q  # Query/Key head dimension
    d_v = config.d_v  # Value head dimension
    is_causal = config.is_causal

    # Use hardware config values (cannot query device due to TLB conflicts with subprocess tests)
    full_grid_rows = hw.grid_rows
    total_compute_cores = hw.sdpa_cores
    total_cores = hw.total_cores

    ccl_cores = full_grid_rows  # Full column height for CCL
    ccl_overhead_pct = (ccl_cores * 100.0) / total_cores

    subdir = "ttnn_ring_joint_sdpa_performance"
    perf_results = []

    # Get chunk size combinations to sweep
    if model_name == "wan":
        chunk_combinations = list(product(config.q_chunk_sizes, config.k_chunk_sizes))
    else:
        # MLA: single chunk size pair per sequence length from workloads
        chunk_combinations = [config.get_chunk_sizes(seq_len_per_device)]

    for q_chunk_size, k_chunk_size in chunk_combinations:
        float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]", "PM FPU UTIL (%)"]
        cols = ["ATTRIBUTES"]

        # Test ID format: {model}-{seq_len_per_device}-q{q_chunk}-k{k_chunk}
        test_id = f"{model_name}-{seq_len_per_device}-q{q_chunk_size}-k{k_chunk_size}"
        command = (
            f"pytest tests/nightly/blackhole/ccl/"
            f"test_ring_joint_sdpa.py::"
            f"test_ring_joint_attention_sdpa_sweep_perf_impl"
            f"[{test_id}]"
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
            heads_per_device = nh / hw.tp_size
            utilization = compute_ring_joint_utilization(
                local_seq_len, s, d_q, d_v, heads_per_device, duration_ns, effective_cores, is_causal
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
                    "fpu_util_min": fpu_util_min,
                    "fpu_util_max": fpu_util_max,
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

    mm_flops = compute_sdpa_flops(s, s, d_q, d_v, nh, is_causal)

    # Print summary table
    print(f"\n{'='*190}")
    print(
        f"Ring Joint Attention Performance Sweep ({model_name.upper()}): b={b}, nh={nh}, s={s}, d_q={d_q}, d_v={d_v}, causal={is_causal}"
    )
    print(f"Architecture: {hw.arch_type}, Ring size: {ring_size} devices")
    print(f"Total MM FLOPs (all devices): {mm_flops:,} ({mm_flops/1e9:.2f} GFLOPs)")
    print(f"Per-device workload: Q={s // ring_size} tokens, K/V={s} tokens (via ring), {nh} heads")
    print(f"Core Allocation: {total_compute_cores} compute + {ccl_cores} CCL = {total_cores} total cores")
    print(f"{'='*190}")
    header = "| Rank | q_chunk | k_chunk | Duration (ms) | Compute Used | Compute Idle | Compute Util | CCL Cores | Ring Eff | Iters/Core | Pad Waste | Slot Waste | FPU Util (%)  | Math Util |"
    sep = "|------|---------|---------|---------------|--------------|--------------|--------------|-----------|----------|------------|-----------|------------|---------------|-----------|"
    print(header)
    print(sep)

    for rank, result in enumerate(valid_results, 1):
        fpu_range = f"{result['fpu_util_min']:.1f}-{result['fpu_util_max']:.1f}"
        print(
            f"| {rank:4d} | {result['q_chunk_size']:7d} | {result['k_chunk_size']:7d} | {result['duration_ms']:13.3f} | "
            f"{result['cores_used']:12d} | {result['cores_idle']:12d} | {result['compute_util_pct']:11.0f}% | "
            f"{result['ccl_cores']:9d} | {result['ring_efficiency']:7.0f}% | {result['iters_per_core']:10d} | "
            f"{result['total_waste_pct']:8.1f}% | {result['slot_waste_pct']:9.1f}% | {fpu_range:>13} | {result['utilization']:8.1f}% |"
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
