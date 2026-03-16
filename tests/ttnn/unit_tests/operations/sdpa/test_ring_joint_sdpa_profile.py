# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test file for single-device profiling of ring_joint_sdpa (causal+balanced mode).

Phase 1: Validates helper functions and PyTorch reference.
Phase 2: Tests the actual profiling op on device.

The profiling op measures pure compute time by simulating what one device
in a causal+balanced ring would compute, using pre-staged KV data instead of
actual ring communication.
"""

import torch
import pytest
from typing import List, Tuple
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


# ============================================================================
# Helper Functions: Chunk Ordering
# ============================================================================


def create_balanced_chunk_order(ring_size: int) -> List[int]:
    """
    Create balanced chunk order for sequence reordering.

    For causal attention, early chunks have less work (attend to fewer tokens)
    while late chunks have more work. The zigzag pattern pairs early chunks
    with late chunks to balance workload across devices:

    ring_size=2: [0, 3, 1, 2] -> Device 0 gets [0,3], Device 1 gets [1,2]
    ring_size=4: [0, 7, 1, 6, 2, 5, 3, 4]

    Args:
        ring_size: Number of devices in the ring

    Returns:
        List of chunk indices in balanced order
    """
    num_chunks = 2 * ring_size
    balanced_order = []
    left, right = 0, num_chunks - 1
    for i in range(num_chunks):
        if i % 2 == 0:
            balanced_order.append(left)
            left += 1
        else:
            balanced_order.append(right)
            right -= 1
    return balanced_order


def get_device_chunk_indices(ring_index: int, ring_size: int) -> List[int]:
    """
    Get original chunk indices assigned to a device.

    Each device gets 2 chunks from the balanced order:
    - Device 0 in ring_size=2: [0, 3]
    - Device 1 in ring_size=2: [1, 2]

    Args:
        ring_index: This device's position in the ring (0 to ring_size-1)
        ring_size: Number of devices in the ring

    Returns:
        List of chunk indices for this device
    """
    chunk_order = create_balanced_chunk_order(ring_size)
    chunks_per_device = 2
    start_pos = ring_index * chunks_per_device
    return chunk_order[start_pos : start_pos + chunks_per_device]


# ============================================================================
# Helper Functions: Ring Arrival Order
# ============================================================================


def get_ring_id_arrival_order(ring_index: int, ring_size: int) -> List[int]:
    """
    Compute ring_id arrival order for Linear topology.

    Simulates the fused_op_receiver bidirectional alternating logic from
    fused_op_receiver.hpp. The arrival order determines which device's KV
    is processed at each ring iteration.

    For Linear topology:
    - expected_inputs = (from_forward, from_backward)
    - from_forward = ring_size - 1 - ring_index
    - from_backward = ring_index
    - Alternates between directions when both have pending inputs

    ring_size=2:
      Device 0: [0, 1]
      Device 1: [1, 0]

    ring_size=4:
      Device 0: [0, 1, 2, 3]
      Device 1: [1, 2, 0, 3]
      Device 2: [2, 3, 1, 0]
      Device 3: [3, 2, 1, 0]

    Args:
        ring_index: This device's position in the ring
        ring_size: Number of devices in the ring

    Returns:
        List of ring_ids in arrival order
    """
    from_forward = ring_size - 1 - ring_index
    from_backward = ring_index
    expected_inputs = (from_forward, from_backward)

    received_inputs = [0, 0]
    curr_dir = 0
    arrival_order = []

    for transfer_idx in range(ring_size):
        if transfer_idx == 0:
            # First iteration: process local KV
            sender_ring_id = ring_index
        else:
            # Subsequent iterations: receive from alternating directions
            received_inputs[curr_dir] += 1
            if curr_dir == 1:  # Backward direction
                sender_ring_id = (ring_index - received_inputs[curr_dir] + ring_size) % ring_size
            else:  # Forward direction
                sender_ring_id = (ring_index + received_inputs[curr_dir]) % ring_size

        arrival_order.append(sender_ring_id)

        # Determine next direction
        if transfer_idx == 0:
            if expected_inputs[curr_dir] == 0:
                curr_dir = 1 - curr_dir
        else:
            next_dir = 1 - curr_dir
            if received_inputs[next_dir] < expected_inputs[next_dir]:
                curr_dir = next_dir

    return arrival_order


def get_kv_arrival_chunk_indices(ring_index: int, ring_size: int) -> List[Tuple[int, List[int]]]:
    """
    Get (ring_id, chunk_indices) pairs in arrival order.

    Combines ring arrival order with chunk assignment to determine which
    KV chunks arrive at each ring iteration.

    For device 0, ring_size=2:
      [(0, [0,3]), (1, [1,2])]
      First iteration: local KV (chunks 0,3)
      Second iteration: KV from device 1 (chunks 1,2)

    Args:
        ring_index: This device's position in the ring
        ring_size: Number of devices in the ring

    Returns:
        List of (ring_id, chunk_indices) tuples in arrival order
    """
    ring_id_order = get_ring_id_arrival_order(ring_index, ring_size)
    return [(rid, get_device_chunk_indices(rid, ring_size)) for rid in ring_id_order]


# ============================================================================
# Data Preparation Functions
# ============================================================================


def extract_chunks(tensor: torch.Tensor, chunk_indices: List[int], chunk_size: int, seq_dim: int = 2) -> torch.Tensor:
    """
    Extract and concatenate specified chunks from tensor.

    Args:
        tensor: Input tensor with shape [b, nh, seq_len, d]
        chunk_indices: List of chunk indices to extract
        chunk_size: Size of each chunk along seq_dim
        seq_dim: Dimension along which to extract chunks (default: 2)

    Returns:
        Concatenated chunks along seq_dim
    """
    chunks = []
    for idx in chunk_indices:
        start = idx * chunk_size
        end = start + chunk_size
        if seq_dim == 2:
            chunks.append(tensor[:, :, start:end, :])
        else:
            raise ValueError(f"Unsupported seq_dim: {seq_dim}")
    return torch.cat(chunks, dim=seq_dim)


def build_gathered_kv_buffer(kv_full: torch.Tensor, ring_index: int, ring_size: int, chunk_size: int) -> torch.Tensor:
    """
    Build gathered KV buffer in ring arrival order.

    Layout: [local_kv | ring_iter_1_kv | ring_iter_2_kv | ...]

    This simulates what the profiling op would read from DRAM instead of
    receiving via ring all-gather. Local KV is included at the start for
    simpler kernel indexing.

    Args:
        kv_full: Full K or V tensor with shape [b, nh, seq_len, d]
        ring_index: This device's position in the ring
        ring_size: Number of devices in the ring
        chunk_size: Size of each chunk

    Returns:
        KV buffer with all chunks in arrival order
    """
    kv_arrival = get_kv_arrival_chunk_indices(ring_index, ring_size)
    chunks = []
    for ring_id, chunk_indices in kv_arrival:
        chunks.append(extract_chunks(kv_full, chunk_indices, chunk_size))
    return torch.cat(chunks, dim=2)


# ============================================================================
# PyTorch Reference Implementation
# ============================================================================


def compute_causal_balanced_reference(
    Q_full: torch.Tensor,
    K_full: torch.Tensor,
    V_full: torch.Tensor,
    ring_index: int,
    ring_size: int,
) -> torch.Tensor:
    """
    Compute expected output for a device in causal+balanced ring attention.

    Approach:
    1. Compute full causal attention in original order (as if single device)
    2. Extract output for this device's Q chunk positions
    3. Return in device's local order (matching profiling op output layout)

    The key insight is that causal attention is well-defined: output[i] only
    depends on Q[i] and K[:i+1], V[:i+1]. So we can compute full attention
    and extract the relevant slices.

    Args:
        Q_full: Full Q tensor [b, nh, seq_len, d]
        K_full: Full K tensor [b, nh, seq_len, d]
        V_full: Full V tensor [b, nh, seq_len, d]
        ring_index: This device's position in the ring
        ring_size: Number of devices in the ring

    Returns:
        Expected output [b, nh, local_seq_len, d] for this device
    """
    # Compute full causal attention
    full_output = torch.nn.functional.scaled_dot_product_attention(Q_full, K_full, V_full, is_causal=True)

    # Get this device's chunk indices
    seq_len = Q_full.shape[2]
    chunk_size = seq_len // (2 * ring_size)
    device_chunks = get_device_chunk_indices(ring_index, ring_size)

    # Extract output for this device's Q positions (in device's local order)
    return extract_chunks(full_output, device_chunks, chunk_size)


# ============================================================================
# Utility Functions for Testing
# ============================================================================


def fa_rand(*shape) -> torch.Tensor:
    """
    Generate random tensor for flash attention testing.

    Uses a mixture of normal distributions to create varied values
    that test edge cases better than pure normal distribution.
    """
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


# ============================================================================
# Unit Tests: Helper Functions
# ============================================================================


class TestBalancedChunkOrder:
    """Tests for create_balanced_chunk_order."""

    def test_ring_size_2(self):
        """ring_size=2 should produce [0, 3, 1, 2]."""
        result = create_balanced_chunk_order(ring_size=2)
        assert result == [0, 3, 1, 2]

    def test_ring_size_4(self):
        """ring_size=4 should produce [0, 7, 1, 6, 2, 5, 3, 4]."""
        result = create_balanced_chunk_order(ring_size=4)
        assert result == [0, 7, 1, 6, 2, 5, 3, 4]

    def test_ring_size_8(self):
        """ring_size=8 should produce 16 chunks in zigzag order."""
        result = create_balanced_chunk_order(ring_size=8)
        expected = [0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8]
        assert result == expected

    def test_length_is_2x_ring_size(self):
        """Output length should always be 2 * ring_size."""
        for ring_size in [1, 2, 4, 8, 16]:
            result = create_balanced_chunk_order(ring_size)
            assert len(result) == 2 * ring_size


class TestGetDeviceChunkIndices:
    """Tests for get_device_chunk_indices."""

    def test_ring_size_2_device_0(self):
        """Device 0 in ring_size=2 should get chunks [0, 3]."""
        result = get_device_chunk_indices(ring_index=0, ring_size=2)
        assert result == [0, 3]

    def test_ring_size_2_device_1(self):
        """Device 1 in ring_size=2 should get chunks [1, 2]."""
        result = get_device_chunk_indices(ring_index=1, ring_size=2)
        assert result == [1, 2]

    def test_ring_size_4_all_devices(self):
        """All devices in ring_size=4 should get their expected chunks."""
        expected = {
            0: [0, 7],
            1: [1, 6],
            2: [2, 5],
            3: [3, 4],
        }
        for ring_index, expected_chunks in expected.items():
            result = get_device_chunk_indices(ring_index, ring_size=4)
            assert result == expected_chunks, f"Device {ring_index} failed"

    def test_all_chunks_covered(self):
        """All chunk indices should be assigned exactly once across devices."""
        for ring_size in [2, 4, 8]:
            all_chunks = []
            for ring_index in range(ring_size):
                all_chunks.extend(get_device_chunk_indices(ring_index, ring_size))
            all_chunks.sort()
            assert all_chunks == list(range(2 * ring_size))


class TestGetRingIdArrivalOrder:
    """Tests for get_ring_id_arrival_order."""

    def test_ring_size_2_device_0(self):
        """Device 0 in ring_size=2: arrival order [0, 1]."""
        result = get_ring_id_arrival_order(ring_index=0, ring_size=2)
        assert result == [0, 1]

    def test_ring_size_2_device_1(self):
        """Device 1 in ring_size=2: arrival order [1, 0]."""
        result = get_ring_id_arrival_order(ring_index=1, ring_size=2)
        assert result == [1, 0]

    def test_ring_size_4_all_devices(self):
        """All devices in ring_size=4 should have correct arrival order."""
        expected = {
            0: [0, 1, 2, 3],
            1: [1, 2, 0, 3],
            2: [2, 3, 1, 0],
            3: [3, 2, 1, 0],
        }
        for ring_index, expected_order in expected.items():
            result = get_ring_id_arrival_order(ring_index, ring_size=4)
            assert result == expected_order, f"Device {ring_index} failed"

    def test_first_arrival_is_self(self):
        """First arrival should always be the device itself (local KV)."""
        for ring_size in [2, 4, 8]:
            for ring_index in range(ring_size):
                result = get_ring_id_arrival_order(ring_index, ring_size)
                assert result[0] == ring_index

    def test_all_devices_visited(self):
        """Each device should receive KV from all ring_ids exactly once."""
        for ring_size in [2, 4, 8]:
            for ring_index in range(ring_size):
                result = get_ring_id_arrival_order(ring_index, ring_size)
                assert sorted(result) == list(range(ring_size))


class TestGetKvArrivalChunkIndices:
    """Tests for get_kv_arrival_chunk_indices."""

    def test_ring_size_2_device_0(self):
        """Device 0 in ring_size=2: [(0, [0,3]), (1, [1,2])]."""
        result = get_kv_arrival_chunk_indices(ring_index=0, ring_size=2)
        expected = [(0, [0, 3]), (1, [1, 2])]
        assert result == expected

    def test_ring_size_2_device_1(self):
        """Device 1 in ring_size=2: [(1, [1,2]), (0, [0,3])]."""
        result = get_kv_arrival_chunk_indices(ring_index=1, ring_size=2)
        expected = [(1, [1, 2]), (0, [0, 3])]
        assert result == expected


class TestBuildGatheredKvBuffer:
    """Tests for build_gathered_kv_buffer."""

    def test_buffer_shape(self):
        """Gathered buffer should have full sequence length."""
        b, nh, seq_len, d = 1, 2, 256, 64
        ring_size = 2
        chunk_size = seq_len // (2 * ring_size)

        kv_full = torch.randn(b, nh, seq_len, d)
        result = build_gathered_kv_buffer(kv_full, ring_index=0, ring_size=ring_size, chunk_size=chunk_size)

        assert result.shape == (b, nh, seq_len, d)

    def test_buffer_contains_all_data(self):
        """Gathered buffer should contain all original data (reordered)."""
        b, nh, seq_len, d = 1, 1, 128, 32
        ring_size = 2
        chunk_size = seq_len // (2 * ring_size)  # 32

        # Create tensor with unique values per position
        kv_full = torch.arange(seq_len).float().view(1, 1, seq_len, 1).expand(b, nh, seq_len, d)

        result = build_gathered_kv_buffer(kv_full, ring_index=0, ring_size=ring_size, chunk_size=chunk_size)

        # For device 0, ring_size=2:
        # Arrival order: [0, 1]
        # Device 0 chunks: [0, 3], Device 1 chunks: [1, 2]
        # Expected order: chunks [0, 3, 1, 2] -> positions [0:32, 96:128, 32:64, 64:96]
        expected_positions = list(range(0, 32)) + list(range(96, 128)) + list(range(32, 64)) + list(range(64, 96))
        expected = torch.tensor(expected_positions).float().view(1, 1, seq_len, 1).expand(b, nh, seq_len, d)

        torch.testing.assert_close(result, expected)


# ============================================================================
# Tests: PyTorch Reference Implementation
# ============================================================================


class TestComputeCausalBalancedReference:
    """Tests for compute_causal_balanced_reference."""

    def test_output_shape(self):
        """Output shape should be [b, nh, local_seq_len, d]."""
        b, nh, seq_len, d = 1, 2, 256, 64
        ring_size = 2
        local_seq_len = seq_len // ring_size

        Q_full = fa_rand(b, nh, seq_len, d)
        K_full = fa_rand(b, nh, seq_len, d)
        V_full = fa_rand(b, nh, seq_len, d)

        result = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index=0, ring_size=ring_size)

        assert result.shape == (b, nh, local_seq_len, d)

    def test_extracts_correct_positions(self):
        """Reference should extract output at device's Q positions."""
        b, nh, seq_len, d = 1, 2, 256, 64
        ring_size = 2
        chunk_size = seq_len // (2 * ring_size)  # 64

        Q_full = fa_rand(b, nh, seq_len, d)
        K_full = fa_rand(b, nh, seq_len, d)
        V_full = fa_rand(b, nh, seq_len, d)

        # Compute full attention
        full_output = torch.nn.functional.scaled_dot_product_attention(Q_full, K_full, V_full, is_causal=True)

        # Device 0 gets chunks [0, 3] (positions [0:64, 192:256])
        expected = torch.cat([full_output[:, :, 0:64, :], full_output[:, :, 192:256, :]], dim=2)

        result = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index=0, ring_size=ring_size)

        torch.testing.assert_close(result, expected)

    def test_all_devices_cover_full_output(self):
        """Output from all devices should reconstruct full attention (after reordering)."""
        b, nh, seq_len, d = 1, 2, 256, 64
        ring_size = 2
        chunk_size = seq_len // (2 * ring_size)

        Q_full = fa_rand(b, nh, seq_len, d)
        K_full = fa_rand(b, nh, seq_len, d)
        V_full = fa_rand(b, nh, seq_len, d)

        # Compute full attention
        full_output = torch.nn.functional.scaled_dot_product_attention(Q_full, K_full, V_full, is_causal=True)

        # Get outputs from all devices
        outputs = {}
        for ring_index in range(ring_size):
            outputs[ring_index] = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index, ring_size)
            device_chunks = get_device_chunk_indices(ring_index, ring_size)

        # Reconstruct full output by placing each device's output at correct positions
        reconstructed = torch.zeros_like(full_output)
        for ring_index in range(ring_size):
            device_chunks = get_device_chunk_indices(ring_index, ring_size)
            device_output = outputs[ring_index]
            for local_idx, chunk_idx in enumerate(device_chunks):
                start = chunk_idx * chunk_size
                end = start + chunk_size
                local_start = local_idx * chunk_size
                local_end = local_start + chunk_size
                reconstructed[:, :, start:end, :] = device_output[:, :, local_start:local_end, :]

        torch.testing.assert_close(reconstructed, full_output)


# ============================================================================
# Integration Test: Full Pipeline (Placeholder for Op)
# ============================================================================


class TestRingJointSdpaProfile:
    """
    Integration tests for ring_joint_sdpa_profile op.

    These tests validate the full pipeline from data preparation to output
    comparison. Currently they only test the PyTorch reference; the actual
    op call will be enabled in Phase 2.
    """

    @pytest.mark.parametrize("ring_index", [0, 1])
    def test_ring_joint_sdpa_profile_ring2(self, ring_index: int):
        """
        Test profiling logic for ring_size=2.

        This test validates:
        1. Local Q/K/V extraction
        2. Gathered KV buffer construction
        3. PyTorch reference computation

        TODO (Phase 2): Enable actual op call when implemented.
        """
        # Config
        ring_size = 2
        b, nh, seq_len, d = 1, 2, 256, 64
        chunk_size = seq_len // (2 * ring_size)  # 64

        # Create full tensors
        Q_full = fa_rand(b, nh, seq_len, d)
        K_full = fa_rand(b, nh, seq_len, d)
        V_full = fa_rand(b, nh, seq_len, d)

        # Prepare local Q (this device's chunks)
        device_chunks = get_device_chunk_indices(ring_index, ring_size)
        Q_local = extract_chunks(Q_full, device_chunks, chunk_size)
        K_local = extract_chunks(K_full, device_chunks, chunk_size)
        V_local = extract_chunks(V_full, device_chunks, chunk_size)

        # Verify local tensor shapes
        local_seq_len = seq_len // ring_size
        assert Q_local.shape == (b, nh, local_seq_len, d)
        assert K_local.shape == (b, nh, local_seq_len, d)
        assert V_local.shape == (b, nh, local_seq_len, d)

        # Prepare gathered KV (all KV in arrival order)
        K_gathered = build_gathered_kv_buffer(K_full, ring_index, ring_size, chunk_size)
        V_gathered = build_gathered_kv_buffer(V_full, ring_index, ring_size, chunk_size)

        # Verify gathered buffer shapes
        assert K_gathered.shape == (b, nh, seq_len, d)
        assert V_gathered.shape == (b, nh, seq_len, d)

        # Compute expected output via reference
        expected = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index, ring_size)
        assert expected.shape == (b, nh, local_seq_len, d)

        # TODO (Phase 2): Call actual profiling op
        # tt_output, _, tt_lse = ttnn.transformer.ring_joint_sdpa_profile(
        #     tt_Q_local, tt_K_local, tt_V_local,
        #     gathered_k=tt_K_gathered, gathered_v=tt_V_gathered,
        #     ring_size=ring_size, ring_index=ring_index,
        #     is_causal=True, is_balanced=True, logical_n=seq_len,
        #     program_config=program_config,
        # )
        # torch.testing.assert_close(ttnn.to_torch(tt_output), expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("ring_index", [0, 1, 2, 3])
    def test_ring_joint_sdpa_profile_ring4(self, ring_index: int):
        """
        Test profiling logic for ring_size=4.

        Validates the more complex arrival order and chunk assignment
        with 4 devices.
        """
        # Config
        ring_size = 4
        b, nh, seq_len, d = 1, 2, 512, 64
        chunk_size = seq_len // (2 * ring_size)  # 64

        # Create full tensors
        Q_full = fa_rand(b, nh, seq_len, d)
        K_full = fa_rand(b, nh, seq_len, d)
        V_full = fa_rand(b, nh, seq_len, d)

        # Prepare local Q (this device's chunks)
        device_chunks = get_device_chunk_indices(ring_index, ring_size)
        Q_local = extract_chunks(Q_full, device_chunks, chunk_size)

        # Prepare gathered KV
        K_gathered = build_gathered_kv_buffer(K_full, ring_index, ring_size, chunk_size)
        V_gathered = build_gathered_kv_buffer(V_full, ring_index, ring_size, chunk_size)

        # Compute expected output
        expected = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index, ring_size)

        # Verify shapes
        local_seq_len = seq_len // ring_size
        assert Q_local.shape == (b, nh, local_seq_len, d)
        assert expected.shape == (b, nh, local_seq_len, d)
        assert K_gathered.shape == (b, nh, seq_len, d)


# ============================================================================
# Device Tests: Actual Op Execution (Phase 2)
# ============================================================================


@pytest.mark.parametrize(
    "ring_size, ring_index",
    [
        (2, 0),
        (2, 1),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
    ],
)
def test_ring_joint_sdpa_profile_device(device, ring_size: int, ring_index: int):
    """
    Test ring_joint_sdpa_profile op on actual device.

    This test validates:
    1. Op executes without errors
    2. Output matches PyTorch reference (PCC > 0.99)
    """
    # Config
    b, nh, d = 1, 8, 64
    seq_len = 128 * ring_size  # Total sequence length
    local_seq_len = seq_len // ring_size
    chunk_size = seq_len // (2 * ring_size)

    # Program config
    q_chunk_size = 64
    k_chunk_size = 64
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create full tensors
    Q_full = fa_rand(b, nh, seq_len, d)
    K_full = fa_rand(b, nh, seq_len, d)
    V_full = fa_rand(b, nh, seq_len, d)

    # Prepare local Q/K/V (this device's chunks)
    device_chunks = get_device_chunk_indices(ring_index, ring_size)
    Q_local = extract_chunks(Q_full, device_chunks, chunk_size)
    K_local = extract_chunks(K_full, device_chunks, chunk_size)
    V_local = extract_chunks(V_full, device_chunks, chunk_size)

    # Prepare gathered KV (all KV in arrival order)
    K_gathered = build_gathered_kv_buffer(K_full, ring_index, ring_size, chunk_size)
    V_gathered = build_gathered_kv_buffer(V_full, ring_index, ring_size, chunk_size)

    # Compute expected output via PyTorch reference
    expected = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index, ring_size)

    # Move tensors to device
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    dtype = ttnn.bfloat16

    tt_Q = ttnn.from_torch(Q_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_K = ttnn.from_torch(K_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_V = ttnn.from_torch(V_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_K_gathered = ttnn.from_torch(
        K_gathered, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )
    tt_V_gathered = ttnn.from_torch(
        V_gathered, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )

    # Call the profiling op
    tt_output, _, tt_lse = ttnn.transformer.ring_joint_sdpa_profile(
        tt_Q,
        tt_K,
        tt_V,
        tt_K_gathered,
        tt_V_gathered,
        ring_size=ring_size,
        ring_index=ring_index,
        logical_n=seq_len,
        program_config=program_config,
        is_causal=True,
        is_balanced=True,
        compute_kernel_config=compute_kernel_config,
    )

    # Convert back to torch and compare
    output = ttnn.to_torch(tt_output)
    output = output[:, :, :local_seq_len, :]  # Remove any tile padding

    # Compare with expected (PCC > 0.99)
    passing, pcc_val = comp_pcc(expected, output, 0.99)
    assert passing, f"PCC {pcc_val} is below threshold 0.99 for ring_size={ring_size}, ring_index={ring_index}"


# ============================================================================
# Extended Test Coverage (Phase 3)
# ============================================================================


@pytest.mark.parametrize("ring_size", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("ring_index_ratio", [0.0, 0.5, 1.0])
def test_ring_joint_sdpa_profile_ring_sizes(device, ring_size: int, ring_index_ratio: float):
    """
    Test ring_joint_sdpa_profile across various ring sizes.

    Tests first, middle, and last device positions for each ring size.
    This validates the ProfileRingIndexer logic handles all ring topologies.
    """
    ring_index = int(ring_index_ratio * (ring_size - 1))

    # Config - use fixed parameters, scale seq_len with ring_size
    b, nh, d = 1, 8, 64
    q_chunk_size = 64
    k_chunk_size = 64
    # seq_len must be divisible by 2*ring_size*chunk_size
    seq_len = 2 * ring_size * q_chunk_size  # Minimum valid seq_len
    local_seq_len = seq_len // ring_size
    chunk_size = seq_len // (2 * ring_size)

    # Program config
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create full tensors
    Q_full = fa_rand(b, nh, seq_len, d)
    K_full = fa_rand(b, nh, seq_len, d)
    V_full = fa_rand(b, nh, seq_len, d)

    # Prepare local Q/K/V (this device's chunks)
    device_chunks = get_device_chunk_indices(ring_index, ring_size)
    Q_local = extract_chunks(Q_full, device_chunks, chunk_size)
    K_local = extract_chunks(K_full, device_chunks, chunk_size)
    V_local = extract_chunks(V_full, device_chunks, chunk_size)

    # Prepare gathered KV (all KV in arrival order)
    K_gathered = build_gathered_kv_buffer(K_full, ring_index, ring_size, chunk_size)
    V_gathered = build_gathered_kv_buffer(V_full, ring_index, ring_size, chunk_size)

    # Compute expected output via PyTorch reference
    expected = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index, ring_size)

    # Move tensors to device
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    dtype = ttnn.bfloat16

    tt_Q = ttnn.from_torch(Q_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_K = ttnn.from_torch(K_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_V = ttnn.from_torch(V_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_K_gathered = ttnn.from_torch(
        K_gathered, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )
    tt_V_gathered = ttnn.from_torch(
        V_gathered, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )

    # Call the profiling op
    tt_output, _, tt_lse = ttnn.transformer.ring_joint_sdpa_profile(
        tt_Q,
        tt_K,
        tt_V,
        tt_K_gathered,
        tt_V_gathered,
        ring_size=ring_size,
        ring_index=ring_index,
        logical_n=seq_len,
        program_config=program_config,
        is_causal=True,
        is_balanced=True,
        compute_kernel_config=compute_kernel_config,
    )

    # Convert back to torch and compare
    output = ttnn.to_torch(tt_output)
    output = output[:, :, :local_seq_len, :]  # Remove any tile padding

    # Compare with expected (PCC > 0.99)
    passing, pcc_val = comp_pcc(expected, output, 0.99)
    assert passing, f"PCC {pcc_val} < 0.99 for ring_size={ring_size}, ring_index={ring_index}"


@pytest.mark.parametrize(
    "q_chunk_size, k_chunk_size",
    [
        (64, 64),  # Default
        (128, 128),  # Larger chunks
        (64, 128),  # Asymmetric: smaller Q, larger K
        (128, 256),  # Large K chunks
    ],
)
def test_ring_joint_sdpa_profile_chunk_sizes(device, q_chunk_size: int, k_chunk_size: int):
    """
    Test ring_joint_sdpa_profile with various chunk size configurations.

    Validates the op handles different q_chunk_size and k_chunk_size combinations.
    """
    ring_size = 4
    ring_index = 0

    # Config - seq_len must be divisible by 2*ring_size*max(q_chunk, k_chunk)
    b, nh, d = 1, 8, 64
    min_chunk_multiple = max(q_chunk_size, k_chunk_size)
    seq_len = 2 * ring_size * min_chunk_multiple
    local_seq_len = seq_len // ring_size
    chunk_size = seq_len // (2 * ring_size)

    # Program config
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create full tensors
    Q_full = fa_rand(b, nh, seq_len, d)
    K_full = fa_rand(b, nh, seq_len, d)
    V_full = fa_rand(b, nh, seq_len, d)

    # Prepare local Q/K/V (this device's chunks)
    device_chunks = get_device_chunk_indices(ring_index, ring_size)
    Q_local = extract_chunks(Q_full, device_chunks, chunk_size)
    K_local = extract_chunks(K_full, device_chunks, chunk_size)
    V_local = extract_chunks(V_full, device_chunks, chunk_size)

    # Prepare gathered KV (all KV in arrival order)
    K_gathered = build_gathered_kv_buffer(K_full, ring_index, ring_size, chunk_size)
    V_gathered = build_gathered_kv_buffer(V_full, ring_index, ring_size, chunk_size)

    # Compute expected output via PyTorch reference
    expected = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index, ring_size)

    # Move tensors to device
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    dtype = ttnn.bfloat16

    tt_Q = ttnn.from_torch(Q_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_K = ttnn.from_torch(K_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_V = ttnn.from_torch(V_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_K_gathered = ttnn.from_torch(
        K_gathered, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )
    tt_V_gathered = ttnn.from_torch(
        V_gathered, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )

    # Call the profiling op
    tt_output, _, tt_lse = ttnn.transformer.ring_joint_sdpa_profile(
        tt_Q,
        tt_K,
        tt_V,
        tt_K_gathered,
        tt_V_gathered,
        ring_size=ring_size,
        ring_index=ring_index,
        logical_n=seq_len,
        program_config=program_config,
        is_causal=True,
        is_balanced=True,
        compute_kernel_config=compute_kernel_config,
    )

    # Convert back to torch and compare
    output = ttnn.to_torch(tt_output)
    output = output[:, :, :local_seq_len, :]  # Remove any tile padding

    # Compare with expected (PCC > 0.99)
    passing, pcc_val = comp_pcc(expected, output, 0.99)
    assert passing, f"PCC {pcc_val} < 0.99 for q_chunk={q_chunk_size}, k_chunk={k_chunk_size}"


@pytest.mark.parametrize("seq_len_per_device", [128, 256, 512])
def test_ring_joint_sdpa_profile_seq_lengths(device, seq_len_per_device: int):
    """
    Test ring_joint_sdpa_profile with various sequence lengths.

    Validates the op handles different total sequence lengths
    (128-512 tokens per device, 512-2048 total).
    """
    ring_size = 4
    ring_index = 0

    # Config
    b, nh, d = 1, 8, 64
    q_chunk_size = 64
    k_chunk_size = 64
    seq_len = seq_len_per_device * ring_size  # Total sequence length
    local_seq_len = seq_len // ring_size
    chunk_size = seq_len // (2 * ring_size)

    # Program config
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create full tensors
    Q_full = fa_rand(b, nh, seq_len, d)
    K_full = fa_rand(b, nh, seq_len, d)
    V_full = fa_rand(b, nh, seq_len, d)

    # Prepare local Q/K/V (this device's chunks)
    device_chunks = get_device_chunk_indices(ring_index, ring_size)
    Q_local = extract_chunks(Q_full, device_chunks, chunk_size)
    K_local = extract_chunks(K_full, device_chunks, chunk_size)
    V_local = extract_chunks(V_full, device_chunks, chunk_size)

    # Prepare gathered KV (all KV in arrival order)
    K_gathered = build_gathered_kv_buffer(K_full, ring_index, ring_size, chunk_size)
    V_gathered = build_gathered_kv_buffer(V_full, ring_index, ring_size, chunk_size)

    # Compute expected output via PyTorch reference
    expected = compute_causal_balanced_reference(Q_full, K_full, V_full, ring_index, ring_size)

    # Move tensors to device
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    dtype = ttnn.bfloat16

    tt_Q = ttnn.from_torch(Q_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_K = ttnn.from_torch(K_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_V = ttnn.from_torch(V_local, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device)
    tt_K_gathered = ttnn.from_torch(
        K_gathered, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )
    tt_V_gathered = ttnn.from_torch(
        V_gathered, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )

    # Call the profiling op
    tt_output, _, tt_lse = ttnn.transformer.ring_joint_sdpa_profile(
        tt_Q,
        tt_K,
        tt_V,
        tt_K_gathered,
        tt_V_gathered,
        ring_size=ring_size,
        ring_index=ring_index,
        logical_n=seq_len,
        program_config=program_config,
        is_causal=True,
        is_balanced=True,
        compute_kernel_config=compute_kernel_config,
    )

    # Convert back to torch and compare
    output = ttnn.to_torch(tt_output)
    output = output[:, :, :local_seq_len, :]  # Remove any tile padding

    # Compare with expected (PCC > 0.99)
    passing, pcc_val = comp_pcc(expected, output, 0.99)
    assert passing, f"PCC {pcc_val} < 0.99 for seq_len_per_device={seq_len_per_device}"


# ============================================================================
# Production-Scale Profile Test (Phase 5)
# ============================================================================


@pytest.mark.parametrize("k_chunk_size", [64, 128, 256, 512])
@pytest.mark.parametrize("q_chunk_size", [64, 128, 256, 512])
@pytest.mark.parametrize(
    "total_seq",
    [
        pytest.param(102400, id="100k"),
        pytest.param(131072, id="128k"),
    ],
)
@pytest.mark.parametrize("ring_index", list(range(32)))
def test_ring_joint_sdpa_profile_production_scale(
    device, ring_index: int, total_seq: int, q_chunk_size: int, k_chunk_size: int
):
    """
    Profile ring_joint_sdpa with production-scale dimensions.

    No correctness check - profiling only. This test validates the op
    can execute at production scale and generates Tracy profile data.

    Configuration:
    - GQA: 32 Q heads, 1 KV head (shared across Q heads)
    - Q: [1, 32, local_seq, 576] BFLOAT16
    - K: [1, 1, local_seq, 576] BFLOAT8_B
    - V: [1, 32, local_seq, 128] BFLOAT8_B
    - ring_size=32
    """
    # Production config
    b = 1
    nh_q, nh_k, nh_v = 32, 1, 32  # GQA: 32 Q heads, 1 KV head
    d_qk, d_v = 576, 128  # Q/K dim=576, V dim=128
    ring_size = 32
    local_seq = total_seq // ring_size

    # Program config - use device's actual grid size
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create random tensors (no need for real data since we skip correctness)
    Q_local = torch.randn(b, nh_q, local_seq, d_qk)
    K_local = torch.randn(b, nh_k, local_seq, d_qk)
    V_local = torch.randn(b, nh_v, local_seq, d_v)
    K_gathered = torch.randn(b, nh_k, total_seq, d_qk)
    V_gathered = torch.randn(b, nh_v, total_seq, d_v)

    # Move to device with appropriate dtypes
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    tt_Q = ttnn.from_torch(
        Q_local, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )
    tt_K = ttnn.from_torch(
        K_local, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )
    tt_V = ttnn.from_torch(
        V_local, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )
    tt_K_gathered = ttnn.from_torch(
        K_gathered, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )
    tt_V_gathered = ttnn.from_torch(
        V_gathered, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device
    )

    # Run profile op - catch L1 OOM as expected for large chunk sizes
    try:
        tt_output, _, tt_lse = ttnn.transformer.ring_joint_sdpa_profile(
            tt_Q,
            tt_K,
            tt_V,
            tt_K_gathered,
            tt_V_gathered,
            ring_size=ring_size,
            ring_index=ring_index,
            logical_n=total_seq,
            program_config=program_config,
            is_causal=True,
            is_balanced=True,
            compute_kernel_config=compute_kernel_config,
        )

        # Success - verify and log
        assert tt_output is not None
        assert tt_lse is not None
        print(
            f"PASS: ring_index={ring_index}, total_seq={total_seq}, local_seq={local_seq}, q_chunk={q_chunk_size}, k_chunk={k_chunk_size}"
        )
        print(f"  Q: {list(Q_local.shape)} -> tt_Q: {tt_Q.shape}")
        print(f"  K_gathered: {list(K_gathered.shape)} -> tt_K_gathered: {tt_K_gathered.shape}")
        print(f"  Output: {tt_output.shape}")

    except RuntimeError as e:
        if "beyond max L1 size" in str(e):
            # Expected OOM for large chunk sizes - skip
            print(f"OOM: ring_index={ring_index}, q_chunk={q_chunk_size}, k_chunk={k_chunk_size}")
            pytest.skip(f"L1 OOM: q={q_chunk_size}, k={k_chunk_size}")
        else:
            raise


# ============================================================================
# Main entry point for direct execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
