# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import random

import torch
from loguru import logger
from ttnn.operations.ccl import MoEActivationFunction

import ttnn
from tt_py_test_utils_common.all_to_all_combine_ref import get_batch_cluster_idxr, get_cluster_dims
from tt_py_test_utils_common.all_to_all_dispatch import tt_to_torch_dtype
from tt_py_test_utils_common.selective_combine import device_mesh_iterator
from tt_py_test_utils_common.utility_functions import comp_pcc


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------
def validate_per_expert_tokens(
    mesh_device,
    experts_per_device,
    num_devices,
    per_expert_total_tokens_output_tensor,
    expert_token_counts,
    worker_mcast_bbox=None,
):
    logger.debug(f"\n========== Per Expert Total Tokens Tensor Validation ==========")
    per_expert_tokens_all_passed = True

    # L1 alignment constant (16 bytes)
    l1_alignment = 16

    # Validate shape: [num_devices, num_cores, aligned_row_elements]
    # Row is experts_per_device uint32s, aligned to 16 bytes. Replicated on every core
    per_expert_row_bytes = ((experts_per_device * 4 + l1_alignment - 1) // l1_alignment) * l1_alignment
    per_expert_row_elements = per_expert_row_bytes // 4
    core_range = mesh_device.compute_with_storage_grid_size()
    num_cores = core_range.x * core_range.y
    expected_per_expert_shape = (num_devices * num_cores, per_expert_row_elements)

    # Convert per_expert_total_tokens tensor to torch
    # Shape per device: [num_cores (70), aligned_elements] as uint32
    per_expert_total_tokens_torch = ttnn.to_torch(
        per_expert_total_tokens_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    logger.debug(f"Per expert total tokens torch shape: {per_expert_total_tokens_torch.shape}")

    assert per_expert_total_tokens_torch.shape == expected_per_expert_shape, (
        f"per_expert_total_tokens shape mismatch: expected {expected_per_expert_shape}, "
        f"got {per_expert_total_tokens_torch.shape}"
    )

    per_expert_total_tokens_torch = per_expert_total_tokens_torch.reshape(
        (num_devices, num_cores, per_expert_row_elements)
    )
    # The op allocates the per_expert_total_tokens output sharded across the FULL
    # compute_with_storage_grid_size() (see `compute_output_specs` in
    # moe_compute_device_operation.cpp), but
    # the kernel only multicasts the counts to its `all_worker_cores_bounding_box`
    # (= tilize + matmul + combine cores). On WH 6U the op uses the full grid, so every
    # core is inside the bbox and every shard slot gets the count via mcast. On BH single
    # LB the op uses ~110 of the grid's 130 cores, so shard slots outside the bbox hold
    # the initial-allocation zero. When `worker_mcast_bbox` is provided, skip out-of-bbox
    # slots explicitly (strict in-bbox check). When omitted (legacy callers), fall back
    # to the heuristic: treat actual=0/expected>0 as off-bbox; flag any other mismatch.
    all_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(core_range.x - 1, core_range.y - 1),
            ),
        }
    )
    grid_cores_row_major = ttnn.corerange_to_cores(all_core_range_set, row_wise=True)
    assert len(grid_cores_row_major) == num_cores

    if worker_mcast_bbox is not None:

        def core_in_mcast_bbox(core_coord):
            return (
                worker_mcast_bbox.start.x <= core_coord.x <= worker_mcast_bbox.end.x
                and worker_mcast_bbox.start.y <= core_coord.y <= worker_mcast_bbox.end.y
            )

        cores_per_device_validated = 0
        for device_idx in range(num_devices):
            for c, core_coord in enumerate(grid_cores_row_major):
                if not core_in_mcast_bbox(core_coord):
                    continue

                device_counts = per_expert_total_tokens_torch[device_idx][c]
                for local_exp_idx in range(experts_per_device):
                    expected_count = expert_token_counts[device_idx, local_exp_idx].item()
                    actual_count = device_counts[local_exp_idx].item()

                    if actual_count != expected_count:
                        logger.warning(
                            f"  Device {device_idx}, Core {c} ({core_coord.x},{core_coord.y}), "
                            f"Expert {local_exp_idx}: count mismatch - expected {expected_count}, got {actual_count}"
                        )
                        per_expert_tokens_all_passed = False
                    else:
                        logger.debug(
                            f"  Device {device_idx}, Core {c} ({core_coord.x},{core_coord.y}), "
                            f"Expert {local_exp_idx}: count={actual_count} PASSED"
                        )
                if device_idx == 0:
                    cores_per_device_validated += 1
    else:
        # Legacy heuristic: treat actual=0 with expected>0 as "core not in mcast bbox"
        # and skip; only flag truly inconsistent values.
        cores_per_device_validated = 0
        for device_idx in range(num_devices):
            for c in range(num_cores):
                device_counts = per_expert_total_tokens_torch[device_idx][c]

                core_in_bbox = False
                for local_exp_idx in range(experts_per_device):
                    expected_count = expert_token_counts[device_idx, local_exp_idx].item()
                    actual_count = device_counts[local_exp_idx].item()

                    if actual_count == 0 and expected_count > 0:
                        continue

                    core_in_bbox = True
                    if actual_count != expected_count:
                        logger.warning(
                            f"  Device {device_idx}, Core {c}, Expert {local_exp_idx}: "
                            f"count mismatch - expected {expected_count}, got {actual_count}"
                        )
                        per_expert_tokens_all_passed = False
                    else:
                        logger.debug(
                            f"  Device {device_idx}, Core {c}, Expert {local_exp_idx}: count={actual_count} PASSED"
                        )
                if core_in_bbox and device_idx == 0:
                    cores_per_device_validated += 1

    # Sanity: ensure the mcast actually reached a meaningful number of cores per device.
    # (Catches the regression where the mcast bbox shrinks to 0 by accident.)
    if cores_per_device_validated == 0:
        logger.warning("No cores received per-expert counts via mcast — bbox calculation broken?")
        per_expert_tokens_all_passed = False
    else:
        logger.debug(
            f"Per-expert counts validated on {cores_per_device_validated} cores per device "
            f"(of {num_cores} grid cores total)"
        )

    return per_expert_tokens_all_passed


def validate_activation(
    mesh_device, experts_per_device, num_devices, expert_activation_output_tensor, golden_activation
):
    logger.debug(f"\n========== Expert Activation Tensor Validation ==========")
    activation_all_passed = True

    # Row size in uint32 elements (aligned to 16 bytes = 4 uint32s)
    row_elements_unaligned = 2 * experts_per_device + 1  # token_id + k_indices + scores
    row_bytes_unaligned = row_elements_unaligned * 4
    aligned_row_bytes = ((row_bytes_unaligned + 15) // 16) * 16
    aligned_row_elements = aligned_row_bytes // 4

    # Convert expert_activation tensor to torch
    # Shape per device: [1, total_bytes / 4] as uint32
    expert_activation_torch = ttnn.to_torch(
        expert_activation_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    logger.debug(f"Expert activation torch shape: {expert_activation_torch.shape}")

    for device_idx in range(num_devices):
        golden_rows = golden_activation[device_idx]
        num_expected_rows = len(golden_rows)

        # Extract this device's activation data
        # The tensor is flattened, so we need to parse rows
        device_activation = expert_activation_torch[device_idx].flatten().to(torch.int64)
        max_rows = len(device_activation) // aligned_row_elements

        logger.debug(
            f"Device {device_idx}: expecting {num_expected_rows} activated tokens, tensor has space for {max_rows} rows"
        )

        # Validate each row in sequential order (kernel preserves global token order)
        for row_idx, golden_row in enumerate(golden_rows):
            if row_idx >= max_rows:
                logger.warning(f"  Device {device_idx}: row {row_idx} out of bounds (max {max_rows})")
                activation_all_passed = False
                break

            row_start = row_idx * aligned_row_elements

            # Extract token_id
            actual_token_id = device_activation[row_start].item()
            expected_token_id = golden_row["token_id"]

            if actual_token_id != expected_token_id:
                logger.warning(
                    f"  Device {device_idx}, Row {row_idx}: token_id mismatch - "
                    f"expected {expected_token_id}, got {actual_token_id}"
                )
                activation_all_passed = False
                continue

            # Validate k_indices and scores for each local expert
            for local_exp_idx in range(experts_per_device):
                expected_k = golden_row["k_indices"][local_exp_idx]
                expected_score = golden_row["scores"][local_exp_idx]

                if expected_k >= 0:  # This expert was selected
                    actual_k = device_activation[row_start + 1 + local_exp_idx].item()
                    actual_score_bits = device_activation[row_start + 1 + experts_per_device + local_exp_idx].item()

                    # Convert score bits back to bfloat16 then float
                    # The score is stored as uint16 in the lower bits of uint32
                    actual_score_bf16 = torch.tensor([actual_score_bits & 0xFFFF], dtype=torch.int16).view(
                        torch.bfloat16
                    )
                    actual_score = actual_score_bf16.item()

                    if actual_k != expected_k:
                        logger.warning(
                            f"  Device {device_idx}, Row {row_idx}, Expert {local_exp_idx}: "
                            f"k_index mismatch - expected {expected_k}, got {actual_k}"
                        )
                        activation_all_passed = False

                    # Compare scores with tolerance (bfloat16 precision)
                    if abs(actual_score - expected_score) > 1e-2:
                        logger.warning(
                            f"  Device {device_idx}, Row {row_idx}, Expert {local_exp_idx}: "
                            f"score mismatch - expected {expected_score:.4f}, got {actual_score:.4f}"
                        )
                        activation_all_passed = False

    return activation_all_passed


def validate_e_t(mesh_device, total_tokens, experts_per_device, num_devices, e_t_output_tensor, golden_e_t):
    logger.debug(f"\n========== E-T (Expert-to-Token) Tensor Validation ==========")
    e_t_all_passed = True

    # L1 alignment constant (16 bytes)
    l1_alignment = 16

    # Each entry is 16B aligned (4 uint32s per token ID)
    e_t_entry_size_bytes = ((4 + l1_alignment - 1) // l1_alignment) * l1_alignment  # align sizeof(uint32_t) to 16B
    e_t_entry_elements = e_t_entry_size_bytes // 4  # elements per entry (4 for 16B alignment)

    # Validate shape: [num_devices * experts_per_device, e_t_row_elements]
    # Each expert has (total_tokens + 1) entries (tokens + sentinel), each entry is 16B aligned
    e_t_row_elements = (total_tokens + 1) * e_t_entry_elements
    expected_e_t_shape = (num_devices * experts_per_device, e_t_row_elements)

    # Convert e_t tensor to torch
    # Shape per device: [experts_per_device, e_t_row_elements] as uint32
    e_t_torch = ttnn.to_torch(e_t_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    logger.debug(f"E-T torch shape: {e_t_torch.shape}")

    if e_t_torch.shape != expected_e_t_shape:
        logger.warning(f"e_t shape mismatch: expected {expected_e_t_shape}, got {e_t_torch.shape}")
        return False

    for device_idx in range(num_devices):
        for local_exp_idx in range(experts_per_device):
            expected_tokens = golden_e_t[device_idx][local_exp_idx]
            num_expected_tokens = len(expected_tokens)

            # Get the row for this expert from this device
            # Each device has experts_per_device rows in the tensor
            row_idx = device_idx * experts_per_device + local_exp_idx
            device_expert_row = e_t_torch[row_idx].flatten().to(torch.int64)

            # Validate each token in the e_t list
            tokens_match = True
            for i, expected_token_id in enumerate(expected_tokens):
                # Each entry is at 16B (e_t_entry_elements) offset
                actual_token_id = device_expert_row[i * e_t_entry_elements].item()

                if actual_token_id != expected_token_id:
                    logger.warning(
                        f"  Device {device_idx}, Expert {local_exp_idx}, Entry {i}: "
                        f"token_id mismatch - expected {expected_token_id}, got {actual_token_id}"
                    )
                    tokens_match = False
                    e_t_all_passed = False

            # Validate sentinel (-1) at end of list
            sentinel_idx = num_expected_tokens * e_t_entry_elements
            if sentinel_idx < len(device_expert_row):
                sentinel_value = device_expert_row[sentinel_idx].item()
                is_sentinel = (sentinel_value == -1) or (sentinel_value == 0xFFFFFFFF)

                if not is_sentinel:
                    logger.warning(
                        f"  Device {device_idx}, Expert {local_exp_idx}: "
                        f"sentinel mismatch - expected -1, got {sentinel_value}"
                    )
                    e_t_all_passed = False
                elif tokens_match:
                    logger.debug(
                        f"  Device {device_idx}, Expert {local_exp_idx}: "
                        f"{num_expected_tokens} tokens validated, sentinel PASSED"
                    )
            else:
                logger.warning(
                    f"  Device {device_idx}, Expert {local_exp_idx}: " f"sentinel index {sentinel_idx} out of bounds"
                )
                e_t_all_passed = False

    return e_t_all_passed


ATOL_THRESHOLD = 700


def validate_combine(layer_id, mesh_device, cluster_axis, tt_combine_output, combine_goldens, pcc_threshold):
    if cluster_axis == 0:
        mesh_shape = tuple(mesh_device.shape)
        # need to roll my own mesh composer here for the transposed ordering
        device_shards = [
            ttnn.to_torch(ittout, mesh_composer=None) for ittout in ttnn.get_device_tensors(tt_combine_output)
        ]
        ordered_shards = []
        for ir in range(mesh_shape[1]):
            for ic in range(mesh_shape[0]):
                ordered_shards.append(device_shards[ic * mesh_shape[1] + ir])
        torch_combine_out = torch.cat(ordered_shards, dim=1)

    else:
        torch_combine_out = ttnn.to_torch(tt_combine_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))

    return validate_combine_torch(layer_id, torch_combine_out, combine_goldens, pcc_threshold)


def validate_combine_torch(layer_id, torch_combine_out, combine_goldens, pcc_threshold):
    """Compare an already-composed combine output [K, tokens, hidden] against the golden rows
    flagged in output_data_map. Slots with no flagged row (a device that owns none of the
    experts a token selected at that k) are skipped."""
    output_ref, output_data_map = combine_goldens

    assert torch_combine_out.shape == output_ref[0].shape
    combine_all_passed = True

    for k in range(torch_combine_out.shape[0]):
        vals, refs = [], []
        for t in range(torch_combine_out.shape[1]):
            if output_data_map[layer_id, k, t].item() == 1:
                vals.append(torch_combine_out[k, t, :])
                refs.append(output_ref[layer_id, k, t, :])

        if not vals:
            continue
        vals = torch.stack(vals)
        refs = torch.stack(refs)
        _, pcc_val = comp_pcc(refs, vals)
        allclose_passed = torch.allclose(refs, vals, atol=ATOL_THRESHOLD)

        if pcc_val < pcc_threshold or not allclose_passed:
            combine_all_passed = False
            logger.warning(f"Layer {layer_id}, k: {k} PCC={pcc_val:.6f}, AllClose passed: {allclose_passed}")
            if not allclose_passed:
                mask = (vals - refs).abs() > ATOL_THRESHOLD
                logger.warning(f"AllClose variation result: {vals[mask]}, ref: {refs[mask]}")
        else:
            logger.debug(f"Combine, layer: {layer_id}, k: {k} PCC={pcc_val:.6f}, AllClose passed: {allclose_passed}")

    return combine_all_passed


# ---------------------------------------------------------------------------
# Tensor creation functions
# ---------------------------------------------------------------------------
def create_torch_w0(L, E, K, N):
    """
    Create torch w0 weight tensor.

    Args:
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension

    Returns:
        torch_w0: Tensor of shape (L, E, K, N)

    Weight initialization controlled by WEIGHT_INIT_MODE env var:
        - "baseline" or "golden": constant weights (0.1)
        - "random_w0w1": random w0/w1, golden w2
        - "random_w2": golden w0/w1, random w2
        - "random_all" (default): all random weights
    """
    mode = os.environ.get("WEIGHT_INIT_MODE", "random_all")

    if mode in ["baseline", "golden", "random_w2"]:
        # Use constant/golden weights for w0
        torch_w0 = torch.ones((L, E, K, N), dtype=torch.bfloat16) * 0.1
        logger.debug(f"[WEIGHT_INIT] w0: GOLDEN (constant 0.1) - mode={mode}")
    else:
        # Use random weights for w0
        torch_w0 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
        logger.debug(f"[WEIGHT_INIT] w0: RANDOM - mode={mode}")

    return torch_w0


def create_torch_w1(L, E, K, N):
    """
    Create torch w1 weight tensor.

    Args:
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension

    Returns:
        torch_w1: Tensor of shape (L, E, K, N)

    Weight initialization controlled by WEIGHT_INIT_MODE env var:
        - "baseline" or "golden": constant weights (0.1)
        - "random_w0w1": random w0/w1, golden w2
        - "random_w2": golden w0/w1, random w2
        - "random_all" (default): all random weights
    """
    mode = os.environ.get("WEIGHT_INIT_MODE", "random_all")

    if mode in ["baseline", "golden", "random_w2"]:
        # Use constant/golden weights for w1
        torch_w1 = torch.ones((L, E, K, N), dtype=torch.bfloat16) * 0.1
        logger.debug(f"[WEIGHT_INIT] w1: GOLDEN (constant 0.1) - mode={mode}")
    else:
        # Use random weights for w1
        torch_w1 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
        logger.debug(f"[WEIGHT_INIT] w1: RANDOM - mode={mode}")

    return torch_w1


def create_torch_w2(L, E, N, K):
    """
    Create torch w2 weight tensor.

    Args:
        L: Number of layers
        E: Number of experts
        N: Intermediate dimension
        K: Output dimension

    Returns:
        torch_w2: Tensor of shape (L, E, N, K)

    Weight initialization controlled by WEIGHT_INIT_MODE env var:
        - "baseline" or "golden": constant weights (0.1)
        - "random_w0w1": random w0/w1, golden w2
        - "random_w2": golden w0/w1, random w2
        - "random_all" (default): all random weights
    """
    mode = os.environ.get("WEIGHT_INIT_MODE", "random_all")

    if mode in ["baseline", "golden", "random_w0w1"]:
        # Use constant/golden weights for w2
        torch_w2 = torch.ones((L, E, N, K), dtype=torch.bfloat16) * 0.1
        logger.debug(f"[WEIGHT_INIT] w2: GOLDEN (constant 0.1) - mode={mode}")
    else:
        # Use random weights for w2
        torch_w2 = torch.rand((L, E, N, K), dtype=torch.bfloat16) - 0.5
        logger.debug(f"[WEIGHT_INIT] w2: RANDOM - mode={mode}")

    return torch_w2


def get_linearized_mesh_coord(num_replicated_devices, cluster_axis, expert_id, experts_per_cluster, experts_per_device):
    if cluster_axis == 0:
        cluster_id = expert_id // experts_per_cluster
        expert_id_within_cluster = expert_id % experts_per_cluster
        device_id_within_cluster = expert_id_within_cluster // experts_per_device

        return device_id_within_cluster * num_replicated_devices + cluster_id
    else:
        return expert_id // experts_per_device


def gen_expert_mapping(
    num_devices, num_replicated_devices, cluster_axis, experts, experts_per_cluster, experts_per_device
):
    """
    Create per-device expert mapping tensor that maps each expert to the device it belongs to.
    Shape: [num_devices, experts] where each entry is the linearized mesh coordinate of the device
    that owns that expert from the perspective of that source device.

    For now, all devices see the same mapping (no replicated experts).
    For 256 experts and 128 devices (2 experts per device):
    expert_mapping[d, e] = e // experts_per_device

    In the future, this can be extended to support replicated experts where each device
    sees the "optimal" device (e.g., shortest distance) for each expert.

    This tensor is replicated on every device (even devices not along the dispatch axis).
    """
    expert_mapping = torch.zeros(1, experts, dtype=torch.uint16)
    for e in range(experts):
        expert_mapping[0, e] = get_linearized_mesh_coord(
            num_replicated_devices, cluster_axis, e, experts_per_cluster, experts_per_device
        )
    # Replicate across all devices (same mapping for now)
    expert_mapping = expert_mapping.repeat(num_devices, 1)
    return expert_mapping


def gen_sparse_buffer_and_indices(
    tokens_per_device, hidden_size, experts, selected_experts_k, mesh_shape, cluster_axis, dtype=torch.bfloat16
):
    """
    Generate the sparse buffer (simulating output from all_to_all_dispatch) and the all-gathered
    expert indices tensor.

    The sparse buffer has shape [num_devices, total_tokens, hidden_size].
    Each device receives tokens from all devices in the dispatch dimension.
    A token is placed in the sparse buffer if the expert it selected lives on that device.
    total_tokens = tokens_per_device * num_dispatch_devices

    The expert indices tensor has shape [num_dispatch_devices, tokens_per_device, selected_experts_k]
    and is all-gathered so each device sees which experts every token selected.

    Returns:
        sparse_buffer: [num_devices, total_tokens, hidden_size] - the sparse input to selective_tilize
        expert_indices: [num_dispatch_devices, tokens_per_device, K] - all-gathered indices
        expert_scores: [num_dispatch_devices, tokens_per_device, K] - all-gathered scores
        original_tokens: [num_dispatch_devices, tokens_per_device, hidden_size] - for verification
    """
    num_devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        num_dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        num_dispatch_devices = mesh_shape[0]
    else:
        num_dispatch_devices = num_devices

    experts_per_device = experts // num_devices

    # Total tokens in sparse buffer = tokens_per_device * num_dispatch_devices
    total_tokens = tokens_per_device * num_dispatch_devices

    # Generate original tokens for each source device
    # Shape: [num_dispatch_devices, tokens_per_device, hidden_size]

    # original_tokens = torch.ones(num_dispatch_devices, tokens_per_device, hidden_size, dtype=dtype)
    # original_tokens = torch.zeros(num_dispatch_devices, tokens_per_device, hidden_size, dtype=dtype)
    original_tokens = torch.rand(num_dispatch_devices, tokens_per_device, hidden_size, dtype=dtype) - 0.5

    # Generate expert indices for each token
    # Shape: [num_dispatch_devices, tokens_per_device, selected_experts_k]
    expert_indices = torch.zeros(num_dispatch_devices, tokens_per_device, selected_experts_k, dtype=torch.uint16)
    for src_device in range(num_dispatch_devices):
        for t in range(tokens_per_device):
            # Each token selects K unique experts
            selected = torch.randperm(experts)[:selected_experts_k]
            expert_indices[src_device, t, :] = selected.to(torch.uint16)

    # Generate expert scores
    # Shape: [num_dispatch_devices, tokens_per_device, selected_experts_k]
    expert_scores = torch.rand(num_dispatch_devices, tokens_per_device, selected_experts_k, dtype=dtype) + 1e-5
    expert_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True)

    # Build the sparse buffer
    # Shape: [num_devices, total_tokens, hidden_size]
    # Initialize with random garbage (tokens not sent to a device will have garbage)
    sparse_buffer = torch.rand(num_devices, total_tokens, hidden_size, dtype=dtype)

    # Place tokens in the sparse buffer based on expert selection
    # Token layout: [src_device_0_token_0, src_device_0_token_1, ..., src_device_1_token_0, ...]
    for src_device in range(num_dispatch_devices):
        for t in range(tokens_per_device):
            token = original_tokens[src_device, t, :]
            token_idx_in_sparse = src_device * tokens_per_device + t

            # For each expert this token selected, place it on the device that owns that expert
            for k in range(selected_experts_k):
                expert_id = expert_indices[src_device, t, k].item()
                target_device = expert_id // experts_per_device
                sparse_buffer[target_device, token_idx_in_sparse, :] = token

    return sparse_buffer, expert_indices, expert_scores, original_tokens


# ---------------------------------------------------------------------------
# Golden computation functions
# ---------------------------------------------------------------------------
def compute_selective_tilize_golden(
    sparse_buffer, expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
):
    """
    Compute the golden output for selective_tilize.

    For each device, we need to produce a dense, tilized output for each of its experts.
    Output shape: [num_devices, experts_per_device, total_tokens, hidden_size]

    Each expert on a device collects all tokens that selected it from all source devices.
    """
    num_devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        num_dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        num_dispatch_devices = mesh_shape[0]
    else:
        num_dispatch_devices = num_devices

    tokens_per_device = expert_indices.shape[1]
    selected_experts_k = expert_indices.shape[2]
    hidden_size = sparse_buffer.shape[2]
    experts = expert_mapping.shape[1]
    experts_per_device = experts // num_devices

    # Total possible tokens that could be sent to any expert
    total_tokens = tokens_per_device * num_dispatch_devices

    # Output: [devices, experts_per_device, total_tokens, hidden_size]
    # Initialize with zeros
    golden_output = torch.zeros(num_devices, experts_per_device, total_tokens, hidden_size, dtype=sparse_buffer.dtype)

    # Track how many tokens each expert has received (for each device)
    expert_token_counts = torch.zeros(num_devices, experts_per_device, dtype=torch.int32)

    # For each token, place it in the output for the experts it selected
    for src_device in range(num_dispatch_devices):
        for t in range(tokens_per_device):
            token_idx_in_sparse = src_device * tokens_per_device + t

            for k in range(selected_experts_k):
                expert_id = expert_indices[src_device, t, k].item()

                # Get the device that owns this expert from the source device's perspective
                target_device = expert_mapping[src_device, expert_id].item()

                # Local expert index on that device
                local_expert_idx = expert_id % experts_per_device

                # Get the token from sparse buffer
                token = sparse_buffer[target_device, token_idx_in_sparse, :]

                # Place in output at the next available slot for this expert
                token_slot = expert_token_counts[target_device, local_expert_idx].item()
                golden_output[target_device, local_expert_idx, token_slot, :] = token
                expert_token_counts[target_device, local_expert_idx] += 1

    return golden_output, expert_token_counts


def compute_expert_activation_golden(expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis):
    """
    Compute the golden expert_activation tensor for each device.

    For each device, the expert_activation tensor contains rows for tokens that selected
    at least one expert on that device. Each row format:
    [token_id, k_idx_0, k_idx_1, ..., score_0, score_1, ...]

    Where:
    - token_id: global token index (src_device * tokens_per_device + local_token_idx)
    - k_idx_e: which of the K selected experts (0..K-1) maps to local expert e, or -1 if not selected
    - score_e: the score for local expert e (as bfloat16 bits in uint32), or 0 if not selected

    The last row is a sentinel with token_id = -1 (0xFFFFFFFF as uint32).

    Returns:
        golden_activation: dict[device_idx] -> list of activation row dicts
        Each row dict has: token_id, k_indices (list), scores (list)
    """
    num_devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        num_dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        num_dispatch_devices = mesh_shape[0]
    else:
        num_dispatch_devices = num_devices

    tokens_per_device = expert_indices.shape[1]
    selected_experts_k = expert_indices.shape[2]
    experts = expert_mapping.shape[1]
    experts_per_device = experts // num_devices

    # Build activation rows for each device
    # golden_activation[device] = list of (token_id, k_indices, scores)
    golden_activation = {d: [] for d in range(num_devices)}

    for src_device in range(num_dispatch_devices):
        for t in range(tokens_per_device):
            global_token_id = src_device * tokens_per_device + t

            # Track which local experts this token activated on each device
            # device -> {local_expert_idx: (k, score)}
            device_activations = {d: {} for d in range(num_devices)}

            for k in range(selected_experts_k):
                expert_id = expert_indices[src_device, t, k].item()
                target_device = expert_mapping[src_device, expert_id].item()
                local_expert_idx = expert_id % experts_per_device
                score = expert_scores[src_device, t, k].item()

                # Store the k-index and score for this local expert
                device_activations[target_device][local_expert_idx] = (k, score)

            # For each device that has at least one activation, add a row
            for device_idx in range(num_devices):
                if device_activations[device_idx]:
                    k_indices = [-1] * experts_per_device  # -1 means not activated
                    scores = [0.0] * experts_per_device

                    for local_exp_idx, (k, score) in device_activations[device_idx].items():
                        k_indices[local_exp_idx] = k
                        scores[local_exp_idx] = score

                    golden_activation[device_idx].append(
                        {
                            "token_id": global_token_id,
                            "k_indices": k_indices,
                            "scores": scores,
                        }
                    )

    return golden_activation, experts_per_device


def compute_e_t_golden(expert_indices, expert_mapping, mesh_shape, cluster_axis):
    """
    Compute the golden e_t (expert-to-token) tensor for each device.

    For each device and each local expert, this builds a list of token IDs that
    activate that expert. The list is stored in sequential order as tokens are
    processed, and is terminated with a -1 sentinel.

    Returns:
        golden_e_t: dict[device_idx] -> dict[local_expert_idx] -> list of token_ids
    """
    num_devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis == 1:
        num_dispatch_devices = mesh_shape[1]
    elif cluster_axis == 0:
        num_dispatch_devices = mesh_shape[0]
    else:
        num_dispatch_devices = num_devices

    tokens_per_device = expert_indices.shape[1]
    selected_experts_k = expert_indices.shape[2]
    experts = expert_mapping.shape[1]
    experts_per_device = experts // num_devices

    # Build e_t lists for each device and each local expert
    # golden_e_t[device][local_expert] = [token_id_0, token_id_1, ...]
    golden_e_t = {d: {e: [] for e in range(experts_per_device)} for d in range(num_devices)}

    for src_device in range(num_dispatch_devices):
        for t in range(tokens_per_device):
            global_token_id = src_device * tokens_per_device + t

            for k in range(selected_experts_k):
                expert_id = expert_indices[src_device, t, k].item()
                target_device = expert_mapping[src_device, expert_id].item()
                local_expert_idx = expert_id % experts_per_device

                # Each token selects K unique experts, so no duplicate tracking needed
                golden_e_t[target_device][local_expert_idx].append(global_token_id)

    return golden_e_t, experts_per_device


# hardcoded for GPT-OSS
def _swiglu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    gate_c = torch.clamp(gate, max=clamp_limit)
    up_c = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def compute_matmul_golden(
    torch_input_ref,
    torch_w0,
    torch_w1,
    torch_w2,
    layers,
    experts,
    devices,
    tokens_per_device,
    hidden,
    torch_b0=None,
    torch_b1=None,
    torch_b2=None,
    activation_type=MoEActivationFunction.SILU,
    activation_limit=None,
):
    if (activation_type == MoEActivationFunction.CLAMPED_SILU) != (activation_limit is not None):
        raise ValueError("activation_limit is required for CLAMPED_SILU and must be omitted for other activations")
    tokens = tokens_per_device * devices

    # (L, D, E/D, T, H) -> (L, E, T, H)
    torch_input_ref = torch_input_ref.reshape(layers, experts, tokens, hidden)

    # in the test setup the expert weights are duplicated over devices, do so here
    # (L, E/D, K, N) -> (L, E, K, N)
    torch_w0 = torch_w0.repeat([1, devices, 1, 1])
    # (L, E/D, K, N) -> (L, E, K, N)
    torch_w1 = torch_w1.repeat([1, devices, 1, 1])
    # (L, E/D, N, K) -> (L, E, N, K)
    torch_w2 = torch_w2.repeat([1, devices, 1, 1])

    # Cast to fp32 for CPU matmul: torch's bf16 CPU matmul falls through to a
    # single-threaded reference loop on many builds (even with OMP/MKL enabled),
    # making the golden compute take tens of minutes on large shapes. fp32 uses
    # MKL/oneDNN parallel kernels and is ~100x faster here. Result is cast back
    # to the original dtype before return so downstream golden compute is
    # unchanged. Verified on BH single-LB run: bf16 path hung past 15 min in
    # `compute_matmul_golden`; fp32 path completes in seconds.
    _orig_dtype = torch_input_ref.dtype
    torch_input_ref = torch_input_ref.float()
    torch_w0 = torch_w0.float()
    torch_w1 = torch_w1.float()
    torch_w2 = torch_w2.float()

    # Compute gate activations for each expert
    # (L, E, T, K) @ (L, E, K, N) -> (L, E, T, N)
    torch_w0_output_ref = torch_input_ref @ torch_w0
    if torch_b0 is not None:
        # True PyTorch MoE math: x @ W + bias.
        # Bias shape: (L, E, N) - broadcasts across tokens (L, E, T, N) automatically.
        # Weights are replicated per-device, so bias must be too.
        b0 = torch_b0.repeat([1, devices, 1]).float()  # (L, E, N)
        torch_w0_output_ref = torch_w0_output_ref + b0.unsqueeze(2)  # broadcast T dimension

    torch_w1_output_ref = torch_input_ref @ torch_w1
    if torch_b1 is not None:
        # Same reasoning as b0.
        b1 = torch_b1.repeat([1, devices, 1]).float()  # (L, E, N)
        torch_w1_output_ref = torch_w1_output_ref + b1.unsqueeze(2)

    if activation_type == MoEActivationFunction.SILU:
        # SILU: silu(x @ w0) * (x @ w1)
        torch_silu_output_ref = torch.nn.functional.silu(torch_w0_output_ref)
        # (L, E, T, K) @ (L, E, K, N) -> (L, E, T, N)
        torch_intermediate_ref = torch_silu_output_ref * torch_w1_output_ref  # (L, E, T, N)
    elif activation_type == MoEActivationFunction.CLAMPED_SILU:
        torch_intermediate_ref = torch.nn.functional.silu(
            torch_w0_output_ref.clamp(max=activation_limit)
        ) * torch_w1_output_ref.clamp(min=-activation_limit, max=activation_limit)
    elif activation_type == MoEActivationFunction.SWIGLU:
        torch_intermediate_ref = _swiglu_reference(torch_w0_output_ref, torch_w1_output_ref)  # (L, E, T, N)
    elif activation_type == MoEActivationFunction.GELU:
        torch_gelu_output_ref = torch.nn.functional.gelu(torch_w0_output_ref, approximate="tanh")
        torch_intermediate_ref = torch_gelu_output_ref * torch_w1_output_ref  # (L, E, T, N)
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")

    # (L, E, T, N) @ (L, E, N, K) -> (L, E, T, K)
    torch_output_ref = torch_intermediate_ref @ torch_w2
    if torch_b2 is not None:
        # Same reasoning as b0: true PyTorch bias addition, replicated per device.
        b2 = torch_b2.repeat([1, devices, 1]).float()  # (L, E, K)
        torch_output_ref = torch_output_ref + b2.unsqueeze(2)

    # Cast back to the input dtype to keep downstream golden compute unchanged.
    torch_output_ref = torch_output_ref.to(_orig_dtype)

    # pull device dim back out for comparison
    # (L, E, T, H) -> (L, devices, E/devices, T, H)
    return torch_output_ref.reshape(layers, devices, experts // devices, tokens, hidden)


def compute_combine_golden(
    layers,
    experts,
    tokens,
    hidden_size,
    select_experts_k,
    mesh_shape,
    matmul_goldens,
    dense_token_activations,
    cluster_axis,
):
    cluster_factor, cluster_size, devices = get_cluster_dims(cluster_axis, mesh_shape)
    experts_per_device = experts // devices

    output_ref_tensor = torch.zeros(layers, select_experts_k, tokens * cluster_factor, hidden_size).bfloat16()
    output_data_map = torch.zeros(output_ref_tensor.shape[:-1])

    batch_rep_idxr = get_batch_cluster_idxr(cluster_axis, tokens)

    for l in range(layers):
        for m0, m1, d in device_mesh_iterator(mesh_shape):
            activations = dense_token_activations[l][d]
            for e in range(experts_per_device):
                dense_token_index = 0
                for a in activations:
                    if a["k_indices"][e] == -1:
                        continue
                    st = a["token_id"]
                    k = a["k_indices"][e]

                    gt = batch_rep_idxr(m0, m1, st)

                    contrib = matmul_goldens[l, d, e, dense_token_index]
                    output_ref_tensor[l, k, gt] = contrib
                    output_data_map[l, k, gt] = 1

                    dense_token_index += 1

    return output_ref_tensor, output_data_map


def create_sharded_memory_config(core_range_set, tensor_shape, dtype):
    """
    Create an L1 sharded memory config for a tensor to be completely on specified cores.
    """
    total_elements = 1
    for dim in tensor_shape:
        total_elements *= dim

    # Calculate bytes per element
    if dtype == ttnn.uint16:
        bytes_per_element = 2
    elif dtype == ttnn.bfloat16:
        bytes_per_element = 2
    elif dtype == ttnn.float32:
        bytes_per_element = 4
    else:
        bytes_per_element = 2

    # Shard evenly across cores, but for "completely on one core" we use 1 core
    shard_height = tensor_shape[0] if len(tensor_shape) > 0 else 1
    shard_width = tensor_shape[1] if len(tensor_shape) > 1 else total_elements

    shard_spec = ttnn.ShardSpec(
        core_range_set,
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )


@dataclasses.dataclass
class _MoEWeightInputs:
    """Raw torch weights/biases — consumed once (on a cache miss) to build the golden
    and the quantized weight host tensors. Returned alongside the cached
    ``_MoEHostData`` so the device-dependent quantization can run after the (pure)
    golden build; once the bf4 host tensors land on ``_MoEHostData`` they supersede
    these for all downstream use."""

    torch_w0: "torch.Tensor"
    torch_w1: "torch.Tensor"
    torch_w2: "torch.Tensor"
    torch_b0: object
    torch_b1: object
    torch_b2: object


@dataclasses.dataclass
class _MoEHostData:
    """Cached, device-independent artifacts for one MoE compute test shape.

    Two kinds of expensive work are cached here, both reusable across pytest
    parameters that don't change the shape (notably the two ``enable_trace``
    variants of a model, which run back-to-back):

    1. The torch golden references — ``compute_matmul_golden`` dominates host cost.
    2. The quantized (``bfloat4_b``) weight tensors as **host** tensors. Producing
       them (upload → on-device prepare → host round-trip quantize) is expensive
       and needs a live device, but ``HostStorage`` tensors survive ``mesh_device``
       teardown, so each invocation only pays the ``to_device`` re-upload.
    """

    # Per-layer torch inputs (lists indexed by layer_id) — cheap to re-upload via from_torch
    sparse_buffers: list
    expert_indices_list: list
    expert_scores_list: list
    expert_mapping: "torch.Tensor"
    # Golden references
    per_expert_tokens_goldens: list
    activation_goldens: list
    e_t_goldens: list
    matmul_goldens: "torch.Tensor"
    combine_goldens: tuple
    # Quantized bfloat4_b weights as host tensors; consumed each run via ttnn.to_device.
    # Populated after the (device-independent) golden build, since quantization needs a device.
    w0_w1_host: object = None
    w2_host: object = None


@functools.lru_cache(maxsize=1)
@torch.no_grad()
def _build_moe_host_data(
    mesh_shape,
    cluster_axis,
    experts_per_device,
    tokens_per_device,
    selected_experts_k,
    num_layers,
    N,
    hidden_size,
    dtype,
    activation_type,
    has_bias,
):
    """Build all device-independent inputs and golden references for the test.

    Deterministic given the fixed seeds below, so the result is safe to memoize
    (``lru_cache`` above) and reuse across runs that share the same shape. The op-run
    knobs that don't change the data (enable_trace, num_iterations, topology, num_links,
    output shard dims) are intentionally NOT parameters, so trace/non-trace runs of the
    same shape share an entry. ``maxsize=1`` keeps at most one (large) shape's artifacts
    alive: a new shape evicts the previous one.

    The matmul ring size is auto-detected from the live DRAM-bank count (12 on WH,
    7/8 on BH) by the weight-prep helpers, so it is fixed per device and not part of the cache key.
    """
    torch.manual_seed(2003)
    random.seed(2003)

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices
    num_replicated_devices = num_devices // num_dispatch_devices
    total_tokens = tokens_per_device * num_dispatch_devices
    experts = experts_per_device * mesh_shape[cluster_axis]
    experts_per_cluster = experts // num_replicated_devices
    experts_per_device = experts // num_devices

    #### Expert mapping - per-device [num_devices, experts], replicated on every device ###
    expert_mapping = gen_expert_mapping(
        num_devices, num_replicated_devices, cluster_axis, experts, experts_per_cluster, experts_per_device
    )

    sparse_buffers = []
    expert_indices_list = []
    expert_scores_list = []
    per_expert_tokens_goldens = []
    activation_goldens = []
    e_t_goldens = []
    # save the original dense tokens to create matmul goldens
    tilize_golden_layer_outputs = []

    logger.debug(f"Creating goldens and input tensors")

    for layer_id in range(num_layers):
        # Generate test data
        sparse_buffer, expert_indices, expert_scores, _ = gen_sparse_buffer_and_indices(
            tokens_per_device,
            hidden_size,
            experts,
            selected_experts_k,
            mesh_shape,
            cluster_axis,
            dtype=tt_to_torch_dtype(dtype),
        )

        # Compute goldens
        tilize_golden_output, expert_token_counts = compute_selective_tilize_golden(
            sparse_buffer, expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
        )
        logger.debug(f"  expert_token_counts:\n{expert_token_counts}")
        per_expert_tokens_goldens.append(expert_token_counts)
        tilize_golden_layer_outputs.append(tilize_golden_output)

        golden_activation, _ = compute_expert_activation_golden(
            expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
        )
        for d in range(num_devices):
            logger.debug(f"  Device {d} activated tokens: {len(golden_activation[d])}")
        activation_goldens.append(golden_activation)

        golden_e_t, _ = compute_e_t_golden(expert_indices, expert_mapping, mesh_shape, cluster_axis)
        e_t_goldens.append(golden_e_t)

        sparse_buffers.append(sparse_buffer)
        expert_indices_list.append(expert_indices)
        expert_scores_list.append(expert_scores)

    # (L, D, E/D, T, H)
    tilize_golden_outputs = torch.stack(tilize_golden_layer_outputs)
    del tilize_golden_layer_outputs

    logger.debug(f"Done creating tilize goldens and input tensors")

    logger.debug(f"Creating matmul goldens and input tensors")

    torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, N)
    torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, N)
    torch_w2 = create_torch_w2(num_layers, experts_per_device, N, hidden_size)

    # Create bias tensors for validation.
    # The packed bias tile is 32 rows stored in Bfp4_b format, with only row 0
    # populated and the remaining rows zero. The kernel applies bias via
    # matmul(ones(32,32), bias(32,N)), which reproduces the row-0 bias values for
    # each column directly; no extra sum(dim=2)-style adjustment is needed in the
    # golden for this mechanism.
    #
    # Use a small zero-mean normal distribution (float32 draw, cast to bf16) so each
    # element in the tile varies — closer to real expert biases than a single constant.
    # The fixed torch.manual_seed(2003) above makes the draws reproducible.
    #
    # Biases are identical per-device (same as weights which use ReplicateTensorToMesh).
    # The golden's .repeat([1, devices, 1, 1]) in compute_matmul_golden is correct
    # under this assumption.
    torch_b0 = torch_b1 = torch_b2 = None
    if has_bias:
        _bias_std = 0.12
        # True PyTorch bias format: (L, E, N) without tile padding.
        # The _prepare functions will convert to kernel tile format as needed.
        torch_b0 = (torch.randn(num_layers, experts_per_device, N, dtype=torch.float32) * _bias_std).to(torch.bfloat16)
        torch_b1 = (torch.randn(num_layers, experts_per_device, N, dtype=torch.float32) * _bias_std).to(torch.bfloat16)
        torch_b2 = (torch.randn(num_layers, experts_per_device, hidden_size, dtype=torch.float32) * _bias_std).to(
            torch.bfloat16
        )

    # now we can create our golden reference
    # (L, D, E/D, T, H) (block sparse)
    matmul_goldens = compute_matmul_golden(
        tilize_golden_outputs,
        torch_w0,
        torch_w1,
        torch_w2,
        num_layers,
        experts,
        num_devices,
        tokens_per_device,
        hidden_size,
        torch_b0=torch_b0,
        torch_b1=torch_b1,
        torch_b2=torch_b2,
        activation_type=activation_type,
    )

    # compute goldens for combine
    combine_goldens = compute_combine_golden(
        num_layers,
        experts,
        total_tokens,
        hidden_size,
        selected_experts_k,
        mesh_shape,
        matmul_goldens,
        activation_goldens,
        cluster_axis,
    )

    host_data = _MoEHostData(
        sparse_buffers=sparse_buffers,
        expert_indices_list=expert_indices_list,
        expert_scores_list=expert_scores_list,
        expert_mapping=expert_mapping,
        per_expert_tokens_goldens=per_expert_tokens_goldens,
        activation_goldens=activation_goldens,
        e_t_goldens=e_t_goldens,
        matmul_goldens=matmul_goldens,
        combine_goldens=combine_goldens,
    )
    # Raw weights are returned separately: they feed the on-device quantization
    # (which needs a live device, so it happens in _run_moe_compute_impl) and are
    # then discarded — only the resulting host tensors are cached.
    weight_inputs = _MoEWeightInputs(
        torch_w0=torch_w0,
        torch_w1=torch_w1,
        torch_w2=torch_w2,
        torch_b0=torch_b0,
        torch_b1=torch_b1,
        torch_b2=torch_b2,
    )
    return host_data, weight_inputs


def _build_quantized_weight_host_tensors(
    mesh_device, weight_inputs, num_layers, experts_per_device, hidden_size, N, has_bias
):
    """Upload raw weights, run the on-device prepare/interleave, then quantize to
    ``bfloat4_b`` via host — returning the result as **host** tensors (memory_config
    omitted so quantize_weights_via_host skips the re-upload). These persist across
    mesh_device teardown and are landed on each invocation's device via to_device.
    """

    logger.debug(f"Building weights for {hidden_size=} {N=} {experts_per_device=} {has_bias=}")

    def _upload_raw(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_w0_raw = _upload_raw(weight_inputs.torch_w0)
    tt_w1_raw = _upload_raw(weight_inputs.torch_w1)
    tt_w2_raw = _upload_raw(weight_inputs.torch_w2)
    if has_bias:
        tt_b0_raw = _upload_raw(weight_inputs.torch_b0)
        tt_b1_raw = _upload_raw(weight_inputs.torch_b1)
        tt_b2_raw = _upload_raw(weight_inputs.torch_b2)

    # ------------------------------------------------------------------------
    # Prepare w0_w1 tensor (interleaved, padded, and reordered) on device.
    if has_bias:
        tt_w0_w1_prepped = ttnn.experimental.prepare_w0_w1_tensor_with_bias(
            tt_w0_raw,
            tt_w1_raw,
            tt_b0_raw,
            tt_b1_raw,
            L=num_layers,
            E=experts_per_device,
            K=hidden_size,
            N=N,
        )
        ttnn.deallocate(tt_b0_raw)
        ttnn.deallocate(tt_b1_raw)
    else:
        tt_w0_w1_prepped = ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute(
            tt_w0_raw,
            tt_w1_raw,
            L=num_layers,
            E=experts_per_device,
            K=hidden_size,
            N=N,
        )
    ttnn.deallocate(tt_w0_raw)
    ttnn.deallocate(tt_w1_raw)

    # memory_config=None -> quantize_weights_via_host returns a host tensor.
    w0_w1_host = ttnn.experimental.quantize_weights_via_host(tt_w0_w1_prepped, dtype=ttnn.bfloat4_b, memory_config=None)
    ttnn.deallocate(tt_w0_w1_prepped)

    # ------------------------------------------------------------------------
    # Prepare w2 tensor (padded and reordered) on device.
    if has_bias:
        tt_w2_prepped = ttnn.experimental.prepare_w2_tensor_with_bias(
            tt_w2_raw,
            tt_b2_raw,
            L=num_layers,
            E=experts_per_device,
            N=N,
            K=hidden_size,
        )
        ttnn.deallocate(tt_b2_raw)
    else:
        tt_w2_prepped = ttnn.experimental.prepare_w2_tensor_for_moe_compute(
            tt_w2_raw,
            L=num_layers,
            E=experts_per_device,
            N=N,
            K=hidden_size,
        )
    ttnn.deallocate(tt_w2_raw)

    w2_host = ttnn.experimental.quantize_weights_via_host(tt_w2_prepped, dtype=ttnn.bfloat4_b, memory_config=None)
    ttnn.deallocate(tt_w2_prepped)

    return w0_w1_host, w2_host
