# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import math
import os
import pytest
import random
import torch
import ttnn


MESH_GRAPH_DESC_1x16 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x16_torus_graph_descriptor.textproto"
)


def is_mesh_graph_descriptor_set(expected_path):
    """Check if TT_MESH_GRAPH_DESC_PATH is set to the expected path."""
    return os.environ.get("TT_MESH_GRAPH_DESC_PATH") == expected_path


def validate_per_expert_tokens(
    mesh_device, experts_per_device, num_devices, per_expert_total_tokens_output_tensor, expert_token_counts
):
    logger.info(f"\n========== Per Expert Total Tokens Tensor Validation ==========")
    per_expert_tokens_all_passed = True

    # L1 alignment constant (16 bytes)
    l1_alignment = 16

    # Validate shape: [num_devices, aligned_row_elements]
    # Row is experts_per_device uint32s, aligned to 16 bytes
    per_expert_row_bytes = ((experts_per_device * 4 + l1_alignment - 1) // l1_alignment) * l1_alignment
    per_expert_row_elements = per_expert_row_bytes // 4
    expected_per_expert_shape = (num_devices, per_expert_row_elements)

    # Convert per_expert_total_tokens tensor to torch
    # Shape per device: [1, aligned_elements] as uint32
    per_expert_total_tokens_torch = ttnn.to_torch(
        per_expert_total_tokens_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    logger.info(f"Per expert total tokens torch shape: {per_expert_total_tokens_torch.shape}")

    assert per_expert_total_tokens_torch.shape == expected_per_expert_shape, (
        f"per_expert_total_tokens shape mismatch: expected {expected_per_expert_shape}, "
        f"got {per_expert_total_tokens_torch.shape}"
    )

    for device_idx in range(num_devices):
        device_counts = per_expert_total_tokens_torch[device_idx].flatten()

        for local_exp_idx in range(experts_per_device):
            expected_count = expert_token_counts[device_idx, local_exp_idx].item()
            actual_count = device_counts[local_exp_idx].item()

            if actual_count != expected_count:
                logger.warning(
                    f"  Device {device_idx}, Expert {local_exp_idx}: "
                    f"count mismatch - expected {expected_count}, got {actual_count}"
                )
                per_expert_tokens_all_passed = False
            else:
                logger.info(f"  Device {device_idx}, Expert {local_exp_idx}: count={actual_count} PASSED")

    return per_expert_tokens_all_passed


def validate_activation(
    mesh_device, experts_per_device, num_devices, expert_activation_output_tensor, golden_activation
):
    logger.info(f"\n========== Expert Activation Tensor Validation ==========")
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
    logger.info(f"Expert activation torch shape: {expert_activation_torch.shape}")

    for device_idx in range(num_devices):
        golden_rows = golden_activation[device_idx]
        num_expected_rows = len(golden_rows)

        # Extract this device's activation data
        # The tensor is flattened, so we need to parse rows
        device_activation = expert_activation_torch[device_idx].flatten().to(torch.int64)
        max_rows = len(device_activation) // aligned_row_elements

        logger.info(
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

        # Validate sentinel row (token_id = -1 = 0xFFFFFFFF as uint32)
        sentinel_row_idx = num_expected_rows
        if sentinel_row_idx >= max_rows:
            logger.warning(f"  Device {device_idx}: sentinel row {sentinel_row_idx} out of bounds")
            activation_all_passed = False
        else:
            sentinel_row_start = sentinel_row_idx * aligned_row_elements
            sentinel_token_id = device_activation[sentinel_row_start].item()
            # -1 as uint32 (0xFFFFFFFF) becomes -1 when sign-extended to int64
            is_sentinel = (sentinel_token_id == -1) or (sentinel_token_id == 0xFFFFFFFF)

            if not is_sentinel:
                logger.warning(
                    f"  Device {device_idx}: sentinel row token_id mismatch - " f"expected -1, got {sentinel_token_id}"
                )
                activation_all_passed = False
            else:
                logger.info(f"  Device {device_idx}: {num_expected_rows} tokens validated, sentinel PASSED")

    return activation_all_passed


def validate_e_t(mesh_device, total_tokens, experts_per_device, num_devices, e_t_output_tensor, golden_e_t):
    logger.info(f"\n========== E-T (Expert-to-Token) Tensor Validation ==========")
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
    logger.info(f"E-T torch shape: {e_t_torch.shape}")

    if e_t_torch.shape != expected_e_t_shape:
        logger.info(f"e_t shape mismatch: expected {expected_e_t_shape}, got {e_t_torch.shape}")
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
                    logger.info(
                        f"  Device {device_idx}, Expert {local_exp_idx}: "
                        f"{num_expected_tokens} tokens validated, sentinel PASSED"
                    )
            else:
                logger.warning(
                    f"  Device {device_idx}, Expert {local_exp_idx}: " f"sentinel index {sentinel_idx} out of bounds"
                )
                e_t_all_passed = False

    return e_t_all_passed


def prepare_output_tensor_from_combine_writer(
    raw_torch_output,
    active_token_counts,
    all_core_range_set,
    output_shard_cores,
    output_shard_height_dim,
    output_shard_width_dim,
    experts_per_device,
    total_tokens,
    hidden,
):
    # python in doesn't work as expected with list of CoreCoord
    def _output_shard_contains(core):
        for c in output_shard_cores:
            if core.x == c.x and core.y == c.y:
                return True
        return False

    torch.set_printoptions(profile="full")
    output_shards = []
    for i, c in enumerate(ttnn.corerange_to_cores(all_core_range_set, row_wise=True)):
        if _output_shard_contains(c):
            output_shards.append(raw_torch_output[i, :, :, :])

    output_core_shards = torch.stack(output_shards)

    output_shape = (
        output_shard_height_dim,
        output_shard_width_dim,
        experts_per_device,
        total_tokens // output_shard_height_dim,
        hidden // output_shard_width_dim,
    )

    shaped_torch_output = output_core_shards.view(output_shape)

    shaped_torch_output = shaped_torch_output.permute([2, 0, 3, 1, 4]).reshape(
        [experts_per_device, total_tokens, hidden]
    )
    torch_output = torch.zeros([experts_per_device, total_tokens, hidden], dtype=torch.bfloat16)

    for e in range(experts_per_device):
        active_tokens = active_token_counts[e]
        tokens_per_shard = math.ceil(active_tokens / output_shard_height_dim)
        for t in range(active_tokens):
            bt = t // tokens_per_shard
            ot = t % tokens_per_shard

            contrib = shaped_torch_output[e, bt * total_tokens // output_shard_height_dim + ot]

            torch_output[e, t] = contrib

    return torch_output


def validate_matmul(
    layer_id,
    experts_per_device,
    all_core_range_set,
    output_shard_cores,
    output_shard_height_dim,
    output_shard_width_dim,
    total_tokens,
    hidden,
    expert_token_counts,
    torch_output_ref,
    tt_output_tensor,
    mesh_device,
):
    logger.info(f"\n========== Matmul Output Tensor Validation ==========")

    devices = math.prod(mesh_device.shape)

    raw_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # (D * all cores, E/D, T/D, H) -> (D, all cores, E/D, T/D, H)
    # note this shape does not yet match the layout of the underlying data.
    raw_shape = raw_output.shape
    raw_output = raw_output.reshape([devices, raw_shape[0] // devices] + raw_shape[1:])

    reshape_func = functools.partial(
        prepare_output_tensor_from_combine_writer,
        active_token_counts=expert_token_counts,
        all_core_range_set=all_core_range_set,
        output_shard_cores=output_shard_cores,
        output_shard_height_dim=output_shard_height_dim,
        output_shard_width_dim=output_shard_width_dim,
        experts_per_device=experts_per_device,
        total_tokens=total_tokens,
        hidden=hidden,
    )

    # (D, E/devices, T, H)
    reshaped_device_outputs = torch.stack([reshape_func(raw_output[d]) for d in range(devices)])

    matmul_all_passed = True

    MATMUL_PCC_THRESHOLD = 0.988
    for d in range(devices):
        for expert_id in range(experts_per_device):
            # torch_output_ref is (L, D, E/D, T, H)
            torch_layer_output = torch_output_ref[layer_id, d, expert_id, :, :]
            tt_layer_output = tt_to_torch_outputs[d, expert_id, :, :]

            _pcc_passed, pcc_val = comp_pcc(torch_layer_output, tt_layer_output)
            std = torch_layer_output.std().item()
            relative_rmse_val = (
                (torch.nn.functional.mse_loss(torch_layer_output, tt_layer_output).sqrt().item() / std)
                if std != 0
                else 0.0
            )
            allclose_passed, allclose_val = comp_allclose(torch_layer_output, tt_layer_output)

            if pcc < MATMUL_PCC_THRESHOLD:
                matmul_all_passed = False
                logger.warning(f"Layer {layer_id}, Expert {expert_id}: PCC={pcc:.6f}")
            else:
                logger.info(f"Layer {layer_id}, Expert {expert_id}: PCC={pcc:.6f} (Passed)")

    return matmul_all_passed


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
    """
    # torch_w0 = torch.empty((L, E, K, N), dtype=torch.bfloat16)
    # le_val = 1
    # for l in range(L):
    #     for e in range(E):
    #         for k_chunk in range(K // 32):
    #             k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
    #             k_val = k_chunk * 0.001
    #             for n_chunk in range(N // 32):
    #                 n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
    #                 n_val = n_chunk
    #                 torch_w0[l, e, k_start:k_end, n_start:n_end] = (n_val + k_val) * le_val
    #         le_val *= -1

    torch_w0 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
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
    """
    # torch_w1 = torch.empty((L, E, K, N), dtype=torch.bfloat16)
    # le_val = -1
    # for l in range(L):
    #     for e in range(E):
    #         for k_chunk in range(K // 32):
    #             k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
    #             k_val = k_chunk * 0.001
    #             for n_chunk in range(N // 32):
    #                 n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
    #                 n_val = n_chunk
    #                 torch_w1[l, e, k_start:k_end, n_start:n_end] = (n_val + k_val) * le_val
    #         le_val *= -1

    torch_w1 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
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
    """
    # torch_w2 = torch.empty((L, E, N, K), dtype=torch.bfloat16)
    # le_val = 1
    # for l in range(L):
    #     for e in range(E):
    #         for n_chunk in range(N // 32):
    #             n_start, n_end = n_chunk * 32, n_chunk * 32 + 32
    #             n_val = 0.001 * n_chunk
    #             for k_chunk in range(K // 32):
    #                 k_start, k_end = k_chunk * 32, k_chunk * 32 + 32
    #                 k_val = k_chunk
    #                 torch_w2[l, e, n_start:n_end, k_start:k_end] = (n_val + k_val) * le_val
    #         le_val *= -1
    torch_w2 = torch.rand((L, E, N, K), dtype=torch.bfloat16) - 0.5
    return torch_w2


def prepare_w0_w1_tensor(torch_w0, torch_w1, L, E, K, N, ring2cores):
    """
    Prepare the w0_w1 tensor by interleaving chunks of w0 and w1 width-wise.

    Args:
        torch_w0: Weight tensor of shape (L, E, K, N)
        torch_w1: Weight tensor of shape (L, E, K, N)
        L: Number of layers
        E: Number of experts
        K: Input dimension
        N: Output dimension
        ring2cores: Dictionary mapping ring position to (core_coord, dram_bank_id, pad_flag)

    Returns:
        torch_w0_w1_interleaved: Interleaved tensor of shape (L, E, K, 4096)
    """
    Nt = N // ttnn.TILE_SIZE  # 2048 / 32 = 64 chunks per tensor

    # Reshape to expose chunks: (L, E, K, N) -> (L, E, K, Nt, ttnn.TILE_SIZE)
    w0_chunks = torch_w0.view(L, E, K, Nt, ttnn.TILE_SIZE)
    w1_chunks = torch_w1.view(L, E, K, Nt, ttnn.TILE_SIZE)

    # Stack w0 and w1 chunks together: (L, E, K, Nt, 2, ttnn.TILE_SIZE)
    # This puts w0_chunk_i and w1_chunk_i adjacent to each other
    stacked = torch.stack([w0_chunks, w1_chunks], dim=4)

    # Reshape to interleave: (L, E, K, Nt * 2 * ttnn.TILE_SIZE) = (L, E, K, 4096)
    # The order will be: w0_chunk_0, w1_chunk_0, w0_chunk_1, w1_chunk_1, ...
    torch_w0_w1_interleaved = stacked.view(L, E, K, Nt, 2 * ttnn.TILE_SIZE)

    # Permute to move Nt before K: (L, E, K, Nt, 2*TILE) -> (L, E, Nt, K, 2*TILE)
    torch_w0_w1_permuted = torch_w0_w1_interleaved.permute(0, 1, 3, 2, 4)

    each_shard = []

    # Pick appropriate number of column tiles for each core based on the ring position.
    start_tile = 0
    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        num_tiles = 5 if pad_flag else 6
        each_shard.append(torch_w0_w1_permuted[:, :, start_tile : start_tile + num_tiles, :, :])

        if pad_flag:
            each_shard.append(torch.zeros(L, E, 1, K, 2 * ttnn.TILE_SIZE, dtype=torch_w0_w1_permuted.dtype))
        start_tile += num_tiles

    torch_w0_w1_reordered = torch.cat(each_shard, dim=2)  # (L, E, 5 * 8 + 1 * 8 + 6 * 4, K, 64)
    all_groups_per_bank = torch_w0_w1_reordered.view(L, E, 12, -1, K, 2 * ttnn.TILE_SIZE)  # (L, E, 12, 6, K, 64)
    all_groups_per_bank = all_groups_per_bank.permute(2, 0, 1, 3, 4, 5)  # (12, L, E, 6, K, 64)

    # Let us further make the 6 as 3 and 64 as 128.
    torch_w0_w1_pair_2_tiles = all_groups_per_bank.view(12, L, E, 3, -1, K, 2 * ttnn.TILE_SIZE)
    # (12, L, E, 3, 2, K, 64) -> (12, L, E, 3, K, 2, 64)
    torch_w0_w1_pair_2_tiles = torch_w0_w1_pair_2_tiles.permute(0, 1, 2, 3, 5, 4, 6)
    torch_w0_w1_paired = torch_w0_w1_pair_2_tiles.reshape(12, L, E, 3, -1, 4 * ttnn.TILE_SIZE)

    return torch_w0_w1_paired


def prepare_w2_tensor(torch_w2, L, E, N, K, ring2cores):
    """
    Prepare the w2 tensor by padding and reordering tiles.

    Args:
        torch_w2: Weight tensor of shape (L, E, N, K)
        L: Number of layers
        E: Number of experts
        N: Intermediate dimension
        K: Output dimension
        ring2cores: Dictionary mapping ring position to (core_coord, dram_bank_id, pad_flag)

    Returns:
        torch_w2_reordered: Reordered tensor of shape (L, E, N_padded, 7680)
    """
    # Separate the tensor into 4 groups of 4 * 32 tiles and then 1 group of 2/3 * 32 tiles.
    each_shard = []

    start_col = 0
    for ring_pos in range(len(ring2cores)):
        (_, _, pad_flag) = ring2cores[ring_pos]
        last_group_tiles = 3 if pad_flag else 2
        last_group_pad_tiles = 1 if pad_flag else 2

        # Get the first 4 groups of 4 * 32 tiles.
        each_shard.append(torch_w2[:, :, :, start_col : start_col + 4 * 4 * ttnn.TILE_SIZE])
        start_col += 4 * 4 * ttnn.TILE_SIZE
        each_shard.append(torch_w2[:, :, :, start_col : start_col + last_group_tiles * ttnn.TILE_SIZE])
        start_col += last_group_tiles * ttnn.TILE_SIZE

        # Add padding for the last group.
        each_shard.append(torch.zeros(L, E, N, last_group_pad_tiles * ttnn.TILE_SIZE, dtype=torch_w2.dtype))

    torch_w2_reordered = torch.cat(each_shard, dim=-1)  # (L, E, N, 12 * (4 * 4 * 32 + 4 * 32))
    all_groups_per_bank = torch_w2_reordered.view(L, E, N, 12, -1, 4 * ttnn.TILE_SIZE)

    # (L, E, N, 12, 5, 128) -> (12, L, E, 5, N, 128)
    all_groups_per_bank = all_groups_per_bank.permute(3, 0, 1, 4, 2, 5)

    # Group N in terms of tiles first
    N_grouped = all_groups_per_bank.view(
        12, L, E, 5, -1, ttnn.TILE_SIZE, 4 * ttnn.TILE_SIZE
    )  # (12, L, E, 5, 64, 32, 128)

    # Figure out the order of N tiles based on the ring position.
    core_chunk_order = torch.tensor(list(reversed(range(len(ring2cores))))).roll(1)

    # Figure out the starting position for each chunk
    chunk_sizes = [5 if ring2cores[ring_pos][2] else 6 for ring_pos in range(len(ring2cores))]
    chunk_start_positions = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(chunk_sizes, dtype=torch.int32), dim=0)]
    )

    each_shard = []
    # Assemble the number of such N tiles based on the ring position.
    for core_id in range(len(ring2cores)):
        each_chunk = []
        for chunk_id in core_chunk_order:
            start_pos = chunk_start_positions[chunk_id]
            end_pos = chunk_start_positions[chunk_id + 1]
            this_chunk = N_grouped[core_id, :, :, :, start_pos:end_pos, :, :]
            each_chunk.append(this_chunk)
        each_shard.append(torch.cat(each_chunk, dim=3))

        core_chunk_order = core_chunk_order.roll(1)

    N_reordered = torch.stack(each_shard).view(12, L, E, 5, -1, 4 * ttnn.TILE_SIZE)

    # Pad "N" dimension to make it divisible by 7 tiles, since we read 7 tiles at a time.
    Nt = N // ttnn.TILE_SIZE  # 2048 / 32 = 64 chunks per tensor
    N_padding = math.ceil(Nt / 7) * 7 * ttnn.TILE_SIZE - N
    padding = torch.zeros(12, L, E, 5, N_padding, 4 * ttnn.TILE_SIZE, dtype=torch_w2.dtype)
    all_groups_per_bank = torch.cat([N_reordered, padding], dim=4)  # (12, L, E, 5, N + 192, 128)
    return all_groups_per_bank


def tt_to_torch_dtype(tt_dtype):
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    elif tt_dtype == ttnn.bfloat8_b:
        return torch.bfloat16
    elif tt_dtype == ttnn.float32:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {tt_dtype}")


def gen_expert_mapping(experts, mesh_shape, cluster_axis):
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
    num_devices = mesh_shape[0] * mesh_shape[1]
    experts_per_device = experts // num_devices
    expert_mapping = torch.zeros(1, experts, dtype=torch.uint16)
    for e in range(experts):
        expert_mapping[0, e] = e // experts_per_device
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
    original_tokens = torch.rand(num_dispatch_devices, tokens_per_device, hidden_size, dtype=dtype)

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


def compute_matmul_golden(
    torch_input_ref, torch_w0, torch_w1, torch_w2, layers, experts, devices, tokens_per_device, hidden
):
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

    # Compute gate activations for each expert
    # (L, E, T, K) @ (L, E, K, N) -> (L, E, T, N)
    torch_w0_output_ref = torch_input_ref @ torch_w0
    torch_silu_output_ref = torch.nn.functional.silu(torch_w0_output_ref)
    # (L, E, T, K) @ (L, E, K, N) -> (L, E, T, N)
    torch_w1_output_ref = torch_input_ref @ torch_w1
    torch_intermediate_ref = torch_silu_output_ref * torch_w1_output_ref  # (L, E, T, N)

    # (L, E, T, N) @ (L, E, N, K) -> (L, E, T, K)
    torch_output_ref = torch_intermediate_ref @ torch_w2

    # pull device dim back out for comparison
    # (L, E, T, H) -> (L, D, E/D, T, H)
    return torch_output_ref.reshape(layers, devices, experts // devices, tokens, hidden)


def create_sharded_memory_config(core_range_set, tensor_shape, dtype):
    """
    Create an L1 sharded memory config for a tensor to be completely on specified cores.
    """
    num_cores = core_range_set.num_cores()
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

    total_bytes = total_elements * bytes_per_element
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


# Requires TT_MESH_GRAPH_DESC_PATH to be set to the 1x16 mesh descriptor before running
@pytest.mark.skipif(
    not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x16),
    reason=f"Requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_1x16}",
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((1, 16), (1, 16), id="1x16_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("tokens_per_device", [32])  # Collapsed batch * seq_len
@pytest.mark.parametrize("experts", [2 * 16])  # 32 experts for 16 devices = 2 experts per device
@pytest.mark.parametrize(
    "selected_experts_k, num_layers, num_iterations",
    [(1, 1, 1)]
    #    "selected_experts_k, num_layers, num_iterations", [(1, 1, 1), (8, 5, 1)], ids=["perf", "accuracy"]
)
@pytest.mark.parametrize("N, hidden_size", [(2048, 7168)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("enable_trace", [True, False])
@pytest.mark.parametrize("output_shard_height_dim", [4])
@pytest.mark.parametrize("output_shard_width_dim", [4])
def test_moe_compute(
    mesh_device,
    mesh_shape,
    cluster_axis,
    tokens_per_device,
    experts,
    selected_experts_k,
    num_layers,
    num_iterations,
    N,
    hidden_size,
    output_shard_height_dim,
    output_shard_width_dim,
    dtype,
    enable_trace,
    device_params,
):
    """
    This test:
    1. Generates a sparse buffer (simulating output from all_to_all_dispatch)
    2. Generates all-gathered expert indices and scores
    3. Generates per-device expert mapping
    4. Runs the moe operation
    5. Verifies the outputs against a golden reference
    """
    torch.manual_seed(2005)
    random.seed(2005)

    #########################################
    # TEST SETUP
    #########################################

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices
    total_tokens = tokens_per_device * num_dispatch_devices
    experts_per_device = experts // num_devices

    logger.info(f"Test configuration:")
    logger.info(f"  mesh_shape: {mesh_shape}")
    logger.info(f"  cluster_axis: {cluster_axis}")
    logger.info(f"  num_devices: {num_devices}, num_dispatch_devices: {num_dispatch_devices}")
    logger.info(f"  tokens_per_device: {tokens_per_device}, total_tokens: {total_tokens}")
    logger.info(
        f"  experts: {experts}, selected_experts_k: {selected_experts_k}, experts_per_device: {experts_per_device}"
    )
    logger.info(f"  hidden_size: {hidden_size}")
    logger.info(f"  dtype: {dtype}")

    #########################################
    # CREATE TILIZE INPUT TENSORS AND GOLDENS
    #########################################

    # Drain tilize core is core (6,9) where indices and scores are sharded
    tilize_drain_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9))})

    #### Expert mapping - per-device [num_devices, experts], replicated on every device ###
    # Each device gets its own row after sharding, but since it's replicated,
    # we give each device the full tensor and it uses its own row.
    # Expert mapping is constant across all runs.
    expert_mapping = gen_expert_mapping(experts, mesh_shape, cluster_axis)
    expert_mapping_mem_config = ttnn.L1_MEMORY_CONFIG
    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=expert_mapping_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Sparse memory config
    sparse_mem_config = ttnn.L1_MEMORY_CONFIG

    # Create L1 sharded memory config for indices on drain tilizer core
    # expert_indices shape per device: [tokens_per_device, selected_experts_k] (after shard along dispatch axis)
    # But we need the all-gathered version, so shape is [num_dispatch_devices * tokens_per_device, selected_experts_k]
    # which is [total_tokens, selected_experts_k]
    expert_indices_shard_shape = [total_tokens, selected_experts_k]
    expert_indices_mem_config = create_sharded_memory_config(tilize_drain_core, expert_indices_shard_shape, ttnn.uint16)

    # Create L1 sharded memory config for indices and scores on drain tilizer core
    expert_scores_shard_shape = [total_tokens, selected_experts_k]
    expert_scores_mem_config = create_sharded_memory_config(tilize_drain_core, expert_scores_shard_shape, dtype)

    tt_sparse_buffers = []
    tt_expert_indices_buffers = []
    tt_expert_scores_buffers = []

    per_expert_tokens_goldens = []
    activation_goldens = []
    e_t_goldens = []

    # save the original dense token to create matmul goldens
    tilize_golden_layer_outputs = []

    logger.info(f"Creating goldens and input tensors")

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
        logger.info(f"  expert_token_counts:\n{expert_token_counts}")
        per_expert_tokens_goldens.append(expert_token_counts)
        tilize_golden_layer_outputs.append(tilize_golden_output)

        golden_activation, experts_per_device_check = compute_expert_activation_golden(
            expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
        )
        for d in range(num_devices):
            logger.info(f"  Device {d} activated tokens: {len(golden_activation[d])}")
        activation_goldens.append(golden_activation)

        golden_e_t, _ = compute_e_t_golden(expert_indices, expert_mapping, mesh_shape, cluster_axis)
        e_t_goldens.append(golden_e_t)

        # Create input tensors
        # NOTE:
        # - when running multiple layers we initially create tt_sparse_buffer, tt_expert_indices and tt_expert_scores in DRAM, we'll move to L1 before running moe_compute
        # - we're extremely tight on L1 for a single invocation of the op
        if num_layers == 1:
            init_sparse_mem_config = sparse_mem_config
            init_expert_indices_mem_config = expert_indices_mem_config
            init_expert_scores_mem_config = expert_scores_mem_config
        else:
            init_sparse_mem_config = ttnn.DRAM_MEMORY_CONFIG
            init_expert_indices_mem_config = ttnn.DRAM_MEMORY_CONFIG
            init_expert_scores_mem_config = ttnn.DRAM_MEMORY_CONFIG

        ### Sparse buffer is sharded across devices (dim 0) ###
        tt_sparse_buffer = ttnn.from_torch(
            sparse_buffer,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=init_sparse_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_sparse_buffers.append(tt_sparse_buffer)

        ### Expert indices - all-gathered (replicated on all devices) ###
        # Shape: [num_dispatch_devices, tokens_per_device, K]
        # Flatten to [num_dispatch_devices * tokens_per_device, K] = [total_tokens, K] per device
        # Replicate on all devices
        expert_indices_flat = expert_indices.reshape(total_tokens, selected_experts_k)
        expert_indices_replicated = expert_indices_flat.unsqueeze(0).repeat(num_devices, 1, 1)
        tt_expert_indices = ttnn.from_torch(
            expert_indices_replicated,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=init_expert_indices_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_expert_indices_buffers.append(tt_expert_indices)

        ### Expert scores - same distribution as indices ###
        expert_scores_flat = expert_scores.reshape(total_tokens, selected_experts_k)
        expert_scores_replicated = expert_scores_flat.unsqueeze(0).repeat(num_devices, 1, 1)
        tt_expert_scores = ttnn.from_torch(
            expert_scores_replicated,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=init_expert_scores_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_expert_scores_buffers.append(tt_expert_scores)
    # (L, D, E/D, T, H)
    tilize_golden_outputs = torch.stack(tilize_golden_layer_outputs)
    del tilize_golden_layer_outputs

    logger.info(f"Done creating goldens and input tensors")

    #########################################
    # CREATE MATMUL INPUT TENSORS
    #########################################
    logger.info(f"Creating matmul goldens and input tensors")

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    MATMUL_FULL_CORES = {0, 1, 8, 9}
    MATMUL_PAD_CORES = {2, 3, 4, 5, 6, 7, 10, 11}

    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, 0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

    # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        # key: ring_pos, value: (core_coord, dram_bank_id, pad_flag)
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in MATMUL_PAD_CORES else 0)

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, N)
    torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, N)
    torch_w2 = create_torch_w2(num_layers, experts_per_device, N, hidden_size)

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
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w0_w1
    # Tensor shape: (num_layers, experts_per_device, hidden_size, 4608) -> padded and reordered to (12, num_layers, experts_per_device, 6, hidden_size, 64)
    # ------------------------------------------------------------------------
    w0_w1_shard_height = num_layers * experts_per_device * 3 * hidden_size
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # Tensor shape: (num_layers, experts_per_device, N, hidden_size) -> padded and reordered to (12, num_layers, experts_per_device, 5, N + 192, 128)
    # ------------------------------------------------------------------------
    w2_shard_height = num_layers * experts_per_device * 5 * (N + 192)
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    # ------------------------------------------------------------------------
    # Prepare w0_w1 tensor (interleaved, padded, and reordered)
    torch_w0_w1_reordered = prepare_w0_w1_tensor(
        torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, ring2cores
    )

    # Create tt_w0_w1 tensor with DRAM sharding
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ------------------------------------------------------------------------
    # Prepare w2 tensor (padded and reordered)
    torch_w2_reordered = prepare_w2_tensor(torch_w2, num_layers, experts_per_device, N, hidden_size, ring2cores)

    # Create tt_w2 tensor with DRAM sharding
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    #########################################
    # RUN OP
    #########################################

    def run_op():
        moe_compute_outputs = []

        for layer_id in range(num_layers):
            # if only running a single layer, we can fit the single set of inputs in L1 initially
            # otherwise with multiple layers, and multiple sets of inputs, we need to move inputs into L1 before a given
            if num_layers == 1:
                tt_sparse_buffer = tt_sparse_buffers[0]
                tt_expert_indices = tt_expert_indices_buffers[0]
                tt_expert_scores = tt_expert_scores_buffers[0]
            else:
                tt_sparse_buffer = ttnn.to_memory_config(tt_sparse_buffers[layer_id], memory_config=sparse_mem_config)
                tt_expert_indices = ttnn.to_memory_config(
                    tt_expert_indices_buffers[layer_id], memory_config=expert_indices_mem_config
                )
                tt_expert_scores = ttnn.to_memory_config(
                    tt_expert_scores_buffers[layer_id], memory_config=expert_scores_mem_config
                )

            # run the op
            (
                l1_per_expert_total_tokens_output_tensor,
                l1_expert_activation_output_tensor,
                l1_e_t_output_tensor,
                _,  # tile layout output of selective tilize (same buffer as output)
                l1_output_tensor,
            ) = ttnn.experimental.moe_compute(
                tt_sparse_buffer,
                tt_expert_indices,
                tt_expert_scores,
                tt_expert_mapping,
                tt_w0_w1,
                tt_w2,
                layer_id=layer_id,
                cluster_axis=cluster_axis,
            )

            # deallocate L1 inputs
            # if running with multiple layers, we have to deallocate previous inputs to free up L1 space
            # we still have the DRAM version of the input tensor after deallocating the L1 version
            if num_layers != 1:
                ttnn.deallocate(tt_sparse_buffer)
                ttnn.deallocate(tt_expert_indices)
                ttnn.deallocate(tt_expert_scores)

            # convert outputs to DRAM (we don't have enough L1 space to leave outputs in L1 when running multiple invocations)
            dram_per_expert_total_tokens_output_tensor = ttnn.to_memory_config(
                l1_per_expert_total_tokens_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            dram_expert_activation_output_tensor = ttnn.to_memory_config(
                l1_expert_activation_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            dram_e_t_output_tensor = ttnn.to_memory_config(l1_e_t_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            dram_output_tensor = ttnn.to_memory_config(l1_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # # deallocate L1 outputs
            ttnn.deallocate(l1_per_expert_total_tokens_output_tensor)
            ttnn.deallocate(l1_expert_activation_output_tensor)
            ttnn.deallocate(l1_e_t_output_tensor)
            ttnn.deallocate(l1_output_tensor)

            # save outputs to verify later
            moe_compute_output = (
                dram_per_expert_total_tokens_output_tensor,
                dram_expert_activation_output_tensor,
                dram_e_t_output_tensor,
                dram_output_tensor,
            )
            moe_compute_outputs.append(moe_compute_output)

        return moe_compute_outputs

    logger.info(f"\n========== Running op ==========")
    moe_compute_outputs = []
    if enable_trace:
        # Compile the op
        for i in range(num_iterations):
            run_op()
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iterations):
            moe_compute_output = run_op()
            moe_compute_outputs.append(moe_compute_output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")
    else:
        for i in range(num_iterations):
            moe_compute_output = run_op()
            moe_compute_outputs.append(moe_compute_output)

    #########################################
    # VALIDATE OUTPUTS PER LAYER
    #########################################
    logger.info(f"\n========== Starting Validation ==========")

    all_core_grid = device.compute_with_storage_grid_size()
    all_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(all_core_grid.x - 1, all_core_grid.y - 1),
            ),
        }
    )

    output_shard_cores = ttnn.get_moe_combine_cores(mesh_device)

    per_expert_tokens_all_passed = True
    activation_all_passed = True
    e_t_all_passed = True
    matmul_all_passed = True
    for i in range(num_iterations):
        for layer_id in range(num_layers):
            (
                per_expert_total_tokens_output_tensor,
                expert_activation_output_tensor,
                e_t_output_tensor,
                output_tensor,
            ) = moe_compute_outputs[i][layer_id]

            logger.info(f"\n========== Iteration {i} Layer {layer_id} Validation ==========")
            logger.info(f"Per expert total tokens tensor shape: {per_expert_total_tokens_output_tensor.shape}")
            logger.info(f"Expert activation tensor shape: {expert_activation_output_tensor.shape}")
            logger.info(f"E-T (expert-to-token) tensor shape: {e_t_output_tensor.shape}")
            logger.info(f"Output tensor shape: {output_tensor.shape}")

            # ========== Per Expert Total Tokens Tensor Validation ==========
            expert_token_counts = per_expert_tokens_goldens[layer_id]
            if not validate_per_expert_tokens(
                mesh_device, experts_per_device, num_devices, per_expert_total_tokens_output_tensor, expert_token_counts
            ):
                per_expert_tokens_all_passed = False

            # ========== Expert Activation Tensor Validation ==========
            golden_activation = activation_goldens[layer_id]
            if not validate_activation(
                mesh_device, experts_per_device, num_devices, expert_activation_output_tensor, golden_activation
            ):
                activation_all_passed = False

            # ========== E-T (Expert-to-Token) Tensor Validation ==========
            golden_e_t = e_t_goldens[layer_id]
            if not validate_e_t(
                mesh_device, total_tokens, experts_per_device, num_devices, e_t_output_tensor, golden_e_t
            ):
                e_t_all_passed = False

            # ========== Matmul Output Tensor Validation ==========
            if not validate_matmul(
                layer_id,
                experts_per_device,
                all_core_range_set,
                output_shard_cores,
                output_shard_height_dim,
                output_shard_width_dim,
                tokens_per_device * devices,
                hidden_size,
                per_expert_total_tokens_output_tensor,
                matmul_goldens,
                output_tensor,
                mesh_device,
            ):
                matmul_all_passed = False

    # Asserts
    logger.info(f"\n========== Asserts ==========")
    logger.info(f"\nPer Expert Total Tokens Verification: {'PASSED' if per_expert_tokens_all_passed else 'FAILED'}")
    logger.info(f"\nExpert Activation Verification: {'PASSED' if activation_all_passed else 'FAILED'}")
    logger.info(f"\nE-T Tensor Verification: {'PASSED' if e_t_all_passed else 'FAILED'}")
    logger.info(f"\nMatmul Output Tensor Verification: {'PASSED' if matmul_all_passed else 'FAILED'}")

    assert per_expert_tokens_all_passed, "Per expert total tokens tensor verification failed!"
    assert activation_all_passed, "Expert activation tensor verification failed!"
    assert e_t_all_passed, "E-T tensor verification failed!"
    assert matmul_all_passed, "Matmul output tensor verification failed!"
