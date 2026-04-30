# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import functools
from loguru import logger
import math
import os
import pytest
import random
import torch
import ttnn
from ttnn.operations.ccl import MoEActivationFunction

from ttnn.experimental.moe_compute_utils import (
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w0_w1_tensor_with_bias,
    prepare_w2_tensor_for_moe_compute,
    prepare_w2_tensor_with_bias,
    DS_PAD_CORES,
    DS_W0_W1_SHARD_VALS,
    DS_W2_SHARD_VALS,
    GPT_PAD_CORES,
    GPT_W0_W1_SHARD_VALS,
    GPT_W2_SHARD_VALS,
    get_weight_core_shard_maps,
    get_weight_mem_configs,
)

from tests.nightly.tg.ccl.moe.test_selective_combine_6U import device_mesh_iterator
from tests.nightly.t3000.ccl.test_all_to_all_combine import get_batch_cluster_idxr, get_cluster_dims

from models.common.utility_functions import comp_pcc

MESH_GRAPH_DESC_1x16 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x16_torus_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_1x8 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x8_torus_graph_descriptor.textproto"
)

# TODO (AM) this should go in a central location
HIDDEN_TO_SHARD_INFO = {
    7168: (DS_PAD_CORES, DS_W0_W1_SHARD_VALS, DS_W2_SHARD_VALS),
    2880: (GPT_PAD_CORES, GPT_W0_W1_SHARD_VALS, GPT_W2_SHARD_VALS),
}


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

    # Validate shape: [num_devices, num_cores, aligned_row_elements]
    # Row is experts_per_device uint32s, aligned to 16 bytes. Replicated on every core
    per_expert_row_bytes = ((experts_per_device * 4 + l1_alignment - 1) // l1_alignment) * l1_alignment
    per_expert_row_elements = per_expert_row_bytes // 4
    # Note: the bounding box containing tilize, matmul, combine cores spans the whole grid.
    core_range = mesh_device.compute_with_storage_grid_size()
    num_cores = core_range.x * core_range.y
    expected_per_expert_shape = (num_devices * num_cores, per_expert_row_elements)

    # Convert per_expert_total_tokens tensor to torch
    # Shape per device: [num_cores (70), aligned_elements] as uint32
    per_expert_total_tokens_torch = ttnn.to_torch(
        per_expert_total_tokens_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )
    logger.info(f"Per expert total tokens torch shape: {per_expert_total_tokens_torch.shape}")

    assert per_expert_total_tokens_torch.shape == expected_per_expert_shape, (
        f"per_expert_total_tokens shape mismatch: expected {expected_per_expert_shape}, "
        f"got {per_expert_total_tokens_torch.shape}"
    )

    per_expert_total_tokens_torch = per_expert_total_tokens_torch.reshape(
        (num_devices, num_cores, per_expert_row_elements)
    )
    for device_idx in range(num_devices):
        for c in range(num_cores):
            device_counts = per_expert_total_tokens_torch[device_idx][c]

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
    device_idx,
    all_core_range_set,
    output_shard_cores,
    output_shard_height_dim,
    output_shard_width_dim,
    experts_per_device,
    hidden,
):
    all_output_shards = {}

    for i, c in enumerate(ttnn.corerange_to_cores(all_core_range_set, row_wise=True)):
        all_output_shards[(c.x, c.y)] = raw_torch_output[i]

    combine_output_shards = [all_output_shards[c.x, c.y] for c in output_shard_cores]
    output_shard_tensor = torch.stack(combine_output_shards)

    # Consistent with token_offset logic in program factories
    assert output_shard_tensor.numel() % (experts_per_device * hidden) == 0
    buffer_size_total_tokens = output_shard_tensor.numel() // (experts_per_device * hidden)

    output_shape = (
        output_shard_height_dim,
        output_shard_width_dim,
        experts_per_device,
        buffer_size_total_tokens // output_shard_height_dim,
        hidden // output_shard_width_dim,
    )

    shaped_torch_output = output_shard_tensor.view(output_shape)

    shaped_torch_output = shaped_torch_output.permute([2, 0, 3, 1, 4]).reshape(
        [experts_per_device, buffer_size_total_tokens, hidden]
    )
    torch_output = torch.zeros([experts_per_device, buffer_size_total_tokens, hidden], dtype=torch.bfloat16)

    for e in range(experts_per_device):
        active_tokens = active_token_counts[e].item()
        tokens_per_shard_chunk = active_tokens // output_shard_height_dim
        tokens_per_shard_rem = active_tokens % output_shard_height_dim

        output_token_shard = 0
        output_token_shard_row = 0
        for t in range(active_tokens):
            contrib = shaped_torch_output[
                e, output_token_shard * buffer_size_total_tokens // output_shard_height_dim + output_token_shard_row
            ]

            torch_output[e, t] = contrib

            if output_token_shard_row == (
                tokens_per_shard_chunk if output_token_shard < tokens_per_shard_rem else tokens_per_shard_chunk - 1
            ):
                output_token_shard += 1
                output_token_shard_row = 0
            else:
                output_token_shard_row += 1

    return torch_output


# Matmul with bias: LoFi + bf16/bfp4 on device can land just under PCC_THRESHOLD (e.g. ~0.987994 on one
# device/expert) while still tracking golden closely; combine PCC stays above 0.988.
PCC_THRESHOLD_MATMUL_WITH_BIAS = 0.98799
ATOL_THRESHOLD = 700
SWIGLU_PCC_THRESHOLD = 0.984
SILU_PCC_THRESHOLD = 0.988


def _get_base_pcc_threshold(activation_type, has_bias):
    # Determine PCC threshold based on activation type
    # Note: this threshold is applicable for checking a block of 32 tokens, smaller matrices will need a lower threshold
    # https://github.com/tenstorrent/tt-metal/blob/368efa1f7062704b8e885aa72dae115e91320032/tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py#L438
    act_threshold = None
    if activation_type == MoEActivationFunction.SWIGLU:
        act_threshold = SWIGLU_PCC_THRESHOLD
    elif activation_type == MoEActivationFunction.SILU:  # SILU
        act_threshold = SILU_PCC_THRESHOLD
    else:
        raise TypeError("Invalid Activation type")

    bias_threshold = None
    if has_bias:
        bias_threshold = PCC_THRESHOLD_MATMUL_WITH_BIAS
    else:
        bias_threshold = act_threshold

    return min(bias_threshold, act_threshold)


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
    base_pcc_threshold,
    *,
    has_bias: bool = False,
):
    logger.info(f"\n========== Matmul Output Tensor Validation ==========")

    devices = math.prod(mesh_device.shape)

    raw_output = ttnn.to_torch(tt_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # (D * all cores, E/D, T/D, H) -> (D, all cores, E/D, T/D, H)
    # note this shape does not yet match the layout of the underlying data.
    raw_shape = list(raw_output.shape)
    raw_output = raw_output.reshape([devices, raw_shape[0] // devices] + raw_shape[1:])

    reshape_func = functools.partial(
        prepare_output_tensor_from_combine_writer,
        all_core_range_set=all_core_range_set,
        output_shard_cores=output_shard_cores,
        output_shard_height_dim=output_shard_height_dim,
        output_shard_width_dim=output_shard_width_dim,
        experts_per_device=2,  # always 2 for double buffer
        hidden=hidden,
    )

    matmul_all_passed = True
    # Calculate which experts are still in the double buffer
    # Buffer toggles for each expert: 0->1->0->1...
    # So for N experts, the last 2 experts in the buffer are:
    # - If N is even: buffer 0 has expert N-2, buffer 1 has expert N-1
    # - If N is odd: buffer 0 has expert N-1, buffer 1 has expert N-2
    experts_to_check = []
    if experts_per_device == 1:
        experts_to_check = [(0, 0)]  # Only one expert in buffer 0
    elif experts_per_device == 2:
        experts_to_check = [(0, 0), (1, 1)]  # Expert 0 in buffer 0, expert 1 in buffer 1
    else:
        # For >2 experts, determine which 2 experts remain in the buffer
        if experts_per_device % 2 == 0:
            # Even number of experts
            experts_to_check = [(experts_per_device - 2, 0), (experts_per_device - 1, 1)]
        else:
            # Odd number of experts
            experts_to_check = [(experts_per_device - 1, 0), (experts_per_device - 2, 1)]

    logger.info(f"Checking experts in double buffer: {experts_to_check}")

    # Build buffer token counts based on which experts are actually in the buffer
    reshaped_device_outputs = []
    for d in range(devices):
        buffer_token_counts = torch.zeros(2, dtype=expert_token_counts[d].dtype)
        for expert_id, buffer_idx in experts_to_check:
            buffer_token_counts[buffer_idx] = expert_token_counts[d][expert_id]
        reshaped_device_outputs.append(reshape_func(raw_output[d], buffer_token_counts, d))
    reshaped_device_outputs = torch.stack(reshaped_device_outputs)

    for d in range(devices):
        for expert_id, buffer_idx in experts_to_check:
            active_tokens = expert_token_counts[d, expert_id].item()
            # torch_output_ref is (L, D, E/D, T, H)
            torch_layer_output = torch_output_ref[layer_id, d, expert_id, :active_tokens, :]
            # The buffer position determines where to read from in the output
            tt_layer_output = reshaped_device_outputs[d, buffer_idx, :active_tokens, :]

            _pcc_passed, pcc_val = comp_pcc(torch_layer_output, tt_layer_output)
            allclose_passed = torch.allclose(torch_layer_output, tt_layer_output, atol=ATOL_THRESHOLD)
            std = torch_layer_output.std().item()
            relative_rmse_val = (
                (torch.nn.functional.mse_loss(torch_layer_output, tt_layer_output).sqrt().item() / std)
                if std != 0
                else 0.0
            )

            # Base PCC threshold is valid for 32xhidden, for comparing smaller matrices, a looser threshold is valid
            pcc_threshold = base_pcc_threshold if active_tokens >= 16 else base_pcc_threshold - 0.001
            if pcc_val < pcc_threshold:
                matmul_all_passed = False

                logger.warning(
                    f"Layer {layer_id}, Expert {expert_id} (buffer {buffer_idx}): PCC={pcc_val:.6f} RMSE: {relative_rmse_val}"
                    f" Allclose passed: {allclose_passed}"
                )

                if not allclose_passed:
                    mask = (tt_layer_output - torch_layer_output).abs() > ATOL_THRESHOLD
                    logger.warning(
                        f"AllClose variation result: {tt_layer_output[mask]}, ref: {torch_layer_output[mask]} indices: {mask.nonzero(as_tuple=True)}"
                    )
            else:
                logger.info(
                    f"Layer {layer_id}, Expert {expert_id} (buffer {buffer_idx}): PCC={pcc_val:.6f} RMSE: {relative_rmse_val} (Passed)"
                    f" Allclose passed: {allclose_passed}"
                )

    return matmul_all_passed


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

    output_ref, output_data_map = combine_goldens

    assert torch_combine_out.shape == output_ref[0].shape
    combine_all_passed = True

    for k in range(torch_combine_out.shape[0]):
        vals, refs = [], []
        for t in range(torch_combine_out.shape[1]):
            if output_data_map[layer_id, k, t].item() == 1:
                vals.append(torch_combine_out[k, t, :])
                refs.append(output_ref[layer_id, k, t, :])

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
            logger.info(f"Combine, layer: {layer_id}, k: {k} PCC={pcc_val:.6f}, AllClose passed: {allclose_passed}")

    return combine_all_passed


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
        logger.info(f"[WEIGHT_INIT] w0: GOLDEN (constant 0.1) - mode={mode}")
    else:
        # Use random weights for w0
        torch_w0 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
        logger.info(f"[WEIGHT_INIT] w0: RANDOM - mode={mode}")

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
        logger.info(f"[WEIGHT_INIT] w1: GOLDEN (constant 0.1) - mode={mode}")
    else:
        # Use random weights for w1
        torch_w1 = torch.rand((L, E, K, N), dtype=torch.bfloat16) - 0.5
        logger.info(f"[WEIGHT_INIT] w1: RANDOM - mode={mode}")

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
        logger.info(f"[WEIGHT_INIT] w2: GOLDEN (constant 0.1) - mode={mode}")
    else:
        # Use random weights for w2
        torch_w2 = torch.rand((L, E, N, K), dtype=torch.bfloat16) - 0.5
        logger.info(f"[WEIGHT_INIT] w2: RANDOM - mode={mode}")

    return torch_w2


def tt_to_torch_dtype(tt_dtype):
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    elif tt_dtype == ttnn.bfloat8_b:
        return torch.bfloat16
    elif tt_dtype == ttnn.float32:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {tt_dtype}")


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
    if torch_b0 is not None:
        # True PyTorch MoE math: x @ W + bias.
        # Bias shape: (L, E, N) - broadcasts across tokens (L, E, T, N) automatically.
        # Weights are replicated per-device, so bias must be too.
        b0 = torch_b0.repeat([1, devices, 1])  # (L, E, N)
        torch_w0_output_ref = torch_w0_output_ref + b0.unsqueeze(2)  # broadcast T dimension

    torch_w1_output_ref = torch_input_ref @ torch_w1
    if torch_b1 is not None:
        # Same reasoning as b0.
        b1 = torch_b1.repeat([1, devices, 1])  # (L, E, N)
        torch_w1_output_ref = torch_w1_output_ref + b1.unsqueeze(2)

    if activation_type == MoEActivationFunction.SILU:
        # SILU: silu(x @ w0) * (x @ w1)
        torch_silu_output_ref = torch.nn.functional.silu(torch_w0_output_ref)
        # (L, E, T, K) @ (L, E, K, N) -> (L, E, T, N)
        torch_intermediate_ref = torch_silu_output_ref * torch_w1_output_ref  # (L, E, T, N)
    elif activation_type == MoEActivationFunction.SWIGLU:
        torch_intermediate_ref = _swiglu_reference(torch_w0_output_ref, torch_w1_output_ref)  # (L, E, T, N)
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")

    # (L, E, T, N) @ (L, E, N, K) -> (L, E, T, K)
    torch_output_ref = torch_intermediate_ref @ torch_w2
    if torch_b2 is not None:
        # Same reasoning as b0: true PyTorch bias addition.
        b2 = torch_b2.repeat([1, devices, 1])  # (L, E, K)
        torch_output_ref = torch_output_ref + b2.unsqueeze(2)

    # pull device dim back out for comparison
    # (L, E, T, H) -> (L, D, E/D, T, H)
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


@torch.no_grad()
def run_moe_compute_test(
    mesh_device,
    mesh_shape,
    cluster_axis,
    experts_per_device,
    tokens_per_device,
    selected_experts_k,
    num_layers,
    num_iterations,
    N,
    hidden_size,
    output_height_shard_dim,
    output_width_shard_dim,
    dtype,
    enable_trace,
    activation_type,
    has_bias,
):
    """
    Core test execution helper function.
    """
    torch.manual_seed(2003)
    random.seed(2003)

    experts = experts_per_device * mesh_shape[cluster_axis]

    #########################################
    # TEST SETUP
    #########################################

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices
    num_replicated_devices = num_devices // num_dispatch_devices
    total_tokens = tokens_per_device * num_dispatch_devices
    experts_per_cluster = experts // num_replicated_devices
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
    logger.info(f"  num_iterations: {num_iterations}")
    logger.info(f"  enable_trace: {enable_trace}")
    logger.info(f"  activation_type: {activation_type}")

    #########################################
    # CREATE TILIZE INPUT TENSORS AND GOLDENS
    #########################################

    # Drain tilize core is core (6,9) where indices and scores are sharded
    tilize_drain_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9))})

    #### Expert mapping - per-device [num_devices, experts], replicated on every device ###
    # Each device gets its own row after sharding, but since it's replicated,
    # we give each device the full tensor and it uses its own row.
    # Expert mapping is constant across all runs.
    expert_mapping = gen_expert_mapping(
        num_devices, num_replicated_devices, cluster_axis, experts, experts_per_cluster, experts_per_device
    )
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

    w0_w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(
        mesh_device, *HIDDEN_TO_SHARD_INFO[hidden_size]
    )

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
    # test_moe_compute already fixed torch.manual_seed(2003) so draws are reproducible.
    #
    # Biases are identical per-device (same as weights which use ReplicateTensorToMesh).
    # The golden's .repeat([1, devices, 1, 1]) in compute_matmul_golden is correct
    # under this assumption.
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
        torch_b0=torch_b0 if has_bias else None,
        torch_b1=torch_b1 if has_bias else None,
        torch_b2=torch_b2 if has_bias else None,
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

    # Get memory configurations for weights (handles bias padding)
    w0_w1_mem_config, w2_mem_config, K_for_shard, w2_N_total = get_weight_mem_configs(
        num_layers,
        experts_per_device,
        hidden_size,
        N,
        w0_w1_shard_map,
        w2_shard_map,
        dram_core_range_set,
        has_bias=has_bias,
    )

    # ------------------------------------------------------------------------
    # Prepare w0_w1 tensor (interleaved, padded, and reordered)
    if has_bias:
        torch_w0_w1_reordered = prepare_w0_w1_tensor_with_bias(
            torch_w0, torch_w1, torch_b0, torch_b1, num_layers, experts_per_device, hidden_size, N, w0_w1_shard_map
        )
    else:
        torch_w0_w1_reordered = prepare_w0_w1_tensor_for_moe_compute(
            torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, w0_w1_shard_map
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
    if has_bias:
        torch_w2_reordered = prepare_w2_tensor_with_bias(
            torch_w2, torch_b2, num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
        )
    else:
        torch_w2_reordered = prepare_w2_tensor_for_moe_compute(
            torch_w2, num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
        )

    if False:  # has_bias:
        # Verify prepare_w2_tensor_with_bias correctness:
        # The bias tile occupies element rows [N:N+TILE_SIZE] in the N dimension (tile Nt).
        # It should be non-zero (bias was appended) and must NOT appear in the weight rows [0:N]
        # (bias tile is at position Nt, after all ring-rotated weight tiles).
        #
        # Note: different cores receive different K-column slices, so bias row values
        # differ per core. The invariant is positional: bias is at N, not ring-rotated
        # into an earlier N position.
        bias_rows = torch_w2_reordered[:, :, :, :, N : N + ttnn.TILE_SIZE, :]  # (12, L, E, 5, 32, 128)
        assert bias_rows.abs().max() > 0.1, (
            "W2 bias row (at N-dim position N:N+TILE_SIZE) appears to be all zeros — "
            "bias may not have been appended at the correct position"
        )
        # Groups 0-3 (dim 3, indices 0:4) are fully populated from unpadded bias data for all
        # 12 cores; group 4 may have trailing padding zeros for some cores, so exclude it.
        first_four_groups = bias_rows[:, :, :, :4, :, :]  # (12, L, E, 4, 32, 128)
        assert torch.isfinite(first_four_groups).all(), "W2 bias row groups 0-3 contain non-finite values"
        assert (
            first_four_groups.abs().max() > 1e-3
        ), "W2 bias row groups 0-3 appear all-near-zero — bias may not be packed at N:N+TILE_SIZE"

    # Create tt_w2 tensor with DRAM sharding
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    output_shard_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device, output_height_shard_dim, output_width_shard_dim
    )
    combine_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in output_shard_cores])
    combine_barrier_semaphore = ttnn.create_global_semaphore(mesh_device, combine_core_range_set, 0)
    mux_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange((1, 1), (3, 3))])

    torch_combine_output_tensor = torch.zeros([selected_experts_k, total_tokens, hidden_size], dtype=torch.bfloat16)
    tt_combine_output_tensors = [
        ttnn.from_torch(
            torch_combine_output_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        for _ in range(num_layers)
    ]

    #########################################
    # HELPER FUNCTIONS
    #########################################

    def prepare_layer_inputs(layer_id):
        """Prepare inputs for a specific layer by moving from DRAM to L1"""
        if num_layers == 1:
            # Already in L1
            return tt_sparse_buffers[0], tt_expert_indices_buffers[0], tt_expert_scores_buffers[0]
        else:
            tt_sparse_buffer = ttnn.to_memory_config(tt_sparse_buffers[layer_id], memory_config=sparse_mem_config)
            tt_expert_indices = ttnn.to_memory_config(
                tt_expert_indices_buffers[layer_id], memory_config=expert_indices_mem_config
            )
            tt_expert_scores = ttnn.to_memory_config(
                tt_expert_scores_buffers[layer_id], memory_config=expert_scores_mem_config
            )
            return tt_sparse_buffer, tt_expert_indices, tt_expert_scores

    def deallocate_layer_inputs(tt_sparse_buffer, tt_expert_indices, tt_expert_scores):
        """Deallocate L1 inputs if using multiple layers"""
        if num_layers != 1:
            ttnn.deallocate(tt_sparse_buffer)
            ttnn.deallocate(tt_expert_indices)
            ttnn.deallocate(tt_expert_scores)

    def convert_outputs_to_dram(outputs):
        """Convert L1 outputs to DRAM and deallocate L1 versions"""
        l1_per_expert, l1_activation, l1_e_t, _, l1_matmul, combine = outputs

        # Convert to DRAM
        dram_per_expert = ttnn.to_memory_config(l1_per_expert, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dram_activation = ttnn.to_memory_config(l1_activation, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dram_e_t = ttnn.to_memory_config(l1_e_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dram_matmul = ttnn.to_memory_config(l1_matmul, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Deallocate L1 versions
        ttnn.deallocate(l1_per_expert)
        ttnn.deallocate(l1_activation)
        ttnn.deallocate(l1_e_t)
        ttnn.deallocate(l1_matmul)

        return (dram_per_expert, dram_activation, dram_e_t, dram_matmul, combine)

    def run_op_inner(tt_sparse_buffer, tt_expert_indices, tt_expert_scores, layer_id):
        """Core moe_compute operation"""
        return ttnn.experimental.moe_compute(
            tt_sparse_buffer,
            tt_expert_indices,
            tt_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=layer_id,
            output_height_shard_dim=output_height_shard_dim,
            has_bias=has_bias,
            cluster_axis=cluster_axis,
            mux_core_range_set=mux_core_range_set,
            optional_output_tensor=tt_combine_output_tensors[layer_id],
            optional_cross_device_semaphore=combine_barrier_semaphore,
            activation_type=activation_type,
        )

    #########################################
    # RUN OP
    #########################################

    def run_op():
        moe_compute_outputs = []

        for layer_id in range(num_layers):
            # Prepare layer inputs
            tt_sparse_buffer, tt_expert_indices, tt_expert_scores = prepare_layer_inputs(layer_id)

            # Run core moe_compute operation
            outputs = run_op_inner(tt_sparse_buffer, tt_expert_indices, tt_expert_scores, layer_id)

            # Deallocate L1 inputs
            deallocate_layer_inputs(tt_sparse_buffer, tt_expert_indices, tt_expert_scores)

            # Convert outputs to DRAM and save
            moe_compute_output = convert_outputs_to_dram(outputs)
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
        # Non-trace execution
        for i in range(num_iterations):
            moe_compute_output = run_op()
            ttnn.synchronize_device(mesh_device)
            moe_compute_outputs.append(moe_compute_output)

    #########################################
    # VALIDATE OUTPUTS PER LAYER
    #########################################
    logger.info(f"\n========== Starting Validation ==========")

    all_core_grid = mesh_device.compute_with_storage_grid_size()
    all_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(all_core_grid.x - 1, all_core_grid.y - 1),
            ),
        }
    )

    base_pcc_threshold = _get_base_pcc_threshold(activation_type, has_bias)

    output_shard_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device, output_height_shard_dim, output_width_shard_dim
    )
    per_expert_tokens_all_passed = True
    activation_all_passed = True
    e_t_all_passed = True
    matmul_all_passed = True
    combine_all_passed = True
    for i in range(num_iterations):
        for layer_id in range(num_layers):
            (
                per_expert_total_tokens_output_tensor,
                expert_activation_output_tensor,
                e_t_output_tensor,
                matmul_output_tensor,
                combine_output_tensor,
            ) = moe_compute_outputs[i][layer_id]

            logger.info(f"\n========== Iteration {i} Layer {layer_id} Validation ==========")
            logger.info(f"Per expert total tokens tensor shape: {per_expert_total_tokens_output_tensor.shape}")
            logger.info(f"Expert activation tensor shape: {expert_activation_output_tensor.shape}")
            logger.info(f"E-T (expert-to-token) tensor shape: {e_t_output_tensor.shape}")
            logger.info(f"Matmul Output tensor shape: {matmul_output_tensor.shape}")

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
                output_height_shard_dim,
                output_width_shard_dim,
                total_tokens,
                hidden_size,
                expert_token_counts,
                matmul_goldens,
                matmul_output_tensor,
                mesh_device,
                base_pcc_threshold,
                has_bias=has_bias,
            ):
                matmul_all_passed = False

            if not validate_combine(
                layer_id,
                mesh_device,
                cluster_axis,
                combine_output_tensor,
                combine_goldens,
                base_pcc_threshold,
            ):
                combine_all_passed = False

    # Asserts
    logger.info(f"\n========== Asserts ==========")
    logger.info(f"\nPer Expert Total Tokens Verification: {'PASSED' if per_expert_tokens_all_passed else 'FAILED'}")
    logger.info(f"\nExpert Activation Verification: {'PASSED' if activation_all_passed else 'FAILED'}")
    logger.info(f"\nE-T Tensor Verification: {'PASSED' if e_t_all_passed else 'FAILED'}")
    logger.info(f"\nMatmul Output Tensor Verification: {'PASSED' if matmul_all_passed else 'FAILED'}")
    logger.info(f"\nCombine Output Tensor Verification: {'PASSED' if combine_all_passed else 'FAILED'}")

    assert per_expert_tokens_all_passed, "Per expert total tokens tensor verification failed!"
    assert activation_all_passed, "Expert activation tensor verification failed!"
    assert e_t_all_passed, "E-T tensor verification failed!"
    assert matmul_all_passed, "Matmul output tensor verification failed!"
    assert combine_all_passed, "Combine output tensor verification failed!"


# Test for DeepSeek configuration - requires 1x16 mesh
@pytest.mark.skipif(
    not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x16),
    reason=f"DeepSeek test requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_1x16}",
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
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 16), (1, 16))], indirect=["mesh_device"])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize("test_mode", ["perf", "correctness"])
@pytest.mark.parametrize("has_bias", [False, True])
def test_moe_compute_deepseek(
    mesh_device,
    mesh_shape,
    has_bias,
    enable_trace,
    test_mode,
):
    """Test MoE compute for DeepSeek configuration on 1x16 mesh."""

    # DeepSeek specific configuration
    cluster_axis = 1
    experts_per_device = 2
    tokens_per_device = 32
    N = 2048
    hidden_size = 7168
    output_height_shard_dim = 4
    output_width_shard_dim = 4  # DeepSeekRingConfig::OUTPUT_WIDTH_SHARD_DIM
    dtype = ttnn.bfloat16
    activation_type = MoEActivationFunction.SILU

    # Test mode specific parameters
    if test_mode == "perf":
        selected_experts_k = 1
        num_layers = 1
        num_iterations = 5
    else:  # correctness
        selected_experts_k = 8
        num_layers = 5
        num_iterations = 3

    run_moe_compute_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        cluster_axis=cluster_axis,
        experts_per_device=experts_per_device,
        tokens_per_device=tokens_per_device,
        selected_experts_k=selected_experts_k,
        num_layers=num_layers,
        num_iterations=num_iterations,
        N=N,
        hidden_size=hidden_size,
        output_height_shard_dim=output_height_shard_dim,
        output_width_shard_dim=output_width_shard_dim,
        dtype=dtype,
        enable_trace=enable_trace,
        activation_type=activation_type,
        has_bias=has_bias,
    )


# Test for GPT-OSS configuration - requires 1x8 mesh
@pytest.mark.skipif(
    not is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x8),
    reason=f"GPT-OSS test requires TT_MESH_GRAPH_DESC_PATH={MESH_GRAPH_DESC_1x8}",
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
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 8), (1, 8))], indirect=["mesh_device"])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize("test_mode", ["perf", "correctness"])
@pytest.mark.parametrize("has_bias", [True])
def test_moe_compute_gpt_oss(
    mesh_device,
    mesh_shape,
    enable_trace,
    has_bias,
    test_mode,
):
    """Test MoE compute for GPT-OSS configuration on 1x8 mesh."""

    # GPT-OSS specific configuration
    cluster_axis = 1
    experts_per_device = 4
    tokens_per_device = 32
    N = 2880
    hidden_size = 2880
    output_height_shard_dim = 4
    output_width_shard_dim = 3  # GptRingConfig::OUTPUT_WIDTH_SHARD_DIM
    dtype = ttnn.bfloat16
    activation_type = MoEActivationFunction.SILU

    # Test mode specific parameters
    if test_mode == "perf":
        selected_experts_k = 1
        num_layers = 1
        num_iterations = 5
    else:  # correctness
        selected_experts_k = 8
        num_layers = 5
        num_iterations = 3

    run_moe_compute_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        cluster_axis=cluster_axis,
        experts_per_device=experts_per_device,
        tokens_per_device=tokens_per_device,
        selected_experts_k=selected_experts_k,
        num_layers=num_layers,
        num_iterations=num_iterations,
        N=N,
        hidden_size=hidden_size,
        output_height_shard_dim=output_height_shard_dim,
        output_width_shard_dim=output_width_shard_dim,
        dtype=dtype,
        enable_trace=enable_trace,
        activation_type=activation_type,
        has_bias=has_bias,
    )
