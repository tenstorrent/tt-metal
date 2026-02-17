# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose

PCC_THRESHOLD = 0.97

"""
We have a total of 12 cores adjacent to 12 DRAM banks. Out of these, 8 cores process 2/3rd of the K
tiles for just one N tile. The remaining 4 cores process 1/3rd of the K tiles for 2 N tiles.
Since the 4 cores send their partial to the other 8 cores for final value, we identify these 4 cores
as the SEND_CORES and the remaining 4 cores as the RECV_CORES.
"""
SEND_CORES = (0, 3, 6, 9)
RECV_CORES = (1, 2, 4, 5, 7, 8, 10, 11)


def create_torch_input(L, in0_num_cores, M, K):
    """
    Create torch input tensor with random values per layer.

    Args:
        L: Number of layers
        in0_num_cores: Number of input cores the tensor is replicated across
        M: Number of tokens
        K: Hidden dimension

    Returns:
        torch_input: Tensor of shape (L, in0_num_cores, M, K)
    """
    torch_input = torch.rand((L, 1, M, K), dtype=torch.bfloat16) - 0.5
    torch_input = torch_input.repeat(1, in0_num_cores, 1, 1)
    return torch_input


def create_torch_w(L, K, N):
    """
    Create torch weight tensor with random values.

    Args:
        L: Number of layers
        K: Hidden dimension
        N: Output dimension

    Returns:
        torch_w: Tensor of shape (L, K, N)
    """
    torch_w = torch.rand((L, K, N), dtype=torch.bfloat16) - 0.5
    return torch_w


def create_torch_bias(L, N):
    """
    Create torch bias tensor with random values.

    Args:
        L: Number of layers
        N: Output dimension

    Returns:
        torch_bias: Tensor of shape (L, N)
    """
    torch_bias = torch.rand((L, N), dtype=torch.bfloat16)
    return torch_bias


def prepare_w_tensor(torch_w, torch_bias, L, K, N, ring2cores):
    """
    Prepare the w tensor and bias tensor by padding and reordering tiles.

    Args:
        torch_w: Weight tensor of shape (L, K, N)
        torch_bias: Bias tensor of shape (L, N)
        L: Number of layers
        K: Input dimension
        N: Output dimension
        ring2cores: Dictionary mapping ring position to (core_coord, dram_bank_id, send_flag)

    Returns:
        torch_w: Tensor of shape (L, K, N)
    """
    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    # 8 cores get 2/3rd of K tiles and 1 N tile -> Type 1 (send flag is 0)
    # 4 cores get 1/3rd of K tiles and 2 N tiles -> Type 2 (send flag is 1)
    # Every third core is of type 2.
    w_tile_view = torch_w.view(L, Kt, ttnn.TILE_SIZE, Nt, ttnn.TILE_SIZE)

    # For the 8 cores, we append values from the bias tensor at the end, so it can be read in the
    # same DRAM transaction, optimally without any additional overhead.
    bias_tile_view = torch_bias.view(L, Nt, ttnn.TILE_SIZE)

    each_shard = []

    current_N_tile = 0
    for ring_pos in range(len(ring2cores)):
        _, _, send_flag = ring2cores[ring_pos]

        if send_flag:
            # Type 2: Last 72 K tiles for 2 N tiles
            first_chunk = w_tile_view[:, -72:, :, current_N_tile, :]
            second_chunk = w_tile_view[:, -72:, :, current_N_tile + 1, :]

            # Interleave the two chunks, one tile each on width dimension.
            interleaved = torch.stack([first_chunk, second_chunk], dim=3)

            # Reshape to interleave: (L, E, K, Nt * 2 * ttnn.TILE_SIZE) = (L, E, K, 4096)
            # The order will be: w0_chunk_0, w1_chunk_0, w0_chunk_1, w1_chunk_1, ...
            interleaved_chunks = interleaved.view(L, 72, ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE)

            # Since we want the shard height to be same on all 12 cores, we add some padding here.
            padding = torch.zeros(L, 5, ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE, dtype=torch_w.dtype)
            torch_w_with_padding = torch.cat([interleaved_chunks, padding], dim=1)
            each_shard.append(torch_w_with_padding)
        else:
            # Type 1: First 2 * 76 K tiles for 1 N tile
            all_tiles = w_tile_view[:, : 2 * 76, :, current_N_tile, :]

            # Separate the even and odd tiles.
            even_tiles = all_tiles[:, ::2, :, :]
            odd_tiles = all_tiles[:, 1::2, :, :]
            interleaved = torch.cat([even_tiles, odd_tiles], dim=-1)

            # Put one each of even and odd tiles in width dimension.
            all_tiles = interleaved.reshape(L, 76, ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE)

            # Create the bias tile with zero padding.
            bias_tile = torch.zeros((L, 1, ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE), dtype=torch_bias.dtype)

            # Add data from bias tensor to the bias tile.
            bias_tile[:, 0, 0, : ttnn.TILE_SIZE] = bias_tile_view[:, current_N_tile, :]

            # Append the bias tensor to the end of the all tiles.
            w_bias_tile = torch.cat([all_tiles, bias_tile], dim=1)
            each_shard.append(w_bias_tile)
            current_N_tile += 1

    torch_w_all_banks = torch.stack(each_shard, dim=0)
    return torch_w_all_banks


def prepare_output_tensor(tt_output, ring2cores):
    """
    Prepare the output tensor by picking the appropriate tiles from the cores that have the final data.

    Args:
        tt_output: Tensor of shape (M, in0_num_cores * ttnn.TILE_SIZE)
        ring2cores: Dictionary mapping ring position to (core_coord, dram_bank_id, send_flag)

    Returns:
        tt_values: Tensor of shape (M, 8)
    """

    each_shard = []
    current_column = 0
    for ring_pos in range(len(ring2cores)):
        _, _, send_flag = ring2cores[ring_pos]
        if not send_flag:
            each_shard.append(tt_output[:, current_column : current_column + ttnn.TILE_SIZE])
        current_column += ttnn.TILE_SIZE

    # --------------------------------------------------------------------------
    # The following snippet is to be used if we want to return just the matmul
    # output, without the final selection of top 8 experts.
    # So we retain it for reference, but not used in the test.
    # --------------------------------------------------------------------------
    # output = torch.cat(each_shard, dim=1)

    # # Get the 32 scores values from each tile.
    # f1_scores = output.view(output.shape[0], -1, ttnn.TILE_SIZE)[3, :, :16]
    # f2_scores = output.view(output.shape[0], -1, ttnn.TILE_SIZE)[4, :, :16]

    # group_scores = torch.cat([f1_scores, f2_scores], dim=-1)
    # return group_scores.transpose(0, 1)
    # --------------------------------------------------------------------------

    # Only the last core has the values in the first 8 rows of the first 2 faces of the tile
    tt_values = each_shard[-1][:8, :].transpose(0, 1)
    tt_as_bf16_indices = each_shard[-1][8:16, :].transpose(0, 1).view(torch.uint16)

    # Initialize an empty array of shape tt_indices
    tt_indices = torch.empty(tt_as_bf16_indices.shape, dtype=torch.uint16)
    for m, k in itertools.product(range(tt_as_bf16_indices.shape[0]), range(tt_as_bf16_indices.shape[1])):
        tt_indices[m, k] = tt_as_bf16_indices[m, k].item() >> 7

    return tt_values, tt_indices


def get_accuracy_metrics(torch_output, tt_output):
    _pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    std = torch_output.std().item()
    relative_rmse_val = (torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std) if std != 0 else 0.0
    allclose_passed, allclose_val = comp_allclose(torch_output, tt_output, rtol=2e-2, atol=1e-1)
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
        "allclose": allclose_passed,
        "allclose_val": allclose_val,
    }


def run_test_moe_mm(device, M, K, N, L, C, check_accuracy, dump_outputs):
    torch.manual_seed(0)

    logger.info(
        f"Running test_moe_mm with M={M}, K={K}, N={N}, L={L}, C={C}, check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)

    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

    # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        # key: ring_pos, value: (core_coord, dram_bank_id, send_flag)
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in SEND_CORES else 0)

    in0_core_range = [ttnn.CoreRange(in0_core_coord, in0_core_coord) for in0_core_coord in in0_core_coords_sorted]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------
    in0_dtype = ttnn.bfloat16
    w_dtype = ttnn.bfloat16
    num_dram_banks = len(in0_core_coords)

    dram_core_coords = [ttnn.CoreCoord(core2dram[in0_core_coord], 0) for in0_core_coord in in0_core_coords_sorted]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    # --------------------------------------------------------------------------
    # Tensor shapes and memory configurations
    # --------------------------------------------------------------------------
    # Define tensor shapes - same for both accuracy and performance testing
    input_shape = (in0_num_cores, M, K)

    in0_shard_spec = ttnn.ShardSpec(
        grid=in0_core_range_set,
        shard_shape=(M, K),
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Each core gets a copy of the original (M, K) input
    input_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
    )

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w
    # Tensor shape: (L, K, N) -> Sharded across N cores
    # ------------------------------------------------------------------------
    w_shard_height = L * (76 + 1) * ttnn.TILE_SIZE
    w_shard_width = 2 * ttnn.TILE_SIZE

    w_shard_spec = ttnn.ShardSpec(dram_core_range_set, (w_shard_height, w_shard_width), ttnn.ShardOrientation.ROW_MAJOR)

    w_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for output
    # Tensor shape: (M, N) -> Sharded across 8 cores
    # ------------------------------------------------------------------------
    output_shard_height = M
    output_shard_width = ttnn.TILE_SIZE
    output_shard_spec = ttnn.ShardSpec(
        in0_core_range_set, (output_shard_height, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    tt_output = ttnn.empty(
        (M, in0_num_cores * ttnn.TILE_SIZE),
        dtype=in0_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_mem_config,
    )

    # ------------------------------------------------------------------------
    # Prepare the tensors
    # --------------------------------------------------------------------------
    if check_accuracy:
        torch_input = create_torch_input(L, in0_num_cores, M, K)
        torch_w = create_torch_w(L, K, N)
        torch_bias = create_torch_bias(L, N)

        # ------------------------------------------------------------------------
        # Prepare w tensor (padded, and reordered)
        torch_w_reordered = prepare_w_tensor(torch_w, torch_bias, L, K, N, ring2cores)
        # Create tt_w tensor with DRAM sharding
        tt_w = ttnn.from_torch(
            torch_w_reordered,
            dtype=w_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w_mem_config,
        )
    else:
        tt_input = ttnn.empty(
            input_shape,
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
        )
        tt_w = ttnn.empty(
            (num_dram_banks, L, w_shard_height, w_shard_width),
            dtype=w_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w_mem_config,
        )

    # --------------------------------------------------------------------------
    # Run the operation
    # --------------------------------------------------------------------------
    # Collect accuracy metrics for all layers and experts
    all_outputs = []
    all_accuracy_metrics = {}

    for layer_id, column_id in itertools.product(range(L), range(C)):
        if check_accuracy:
            tt_input = ttnn.from_torch(
                torch_input[layer_id],
                dtype=in0_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=input_sharded_mem_config,
            )

        ttnn.experimental.deepseek.moe.moe_gate_mm(
            tt_input,
            w_tensor=tt_w,
            output_tensor=tt_output,
            layer_id=layer_id,
            column_id=column_id,
        )

        tt_to_torch_output = ttnn.to_torch(tt_output)
        all_outputs.append(tt_to_torch_output)

    tt_to_torch_outputs = torch.stack(all_outputs)

    if check_accuracy:
        with torch.no_grad():
            # Reference calculation to match TT output shape (2*M, N) = (E*M, N)
            # Use first 2*M rows of input (one copy of the original replicated input)
            torch_input_ref = torch_input[:, 0, ...]

            # 1. Linear projection: scores = x @ weight
            # (L, M, K) @ (L, K, N) -> (L, M, N)
            torch_mm_out = torch_input_ref @ torch_w

            # 2. Sigmoid activation: scores = sigmoid(scores)
            torch_sigmoid_out = torch.nn.functional.sigmoid(torch_mm_out)

            # 3. Store original scores: original_scores = scores
            torch_original_scores = torch_sigmoid_out.clone()

            # 4. Add bias: scores = scores + bias
            torch_bias_out = torch_sigmoid_out + torch_bias[:, None, :]

            # 5. Reshape to groups of 8
            num_groups = 8
            torch_bias_out = torch_bias_out.reshape(L, M, num_groups, -1)

            # 6. Compute group scores: sum of top-2 scores for each group
            torch_topk_out = torch.topk(torch_bias_out, k=2, dim=-1)[0].sum(dim=-1)

            # 7. Select top groups: 4 group indices for each token
            torch_top4_groups = torch.topk(torch_topk_out, k=4, dim=-1)[1]

            # 7a. Get bitmask for each token, of 8 bits, 1 for each group
            torch_group_bitmask = (
                (1 << torch_top4_groups[:, :, 0])
                + (1 << torch_top4_groups[:, :, 1])
                + (1 << torch_top4_groups[:, :, 2])
                + (1 << torch_top4_groups[:, :, 3])
            )

            # 8. Create group mask: mask = scatter(ones, indices)
            torch_group_mask = torch.zeros((L, M, num_groups), dtype=torch.bool)
            torch_group_mask.scatter_(2, torch_top4_groups, 1)

            # 9. Mask and flatten scores
            torch_masked_scores = (torch_bias_out * torch_group_mask.unsqueeze(-1)).flatten(2)

            # 10. Select top 8 experts
            torch_top8_values, torch_top8_indices = torch.topk(torch_masked_scores, k=8, dim=-1)

            # 11. Gather original scores: weights = original_scores.gather(1, indices)
            torch_weights = torch_original_scores.gather(-1, torch_top8_indices)

            # 12. Normalize weights: weights = weights / weights.sum(dim=-1, keepdim=True)
            torch_weights_scaled = 2.5 * (torch_weights / torch_weights.sum(dim=-1, keepdim=True))

            # 13. Create a token -> experts bitmask for each column
            group_idx, bit_idx = torch_top8_indices // ttnn.TILE_SIZE, torch_top8_indices % ttnn.TILE_SIZE
            bit_values = (1 << bit_idx).to(torch.int32)

            torch_bitmask = torch.zeros(L, M, 8, dtype=torch.int32)

            for i in range(8):
                mask = group_idx == i
                masked_bits = torch.where(mask, bit_values, 0)
                # OR reduction along N dimension
                torch_bitmask[:, :, i] = masked_bits.sum(dim=2)

        # Calculate accuracy metrics for each layer
        for layer_id, column_id in itertools.product(range(L), range(C)):
            torch_layer_values = torch_top8_values[layer_id, :, :]
            torch_layer_indices = torch_top8_indices[layer_id, :, :]
            tt_layer_output = tt_to_torch_outputs[C * layer_id + column_id, :, :]
            tt_values, tt_indices = prepare_output_tensor(tt_layer_output, ring2cores)
            layer_metrics = get_accuracy_metrics(torch_layer_values, tt_values)
            all_accuracy_metrics[layer_id] = layer_metrics

    if dump_outputs:
        torch.set_printoptions(profile="full")
        var2filename = {
            tt_to_torch_outputs: "tt_w_output_act.txt",
        }
        if check_accuracy:
            var2filename[torch_top8_values] = "torch_w_output_ref.txt"

        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

    return all_accuracy_metrics


SHAPE2TIME = {
    (32, 7168, 256, 1, 1): 27,
}


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K, N, L, C",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True], ids=["check_accuracy_True"])
@pytest.mark.parametrize("dump_outputs", [False], ids=["dump_outputs_False"])
def test_moe_mm(device, M, K, N, L, C, check_accuracy, dump_outputs):
    accuracy_metrics = run_test_moe_mm(
        device,
        M,
        K,
        N,
        L,
        C,
        check_accuracy,
        dump_outputs,
    )

    passing = True
    # Print the layers that did not pass the PCC check
    for layer_id, metrics in accuracy_metrics.items():
        if metrics["pcc"] < PCC_THRESHOLD:
            passing = False
            logger.warning(f"Layer {layer_id}: PCC={metrics['pcc']:.6f}")
        else:
            logger.info(f"Layer {layer_id}: PCC={metrics['pcc']:.6f} (Passed)")

    assert passing, f"Some layers did not pass the PCC/Allclose check"
