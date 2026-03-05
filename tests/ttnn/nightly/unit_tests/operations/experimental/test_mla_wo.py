# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose

PCC_THRESHOLD = 0.999

"""
There are total of 512 tiles in K dimension for the weight tensor. These are split non-uniformly
across the 12 DRAM banks. Each DRAM bank gets a contiguous block of tiles in K dimension, with the
the number of tiles assigned as below.
"""
BANK2K_TILES = {
    0: 44,
    1: 44,
    2: 42,
    3: 42,
    4: 42,
    5: 42,
    6: 42,
    7: 42,
    8: 44,
    9: 44,
    10: 42,
    11: 42,
}

MAX_K_TILES_PER_BANK = max(BANK2K_TILES.values())


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


def prepare_w_tensor(torch_w, L, K, N, num_dram_banks):
    """
    Prepare the w tensor and bias tensor by padding and reordering tiles.

    Args:
        torch_w: Weight tensor of shape (L, K, N)
        L: Number of layers
        K: Input dimension
        N: Output dimension
        num_dram_banks: Number of DRAM banks

    Returns:
        torch_w: Tensor of shape (L, K, N)
    """
    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)

    # Reshape to expose tiles: (L, K, N) -> (L, Kt, -1, Nt, ttnn.TILE_SIZE)
    w_tile_view = torch_w.view(L, Kt, ttnn.TILE_SIZE, Nt, ttnn.TILE_SIZE)

    each_shard = []

    current_K_tile = 0

    for dram_bank_id in range(num_dram_banks):
        num_K_tiles = BANK2K_TILES[dram_bank_id]
        padding_tiles = MAX_K_TILES_PER_BANK - num_K_tiles

        this_bank_tiles = w_tile_view[:, current_K_tile : current_K_tile + num_K_tiles, :, :, :]
        this_bank_blocks = this_bank_tiles.view(L, num_K_tiles, ttnn.TILE_SIZE, Nt // 7, 7 * ttnn.TILE_SIZE)

        # this_bank_blocks -> (L, Nt // 7, num_K_tiles, ttnn.TILE_SIZE, 7 * ttnn.TILE_SIZE)
        this_bank_blocks = this_bank_blocks.permute(0, 3, 1, 2, 4)

        this_bank_blocks = this_bank_blocks.reshape(L, Nt * num_K_tiles // 7, ttnn.TILE_SIZE, 7 * ttnn.TILE_SIZE)

        this_bank_padding = torch.zeros(
            L, Nt * padding_tiles // 7, ttnn.TILE_SIZE, 7 * ttnn.TILE_SIZE, dtype=torch_w.dtype
        )

        this_bank_data = torch.cat([this_bank_blocks, this_bank_padding], dim=1)
        each_shard.append(this_bank_data)
        current_K_tile += num_K_tiles

    torch_w_all_banks = torch.stack(each_shard, dim=1)
    return torch_w_all_banks.view(L, num_dram_banks, -1, 7 * ttnn.TILE_SIZE)


def prepare_output_tensor(tt_output):
    """
    Prepare the output tensor by picking the appropriate tiles from the cores that have the final data.

    Args:
        tt_output: Tensor of shape (M, N)

    Returns:
        tt_output: Tensor of shape (M, N)
    """

    each_shard = []

    for chunk_id, core_id in itertools.product(range(4), range(7)):
        chunk_offset = chunk_id * ttnn.TILE_SIZE
        core_offset = core_id * 4 * ttnn.TILE_SIZE

        this_data = tt_output[:, core_offset + chunk_offset : core_offset + chunk_offset + ttnn.TILE_SIZE]
        each_shard.append(this_data)

    return torch.cat(each_shard, dim=1)


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


def run_test_mla_wo(device, M, K, N, L, check_accuracy, dump_outputs):
    torch.manual_seed(0)

    logger.info(
        f"Running test_mla_wo with M={M}, K={K}, N={N}, L={L}, check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # --------------------------------------------------------------------------
    # Shard grid for input
    # --------------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    in0_num_cores = len(in0_core_coords)
    in0_core_range = [ttnn.CoreRange(in0_core_coord, in0_core_coord) for in0_core_coord in in0_core_coords]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    # --------------------------------------------------------------------------
    # Shard grid for output
    # --------------------------------------------------------------------------
    all_cores = device.compute_with_storage_grid_size()

    # Get the in0 cores as a tuple of (x, y) coordinates.
    in0_raw_coords = [(in0_core_coord.x, in0_core_coord.y) for in0_core_coord in in0_core_coords]

    # Start at the bottom right and pick the 7 cores that are not in the dram adjacent cores.
    out_core_coords = []
    for y, x in itertools.product(range(all_cores.y - 1, -1, -1), range(all_cores.x - 1, -1, -1)):
        if (x, y) not in in0_raw_coords:
            out_core_coords.append(ttnn.CoreCoord(x, y))
            if len(out_core_coords) == 7:
                break
    out_core_range = [ttnn.CoreRange(out_core_coord, out_core_coord) for out_core_coord in out_core_coords]
    out_core_range_set = ttnn.CoreRangeSet(out_core_range)

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------
    in0_dtype = ttnn.bfloat16
    w_dtype = ttnn.bfloat8_b
    num_dram_banks = len(in0_core_coords)

    dram_core_coords = [ttnn.CoreCoord(dram_bank_id, 0) for dram_bank_id in range(num_dram_banks)]
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
    # Tensor shape: (L, K, N) -> sharded over K dimension (with padding)
    # ------------------------------------------------------------------------
    w_shard_height = L * MAX_K_TILES_PER_BANK * N // 7
    w_shard_width = 7 * ttnn.TILE_SIZE

    w_shard_spec = ttnn.ShardSpec(dram_core_range_set, (w_shard_height, w_shard_width), ttnn.ShardOrientation.ROW_MAJOR)

    w_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for output
    # Tensor shape: (M, N) -> Sharded across 8 cores
    # ------------------------------------------------------------------------
    output_shard_height = M
    output_shard_width = N // 7
    output_shard_spec = ttnn.ShardSpec(
        out_core_range_set, (output_shard_height, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    tt_output = ttnn.empty(
        (M, N),
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

        # ------------------------------------------------------------------------
        # Prepare w tensor (padded, and reordered)
        torch_w_reordered = prepare_w_tensor(torch_w, L, K, N, num_dram_banks)
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
            [L, num_dram_banks] + w_shard_spec.shape,
            dtype=w_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=w_mem_config,
        )

    # --------------------------------------------------------------------------
    # Run the operation
    # --------------------------------------------------------------------------
    # Collect accuracy metrics for all layers
    all_outputs = []
    all_accuracy_metrics = {}

    for layer_id in range(L):
        if check_accuracy:
            tt_input = ttnn.from_torch(
                torch_input[layer_id],
                dtype=in0_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=input_sharded_mem_config,
            )

        ttnn.experimental.deepseek.mla.matmul_wo(
            tt_input,
            w_tensor=tt_w[layer_id],
            output_tensor=tt_output,
            layer_id=layer_id,
        )

        tt_to_torch_output = ttnn.to_torch(tt_output)
        all_outputs.append(tt_to_torch_output)

    tt_to_torch_outputs = torch.stack(all_outputs)

    if check_accuracy:
        with torch.no_grad():
            torch_input_ref = torch_input[:, 0, ...]

            # (L, M, K) @ (L, K, N) -> (L, M, N)
            torch_mm_out = torch_input_ref @ torch_w

        # Calculate accuracy metrics for each layer
        for layer_id in range(L):
            torch_ref_layer_out = torch_mm_out[layer_id, :, :]
            tt_layer_out = tt_to_torch_outputs[layer_id, :, :]
            torch_act_layer_out = prepare_output_tensor(tt_layer_out)
            layer_metrics = get_accuracy_metrics(torch_ref_layer_out, torch_act_layer_out)
            all_accuracy_metrics[layer_id] = layer_metrics

    if dump_outputs:
        torch.set_printoptions(profile="full")
        var2filename = {
            torch_input: f"torch_input.txt",
            torch_w: f"torch_w.txt",
            torch_mm_out: f"torch_ref_out.txt",
            tt_to_torch_outputs: f"torch_act_out.txt",
        }

        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

    return all_accuracy_metrics


SHAPE2TIME = {
    (32, 16384, 896, 1): 65.0,
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
    "M, K, N, L",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_mla_wo(device, M, K, N, L, check_accuracy, dump_outputs):
    accuracy_metrics = run_test_mla_wo(
        device,
        M,
        K,
        N,
        L,
        check_accuracy,
        dump_outputs,
    )

    if not check_accuracy:
        return

    passing = True
    # Print the layers that did not pass the PCC check
    for layer_id, metrics in accuracy_metrics.items():
        if metrics["pcc"] < PCC_THRESHOLD:
            passing = False
            logger.warning(f"Layer {layer_id}: PCC={metrics['pcc']:.6f}")
        else:
            logger.info(f"Layer {layer_id}: PCC={metrics['pcc']:.6f} (Passed)")

    assert passing, f"Some layers did not pass the PCC/Allclose check"
