# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose

PCC_THRESHOLD = 0.998

"""
The kernel reads 7 tiles per transaction.
"""
W_TILES_PER_TXN = 7

"""
# SMALL_N_TILES_PER_CORE and LARGE_N_TILES_PER_CORE explain core groupings used for splitting the N dimension
#
# The MLA WQKV-AB kernel shards the N dimension (output channels) across multiple device cores. To maximize memory bandwidth
# and balance workloads, the N tiles are split unevenly: 'small' cores handle a smaller number of N tiles, and 'large' cores
# handle a larger number. Typically, half the cores are assigned SMALL_N_TILES_PER_CORE (here, 5), and the other half are assigned
# LARGE_N_TILES_PER_CORE (here, 6):
#
#   - num_cores // 2 cores will process SMALL_N_TILES_PER_CORE = 5 N tiles each
#   - (num_cores - num_cores // 2) cores will process LARGE_N_TILES_PER_CORE = 6 N tiles each
#
# This uneven sharding allows for better balancing in cases where N is not exactly divisible by the core count, ensuring all required tiles are processed.
#
# MAX_N_TILES_PER_CORE gives the maximum tiles any core may process (6 in this setup).

"""
SMALL_N_TILES_PER_CORE = 5
LARGE_N_TILES_PER_CORE = 6
MAX_N_TILES_PER_CORE = max(SMALL_N_TILES_PER_CORE, LARGE_N_TILES_PER_CORE)


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


def n_tiles_for_core(core_id, num_cores):
    """
    Get the number of N tiles for a given core.
    Args:
        core_id: Core ID
        num_cores: Number of cores

    Returns:
        Number of N tiles for the core
    """
    return SMALL_N_TILES_PER_CORE if core_id < (num_cores // 2) else LARGE_N_TILES_PER_CORE


def prepare_w_tensor(torch_w, L, K, N, num_dram_banks):
    """
    Prepare w tensor shards for mla_wqkv_ab pure matmul path.

    Args:
        torch_w: Weight tensor of shape (L, K, N)
        L: Number of layers
        K: Input dimension
        N: Output dimension
        num_dram_banks: Number of DRAM banks / worker cores

    Returns:
        Tensor of shape (num_dram_banks, L, H, W) laid out as 7-tile packets.
    """
    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    w_tile_view = torch_w.view(L, Kt, ttnn.TILE_SIZE, Nt, ttnn.TILE_SIZE)

    each_shard = []
    max_w_tiles_per_core = Kt * MAX_N_TILES_PER_CORE

    # We don't need padding in height dimension, if this is true
    assert max_w_tiles_per_core % W_TILES_PER_TXN == 0

    current_n_tile = 0
    for core_id in range(num_dram_banks):
        n_tiles_this_core = n_tiles_for_core(core_id, num_dram_banks)
        this_core_tiles = w_tile_view[:, :, :, current_n_tile : current_n_tile + n_tiles_this_core, :]
        current_n_tile += n_tiles_this_core

        # Put this_core_tiles together, in the last dimension.
        this_core_tiles = this_core_tiles.reshape(L, Kt, ttnn.TILE_SIZE, n_tiles_this_core * ttnn.TILE_SIZE)

        # Pad smaller-N cores so all DRAM shards keep a uniform shape.
        pad_tiles = LARGE_N_TILES_PER_CORE - n_tiles_this_core
        this_core_padding = torch.zeros(L, Kt, ttnn.TILE_SIZE, pad_tiles * ttnn.TILE_SIZE, dtype=torch_w.dtype)
        this_core_data = torch.cat([this_core_tiles, this_core_padding], dim=-1)

        each_shard.append(this_core_data)

    assert current_n_tile == Nt
    return torch.stack(each_shard, dim=1)


def prepare_output_tensor(tt_output, num_dram_banks):
    """
    Prepare output by extracting the valid 5/6 N tiles per core.

    Args:
        tt_output: Tensor of shape (M, num_dram_banks * 6 * ttnn.TILE_SIZE)
        num_dram_banks: Number of DRAM banks / worker cores

    Returns:
        Tensor of shape (M, N)
    """
    each_shard = []
    shard_width = MAX_N_TILES_PER_CORE * ttnn.TILE_SIZE

    core_offset = 0
    for core_id in range(num_dram_banks):
        n_tiles_this_core = n_tiles_for_core(core_id, num_dram_banks)
        each_shard.append(tt_output[:, core_offset : core_offset + n_tiles_this_core * ttnn.TILE_SIZE])
        core_offset += MAX_N_TILES_PER_CORE * ttnn.TILE_SIZE
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


def run_test_mla_wqkv_ab(device, M, K, N, L, check_accuracy, dump_outputs):
    # torch.manual_seed(0)

    logger.info(
        f"Running test_mla_wqkv_ab with M={M}, K={K}, N={N}, L={L}, check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    in0_core_coords = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)

    in0_num_cores = len(in0_core_coords)
    in0_core_range = [ttnn.CoreRange(in0_core_coord, in0_core_coord) for in0_core_coord in in0_core_coords]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

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
    # Tensor shape: (L, K, N) -> sharded over cores with fixed max shard shape.
    # ------------------------------------------------------------------------
    w_shard_height = K
    w_shard_width = MAX_N_TILES_PER_CORE * ttnn.TILE_SIZE

    w_shard_spec = ttnn.ShardSpec(dram_core_range_set, (w_shard_height, w_shard_width), ttnn.ShardOrientation.ROW_MAJOR)

    w_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for output
    # Tensor shape: (M, N_padded) -> one fixed-width output shard per core.
    # ------------------------------------------------------------------------
    output_shard_height = M
    output_shard_width = MAX_N_TILES_PER_CORE * ttnn.TILE_SIZE
    output_shard_spec = ttnn.ShardSpec(
        in0_core_range_set, (output_shard_height, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    tt_output = ttnn.empty(
        (M, in0_num_cores * output_shard_width),
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
            (num_dram_banks, L, w_shard_height, w_shard_width),
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

        ttnn.experimental.deepseek.mla.mla_wqkv_ab(
            tt_input,
            w_tensor=tt_w,
            output_tensor=tt_output,
            layer_id=layer_id,
        )

        tt_to_torch_output = ttnn.to_torch(tt_output)
        all_outputs.append(tt_to_torch_output)

    tt_to_torch_outputs = torch.stack(all_outputs)

    if check_accuracy:
        with torch.no_grad():
            torch_input_ref = torch_input[:, 0, ...]
            torch_mm_out = torch_input_ref @ torch_w

        # Calculate accuracy metrics for each layer
        for layer_id in range(L):
            torch_ref_layer = torch_mm_out[layer_id, :, :]
            tt_layer_output = tt_to_torch_outputs[layer_id, :, :]
            tt_values = prepare_output_tensor(tt_layer_output, num_dram_banks)
            layer_metrics = get_accuracy_metrics(torch_ref_layer, tt_values)
            all_accuracy_metrics[layer_id] = layer_metrics

    if dump_outputs:
        torch.set_printoptions(profile="full")
        var2filename = {
            tt_to_torch_outputs: "tt_wqkv_ab_output_act.txt",
        }
        if check_accuracy:
            var2filename[torch_mm_out] = "torch_wqkv_ab_output_ref.txt"

        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

    return all_accuracy_metrics


SHAPE2TIME = {
    (32, 7168, 2112, 1): 68,
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
@pytest.mark.parametrize("dump_outputs", [False], ids=["dump_outputs_False"])
def test_mla_wqkv_ab(device, M, K, N, L, check_accuracy, dump_outputs):
    accuracy_metrics = run_test_mla_wqkv_ab(
        device,
        M,
        K,
        N,
        L,
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
