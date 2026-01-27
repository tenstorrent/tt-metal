# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, comp_allclose

from tracy.process_model_log import (
    get_latest_ops_log_filename,
    run_device_profiler,
)

PCC_THRESHOLD = 0.999


def create_torch_input(L, in0_num_cores, M, K):
    """
    Create torch input tensor with unique integer values per layer/expert.

    Args:
        L: Number of layers
        in0_num_cores: Number of input cores
        M: Sequence length
        K: Input dimension

    Returns:
        torch_input: Tensor of shape (L, in0_num_cores, M, K)
    """
    # torch_input = torch.empty((L, in0_num_cores, E, M, K), dtype=torch.bfloat16)
    # le_val = 1
    # for layer in range(L):
    #     for expert in range(E):
    #         for k_chunk_id in range(K // 32):
    #             k_start, k_end = k_chunk_id * 32, k_chunk_id * 32 + 32
    #             chunk_value = le_val * 0.001 * k_chunk_id
    #             torch_input[layer, :, expert, :, k_start:k_end] = chunk_value
    #         le_val *= -1
    # torch_input = 0.25 * 0.25 *torch.ones((L, in0_num_cores, E, M, K), dtype=torch.bfloat16)
    # torch_input = torch.empty((L, in0_num_cores, E, M, K), dtype=torch.bfloat16)
    # k_half = K // 2
    # # Interleave the positive and negatives
    # for i in range(K):
    #     if i % 2 == 0:
    #         torch_input[..., i] = 0.25
    #     else:
    #         torch_input[..., i] = -0.25
    # torch_input = (1 / 1024) * torch.ones((L, in0_num_cores, 2, M, K), dtype=torch.bfloat16)
    torch_input = torch.rand((L, M, K), dtype=torch.bfloat16) - 0.5
    torch_input = torch_input.unsqueeze(1).repeat(1, in0_num_cores, 1, 1)
    return torch_input


def create_torch_w(L, K, N):
    """
    Create torch w0 weight tensor.

    Args:
        L: Number of layers
        K: Input dimension
        N: Output dimension

    Returns:
        torch_w: Tensor of shape (L, K, N)
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

    torch_w = torch.rand((L, K, N), dtype=torch.bfloat16) - 0.5
    return torch_w


def prepare_w_tensor(torch_w, L, K, N):
    """
    Prepare the w tensor by padding and reordering tiles.

    Args:
        torch_w: Weight tensor of shape (L, K, N)
        L: Number of layers
        K: Input dimension
        N: Output dimension

    Returns:
        torch_w: Tensor of shape (L, E, K, 4096)
    """
    return torch_w


def get_accuracy_metrics(torch_output, tt_output):
    _pcc_passed, pcc_val = comp_pcc(torch_output, tt_output)
    std = torch_output.std().item()
    relative_rmse_val = (torch.nn.functional.mse_loss(torch_output, tt_output).sqrt().item() / std) if std != 0 else 0.0
    allclose_passed, allclose_val = comp_allclose(torch_output, tt_output)
    return {
        "pcc": pcc_val,
        "relative_rmse": relative_rmse_val,
        "allclose": allclose_passed,
        "allclose_val": allclose_val,
    }


def run_test_moe_mm(device, M, K, N, L, check_accuracy, dump_outputs):
    logger.info(
        f"Running test_moe_mm with M={M}, K={K}, N={N}, L={L}, check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, 0)
    # Pick only the first 8 cores
    in0_core_coords = in0_core_coords[:8]

    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

    in0_core_range = [ttnn.CoreRange(in0_core_coord, in0_core_coord) for in0_core_coord in in0_core_coords]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    # --------------------------------------------------------------------------
    # Constants
    # --------------------------------------------------------------------------
    in0_dtype = ttnn.bfloat16
    w_dtype = ttnn.bfloat16
    num_dram_banks = 8

    dram_core_coords = [ttnn.CoreCoord(core2dram[in0_core_coord], 0) for in0_core_coord in in0_core_coords]
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
    w_shard_height = L * K * 1
    w_shard_width = ttnn.TILE_SIZE

    w_shard_spec = ttnn.ShardSpec(dram_core_range_set, (w_shard_height, w_shard_width), ttnn.ShardOrientation.ROW_MAJOR)

    w_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, w_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for output
    # Tensor shape: (M, N) -> Sharded across N cores
    # ------------------------------------------------------------------------
    output_shard_height = M
    output_shard_width = ttnn.TILE_SIZE
    output_shard_spec = ttnn.ShardSpec(
        in0_core_range_set, (output_shard_height, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR
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
        torch_w_reordered = prepare_w_tensor(torch_w, L, K, N)
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
            (L, K, N),
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

    for layer_id in range(L):
        if check_accuracy:
            tt_input = ttnn.from_torch(
                torch_input[layer_id],
                dtype=in0_dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=input_sharded_mem_config,
            )

        _tt_output = ttnn.experimental.moe_gate_mm(
            tt_input,
            w_tensor=tt_w,
            output_tensor=tt_output,
            layer_id=layer_id,
        )

        # Output is produced in-place on the input tensor
        tt_to_torch_output = ttnn.to_torch(tt_output)
        all_outputs.append(tt_to_torch_output)

    tt_to_torch_outputs = torch.stack(all_outputs)

    if check_accuracy:
        with torch.no_grad():
            # Reference calculation to match TT output shape (2*M, N) = (E*M, N)
            # Use first 2*M rows of input (one copy of the original replicated input)
            torch_input_ref = torch_input[:, 0, ...]

            # Compute gate activations for each expert
            # (L, M, K) @ (L, K, N) -> (L, M, N)
            torch_w_output_ref = torch_input_ref @ torch_w

        # Calculate accuracy metrics for each layer
        for layer_id in range(L):
            torch_layer_output = torch_w_output_ref[layer_id, :, :]
            tt_layer_output = tt_to_torch_outputs[layer_id, :, :]
            layer_metrics = get_accuracy_metrics(torch_layer_output, tt_layer_output)
            all_accuracy_metrics[layer_id] = layer_metrics

    if dump_outputs:
        torch.set_printoptions(profile="full")
        var2filename = {
            torch_w_output_ref: f"torch_w_output_ref.txt",
            tt_to_torch_outputs: f"tt_w_output_act.txt",
        }

        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

    return all_accuracy_metrics


SHAPE2TIME = {
    (32, 7168, 256, 1): 25.0,
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
def test_moe_mm(device, M, K, N, L, check_accuracy, dump_outputs):
    accuracy_metrics = run_test_moe_mm(
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


@pytest.mark.parametrize(
    "M, K, N, L",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_moe_mm_performance(M, K, N, L, check_accuracy, dump_outputs):
    command = f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_mm.py::test_moe_mm[dump_outputs_{dump_outputs}-check_accuracy_{check_accuracy}-M={M}-K={K}-N={N}-L={L}-dispatch_row]"
    run_device_profiler(command, "ttnn_moe_mm_performance", device_analysis_types=["device_kernel_duration"])

    r = post_process_ops_log("ttnn_moe_mm_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = r["DEVICE KERNEL DURATION [ns]"].sum() / 1000.0
    logger.info(f"Duration per layer: {duration_us / L} us")
    logger.warning(f"Total Duration: {duration_us} us")

    bytes_per_tile = 2048  # bfloat16
    num_cores = 8

    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    w_tiles_per_core = math.ceil(Nt / num_cores) * Kt

    total_bytes_transferred = L * num_cores * w_tiles_per_core * bytes_per_tile
    realized_bandwidth = int(total_bytes_transferred / (duration_us * 1000))
    logger.warning(f"Realized Bandwidth: {realized_bandwidth} GB/s")

    total_tiles_per_core = Kt * Nt
    total_bytes_used = L * total_tiles_per_core * bytes_per_tile
    bandwidth = int(total_bytes_used / (duration_us * 1000))
    logger.warning(f"Useful Bandwidth: {bandwidth} GB/s")

    assert (
        duration_us < SHAPE2TIME[(M, K, N, L)]
    ), f"Performance {duration_us} us is greater than expected {SHAPE2TIME[(M, K, N, L)]} us"


def post_process_ops_log(output_logs_subdir: str, float_columns: list[str]) -> dict[str, float]:
    filename = get_latest_ops_log_filename(output_logs_subdir)

    import pandas as pd

    df = pd.read_csv(filename)
    return {col: df[col].astype(float).to_numpy() for col in float_columns}
