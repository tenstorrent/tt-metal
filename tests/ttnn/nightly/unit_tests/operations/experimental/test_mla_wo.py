# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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

PCC_THRESHOLD = 0.99


def run_test_mla_wo(device, M, K, N, L, check_accuracy, dump_outputs):
    logger.info(
        f"Running test_mla_wo with M={M}, K={K}, N={N}, L={L}, check_accuracy={check_accuracy}, dump_outputs={dump_outputs}"
    )

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, 0)
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
    w_dtype = ttnn.bfloat8_b
    num_dram_banks = 12

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
    # Create DRAM shard spec for w0_w1
    # Tensor shape: (L, K, N) -> padded and reordered to (12, L, 44*32, N)
    # ------------------------------------------------------------------------
    w_shard_height = L * 44 * ttnn.TILE_SIZE
    w_shard_width = N

    w_shard_spec = ttnn.ShardSpec(dram_core_range_set, (w_shard_height, w_shard_width), ttnn.ShardOrientation.ROW_MAJOR)

    w_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w_shard_spec)

    # ------------------------------------------------------------------------
    # Prepare the tensors
    # --------------------------------------------------------------------------
    if check_accuracy:
        assert False, "Accuracy testing not implemented yet"
    else:
        tt_input = ttnn.empty(
            input_shape,
            dtype=in0_dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=input_sharded_mem_config,
        )
        tt_w = ttnn.empty(
            [num_dram_banks] + w_shard_spec.shape,
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
            assert False, "Accuracy testing not implemented yet"

        _tt_output = ttnn.experimental.deepseek.mla.matmul_wo(
            tt_input,
            w_tensor=tt_w,
            output_tensor=tt_input,
            layer_id=layer_id,
        )

    if check_accuracy:
        assert False, "Accuracy testing not implemented yet"

    if dump_outputs:
        torch.set_printoptions(profile="full")
        var2filename = {
            tt_input: f"tt_input.txt",
            tt_w: f"tt_w.txt",
            _tt_output: f"tt_output.txt",
        }

        for var, filename in var2filename.items():
            with open(filename, "w") as f:
                f.write(str(var))

    return all_accuracy_metrics


SHAPE2TIME = {
    (32, 16384, 896, 1): 90.0,
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


@pytest.mark.parametrize(
    "M, K, N, L",
    SHAPE2TIME.keys(),
)
@pytest.mark.parametrize("check_accuracy", [True, False], ids=["check_accuracy_True", "check_accuracy_False"])
@pytest.mark.parametrize("dump_outputs", [True, False], ids=["dump_outputs_True", "dump_outputs_False"])
def test_mla_wo_performance(M, K, N, L, check_accuracy, dump_outputs):
    command = f"pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_mla_wo.py::test_mla_wo[dump_outputs_{dump_outputs}-check_accuracy_{check_accuracy}-M={M}-K={K}-N={N}-L={L}-dispatch_row]"
    run_device_profiler(command, "ttnn_mla_wo_performance", device_analysis_types=["device_kernel_duration"])

    r = post_process_ops_log("ttnn_mla_wo_performance", float_columns=["DEVICE KERNEL DURATION [ns]"])
    duration_us = r["DEVICE KERNEL DURATION [ns]"].sum() / 1000.0
    logger.info(f"Duration per layer: {duration_us / L} us")
    logger.warning(f"Total Duration: {duration_us} us")

    bytes_per_tile = 1024 + 64  # bfloat8_b
    num_cores = 12

    Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
    w_padded_tiles_per_core = 2 * math.ceil(Kt / num_cores / 2) * Nt

    total_bytes_transferred = L * num_cores * w_padded_tiles_per_core * bytes_per_tile
    realized_bandwidth = int(total_bytes_transferred / (duration_us * 1000))
    logger.warning(f"Realized Bandwidth: {realized_bandwidth} GB/s")

    total_tiles = Kt * Nt
    total_bytes_used = L * total_tiles * bytes_per_tile
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
