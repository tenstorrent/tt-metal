# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from loguru import logger
import ttnn

import pytest

import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
    torch2tt_tensor,
    get_devices_for_t3000,
    tt2torch_tensor,
)
import torch

NUM_ITERATIONS = 100000

FF1_HANG_PARAMETRIZATION = (1024, 4608, 18432, 4, 72, 3, 1, 8, NUM_ITERATIONS)

CHIP_ID_TO_COORDINATES_T3K = [None] * 8
CHIP_ID_TO_COORDINATES_T3K[0] = (1, 0)
CHIP_ID_TO_COORDINATES_T3K[1] = (1, 1)
CHIP_ID_TO_COORDINATES_T3K[2] = (2, 1)
CHIP_ID_TO_COORDINATES_T3K[3] = (2, 0)
CHIP_ID_TO_COORDINATES_T3K[4] = (0, 0)
CHIP_ID_TO_COORDINATES_T3K[5] = (0, 1)
CHIP_ID_TO_COORDINATES_T3K[6] = (3, 1)
CHIP_ID_TO_COORDINATES_T3K[7] = (3, 0)


# Used to reproduce issue #8665 with matmul 2D (Falcon 7b matmuls)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (FF1_HANG_PARAMETRIZATION, (1024, 4608, 18432, 4, 72, 3, 1, 1, NUM_ITERATIONS)),
    ids=["ff1-hang", "ff1-pass"],
)
@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_reproduce_matmul_2d_hang(
    num_devices,
    all_devices,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
    determinism_check_enabled=False,
    determinism_check_iterations=1,
):
    torch.manual_seed(1234)

    devices = []
    if num_devices == 8:
        devices = get_devices_for_t3000(all_devices, num_devices)
    else:
        devices = all_devices

    logger.info(f"Running on {num_devices} devices")

    in0_block_shard_spec = ttl.tensor.ShardSpec(
        ttl.tensor.CoreRangeSet(
            {
                ttl.tensor.CoreRange(
                    # Volume must match batch size
                    ttl.tensor.CoreCoord(0, 0),
                    ttl.tensor.CoreCoord(7, 7),
                ),
            }
        ),
        [
            128,
            576,
        ],
        ttl.tensor.ShardOrientation.ROW_MAJOR,
        False,
    )
    in0_block_sharded_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1, in0_block_shard_spec
    )

    dram_interleaved_mem_config = ttl.tensor.MemoryConfig(
        ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM
    )

    in1_mem_config = dram_interleaved_mem_config
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1)

    in0_dtype = ttl.tensor.DataType.BFLOAT16
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT16

    a_shape = [1, 1, seq_len, inner_dim]
    b_shape = [1, 1, inner_dim, weights_n]

    num_activation_tensors = 1
    if determinism_check_enabled:
        # If we are running determinism checks, we want to switch activation tensors
        # every time we complete an iteration of a determinism check, to confirm that
        # device is producing new results, and not just reusing an already existing buffer
        num_activation_tensors = 10

    A = []
    for act in range(num_activation_tensors):
        A.append(torch.randn(a_shape))
    B = torch.randn(b_shape)

    a_t = [[None for _ in range(num_devices)] for _ in range(num_activation_tensors)]
    b_t = []

    for device_idx in range(num_devices):
        for act in range(num_activation_tensors):
            a_t[act][device_idx] = torch2tt_tensor(
                A[act], devices[device_idx], ttl.tensor.Layout.TILE, dram_interleaved_mem_config, in0_dtype
            )
        b_t.append(torch2tt_tensor(B, devices[device_idx], ttl.tensor.Layout.TILE, in1_mem_config, in1_dtype))

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=in_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )

    compute_config = ttl.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttl.tensor.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    num_nd_outputs = [0] * num_devices
    reference_out = []
    if determinism_check_enabled:
        reference_out = [[None for _ in range(num_devices)] for _ in range(num_activation_tensors)]
        for device_idx in range(num_devices):
            # First, convert input to sharded config
            for act in range(num_activation_tensors):
                a_sharded = ttnn.experimental.tensor.interleaved_to_sharded(
                    a_t[act][device_idx], sharded_mem_config=in0_block_sharded_mem_config
                )
                output = ttnn.matmul(
                    a_sharded,
                    b_t[device_idx],
                    program_config=program_config,
                    memory_config=out_mem_config,
                    dtype=out_dtype,
                    compute_kernel_config=compute_config,
                )
                reference_out[act][device_idx] = tt2torch_tensor(output)
                output.deallocate(True)
                a_sharded.deallocate(True)

    current_act_tensor = 0
    sharded_activations_per_device = [None] * num_devices
    out = [None] * num_devices

    for device_idx in range(num_devices):
        sharded_activations_per_device[device_idx] = ttl.tensor.interleaved_to_sharded(
            a_t[current_act_tensor][device_idx], sharded_mem_config=in0_block_sharded_mem_config
        )

    logger.info("Starting iterations")
    # loop_count iterations to test determinism/hang
    for i in range(loop_count):
        # run matmul on all devices
        for device_idx in range(num_devices):
            out[device_idx] = ttnn.matmul(
                sharded_activations_per_device[device_idx],
                b_t[device_idx],
                program_config=program_config,
                memory_config=out_mem_config,
                dtype=out_dtype,
                compute_kernel_config=compute_config,
            )

        # synchronize devices in the order of all_devices fixture list; this list ensures we first synchronize
        # all local devices and then all remote devices, to avoid misinterpreting a hang on the remote device
        # caused by problems on the local device it is attached to
        for device_idx in range(num_devices):
            if num_devices != 1:
                if num_devices == 2:
                    logger.info(f"Start sync device id: {device_idx}")
                if num_devices == 8:
                    logger.info(
                        f"Start sync device id: {device_idx} eth coordinates: {CHIP_ID_TO_COORDINATES_T3K[device_idx]}"
                    )
            else:
                logger.info("Start single device sync:")
            ttl.device.Synchronize(all_devices[device_idx])
            if num_devices != 1:
                if num_devices == 2:
                    logger.info(f"End sync device id: {device_idx}")
                if num_devices == 8:
                    logger.info(
                        f"End sync device id: {device_idx} eth coordinates: {CHIP_ID_TO_COORDINATES_T3K[device_idx]}"
                    )
            else:
                logger.info("End single device sync")

        # check if the output matches the first run output
        if determinism_check_enabled and i % determinism_check_iterations == 0:
            for device_idx in range(num_devices):
                pt_out = tt2torch_tensor(out[device_idx])
                if torch.equal(reference_out[current_act_tensor][device_idx], pt_out):
                    logger.info(f"Device {device_idx} PCC: 1.0")
                else:
                    # for determinism check, we avoid calling comp_pcc func as it is heavy and with too many operations,
                    # part of the code that replaces nans/infs with zeros starts leaking memory, even if deallocation is forced,
                    # so we call it only in case we see tensors are not equal
                    _, pcc = comp_pcc(reference_out[current_act_tensor][device_idx], pt_out)
                    logger.info(f"Device {device_idx} PCC: {pcc}")
                    num_nd_outputs[device_idx] += 1
            current_act_tensor = (current_act_tensor + 1) % num_activation_tensors
            # Deallocate previous sharded activations and shard new ones
            logger.info("Switching activation tensor for new determinism iterations")
            for device_idx in range(num_devices):
                sharded_activations_per_device[device_idx].deallocate(True)
                sharded_activations_per_device[device_idx] = ttl.tensor.interleaved_to_sharded(
                    a_t[current_act_tensor][device_idx], sharded_mem_config=in0_block_sharded_mem_config
                )

        for device_idx in range(num_devices):
            out[device_idx].deallocate(True)

        logger.info(f"Iteration = {i}, done")

    if determinism_check_enabled:
        for device_idx in range(num_devices):
            logger.info(f"Number of non-deterministic outputs on device {device_idx} is {num_nd_outputs[device_idx]}")

    for device_idx in range(num_devices):
        out[device_idx].deallocate(True)

    for device_idx in range(num_devices):
        ttl.device.Synchronize(devices[device_idx])


@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
def test_specific_chip_reproduce_matmul_2d_hang_t3000(all_devices, logical_chip_index, use_program_cache):
    num_devices_t3000 = 8
    if len(all_devices) != num_devices_t3000:
        pytest.skip("Test is only valid for t3000 machines")

    logger.info(
        f"Selecting device id: {logical_chip_index} coordinates: {CHIP_ID_TO_COORDINATES_T3K[logical_chip_index]}"
    )
    target_device = all_devices[logical_chip_index]
    devices = [target_device]

    test_reproduce_matmul_2d_hang(1, devices, 1024, 4608, 18432, 4, 72, 3, 1, 8, NUM_ITERATIONS, use_program_cache)


@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (FF1_HANG_PARAMETRIZATION,),
    ids=[
        "ff1-hang",
    ],
)
@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_determinism(
    num_devices,
    all_devices,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
    determinism_check_iterations,
):
    test_reproduce_matmul_2d_hang(
        num_devices,
        all_devices,
        seq_len,
        inner_dim,
        weights_n,
        per_core_M,
        per_core_N,
        in_block_w,
        out_subblock_h,
        out_subblock_w,
        loop_count,
        use_program_cache,
        determinism_check_enabled=True,
        determinism_check_iterations=determinism_check_iterations,
    )


@pytest.mark.parametrize(
    "logical_chip_index",
    [0, 1, 2, 3, 4, 5, 6, 7],
    ids=[
        "logical_chip0",
        "logical_chip1",
        "logical_chip2",
        "logical_chip3",
        "logical_chip4",
        "logical_chip5",
        "logical_chip6",
        "logical_chip7",
    ],
)
@pytest.mark.parametrize(
    "seq_len, inner_dim, weights_n, per_core_M, per_core_N, in_block_w, out_subblock_h, out_subblock_w, loop_count",
    (FF1_HANG_PARAMETRIZATION,),
    ids=[
        "ff1-hang",
    ],
)
def test_determinism_specific_chip(
    all_devices,
    logical_chip_index,
    seq_len,
    inner_dim,
    weights_n,
    per_core_M,
    per_core_N,
    in_block_w,
    out_subblock_h,
    out_subblock_w,
    loop_count,
    use_program_cache,
    determinism_check_iterations,
):
    num_devices_t3000 = 8
    if len(all_devices) != num_devices_t3000:
        pytest.skip("Test is only valid for t3000 machines")

    logger.info(
        f"Selecting device id: {logical_chip_index} coordinates: {CHIP_ID_TO_COORDINATES_T3K[logical_chip_index]}"
    )
    target_device = all_devices[logical_chip_index]
    devices = [target_device]

    test_reproduce_matmul_2d_hang(
        1,
        devices,
        seq_len,
        inner_dim,
        weights_n,
        per_core_M,
        per_core_N,
        in_block_w,
        out_subblock_h,
        out_subblock_w,
        loop_count,
        use_program_cache,
        determinism_check_enabled=True,
        determinism_check_iterations=determinism_check_iterations,
    )
