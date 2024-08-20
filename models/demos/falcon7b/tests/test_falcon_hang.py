import pytest
from loguru import logger
import ttnn

import tt_lib as ttl
from models.utility_functions import (
    comp_pcc,
    tt2torch_tensor,
    get_devices_for_t3000,
)
import torch

CHIP_ID_TO_COORDINATES_T3K = [None] * 8
CHIP_ID_TO_COORDINATES_T3K[0] = (1, 0)
CHIP_ID_TO_COORDINATES_T3K[1] = (1, 1)
CHIP_ID_TO_COORDINATES_T3K[2] = (2, 1)
CHIP_ID_TO_COORDINATES_T3K[3] = (2, 0)
CHIP_ID_TO_COORDINATES_T3K[4] = (0, 0)
CHIP_ID_TO_COORDINATES_T3K[5] = (0, 1)
CHIP_ID_TO_COORDINATES_T3K[6] = (3, 1)
CHIP_ID_TO_COORDINATES_T3K[7] = (3, 0)


@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_reproduce_lm_head_nd_32(
    all_devices, num_devices, use_program_cache, determinism_check_enabled=False, determinism_check_iterations=1
):
    devices = []
    if num_devices == 8:
        devices = get_devices_for_t3000(all_devices, num_devices)
    else:
        devices = all_devices

    logger.info(f"Running on: {num_devices} devices.")
    in0_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)
    in1_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM)
    out_mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1)

    in0_dtype = ttl.tensor.DataType.BFLOAT8_B
    in1_dtype = ttl.tensor.DataType.BFLOAT8_B
    out_dtype = ttl.tensor.DataType.BFLOAT8_B

    torch.manual_seed(1234)

    seq_len = 32
    a_shape = [1, 1, seq_len, 4544]
    b_shape = [1, 1, 4544, 65024]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a_t = []
    b_t = []

    for device_idx in range(num_devices):
        a_t.append(ttl.tensor.Tensor(A, in0_dtype).to(ttl.tensor.Layout.TILE).to(devices[device_idx], in0_mem_config))
        b_t.append(ttl.tensor.Tensor(B, in1_dtype).to(ttl.tensor.Layout.TILE).to(devices[device_idx], in1_mem_config))

    bias_t = None

    mm_prog_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=1,
        per_core_N=32,
        out_subblock_h=1,
        out_subblock_w=8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    wh_compute_kernel_config = ttnn.experimental.tensor.WormholeComputeKernelConfig(
        math_fidelity=ttnn.experimental.tensor.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    num_nd_outputs = [0] * num_devices
    out = []
    reference_out = []

    for device_idx in range(num_devices):
        out.append(
            ttnn.matmul(
                a_t[device_idx],
                b_t[device_idx],
                program_config=mm_prog_config,
                memory_config=out_mem_config,
                dtype=out_dtype,
                compute_kernel_config=wh_compute_kernel_config,
            )
        )

    if determinism_check_enabled:
        for device_idx in range(num_devices):
            reference_out.append(tt2torch_tensor(out[device_idx]))

    logger.info("Starting iterations")
    for i in range(100000):
        # run matmul on all devices
        for device_idx in range(num_devices):
            out[device_idx].deallocate(True)
            out[device_idx] = ttnn.matmul(
                a_t[device_idx],
                b_t[device_idx],
                program_config=mm_prog_config,
                memory_config=out_mem_config,
                dtype=out_dtype,
                compute_kernel_config=wh_compute_kernel_config,
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
                if torch.equal(reference_out[device_idx], pt_out):
                    logger.info(f"Device {device_idx} PCC: 1.0")
                else:
                    # for determinism check, we avoid calling comp_pcc func as it is heavy and with too many operations,
                    # part of the code that replaces nans/infs with zeros starts leaking memory, even if deallocation is forced,
                    # so we call it only in case we see tensors are not equal
                    _, pcc = comp_pcc(reference_out[device_idx], pt_out)
                    logger.info(f"Device {device_idx} PCC: {pcc}")
                    num_nd_outputs[device_idx] += 1

        logger.info(f"Iteration = {i}")

    if determinism_check_enabled:
        for device_idx in range(num_devices):
            logger.info(f"Number of non-deterministic outputs on device {device_idx} is {num_nd_outputs[device_idx]}")


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
def test_specific_chip_lm_head_nd_32_t3000(all_devices, logical_chip_index, use_program_cache):
    num_devices_t3000 = 8
    if len(all_devices) != num_devices_t3000:
        pytest.skip("Test is only valid for t3000 machines")

    logger.info(
        f"Selecting device id: {logical_chip_index} eth coordinates: {CHIP_ID_TO_COORDINATES_T3K[logical_chip_index]}"
    )
    target_device = all_devices[logical_chip_index]
    devices = [target_device]
    test_reproduce_lm_head_nd_32(devices, 1, use_program_cache)


@pytest.mark.parametrize("num_devices", [1, 2, 8], ids=["1chips", "2chips", "8chips"])
def test_determinism(all_devices, num_devices, use_program_cache, determinism_check_iterations):
    test_reproduce_lm_head_nd_32(
        all_devices,
        num_devices,
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
def test_determinism_specific_chip(all_devices, logical_chip_index, use_program_cache, determinism_check_iterations):
    num_devices_t3000 = 8
    if len(all_devices) != num_devices_t3000:
        pytest.skip("Test is only valid for t3000 machines")

    logger.info(
        f"Selecting device id: {logical_chip_index} eth coordinates: {CHIP_ID_TO_COORDINATES_T3K[logical_chip_index]}"
    )
    target_device = all_devices[logical_chip_index]
    devices = [target_device]

    test_reproduce_lm_head_nd_32(
        devices,
        1,
        use_program_cache,
        determinism_check_enabled=True,
        determinism_check_iterations=determinism_check_iterations,
    )
