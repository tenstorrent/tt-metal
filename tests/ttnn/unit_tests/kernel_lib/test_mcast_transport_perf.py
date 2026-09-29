# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare one-shot multicast and chain latency for the same broad irregular fanout."""

import pytest
import torch
import ttnn
from loguru import logger

from models.perf.device_perf_utils import run_device_perf_detailed
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import (
    KERNEL_DIR,
    core_set,
    inspect_mcast_ct,
    make_cb,
    tile_pattern,
)

pytestmark = pytest.mark.models_device_performance_bare_metal

PERF_FILE = "tests/ttnn/unit_tests/kernel_lib/test_mcast_transport_perf.py"
KERNEL = f"{KERNEL_DIR}/mcast_transport_perf.cpp"
OP = "GenericOpDeviceOperation"
PAYLOAD_PAGES = 8
INVOCATIONS = 20
RECEIVERS = [(x, y) for y in range(8) if y != 4 for x in range(8)]
SENDER = (0, 4)


def _program(device, mode, input_tensor, output_tensor):
    participants = core_set([SENDER, *RECEIVERS])
    transfer = ttnn.TransferMode.Multicast if mode == "multicast" else ttnn.TransferMode.ChainUnicast
    mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0,
            handshake=True,
            data_ready=ttnn.McastDataReady.Flag,
            irregular_receiver_set_mode=transfer,
        ),
        core_set(RECEIVERS),
        len(RECEIVERS),
        ttnn.McastExplicitSenderConfig([[ttnn.CoreCoord(*SENDER)]]),
    )

    compile_time_args = [PAYLOAD_PAGES]
    compile_time_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    compile_time_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    runtime_args = ttnn.RuntimeArgs()
    ranks = {core: rank for rank, core in enumerate(RECEIVERS)}
    for x, y in [SENDER, *RECEIVERS]:
        runtime_args[x][y] = [
            input_tensor.buffer_address(),
            output_tensor.buffer_address(),
            ranks.get((x, y), 0xFFFFFFFF),
        ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=participants,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    descriptor = ttnn.ProgramDescriptor(cbs=[make_cb(index, participants, pages=PAYLOAD_PAGES) for index in (0, 1)])
    mcast.attach(descriptor, "mcast", [kernel], 0)
    metadata = inspect_mcast_ct(kernel)
    assert ((metadata["flags"] >> 3) & 3, metadata["capacity"]) == ((0, 2) if mode == "multicast" else (1, 0))
    descriptor.kernels = [kernel]
    return descriptor


@pytest.mark.parametrize("mode", ["multicast", "chain"])
def test_profile_fixture(device, mode):
    payload = tile_pattern(PAYLOAD_PAGES)
    input_tensor = ttnn.from_torch(
        payload,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(RECEIVERS) * PAYLOAD_PAGES, 1, 32, 32]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    descriptor = _program(device, mode, input_tensor, output_tensor)
    for _ in range(INVOCATIONS):
        ttnn.generic_op([input_tensor, output_tensor], descriptor)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    actual = ttnn.to_torch(output_tensor).reshape(len(RECEIVERS), PAYLOAD_PAGES, 1, 32, 32)
    for rank in range(len(RECEIVERS)):
        assert torch.equal(actual[rank], payload), f"{mode}: receiver rank {rank} data mismatch"


def _device_kernel_ns(mode):
    results = run_device_perf_detailed(
        command=f'pytest "{PERF_FILE}::test_profile_fixture[mode={mode}]" -v',
        subdir=f"mcast_broad_fanout_{mode}",
        cols=["DEVICE KERNEL"],
        op_name=OP,
        warmup_iters=2,
    )
    return results["DEVICE KERNEL"]["AVG"]


def test_multicast_outperforms_chain_on_broad_fanout():
    multicast_ns = _device_kernel_ns("multicast")
    chain_ns = _device_kernel_ns("chain")
    logger.info(
        f"broad-fanout transport | multicast={multicast_ns:.0f} ns | chain={chain_ns:.0f} ns | "
        f"chain/multicast={chain_ns / multicast_ns:.3f}x"
    )
    assert multicast_ns < chain_ns, (
        f"two-rectangle multicast did not outperform the 56-receiver chain: "
        f"{multicast_ns:.0f} ns >= {chain_ns:.0f} ns"
    )
