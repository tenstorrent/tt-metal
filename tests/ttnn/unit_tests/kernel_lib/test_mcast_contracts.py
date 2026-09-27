# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-attached kernel compile contracts; native launchers live in mcast_native_tests.py."""

import pytest
import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import core_set, KERNEL_DIR


@pytest.mark.parametrize(
    "violation,with_dense_group",
    [
        ("chain-receive", False),
        ("chain-receive", True),
        ("wrong-noc", False),
        ("wrong-noc", True),
        ("source-unused", False),
        ("source-data-ready", False),
        ("source-consumer-ready", False),
    ],
)
def test_forwarding_receive_compile_contract(device, expect_error, with_dense_group, violation):
    family = ttnn.McastFamily(
        device,
        ttnn.McastConfig(noc=ttnn.NOC.NOC_1, irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast),
    )
    family.add_group(core_set([(0, 0), (2, 0)]), [ttnn.CoreCoord(0, 0)])
    if with_dense_group:
        family.add_group(core_set([(0, 2), (1, 2)]), [ttnn.CoreCoord(0, 2)])
    family.prepare_arguments()
    receiver = ttnn.CoreCoord(1, 2) if with_dense_group else ttnn.CoreCoord(2, 0)
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_receive_contract.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set([(receiver.x, receiver.y)]),
        compile_time_args=[int(violation != "chain-receive")],
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
    )
    descriptor = ttnn.ProgramDescriptor()
    family.attach(descriptor, "mcast", [kernel])
    ct = list(kernel.compile_time_args)
    offset = dict(kernel.named_compile_time_args)["mcast_ct_offset"]
    if violation.startswith("source-"):
        ct[offset + 11] = {
            "source-unused": 0xFFFFFFFF,
            "source-data-ready": ct[offset + 2],
            "source-consumer-ready": ct[offset + 3],
        }[violation]
        kernel.compile_time_args = ct
    if violation == "wrong-noc":
        # Bypass host validation deliberately to exercise the device compile contract.
        kernel.config = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
        )
    descriptor.kernels = [kernel]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    expected = {
        "chain-receive": "has no member named 'receive'",
        "wrong-noc": "forwarding receiver kernel NoC",
        "source-unused": "distinct signal-source semaphore",
        "source-data-ready": "distinct signal-source semaphore",
        "source-consumer-ready": "distinct signal-source semaphore",
    }[violation]
    with expect_error(RuntimeError, expected):
        ttnn.generic_op([output, output], descriptor)
