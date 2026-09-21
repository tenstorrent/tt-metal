# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-owning compiler checks, invoked by the isolated contract launcher."""

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
        ct[offset + 3] = {
            "source-unused": 0xFFFFFFFF,
            "source-data-ready": ct[offset + 1],
            "source-consumer-ready": ct[offset + 2],
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


@pytest.mark.parametrize(
    "case",
    [
        "optional-sender",
        "optional-receiver",
        "direct-sender",
        "direct-receiver",
        "coordinates",
        "old-tag",
        "old-tag-v2",
        "absent-coordinates",
        "mixed-coordinates",
        "data-ready-semaphore",
        "consumer-ready-semaphore",
        "signal-source-semaphore",
    ],
)
def test_compact_compile_contract(device, expect_error, case):
    sender_placement = case in ("optional-sender", "direct-receiver", "coordinates", "old-tag", "old-tag-v2")
    core = (0, 0) if sender_placement else (1, 0)
    action = {
        "optional-sender": 0,
        "optional-receiver": 0,
        "direct-sender": 1,
        "direct-receiver": 2,
        "coordinates": 3,
        "old-tag": 0,
        "old-tag-v2": 0,
        "absent-coordinates": 3,
        "mixed-coordinates": 4,
        "data-ready-semaphore": 5,
        "consumer-ready-semaphore": 6,
        "signal-source-semaphore": 7,
    }[case]
    family = ttnn.McastFamily(device)
    family.add_group(core_set([(0, 0), (1, 0)]), [ttnn.CoreCoord(0, 0)])
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/pipe_compact_contract.cpp",
        core_ranges=core_set([core]),
        compile_time_args=[action, 0, 0],
        runtime_args=[(ttnn.CoreCoord(*core), [0])],
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
    )
    descriptor = ttnn.ProgramDescriptor()
    if case == "mixed-coordinates":
        kernel.core_ranges = core_set([(0, 0), (1, 0)])
        kernel.runtime_args = [(ttnn.CoreCoord(x, 0), [0]) for x in range(2)]
    if case == "absent-coordinates":
        ttnn.attach_absent(kernel, "mcast")
    else:
        family.attach(descriptor, "mcast", [kernel])
    ct = list(kernel.compile_time_args)
    ct[1] = len(kernel.runtime_args[core[0]][core[1]])
    ct[2] = len(ct)
    if case.startswith("old-tag"):
        ct = ct[:3] + [2 if case.endswith("v2") else 1]  # Reject before reading obsolete/short blocks.
    if case == "mixed-coordinates":
        mapped = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        ct.extend([mapped.x, mapped.y])
    kernel.compile_time_args = ct
    descriptor.kernels = [kernel]
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    if case.startswith("optional-") or case == "mixed-coordinates":
        ttnn.generic_op([output, output], descriptor)
    else:
        expected = {
            "direct-sender": "sender pipe is unavailable",
            "direct-receiver": "receiver pipe is unavailable",
            "coordinates": "Sender coordinates are unavailable",
            "old-tag": "Unsupported multicast wire tag",
            "old-tag-v2": "Unsupported multicast wire tag",
            "absent-coordinates": "has no member named 'sender_x'",
            "data-ready-semaphore": "has no member named 'data_ready'",
            "consumer-ready-semaphore": "has no member named 'consumer_ready'",
            "signal-source-semaphore": "has no member named 'signal_source'",
        }[case]
        with expect_error(RuntimeError, expected):
            ttnn.generic_op([output, output], descriptor)
