# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Invalid chain-receiver call sites must fail in the real kernel compiler, before dispatch."""

import pytest
import ttnn
from tests.ttnn.unit_tests.kernel_lib.test_mcast_family import _cores


@pytest.mark.parametrize("with_dense_group", [False, True], ids=["irregular", "with-dense-group"])
@pytest.mark.parametrize("violation", ["chain-receive", "wrong-noc"])
def test_forwarding_receive_compile_contract(device, expect_error, with_dense_group, violation):
    groups = [ttnn.McastGroup(_cores([(0, 0), (2, 0)]), [ttnn.CoreCoord(0, 0)])]
    if with_dense_group:
        groups.append(ttnn.McastGroup(_cores([(0, 2), (1, 2)]), [ttnn.CoreCoord(0, 2)]))
    family = ttnn.McastFamily(
        device, groups, ttnn.McastConfig(noc=ttnn.NOC.NOC_1), ttnn.IrregularReceiverSetMode.ChainLink
    )
    receiver = ttnn.CoreCoord(1, 2) if with_dense_group else ttnn.CoreCoord(2, 0)
    runtime = ttnn.RuntimeArgs()
    runtime[receiver.x][receiver.y] = family.runtime_args(receiver)
    kernel = ttnn.KernelDescriptor(
        kernel_source="tests/ttnn/unit_tests/kernel_lib/kernels/pipe_receive_contract.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_cores([(receiver.x, receiver.y)]),
        compile_time_args=list(family.compile_time_args()) + [int(violation != "chain-receive")],
        runtime_args=runtime,
        # The family is on NOC_1; pin the mismatching kernel NoC explicitly instead of relying on the reader default.
        config=(
            ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
            if violation == "wrong-noc"
            else ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        ),
    )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    expected = {
        "chain-receive": "has no member named 'receive'",
        "wrong-noc": "forwarding receiver kernel NoC",
    }[violation]
    with expect_error(RuntimeError, expected):
        ttnn.generic_op(
            [output, output], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=family.owned_semaphores())
        )
