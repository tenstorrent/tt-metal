# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Signal lifetime and first-hop progress without per-round output barriers."""

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import core_set

SIGNAL_LIFETIME_CASES = (
    pytest.param(1, False, 0, (False, False), False, id="noc1-flag-payload-guard-both-external"),
    pytest.param(0, True, 0, (True, False), True, id="noc0-counter-payload-caller-sender-local"),
    pytest.param(1, True, 0, (False, True), False, id="noc1-counter-payload-caller-receiver-external"),
    pytest.param(0, False, 0, (True, True), False, id="noc0-flag-payload-caller-both-external"),
    pytest.param(1, True, 1, (False, False), True, id="noc1-counter-control-guard-both-local"),
    pytest.param(1, False, 1, (True, False), True, id="noc1-flag-control-caller-sender-local"),
    pytest.param(0, False, 1, (False, True), False, id="noc0-flag-control-caller-receiver-external"),
    pytest.param(1, True, 1, (True, True), True, id="noc1-counter-control-caller-both-local"),
    pytest.param(0, True, 2, (False, False), True, id="noc0-counter-mixed-guard-both-local"),
    pytest.param(1, False, 2, (True, False), False, id="noc1-flag-mixed-caller-sender-external"),
    pytest.param(0, True, 2, (False, True), True, id="noc0-counter-mixed-caller-receiver-local"),
    pytest.param(0, True, 2, (True, True), False, id="noc0-counter-mixed-caller-both-external"),
)


def _stress(device, noc, counter, events, guards, includes_sender, reverse_channel=False):
    coords = [(0, 0), (2, 0), (4, 0)]
    cores = core_set(coords)
    signal = ttnn.McastDataReady.Counter if counter else ttnn.McastDataReady.Flag
    noc_id = ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0
    receiver_coords = coords if includes_sender else coords[1:]
    mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=noc_id, data_ready=signal, irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast),
        core_set(receiver_coords),
        len(receiver_coords),
        ttnn.McastExplicitSenderConfig([[ttnn.CoreCoord(*coords[0])]]),
    )
    reverse = None
    if reverse_channel:
        reverse = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=noc_id, data_ready=signal),
            core_set([coords[0]]),
            1,
            ttnn.McastExplicitSenderConfig([[ttnn.CoreCoord(*coords[2])]]),
        )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([3, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ct = [events, int(guards[0]), int(guards[1]), int(includes_sender)]
    ct += list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for rank, (x, y) in enumerate(coords):
        core = ttnn.CoreCoord(x, y)
        rt[x][y] = [output.buffer_address(), rank]
    kernel = ttnn.KernelDescriptor(
        kernel_source="tests/ttnn/unit_tests/kernel_lib/kernels/chain_signal_stress.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor() if noc else ttnn.ReaderConfigDescriptor(),
    )
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i, size in enumerate([65536, 65536, 2048])
    ]
    program = ttnn.ProgramDescriptor(cbs=cbs)
    mcast.attach(program, "chain_mcast", [kernel], 0)
    offset = dict(kernel.named_compile_time_args)["chain_mcast_ct_offset"]
    assert len(kernel.compile_time_args[offset:]) == 4 and kernel.compile_time_args[offset + 3] == 2
    if reverse:
        reverse.attach(program, "reverse_mcast", [kernel], mcast.next_semaphore_id())
    else:
        ttnn.attach_absent_mcast(kernel, "reverse_mcast")
    program.kernels = [kernel]
    # A fresh invocation must initialize semaphore-backed Counter progression again, including cache hits.
    for _ in range(2):
        actual = ttnn.generic_op([output, output], program)
        assert torch.count_nonzero(ttnn.to_torch(actual).contiguous().view(torch.int32)) == 0


@pytest.mark.parametrize("noc,counter,events,guards,includes_sender", SIGNAL_LIFETIME_CASES)
def test_chain_signal_lifetime(device, noc, counter, events, guards, includes_sender):
    _stress(device, noc, counter, events, guards, includes_sender)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
def test_chain_interdependent_channels(device, noc, counter):
    _stress(device, noc, counter, 2, (True, True), False, reverse_channel=True)
