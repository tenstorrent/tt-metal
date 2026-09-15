# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Signal lifetime and first-hop progress without per-round output barriers."""

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import core_set


def _stress(device, noc, counter, events, guards, includes_sender, reverse_channel=False):
    coords = [(0, 0), (2, 0), (4, 0)]
    cores = core_set(coords)
    signal = ttnn.McastDataReady.Counter if counter else ttnn.McastDataReady.Flag
    noc_id = ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0
    family = ttnn.McastFamily(
        device,
        ttnn.McastConfig(noc=noc_id, data_ready=signal, irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast),
    )
    family.add_group(core_set(coords if includes_sender else coords[1:]), [ttnn.CoreCoord(*coords[0])])
    family.prepare_arguments()
    reverse = None
    if reverse_channel:
        reverse = ttnn.McastFamily(device, ttnn.McastConfig(noc=noc_id, data_ready=signal))
        reverse.add_group(core_set([coords[0]]), [ttnn.CoreCoord(*coords[2])])
        reverse.prepare_arguments()
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([3, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ct = [events, int(guards[0]), int(guards[1]), int(includes_sender)]
    ct += list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for rank, (x, y) in enumerate(coords):
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
    family.attach(program, "chain_mcast", [kernel])
    offset = dict(kernel.named_compile_time_args)["chain_mcast_ct_offset"]
    assert len(kernel.compile_time_args[offset:]) == 12 and kernel.compile_time_args[offset + 11] == 2
    if reverse:
        reverse.attach(program, "reverse_mcast", [kernel])
    else:
        ttnn.attach_absent(kernel, "reverse_mcast")
    program.kernels = [kernel]
    # A fresh invocation must initialize semaphore-backed Counter progression again, including cache hits.
    for _ in range(2):
        actual = ttnn.generic_op([output, output], program)
        assert torch.count_nonzero(ttnn.to_torch(actual).contiguous().view(torch.int32)) == 0


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
@pytest.mark.parametrize("events", [0, 1, 2], ids=["payload", "control", "mixed"])
@pytest.mark.parametrize(
    "guards",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["guard-both", "caller-sender", "caller-receiver", "caller-both"],
)
@pytest.mark.parametrize("includes_sender", [False, True], ids=["external", "local-copy"])
def test_chain_signal_lifetime(device, noc, counter, events, guards, includes_sender):
    _stress(device, noc, counter, events, guards, includes_sender)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
def test_chain_interdependent_channels(device, noc, counter):
    _stress(device, noc, counter, 2, (True, True), False, reverse_channel=True)
