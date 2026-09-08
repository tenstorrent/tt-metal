# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn


def _run(device, width, senders, rotating, noc, counter, control, alternating, caller_managed, handshake=True):
    rounds = 4 if handshake else 1
    if device.compute_with_storage_grid_size().x < max(width, max(senders) + 1):
        pytest.skip("requires more worker columns")
    receivers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, 0))])
    sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, 0)) for x in senders])
    participant_width = max(width, max(senders) + 1)
    participants = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(participant_width - 1, 0))])
    config = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
        handshake=handshake,
        data_ready=ttnn.McastDataReady.Counter if counter else ttnn.McastDataReady.Flag,
    )
    mc = ttnn.Mcast2D(
        device,
        receivers,
        (
            ttnn.Mcast2DRotatingSenderConfig(sender_grid=sender_grid)
            if rotating
            else ttnn.Mcast2DFixedSenderConfig(ttnn.CoreCoord(senders[0], 0))
        ),
        config,
    )
    payload = (
        torch.arange(1, rounds + 1, dtype=torch.bfloat16)
        .reshape(rounds, 1, 1, 1)
        .expand(rounds, 1, 32, 32)
        .contiguous()
    )
    input_tensor = ttnn.from_torch(payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([width * rounds, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    ct = list(mc.compile_time_args()) + [rounds, int(alternating), int(caller_managed), int(control)]
    ct += list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct += list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for x in range(participant_width):
        rt[x][0] = [input_tensor.buffer_address(), output_tensor.buffer_address(), x * rounds, int(x < width)] + list(
            mc.runtime_args(ttnn.CoreCoord(x, 0))
        )
    cbs = [
        ttnn.CBDescriptor(
            total_size=2048,
            core_ranges=participants,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in [0, 1]
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source="tests/ttnn/unit_tests/kernel_lib/kernels/pipe_prepared_matrix.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=participants,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor() if noc else ttnn.ReaderConfigDescriptor(),
    )
    result = ttnn.generic_op(
        [input_tensor, output_tensor],
        ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs, semaphores=mc.owned_semaphores()),
    )
    actual = ttnn.to_torch(result).reshape(width, rounds, 1, 32, 32)
    for core in range(width):
        for r in range(rounds):
            if control:
                assert actual[core, r].contiguous().view(torch.int32).flatten()[0].item() == (r + 1 if counter else 1)
            else:
                assert torch.equal(actual[core, r], payload[r])


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("caller_managed", [False, True])
@pytest.mark.parametrize(
    "width,senders,rotating", [(1, [0], False), (2, [0], False), (2, [0, 2], True), (9, [0], False)]
)
def test_alternating_prepared_payload(device, noc, caller_managed, width, senders, rotating):
    _run(device, width, senders, rotating, noc, False, False, True, caller_managed)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize(
    "width,senders,rotating", [(1, [0], False), (2, [0], False), (2, [2], False), (2, [0, 2], True)]
)
def test_prepared_control(device, noc, counter, width, senders, rotating):
    _run(device, width, senders, rotating, noc, counter, True, False, False)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
def test_prepared_no_handshake(device, noc, counter):
    _run(device, 2, [2], False, noc, counter, False, True, True, handshake=False)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("control", [False, True])
def test_mixed_local_only_sender_turn(device, noc, control):
    # The helper-wide remote flag is true, but core 0's own sender turn is LocalCopy.
    # Flag is intentional: the known rotating Counter early-return issue is mock-only.
    _run(device, 1, [0, 1], True, noc, False, control, True, True)
