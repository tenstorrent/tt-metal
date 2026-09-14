# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact family broadcasts, concurrent groups, wrapper-independent public API and spectators."""

import pytest
import torch
import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import attach_for_inspection


def _cores(coords):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(set(coords))]
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("chain_link", [False, True])
def test_non_worker_gap_preserves_worker_holes(device, noc, counter, chain_link):
    # Three logical row segments become four rectangles if the virtual NoC gap is treated
    # as missing receivers. Spectators in the partial rows must still remain untouched.
    receivers = [(x, 0) for x in range(2, 9)] + [(x, 1) for x in range(9)] + [(x, 2) for x in range(4)]
    _run(device, [(receivers, [(2, 0)])], noc=noc, counter=counter, chain_link=chain_link, min_rectangles=3)


@pytest.mark.parametrize("noc", [0, 1])
def test_caller_managed_multicast_receiver(device, noc):
    _run(device, [([(0, 0), (1, 0)], [(0, 0)])], noc=noc, receiver_caller_managed=True)


def _run(
    device,
    specs,
    *,
    noc=0,
    counter=False,
    control=False,
    caller_managed=False,
    receiver_caller_managed=False,
    dynamic=True,
    handshake=True,
    rounds=6,
    zero_ack=False,
    adopted=False,
    chain_link=False,
    large=False,
    mixed_events=False,
    delayed=False,
    min_rectangles=None,
    typed_bindings=False,
):
    # specs are (exact logical receivers, ordered logical senders).
    all_coords = {c for receivers, senders in specs for c in receivers + senders}
    width = max(c[0] for c in all_coords) + 2
    height = max(c[1] for c in all_coords) + 1
    size = device.compute_with_storage_grid_size()
    if width > size.x or height > size.y:
        pytest.skip("requires a larger worker grid")
    dispatch = [(x, y) for y in range(height) for x in range(width)]
    participants = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))])
    chained = chain_link and any(
        len(receivers)
        != (max(x for x, _ in receivers) - min(x for x, _ in receivers) + 1)
        * (max(y for _, y in receivers) - min(y for _, y in receivers) + 1)
        for receivers, _ in specs
    )
    semaphore_count = (2 if handshake else 1) + int(chained)
    config = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
        handshake=handshake,
        data_ready=ttnn.McastDataReady.Counter if counter else ttnn.McastDataReady.Flag,
        ack_count_override=0 if zero_ack else None,
        sem_ids=list(range(semaphore_count)) if adopted else None,
        irregular_receiver_set_mode=(ttnn.TransferMode.ChainUnicast if chain_link else ttnn.TransferMode.Multicast),
    )
    family = ttnn.McastFamily(device, config)
    for receivers, senders in specs:
        family.add_group(_cores(receivers), [ttnn.CoreCoord(*c) for c in senders])
    family.prepare_arguments()
    barrier = ttnn.McastFamily(
        device,
        ttnn.McastConfig(noc=config.noc),
    )
    barrier.add_group(participants, [ttnn.CoreCoord(0, 0)])
    barrier.prepare_arguments()
    max_pages = 20 if large else 2
    payload = (
        torch.arange(1, len(specs) * rounds * max_pages + 1, dtype=torch.bfloat16)
        .reshape(-1, 1, 1, 1)
        .expand(-1, 1, 32, 32)
        .contiguous()
    )
    input_tensor = ttnn.from_torch(payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    stride = rounds * max_pages + 1
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(dispatch) * stride, 1, 32, 32]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    ct = [
        rounds,
        int(control),
        int(caller_managed),
        int(dynamic),
        max_pages,
        int(mixed_events),
        int(delayed),
        int(receiver_caller_managed),
    ]
    ct += list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct += list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for index, (x, y) in enumerate(dispatch):
        group_index = next((i for i, (rx, tx) in enumerate(specs) if (x, y) in rx + tx), None)
        inside = group_index is not None and (x, y) in specs[group_index][0]
        rt[x][y] = [
            input_tensor.buffer_address(),
            output_tensor.buffer_address(),
            (group_index or 0) * rounds * max_pages,
            index * stride,
            int(inside),
            int(group_index is None),
        ]
    kernels = []
    faces = [True, False] if zero_ack else [None]
    sender_cores = {core for _, senders in specs for core in senders}
    for sending in faces:
        selected = [c for c in dispatch if sending is None or (c in sender_cores) == sending]
        face_rt = ttnn.RuntimeArgs()
        for x, y in selected:
            face_rt[x][y] = list(rt[x][y])
        face_ct = list(ct)
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=(
                    "tests/ttnn/unit_tests/kernel_lib/kernels/pipe_family_typed.cpp"
                    if typed_bindings
                    else "tests/ttnn/unit_tests/kernel_lib/kernels/pipe_family.cpp"
                ),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=_cores(selected),
                compile_time_args=face_ct,
                runtime_args=face_rt,
                config=ttnn.WriterConfigDescriptor() if noc else ttnn.ReaderConfigDescriptor(),
            )
        )
    semaphores = (
        [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=family.participating_cores(), initial_value=0)
            for i in range(semaphore_count)
        ]
        if adopted
        else []
    )
    descriptor = ttnn.ProgramDescriptor(semaphores=semaphores)
    family.attach(descriptor, "mcast", kernels[:1] if zero_ack else kernels)
    mcast_ct = dict(kernels[0].named_compile_time_args)["mcast_ct_offset"]
    mcast_rt = dict(kernels[0].named_compile_time_args)["mcast_rt_offset"]
    attached_ct = kernels[0].compile_time_args[mcast_ct:]
    if zero_ack:
        passive = ttnn.McastFamily(
            device,
            ttnn.McastConfig(noc=config.noc, handshake=False, data_ready=config.data_ready, sem_ids=[attached_ct[2]]),
        )
        for receivers, senders in specs:
            passive.add_group(_cores(receivers), [ttnn.CoreCoord(*c) for c in senders])
        passive.prepare_arguments()
        passive.attach(descriptor, "mcast", kernels[1:])
    if chain_link:
        assert (attached_ct[5] >> 3) & 3 == int(chained)
        if chained:
            assert attached_ct[10:] == [0, 2]
        for receivers, senders in specs:
            sender = ttnn.CoreCoord(*senders[0])
            fanout = len(receivers) - int(senders[0] in receivers)
            assert kernels[0].runtime_args[sender.x][sender.y][mcast_rt:][1] == (int(fanout > 0) if chained else fanout)
    if min_rectangles is not None:
        # Guard cases whose point is an irregular mapping: they must not degrade into dense sets on this grid.
        # Chain arguments omit rectangles; inspect the same geometry in multicast mode.
        geometry_family = ttnn.McastFamily(device, ttnn.McastConfig(noc=config.noc))
        for receivers, senders in specs:
            geometry_family.add_group(_cores(receivers), [ttnn.CoreCoord(*c) for c in senders])
        geometry_family.prepare_arguments()
        _, geometry_kernel = attach_for_inspection(geometry_family, participants, config.noc)
        for _, senders in specs:
            x, y = senders[0]
            assert geometry_kernel.runtime_args[x][y][0] >= min_rectangles
    for kernel in kernels:
        ttnn.attach_absent(kernel, "absent_mcast")
    barrier.attach(descriptor, "barrier_mcast", kernels)
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096 * max_pages,
            core_ranges=participants,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in (0, 1)
    ]
    descriptor.cbs = cbs
    descriptor.kernels = kernels
    output = ttnn.generic_op(
        [input_tensor, output_tensor],
        descriptor,
    )
    actual = ttnn.to_torch(output).reshape(len(dispatch), stride, 1, 32, 32)
    for index, coord in enumerate(dispatch):
        if coord not in all_coords:
            assert torch.all(actual[index, -1].contiguous().view(torch.int32) == 0x5A5A5A5A), coord
        for group_index, (receivers, _) in enumerate(specs):
            if coord not in receivers:
                continue
            for r in range(rounds):
                pages = (max_pages if r % 2 else 1) if dynamic else 1
                if control or (mixed_events and r % 2):
                    assert actual[index, max_pages * r].contiguous().view(torch.int32).flatten()[0].item() == (
                        r + 1 if counter else 7 + (r % 3 if mixed_events else 0)
                    )
                else:
                    for p in range(pages):
                        assert torch.equal(
                            actual[index, max_pages * r + p],
                            payload[group_index * rounds * max_pages + max_pages * r + p],
                        ), (
                            coord,
                            r,
                            p,
                        )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("caller_managed", [False, True])
def test_fixed_families(device, noc, counter, control, caller_managed):
    _run(
        device,
        [
            ([(0, 0), (2, 0), (3, 0), (0, 1)], [(0, 0)]),
            ([(3, 2), (4, 2)], [(2, 2)]),
            ([(1, 3)], [(1, 3)]),
        ],
        noc=noc,
        counter=counter,
        control=control,
        caller_managed=caller_managed,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("control", [False, True])
def test_rotating_families(device, noc, counter, control):
    _run(
        device,
        [
            ([(0, 0), (2, 0), (3, 0), (0, 1)], [(0, 0), (4, 0)]),
            ([(1, 2)], [(1, 2), (3, 2)]),
            ([(0, 3), (2, 3)], [(0, 3), (2, 3)]),
        ],
        noc=noc,
        counter=counter,
        control=control,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
def test_family_no_handshake(device, noc, counter):
    _run(device, [([(2, 0), (4, 0)], [(0, 0)])], noc=noc, counter=counter, handshake=False, rounds=1)


@pytest.mark.parametrize("noc", [0, 1])
def test_family_staircase_and_gap(device, noc):
    # Staircase with the sender in its local singleton, plus a dense logical row crossing BH's gap.
    _run(
        device,
        [([(7, 0), (0, 1), (1, 1), (2, 1), (0, 2)], [(7, 0)]), ([(x, 3) for x in range(9)], [(0, 3)])],
        noc=noc,
        counter=True,
    )


@pytest.mark.parametrize("noc", [0, 1])
def test_family_three_rectangles(device, noc):
    # Three isolated destinations exercise the maximum supported owning argument array.
    _run(device, [([(0, 0), (2, 0), (4, 0)], [(1, 0)])], noc=noc)


@pytest.mark.parametrize("counter", [False, True])
def test_family_zero_ack(device, counter):
    _run(device, [([(2, 0), (4, 0)], [(0, 0)])], zero_ack=True, counter=counter, rounds=1)


@pytest.mark.parametrize("counter", [False, True])
def test_family_adopted_semaphores(device, counter):
    _run(device, [([(0, 0), (2, 0), (3, 0)], [(0, 0)]), ([(0, 2), (2, 2)], [(0, 2)])], adopted=True, counter=counter)
