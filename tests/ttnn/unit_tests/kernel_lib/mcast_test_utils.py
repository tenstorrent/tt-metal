# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared host attachment and data checks for multicast integration tests."""

from typing import NamedTuple

import pytest
import torch
import ttnn

KERNEL_DIR = "tests/ttnn/unit_tests/kernel_lib/kernels"
TILE_BYTES = 2048


class Group(NamedTuple):
    receivers: list[tuple[int, int]]
    senders: list[tuple[int, int]]


def core_set(coords):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(set(coords))]
    )


def make_cb(index, cores, *, pages=1, page_bytes=TILE_BYTES, dtype=ttnn.bfloat16):
    return ttnn.CBDescriptor(
        total_size=pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def tile_pattern(pages):
    # Bounded integer values are exactly representable in bf16. Two element bands
    # encode the page, so successive pages remain distinct beyond bf16's 256 limit.
    page = torch.arange(pages, dtype=torch.int32).reshape(-1, 1)
    element = torch.arange(1024, dtype=torch.int32).reshape(1, -1)
    values = (element * 17 + page % 127 + (element // 512) * (page // 127)) % 127 - 63
    return values.to(torch.bfloat16).reshape(pages, 1, 32, 32)


def attach_for_inspection(family, cores, noc=ttnn.NOC.NOC_0, semaphores=()):
    kernel = ttnn.KernelDescriptor(
        kernel_source="inspection-only.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=noc),
    )
    descriptor = ttnn.ProgramDescriptor(semaphores=list(semaphores))
    family.attach(descriptor, "mcast", [kernel])
    return descriptor, kernel


def run_family_case(
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
):
    """Build a real family, then exercise its attached kernel arguments."""
    specs = [Group(*group) for group in specs]
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
        family.add_group(core_set(receivers), [ttnn.CoreCoord(*c) for c in senders])
    family.prepare_arguments()
    _run_channel(
        device,
        family,
        specs,
        config,
        rounds=rounds,
        control=control,
        caller_managed=caller_managed,
        receiver_caller_managed=receiver_caller_managed,
        dynamic=dynamic,
        max_pages=20 if large else 2,
        mixed_events=mixed_events,
        delayed=delayed,
        zero_ack=zero_ack,
        adopted=adopted,
        expected_chain=chained if chain_link else None,
        min_rectangles=min_rectangles,
    )


def run_wrapper_case(
    device,
    *,
    width,
    senders,
    rotating,
    noc=0,
    counter=False,
    control=False,
    alternating=True,
    caller_managed=False,
    handshake=True,
    kind="rectangle",
):
    """Use a wrapper's host assembly with the same payload checks as a family."""

    def coords(indices):
        return [(0, i) if kind == "column" else (i, 0) for i in indices]

    group = Group(coords(range(width)), coords(senders))
    size = device.compute_with_storage_grid_size()
    if any(x >= size.x or y >= size.y for x, y in group.receivers + group.senders):
        pytest.skip("requires a larger worker grid")
    config = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
        handshake=handshake,
        data_ready=ttnn.McastDataReady.Counter if counter else ttnn.McastDataReady.Flag,
    )
    if kind == "rectangle":
        sender_config = (
            ttnn.Mcast2DRotatingSenderConfig(sender_grid=core_set(group.senders))
            if rotating
            else ttnn.Mcast2DFixedSenderConfig(ttnn.CoreCoord(*group.senders[0]))
        )
        helper = ttnn.Mcast2D(device, core_set(group.receivers), sender_config, config)
    else:
        sender_config = (
            ttnn.Mcast1DRotatingSenderConfig(sender_grid=core_set(group.senders))
            if rotating
            else ttnn.Mcast1DFixedSenderConfig(starting_sender_index=senders[0])
        )
        helper = ttnn.Mcast1D(
            device,
            core_set(group.receivers),
            ttnn.Mcast1DShape.PerColumn if kind == "column" else ttnn.Mcast1DShape.PerRow,
            sender_config,
            config,
        )
    _run_channel(
        device,
        helper,
        [group],
        config,
        rounds=4 if handshake else 1,
        control=control,
        caller_managed=caller_managed,
        dynamic=False,
        max_pages=1,
        alternating=alternating,
        round_only_receive=True,
        control_value=1,
        with_barrier=False,
    )


def _run_channel(
    device,
    family,
    specs,
    config,
    *,
    rounds=6,
    control=False,
    caller_managed=False,
    receiver_caller_managed=False,
    dynamic=True,
    max_pages=2,
    mixed_events=False,
    delayed=False,
    zero_ack=False,
    adopted=False,
    expected_chain=None,
    min_rectangles=None,
    alternating=True,
    round_only_receive=False,
    control_value=7,
    with_barrier=True,
):
    # The config getter uses the runtime NOC enum, while ttnn.NOC is the descriptor enum.
    noc = config.noc.value
    counter = config.data_ready == ttnn.McastDataReady.Counter
    # specs are (exact logical receivers, ordered logical senders).
    all_coords = {c for receivers, senders in specs for c in receivers + senders}
    width = max(c[0] for c in all_coords) + (2 if with_barrier else 1)
    height = max(c[1] for c in all_coords) + 1
    size = device.compute_with_storage_grid_size()
    if width > size.x or height > size.y:
        pytest.skip("requires a larger worker grid")
    dispatch = [(x, y) for y in range(height) for x in range(width)]
    participants = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))])
    payload = tile_pattern(len(specs) * rounds * max_pages)
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
        int(alternating),
        int(round_only_receive),
        control_value,
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
                kernel_source=f"{KERNEL_DIR}/pipe_family.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=core_set(selected),
                compile_time_args=face_ct,
                runtime_args=face_rt,
                config=ttnn.WriterConfigDescriptor() if noc else ttnn.ReaderConfigDescriptor(),
            )
        )
    semaphores = (
        [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=family.participating_cores(), initial_value=0)
            for i in config.sem_ids
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
            passive.add_group(core_set(receivers), [ttnn.CoreCoord(*c) for c in senders])
        passive.prepare_arguments()
        passive.attach(descriptor, "mcast", kernels[1:])
    if expected_chain is not None:
        assert (attached_ct[5] >> 3) & 3 == int(expected_chain)
        if expected_chain:
            assert attached_ct[10:] == [0, 2]
        for receivers, senders in specs:
            sender = ttnn.CoreCoord(*senders[0])
            fanout = len(receivers) - int(senders[0] in receivers)
            assert kernels[0].runtime_args[sender.x][sender.y][mcast_rt:][1] == (
                int(fanout > 0) if expected_chain else fanout
            )
    if min_rectangles is not None:
        # Guard cases whose point is an irregular mapping: they must not degrade into dense sets on this grid.
        # Chain arguments omit rectangles; inspect the same geometry in multicast mode.
        geometry_family = ttnn.McastFamily(device, ttnn.McastConfig(noc=config.noc))
        for receivers, senders in specs:
            geometry_family.add_group(core_set(receivers), [ttnn.CoreCoord(*c) for c in senders])
        geometry_family.prepare_arguments()
        _, geometry_kernel = attach_for_inspection(geometry_family, participants, config.noc)
        for _, senders in specs:
            x, y = senders[0]
            assert geometry_kernel.runtime_args[x][y][0] >= min_rectangles
    for kernel in kernels:
        ttnn.attach_absent(kernel, "absent_mcast")
    if with_barrier:
        barrier = ttnn.McastFamily(device, ttnn.McastConfig(noc=config.noc))
        barrier.add_group(participants, [ttnn.CoreCoord(0, 0)])
        barrier.prepare_arguments()
        barrier.attach(descriptor, "barrier_mcast", kernels)
    else:
        for kernel in kernels:
            ttnn.attach_absent(kernel, "barrier_mcast")
    descriptor.cbs = [make_cb(i, participants, pages=2 * max_pages) for i in (0, 1)]
    descriptor.kernels = kernels
    output = ttnn.generic_op(
        [input_tensor, output_tensor],
        descriptor,
    )
    actual = ttnn.to_torch(output).reshape(len(dispatch), stride, 1, 32, 32)
    for index, coord in enumerate(dispatch):
        if with_barrier and coord not in all_coords:
            assert torch.all(actual[index, -1].contiguous().view(torch.int32) == 0x5A5A5A5A), coord
        for group_index, (receivers, _) in enumerate(specs):
            if coord not in receivers:
                continue
            for r in range(rounds):
                pages = (max_pages if r % 2 else 1) if dynamic else 1
                if control or (mixed_events and r % 2):
                    assert actual[index, max_pages * r].contiguous().view(torch.int32).flatten()[0].item() == (
                        r + 1 if counter else control_value + (r % 3 if mixed_events else 0)
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
