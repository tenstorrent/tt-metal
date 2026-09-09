# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact family broadcasts, concurrent groups, wrapper-independent public API and spectators."""

import pytest
import torch
import ttnn


def _cores(coords):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(set(coords))]
    )


def _run(
    device,
    specs,
    *,
    noc=0,
    counter=False,
    control=False,
    caller_managed=False,
    dynamic=True,
    handshake=True,
    rounds=6,
    zero_ack=False,
    adopted=False,
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
    config = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0,
        handshake=handshake,
        data_ready=ttnn.McastDataReady.Counter if counter else ttnn.McastDataReady.Flag,
        ack_count_override=0 if zero_ack else None,
        sem_ids=[0, 1] if adopted else None,
    )
    groups = [
        ttnn.McastGroup(_cores(receivers), senders=[ttnn.CoreCoord(*c) for c in senders])
        for receivers, senders in specs
    ]
    family = ttnn.McastFamily(device, groups, config)
    barrier = ttnn.McastFamily(
        device,
        [ttnn.McastGroup(participants, [ttnn.CoreCoord(0, 0)])],
        ttnn.McastConfig(noc=config.noc, base_sem_id=2 if adopted else family.next_base_sem_id()),
    )
    payload = (
        torch.arange(1, len(specs) * rounds * 2 + 1, dtype=torch.bfloat16)
        .reshape(-1, 1, 1, 1)
        .expand(-1, 1, 32, 32)
        .contiguous()
    )
    input_tensor = ttnn.from_torch(payload, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    stride = rounds * 2 + 1
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(dispatch) * stride, 1, 32, 32]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    ct = list(family.compile_time_args()) + [0] + list(barrier.compile_time_args())
    ct += [rounds, int(control), int(caller_managed), int(dynamic)]
    ct += list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    ct += list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    for index, (x, y) in enumerate(dispatch):
        group_index = next((i for i, (rx, tx) in enumerate(specs) if (x, y) in rx + tx), None)
        inside = group_index is not None and (x, y) in specs[group_index][0]
        rt[x][y] = [
            input_tensor.buffer_address(),
            output_tensor.buffer_address(),
            (group_index or 0) * rounds * 2,
            index * stride,
            int(inside),
            int(group_index is None),
        ]
        rt[x][y] = (
            list(rt[x][y])
            + list(family.runtime_args(ttnn.CoreCoord(x, y)))
            + list(barrier.runtime_args(ttnn.CoreCoord(x, y)))
        )
    kernels = []
    faces = [True, False] if zero_ack else [None]
    for sending in faces:
        selected = [c for c in dispatch if sending is None or family.is_sender(ttnn.CoreCoord(*c)) == sending]
        face_rt = ttnn.RuntimeArgs()
        for x, y in selected:
            face_rt[x][y] = list(rt[x][y])
        face_ct = list(ct)
        if sending is False:
            face_ct[5] &= ~1  # Receivers do not ack an explicit-zero sender; start barrier protects initialization.
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source="tests/ttnn/unit_tests/kernel_lib/kernels/pipe_family.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=_cores(selected),
                compile_time_args=face_ct,
                runtime_args=face_rt,
                config=ttnn.WriterConfigDescriptor() if noc else ttnn.ReaderConfigDescriptor(),
            )
        )
    semaphores = family.owned_semaphores()
    if adopted:
        assert not semaphores
        semaphores = [
            ttnn.SemaphoreDescriptor(id=i, core_ranges=family.participating_cores(), initial_value=0) for i in (0, 1)
        ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=8192,
            core_ranges=participants,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i in (0, 1)
    ]
    output = ttnn.generic_op(
        [input_tensor, output_tensor],
        ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores + barrier.owned_semaphores()),
    )
    actual = ttnn.to_torch(output).reshape(len(dispatch), stride, 1, 32, 32)
    for index, coord in enumerate(dispatch):
        if coord not in all_coords:
            assert torch.all(actual[index, -1].contiguous().view(torch.int32) == 0x5A5A5A5A), coord
        for group_index, (receivers, _) in enumerate(specs):
            if coord not in receivers:
                continue
            for r in range(rounds):
                pages = 1 + r % 2 if dynamic else 1
                if control:
                    assert actual[index, 2 * r].contiguous().view(torch.int32).flatten()[0].item() == (
                        r + 1 if counter else 7
                    )
                else:
                    for p in range(pages):
                        assert torch.equal(actual[index, 2 * r + p], payload[group_index * rounds * 2 + 2 * r + p]), (
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
