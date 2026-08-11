# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import gc

import torch

import ttnn


def _core(x, y):
    coord = ttnn.CoreCoord(x, y)
    return ttnn.CoreRangeSet({ttnn.CoreRange(coord, coord)})


def _cores(*coords):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords})


def test_fusion_semaphore_bank_nonzero_values_and_offsets(device):
    core_ranges = [_core(0, 0), _core(2, 0), _cores((0, 0), (2, 0))]
    initial_values = [3, 7, 11]

    bank = ttnn._ttnn.operations.experimental.FusionSemaphoreBank(device, core_ranges, initial_values)

    base_address = bank.tensor.buffer_address()
    assert not bank.tensor.is_per_core_allocated()
    assert list(bank.addresses) == [base_address + 4 * index for index in range(len(initial_values))]

    values = ttnn.to_torch(ttnn.from_device(bank.tensor)).reshape(-1, len(initial_values))
    expected = torch.tensor(initial_values, dtype=torch.uint32).repeat(values.shape[0], 1)
    assert torch.equal(values, expected)


def test_fusion_semaphore_bank_releases_l1(device):
    core_ranges = [_core(0, 0), _core(1, 0)]
    ttnn.synchronize_device(device)
    gc.collect()
    pages_before = len(ttnn._ttnn.reports.get_buffer_pages(device))

    for _ in range(100):
        bank = ttnn._ttnn.operations.experimental.FusionSemaphoreBank(device, core_ranges, [0, 0])
        assert len(bank.addresses) == 2
    del bank
    ttnn.synchronize_device(device)
    gc.collect()

    assert len(ttnn._ttnn.reports.get_buffer_pages(device)) == pages_before


def test_fusion_semaphore_bank_rejects_mismatched_metadata(device, expect_error):
    with expect_error(RuntimeError, "must match"):
        ttnn._ttnn.operations.experimental.FusionSemaphoreBank(device, [_core(0, 0)], [0, 1])


def test_fusion_semaphore_bank_rejects_empty_metadata(device, expect_error):
    with expect_error(RuntimeError, "at least one semaphore"):
        ttnn._ttnn.operations.experimental.FusionSemaphoreBank(device, [], [])


def test_fusion_semaphore_bank_mesh_address_contract(mesh_device):
    bank = ttnn._ttnn.operations.experimental.FusionSemaphoreBank(
        mesh_device,
        [_cores((0, 0), (1, 0)), _core(1, 0)],
        [0, 9],
    )

    assert not bank.tensor.is_per_core_allocated()
    assert len(bank.tensor.device_coords()) == mesh_device.get_num_devices()
    assert list(bank.addresses) == [bank.tensor.buffer_address(), bank.tensor.buffer_address() + 4]
