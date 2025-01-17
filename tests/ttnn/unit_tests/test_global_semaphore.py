# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger


def run_global_semaphore(device):
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    global_sem0 = ttnn.create_global_semaphore(device, tensix_cores0, 1)
    global_sem1 = ttnn.create_global_semaphore(device, tensix_cores1, 2)

    assert ttnn.get_global_semaphore_address(global_sem0) != ttnn.get_global_semaphore_address(global_sem1)

    ttnn.reset_global_semaphore_value(global_sem0, 3)


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_global_semaphore(device, enable_async_mode):
    run_global_semaphore(device)


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_global_semaphore_mesh(mesh_device, enable_async_mode):
    run_global_semaphore(mesh_device)


def run_global_semaphore_same_address(mesh_device):
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    global_sem0 = ttnn.create_global_semaphore(mesh_device.get_devices()[0], tensix_cores0, 0)
    global_sem1 = ttnn.create_global_semaphore(mesh_device.get_devices()[1], tensix_cores0, 0)
    global_sem2 = ttnn.create_global_semaphore(mesh_device.get_devices()[0], tensix_cores0, 0)

    global_sem3 = ttnn.create_global_semaphore_with_same_address(
        mesh_device, tensix_cores0, 0, attempts=10, search_max=False
    )
    addrs0 = ttnn.get_global_semaphore_address(global_sem0)
    addrs1 = ttnn.get_global_semaphore_address(global_sem1)
    addrs2 = ttnn.get_global_semaphore_address(global_sem2)
    addrs3 = ttnn.get_global_semaphore_address(global_sem3)
    logger.debug(f"addrs0: {addrs0}, addrs1: {addrs1}, addrs2: {addrs2}, addrs3: {addrs3}")
    assert len(set(addrs3)) == 1


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_global_semaphore_mesh_same_address(mesh_device, enable_async_mode):
    if len(mesh_device.get_devices()) < 4:
        pytest.skip("requires at least 4 devices to run")
    run_global_semaphore_same_address(mesh_device)
