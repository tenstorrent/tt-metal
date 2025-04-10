# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger


def create_global_semaphore_with_same_address(mesh_device, cores, initial_value):
    semaphore_handles = ttnn.create_global_semaphore_with_same_address(mesh_device, cores, initial_value)
    addrs = ttnn.get_global_semaphore_address(semaphore_handles)
    # assert all addresses are the same
    assert len(set(addrs)) == 1
    return semaphore_handles
