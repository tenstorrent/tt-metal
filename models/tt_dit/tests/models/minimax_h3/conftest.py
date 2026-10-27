# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import ttnn

from ....pipelines.minimax_h3.pipeline_minimax_h3 import minimax_h3_worker_l1_size


@pytest.fixture(scope="function")
def device_params(request):
    """The root `device_params`, plus the `worker_l1_size` every MiniMax-H3 deployment opens with.

    Filled in here rather than in each parametrize list so no H3 test opens a mesh with a layout that is
    not deployed (see `MINIMAX_H3_KERNEL_CONFIG_RING_BYTES`), and at fixture time because the value comes
    from the HAL, which collection must not touch. A test that sets its own `worker_l1_size` keeps it.
    """
    params = dict(getattr(request, "param", {}))
    params.setdefault("worker_l1_size", minimax_h3_worker_l1_size())
    return params


def fence_circular_buffers_below_l1_small(mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Allocate a one-row-per-bank L1 buffer directly under L1_SMALL on every core.

    tt-metal#62798: the static-CB placement check compares a program's CB region only against the full L1
    size and the lowest live *regular* L1 buffer, never L1_SMALL, so with no regular L1 buffer live a
    program's CBs can run into L1_SMALL and overwrite what lives there (H3's audio `conv1d` keeps its
    sliding-window indices in L1_SMALL for the life of the process). Regular L1 is allocated top-down, so
    this buffer, allocated first, sits right under L1_SMALL and the clash check then rejects such a program.
    Costs one L1 alignment unit per core (32 B on Wormhole, 64 B on Blackhole).
    """
    num_banks = ttnn._ttnn.reports.get_device_info(mesh_device).l1_num_banks
    return ttnn.from_torch(
        torch.zeros(num_banks, 8, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


@pytest.fixture(autouse=True)
def l1_small_fence(request):
    """Fence every H3 test mesh so a CB/L1_SMALL overlap fails the test instead of passing silently.

    Test-only: deployments are not fenced, so this turns tt-metal#62798 into a test failure for any H3
    program that would hit it, rather than guarding serving. Drop it once #62798 is fixed in tt-metal.
    """
    if "mesh_device" not in request.fixturenames:
        yield None
        return
    # The mesh was just opened, so this is its first L1 allocation and lands at the top of the regular region.
    fence = fence_circular_buffers_below_l1_small(request.getfixturevalue("mesh_device"))
    yield fence
    ttnn.deallocate(fence)
