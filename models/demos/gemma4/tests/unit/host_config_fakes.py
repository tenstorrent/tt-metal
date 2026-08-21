# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal stand-ins for the objects Gemma4's host-only config tests inspect.

The program-config, shard-spec and memcfg helpers under ``tt/`` only read a
compute grid or a tensor shape, so the host-only tests can hand them these
instead of opening a device. Shared by ``test_ccl_topology``,
``test_prefill_matmul_configs`` and ``test_norm_sharding``; un-prefixed so
pytest does not collect it.
"""


class _FakeComputeGrid:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeMeshDevice:
    def __init__(self, gx=8, gy=8):
        self._grid = _FakeComputeGrid(gx, gy)

    def compute_with_storage_grid_size(self):
        return self._grid


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape
