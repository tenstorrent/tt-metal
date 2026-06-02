# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Contract test for the DYNAMIC ``sub_core_grids`` master compute grid.

``TtQwen36ModelArgs.sub_core_grids`` must be derived from the live device
compute grid (``mesh_device.compute_with_storage_grid_size()``), NOT hard-coded
to the inherited Wormhole 60-core ``(1,0)->(6,9)`` band.

Expected derivation (col 0 reserved for DRAM-adjacent ops):
    CoreRange((1, 0), (grid.x - 1, grid.y - 1))
    => num_cores == (grid.x - 1) * grid.y

This must hold for whatever grid the device reports:
  - WH TG          grid (7, 10)  -> cols 1..6  x 10 = 60 cores
  - BH P150 galaxy grid (12, 10) -> cols 1..11 x 10 = 110 cores

Pure-CPU: the mesh device is mocked.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _FakeMeshDevice:
    """Stand-in for a real ``MeshDevice`` exposing the methods the config
    constructor calls.  ``grid`` parametrizes the reported compute grid."""

    def __init__(self, shape=(8, 4), grid=(12, 10)):
        self.shape = list(shape)
        self._grid = grid

    def get_num_devices(self):
        return self.shape[0] * self.shape[1]

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=self._grid[0], y=self._grid[1])

    def dram_grid_size(self):
        return SimpleNamespace(x=8, y=1)


def _ttnn_patches():
    """Patch the ttnn device-touching entry points the config constructor
    calls, but leave CoreCoord / CoreRange / CoreRangeSet REAL so the grid
    geometry (num_cores, bounding box) can be asserted."""
    import ttnn

    return [
        patch.object(ttnn, "MemoryConfig", lambda *a, **kw: MagicMock(name="MemoryConfig")),
        patch.object(ttnn, "ShardSpec", lambda *a, **kw: MagicMock(name="ShardSpec")),
        patch.object(ttnn, "create_sharded_memory_config", lambda *a, **kw: MagicMock(name="sharded_mem_cfg")),
        patch.object(
            ttnn, "WormholeComputeKernelConfig", lambda *a, **kw: MagicMock(name="WormholeComputeKernelConfig")
        ),
        patch.object(
            ttnn,
            "LayerNormShardedMultiCoreProgramConfig",
            lambda *a, **kw: MagicMock(name="LayerNormShardedMultiCoreProgramConfig"),
        ),
        patch.object(
            ttnn,
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            lambda *a, **kw: MagicMock(name="MatmulMultiCoreReuseMultiCast1DProgramConfig"),
        ),
        patch.object(
            ttnn,
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            lambda *a, **kw: MagicMock(name="MatmulMultiCoreReuseMultiCastProgramConfig"),
        ),
        patch.object(
            ttnn,
            "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
            lambda *a, **kw: MagicMock(name="MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig"),
        ),
        patch.object(ttnn, "MinimalMatmulConfig", lambda *a, **kw: MagicMock(name="MinimalMatmulConfig")),
        patch.object(ttnn, "SDPAProgramConfig", lambda *a, **kw: MagicMock(name="SDPAProgramConfig")),
    ]


def _build_args(grid):
    patches = _ttnn_patches()
    for p in patches:
        p.start()
    try:
        from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

        mesh = _FakeMeshDevice(shape=(8, 4), grid=grid)
        return TtQwen36ModelArgs(mesh_device=mesh, dummy_weights=True)
    finally:
        for p in patches:
            p.stop()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "grid,expected_cores",
    [
        ((12, 10), 110),  # BH P150 galaxy: cols 1..11 x 10
        ((7, 10), 60),  # WH TG: cols 1..6 x 10 (legacy)
    ],
)
def test_sub_core_grids_is_dynamic_full_band(grid, expected_cores):
    """sub_core_grids must span cols 1..grid.x-1 x rows 0..grid.y-1, derived
    from the live device grid — not a hard-coded 60."""
    args = _build_args(grid)
    grid_x, grid_y = grid

    assert (
        args.sub_core_grids.num_cores() == expected_cores
    ), f"grid {grid}: expected {expected_cores} cores, got {args.sub_core_grids.num_cores()}"

    # The band must span exactly col 1 -> grid.x-1, row 0 -> grid.y-1.
    ranges = list(args.sub_core_grids.ranges())
    assert len(ranges) == 1, f"expected a single contiguous CoreRange, got {ranges}"
    r = ranges[0]
    assert r.start.x == 1, f"band must start at col 1, got {r.start.x}"
    assert r.start.y == 0
    assert r.end.x == grid_x - 1, f"band must end at col grid.x-1={grid_x-1}, got {r.end.x}"
    assert r.end.y == grid_y - 1


@pytest.mark.cpu_only
def test_sub_core_grids_p150_is_110_cores():
    """The make-or-break on this box: BH P150 galaxy grid (12,10) -> 110 cores
    (was 60 before the dynamic widening)."""
    args = _build_args((12, 10))
    assert args.sub_core_grids.num_cores() == 110
    assert args.start_core == __import__("ttnn").CoreCoord(1, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
