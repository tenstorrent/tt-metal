# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for allowed_worker_cores on matmul program configs, with and without sub-devices.

Exercises the two-layer core resolution in all non-DRAM-sharded factories:
  Layer 1: sub_device_id → base grid
  Layer 2: allowed_worker_cores → fine constraint

Note on offset grids:
  The 1D and 2D multicast factories compute NOC multicast coordinates relative
  to the core grid's start_core.  Offset-from-origin grids are only exercised
  through the sub-device path (skip_rows >= 1), which is the production use
  case (e.g. reserving row 0 for CCL while computing on remaining rows).
  Using allowed_worker_cores offset from (0,0) *without* a sub-device has not
  been validated for multicast factories and is NOT tested here.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crs(cols: int, rows: int, start_x: int = 0, start_y: int = 0) -> ttnn.CoreRangeSet:
    """Shorthand: rectangular CoreRangeSet of size (cols x rows) starting at (start_x, start_y)."""
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(start_x, start_y),
                ttnn.CoreCoord(start_x + cols - 1, start_y + rows - 1),
            )
        }
    )


def _tensors(device, m, k, n, *, batched_b=False):
    """Random input tensors on device + torch reference."""
    torch.manual_seed(42)
    ta = torch.randn(1, 1, m, k, dtype=torch.bfloat16)
    tb = torch.randn((1, 1, k, n) if batched_b else (k, n), dtype=torch.bfloat16)
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    return a, b, ta @ tb


def _setup_subdevice(device, skip_rows=1):
    """Split device: rows [0..skip_rows-1] = dummy, rest = worker.
    Returns (manager, worker_sub_device_id, worker_cols, worker_rows, start_y).
    """
    grid = device.compute_with_storage_grid_size()
    cols, rows = grid.x, grid.y
    if rows <= skip_rows:
        pytest.skip(f"Need >{skip_rows} rows, device has {rows}")

    dummy = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, skip_rows - 1))})
    worker = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, skip_rows), ttnn.CoreCoord(cols - 1, rows - 1))})
    mgr = device.create_sub_device_manager([ttnn.SubDevice([dummy]), ttnn.SubDevice([worker])], 0)
    device.load_sub_device_manager(mgr)
    device.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
    return mgr, ttnn.SubDeviceId(1), cols, rows - skip_rows, skip_rows


def _teardown(device, mgr):
    device.reset_sub_device_stall_group()
    device.clear_loaded_sub_device_manager()
    device.remove_sub_device_manager(mgr)


# ---------------------------------------------------------------------------
# 1) Auto-config (no explicit program_config)
# ---------------------------------------------------------------------------


class TestAutoConfig:
    def test_basic_matmul(self, device):
        a, b, ref = _tensors(device, 128, 256, 128)
        assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b)), 0.999)

    @pytest.mark.parametrize("skip_rows", [1, 2])
    def test_matmul_on_subdevice(self, device, skip_rows):
        mgr, sd_id, _, _, _ = _setup_subdevice(device, skip_rows)
        try:
            a, b, ref = _tensors(device, 128, 512, 512)
            assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b, sub_device_id=sd_id)), 0.999)
        finally:
            _teardown(device, mgr)

    @pytest.mark.parametrize("skip_rows", [1, 2])
    def test_linear_on_subdevice(self, device, skip_rows):
        mgr, sd_id, _, _, _ = _setup_subdevice(device, skip_rows)
        try:
            a, b, ref = _tensors(device, 128, 512, 512)
            assert_with_pcc(ref, ttnn.to_torch(ttnn.linear(a, b, sub_device_id=sd_id)), 0.999)
        finally:
            _teardown(device, mgr)


# ---------------------------------------------------------------------------
# 2) MatmulMultiCoreReuseProgramConfig  (Factory B)
# ---------------------------------------------------------------------------


class TestReuse:
    M, K, N = 128, 256, 128

    def _cfg(self, awc=None):
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            in0_block_w=self.K // 32,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=self.M // 32,
            per_core_N=self.N // 32,
            allowed_worker_cores=awc,
        )

    def test_default_grid(self, device):
        a, b, ref = _tensors(device, self.M, self.K, self.N, batched_b=True)
        assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b, program_config=self._cfg())), 0.999)

    @pytest.mark.parametrize("gx, gy", [(4, 4), (8, 2)])
    def test_custom_grid_at_origin(self, device, gx, gy):
        a, b, ref = _tensors(device, self.M, self.K, self.N, batched_b=True)
        assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b, program_config=self._cfg(_crs(gx, gy)))), 0.999)

    @pytest.mark.parametrize("skip_rows", [1, 2])
    def test_on_subdevice(self, device, skip_rows):
        mgr, sd_id, cols, wrows, sy = _setup_subdevice(device, skip_rows)
        try:
            a, b, ref = _tensors(device, self.M, self.K, self.N, batched_b=True)
            cfg = self._cfg(_crs(cols, wrows, start_y=sy))
            assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg, sub_device_id=sd_id)), 0.999)
        finally:
            _teardown(device, mgr)


# ---------------------------------------------------------------------------
# 3) MatmulMultiCoreReuseMultiCast1DProgramConfig  (Factory C)
# ---------------------------------------------------------------------------


class TestMcast1D:
    M, K, N = 32, 1024, 1024

    def _cfg(self, mcast_in0, gx, gy, awc=None):
        nc = gx * gy
        if mcast_in0:
            pcM, pcN = self.M // 32, max(1, (self.N // 32) // nc)
        else:
            pcM, pcN = max(1, (self.M // 32) // nc), self.N // 32
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=pcM,
            out_block_w=pcN,
            per_core_M=pcM,
            per_core_N=pcN,
            fuse_batch=True,
            mcast_in0=mcast_in0,
            allowed_worker_cores=awc,
        )

    @pytest.mark.parametrize("mcast_in0", [True, False])
    def test_default_grid(self, device, mcast_in0):
        grid = device.compute_with_storage_grid_size()
        a, b, ref = _tensors(device, self.M, self.K, self.N)
        assert_with_pcc(
            ref, ttnn.to_torch(ttnn.matmul(a, b, program_config=self._cfg(mcast_in0, grid.x, grid.y))), 0.999
        )

    @pytest.mark.parametrize("mcast_in0", [True, False])
    @pytest.mark.parametrize("gx, gy", [(4, 4), (8, 2)])
    def test_custom_grid_at_origin(self, device, mcast_in0, gx, gy):
        a, b, ref = _tensors(device, self.M, self.K, self.N)
        cfg = self._cfg(mcast_in0, gx, gy, _crs(gx, gy))
        assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg)), 0.999)

    @pytest.mark.parametrize("mcast_in0", [True, False])
    @pytest.mark.parametrize("skip_rows", [1, 2])
    def test_on_subdevice(self, device, mcast_in0, skip_rows):
        """Offset grid via sub-device — cores start at row `skip_rows`."""
        mgr, sd_id, cols, wrows, sy = _setup_subdevice(device, skip_rows)
        try:
            a, b, ref = _tensors(device, self.M, self.K, self.N)
            cfg = self._cfg(mcast_in0, cols, wrows, _crs(cols, wrows, start_y=sy))
            assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg, sub_device_id=sd_id)), 0.999)
        finally:
            _teardown(device, mgr)


# ---------------------------------------------------------------------------
# 4) MatmulMultiCoreReuseMultiCastProgramConfig  (Factory D — 2D mcast)
# ---------------------------------------------------------------------------


class TestMcast2D:
    K = 512

    def _cfg(self, gx, gy, awc=None):
        M, N = 32 * gy, 32 * gx
        return (
            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                in0_block_w=2,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=max(1, (M // 32) // gy),
                per_core_N=max(1, (N // 32) // gx),
                transpose_mcast=False,
                allowed_worker_cores=awc,
            ),
            M,
            N,
        )

    def test_default_grid(self, device):
        grid = device.compute_with_storage_grid_size()
        cfg, M, N = self._cfg(grid.x, grid.y)
        a, b, ref = _tensors(device, M, self.K, N)
        assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg)), 0.999)

    @pytest.mark.parametrize("gx, gy", [(4, 4), (4, 2)])
    def test_custom_grid_at_origin(self, device, gx, gy):
        cfg, M, N = self._cfg(gx, gy, _crs(gx, gy))
        a, b, ref = _tensors(device, M, self.K, N)
        assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg)), 0.999)

    @pytest.mark.parametrize("skip_rows", [1, 2])
    def test_on_subdevice(self, device, skip_rows):
        """Offset grid via sub-device — cores start at row `skip_rows`."""
        mgr, sd_id, cols, wrows, sy = _setup_subdevice(device, skip_rows)
        try:
            cfg, M, N = self._cfg(cols, wrows, _crs(cols, wrows, start_y=sy))
            a, b, ref = _tensors(device, M, self.K, N)
            assert_with_pcc(ref, ttnn.to_torch(ttnn.matmul(a, b, program_config=cfg, sub_device_id=sd_id)), 0.999)
        finally:
            _teardown(device, mgr)


# ---------------------------------------------------------------------------
# 5) Validation / property tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_property_roundtrip(self, device):
        awc = _crs(4, 4)
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            transpose_mcast=False,
            allowed_worker_cores=awc,
        )
        assert cfg.allowed_worker_cores is not None
        assert cfg.allowed_worker_cores == awc

    def test_default_is_none(self, device):
        cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
        )
        assert cfg.allowed_worker_cores is None

    def test_repr_contains_allowed_worker_cores(self, device):
        cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            allowed_worker_cores=_crs(4, 4),
        )
        assert "allowed_worker_cores" in repr(cfg)
