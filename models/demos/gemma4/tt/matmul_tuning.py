# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program configurations for single-tile-row decode linears.

``GEMMA4_TUNE_MATMULS`` selects the scopes. Unset enables ``DEFAULT_SCOPES``
(target only) on a single device and nothing on a multi-device mesh, where the
target configs gave no decode gain; ``0`` disables every scope. Draft and CME
tuning stay opt-in.
"""

import os

from loguru import logger

import ttnn

DEFAULT_SCOPES = frozenset({"target"})


def _largest_divisor(value, cap=8):
    for divisor in range(min(value, cap), 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


def _pick_grid(n_tiles, max_x, max_y):
    """Return the largest core rectangle whose core count divides n_tiles."""
    best = (1, 1)
    for grid_y in range(1, max_y + 1):
        for grid_x in range(1, max_x + 1):
            core_count = grid_x * grid_y
            if n_tiles % core_count == 0 and core_count > best[0] * best[1]:
                best = (grid_x, grid_y)
    return best


def derive_decode_1d_config(m, k, n, max_x=8, max_y=8, in0_shard_tiles=None):
    """Return a 1D multicast config for a compatible decode linear."""
    tile_size = ttnn.TILE_SIZE
    if m <= 0 or k <= 0 or n <= 0 or k % tile_size or n % tile_size:
        return None
    m_tiles = -(-m // tile_size)
    k_tiles = k // tile_size
    n_tiles = n // tile_size
    if m_tiles != 1:
        return None
    grid_x, grid_y = _pick_grid(n_tiles, max_x, max_y)
    core_count = grid_x * grid_y
    if core_count < 2 or n_tiles % core_count:
        return None
    in0_block_w = _largest_divisor(k_tiles, cap=8)
    if in0_shard_tiles:
        in0_block_w = _largest_divisor(in0_shard_tiles, cap=min(8, in0_shard_tiles))
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=m_tiles,
        per_core_N=n_tiles // core_count,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


class DecodeMatmulTuner:
    """Derive and cache decode configs, or pass through while disabled."""

    @classmethod
    def from_env(cls, mesh_device=None, scope="draft"):
        """Build a tuner for one scope from ``GEMMA4_TUNE_MATMULS``."""
        raw = os.getenv("GEMMA4_TUNE_MATMULS")
        raw = "" if raw is None else raw.strip().lower()
        if raw in ("1", "true", "yes", "on", "all"):
            enabled = True
        elif raw in ("0", "false", "no", "off"):
            enabled = False
        elif not raw:
            multi_device = mesh_device is not None and mesh_device.get_num_devices() > 1
            enabled = scope in DEFAULT_SCOPES and not multi_device
        else:
            enabled = scope in {item.strip() for item in raw.split(",")}
        return cls(mesh_device, enabled=enabled, label=scope)

    def __init__(self, mesh_device=None, enabled=False, label="mm"):
        # Constructor default is off so a directly built tuner is inert; the
        # environment default lives in ``from_env``/``DEFAULT_SCOPES`` instead.
        self.enabled = bool(enabled)
        self.label = label
        self._cache = {}
        self._max_x, self._max_y = 8, 8
        if self.enabled and mesh_device is not None:
            grid = mesh_device.compute_with_storage_grid_size()
            self._max_x, self._max_y = min(8, grid.x), min(8, grid.y)

    def config_for(self, x, weight):
        if not self.enabled:
            return None
        shard_tiles = self._in0_shard_tiles(x)
        key = (tuple(x.shape), tuple(weight.shape), shard_tiles)
        if key not in self._cache:
            config = derive_decode_1d_config(
                int(x.shape[-2]),
                int(x.shape[-1]),
                int(weight.shape[-1]),
                self._max_x,
                self._max_y,
                in0_shard_tiles=shard_tiles,
            )
            self._cache[key] = config
            logger.info(
                "[mm-tune:{}] M={} K={} N={} -> {}",
                self.label,
                int(x.shape[-2]),
                int(x.shape[-1]),
                int(weight.shape[-1]),
                "tuned" if config else "auto",
            )
        return self._cache[key]

    @staticmethod
    def _in0_shard_tiles(x):
        """Return the per-core input shard width in tiles when available."""
        try:
            if not x.is_sharded():
                return None
            shard_spec = x.memory_config().shard_spec
            return int(shard_spec.shape[1]) // ttnn.TILE_SIZE if shard_spec is not None else None
        except Exception:  # noqa: BLE001 - tensor metadata may be unavailable to mocks
            return None

    def stats(self):
        values = list(self._cache.values())
        return sum(config is not None for config in values), len(values)

    def compute_config_for(self, x, weight):
        """Return the compute config ttnn would pick for this matmul without a program config.

        ttnn lowers its default math fidelity from HiFi2 to LoFi as soon as a program config is
        passed, so a tuned call restates the automatic defaults to change only the blocking. Returns
        None where ttnn's default already matches (both inputs BFP8/BFP4, or both FLOAT32).
        """
        low_precision = (ttnn.bfloat8_b, ttnn.bfloat4_b)
        if x.dtype in low_precision and weight.dtype in low_precision:
            return None
        if x.dtype == ttnn.float32 and weight.dtype == ttnn.float32:
            return None
        return ttnn.init_device_compute_kernel_config(
            x.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def linear(self, x, weight, **kwargs):
        """Call ``ttnn.linear`` while preserving an explicit caller config."""
        if kwargs.get("program_config") is None:
            config = self.config_for(x, weight)
            if config is not None:
                kwargs["program_config"] = config
                if kwargs.get("compute_kernel_config") is None:
                    kwargs["compute_kernel_config"] = self.compute_config_for(x, weight)
        return ttnn.linear(x, weight, **kwargs)


DISABLED = DecodeMatmulTuner(enabled=False)


def resolve(tuner):
    return tuner if tuner is not None else DISABLED
