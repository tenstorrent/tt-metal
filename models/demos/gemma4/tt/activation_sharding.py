# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optional residual activation layout chaining for decode.

RMSNorm can consume and produce one shared WIDTH_SHARDED layout so adjacent
decode operations do not repeatedly convert the residual stream.  The policy is
disabled by default and only applies to a single logical tile row.
"""

import os

import ttnn

#: Minimum tiles per core in the residual-stream shard.  Smaller shards force
#: decode matmuls to use narrow K blocking and can cost more than chaining saves.
MIN_SHARD_TILES = 4


class ActivationSharding:
    """Choose a residual-stream shard spec and convert tensors at its boundaries."""

    def __init__(self, mesh_device=None, enabled=False, min_shard_tiles=MIN_SHARD_TILES, max_cores=8):
        self.enabled = bool(enabled)
        self.min_shard_tiles = min_shard_tiles
        self.max_cores = max_cores
        self.mesh_device = mesh_device
        self._specs = {}

    @classmethod
    def from_env(cls, mesh_device=None, **kwargs):
        raw = (os.getenv("GEMMA4_SHARD_ACTIVATIONS") or "").strip().lower()
        enabled = raw in ("1", "true", "yes", "on")
        min_tiles = os.getenv("GEMMA4_SHARD_MIN_TILES")
        if min_tiles:
            kwargs["min_shard_tiles"] = int(min_tiles)
        return cls(mesh_device, enabled=enabled, **kwargs)

    def cores_for(self, dim):
        """Return the widest legal one-row grid, or ``None`` when inapplicable."""
        if not self.enabled or dim <= 0 or dim % ttnn.TILE_SIZE:
            return None
        tiles = dim // ttnn.TILE_SIZE
        best = None
        for cores in range(1, self.max_cores + 1):
            if tiles % cores == 0 and tiles // cores >= self.min_shard_tiles:
                best = cores
        return best

    def spec(self, dim):
        """Return the shared WIDTH_SHARDED memory config for ``dim``."""
        if dim in self._specs:
            return self._specs[dim]
        cores = self.cores_for(dim)
        spec = None
        if cores:
            spec = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, dim // cores),
                core_grid=ttnn.CoreGrid(x=cores, y=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        self._specs[dim] = spec
        return spec

    def shard_tiles(self, dim):
        """Return tiles per core for ``dim``, or ``None``."""
        cores = self.cores_for(dim)
        return (dim // ttnn.TILE_SIZE) // cores if cores else None

    def applies(self, tensor):
        """Return whether ``tensor`` has a compatible decode shape."""
        return (
            self.enabled
            and hasattr(tensor, "shape")
            and len(tensor.shape) == 4
            and 1 <= int(tensor.shape[-2]) <= ttnn.TILE_SIZE
            and self.spec(int(tensor.shape[-1])) is not None
        )

    def matches(self, tensor):
        """Return whether ``tensor`` already uses this policy's shard spec."""
        if not self.applies(tensor) or not tensor.is_sharded():
            return False
        wanted = self.spec(int(tensor.shape[-1])).shard_spec
        actual = tensor.memory_config().shard_spec
        return actual is not None and list(actual.shape) == list(wanted.shape) and actual.grid == wanted.grid

    def to_stream(self, tensor):
        """Move a compatible tensor into the shared layout."""
        if not self.applies(tensor) or self.matches(tensor):
            return tensor
        return ttnn.to_memory_config(tensor, self.spec(int(tensor.shape[-1])))

    def to_stream_like(self, tensor, other):
        """Match ``tensor`` to ``other`` before a binary operation."""
        if not self.enabled or not hasattr(other, "is_sharded"):
            return tensor
        if other.is_sharded():
            if tensor.is_sharded() and tensor.memory_config() == other.memory_config():
                return tensor
            return ttnn.to_memory_config(tensor, other.memory_config())
        return self.from_stream(tensor) if tensor.is_sharded() else tensor

    def from_stream(self, tensor, memory_config=None):
        """Convert a sharded stream tensor for an incompatible consumer."""
        if not hasattr(tensor, "is_sharded") or not tensor.is_sharded():
            return tensor
        return ttnn.sharded_to_interleaved(tensor, memory_config or ttnn.DRAM_MEMORY_CONFIG)


DISABLED = ActivationSharding(enabled=False)


def resolve(policy):
    return policy if policy is not None else DISABLED
