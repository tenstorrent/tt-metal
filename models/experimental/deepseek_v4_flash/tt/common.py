import os
from contextlib import contextmanager
from typing import Any

import ttnn

# The single-user tile is a 1x32 tile, used when batch size = 1. It avoids the padding of 31
# unused rows, cutting activation and CB footprint 32x.
SINGLE_USER_TILE = ttnn.Tile((1, 32))

# The default 32x32 tile. At batch = 1 the 1x32 tile above must be used instead.
FULL_TILE = ttnn.Tile((32, 32))


# ``ttnn.ReadDeviceProfiler`` is a host call that syncs the device; it must never run inside a
# ``ttnn`` trace capture (which records device ops only and forbids host round-trips / syncs
# mid-capture). The traced decode path reuses the eager ``forward`` helpers below, so route every
# profiler read through this guard and silence it while a trace is being captured.
_IN_TRACE_CAPTURE = False


def _profile(device) -> None:
    """Dump the device profiler for ``device``, unless a ttnn trace capture is running."""
    if not _IN_TRACE_CAPTURE:
        ttnn.ReadDeviceProfiler(device)


# Tracy signposts let the (flat) device-op profile be sliced per decoder-layer sub-module
# (attention, MoE router/experts/shared, hyper-connection, norms): each ``_region`` emits a
# ``<NAME>_START`` / ``<NAME>_END`` host marker around the ops it issues. ``tracy`` only imports
# on a profiler-enabled build, so degrade to a no-op otherwise.
try:
    from tracy import signpost as _tracy_signpost
except Exception:  # pragma: no cover - tracy missing on non-profiling builds
    _tracy_signpost = None

# Master switch for the per-module signposts. Defaults on (a no-op unless the run is captured
# under the Tracy profiler), but can be disabled to drop the host-side call overhead:
# set ``DEEPSEEK_V4_SIGNPOSTS=0``.
_SIGNPOSTS_ENABLED = os.environ.get("DEEPSEEK_V4_SIGNPOSTS", "1") not in ("0", "", "false", "False")


def _signpost(header: str) -> None:
    """Emit one Tracy marker ``header``, unless signposts are off or a trace is capturing."""
    if _SIGNPOSTS_ENABLED and _tracy_signpost is not None and not _IN_TRACE_CAPTURE:
        _tracy_signpost(header=header)


@contextmanager
def _region(name: str):
    """Wrap the enclosed ttnn ops in a Tracy ``<name>_START`` / ``<name>_END`` pair.

    Shape-agnostic: it brackets whatever tensors the enclosed ops build.
    """
    _signpost(f"{name}_START")
    try:
        yield
    finally:
        _signpost(f"{name}_END")


@contextmanager
def _trace_capture_guard():
    """Silence :func:`_profile` while a ttnn trace is being captured.

    Shape-agnostic: it covers the whole enclosed op sequence.
    """
    global _IN_TRACE_CAPTURE
    prev = _IN_TRACE_CAPTURE
    _IN_TRACE_CAPTURE = True
    try:
        yield
    finally:
        _IN_TRACE_CAPTURE = prev


# fp32 accumulation everywhere keeps the long (Dh=512) reductions and the
# softmax/RoPE chains from drifting under bf16; the per-layer PCC test needs it.
_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# The fused ``scaled_dot_product_attention_decode`` op must NOT run with
# ``fp32_dest_acc_en=True``: for this attention shape (head_dim=256, MQA with a
# single shared K==V head) that flag makes the kernel emit garbage (PCC ~0.36 vs
# the manual softmax). HiFi4 with bf16 dest accumulation matches the manual path
# at PCC ~0.9999. ``packer_l1_acc`` is safe to keep on.
_HIFI4_SDPA = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Additive-mask "-inf": a finite bf16-representable floor. Masked logits feed
# ``exp(x - max)`` which underflows to 0 for both this and a true ``-inf``, but
# the finite value avoids ``inf - inf -> NaN`` if a whole row were masked.
_MASK_NEG = -1.0e9


class DeepSeekV4Module:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """``forward(*args, **kwds)`` -- lets every sub-module be called like an ``nn.Module``."""
        return self.forward(*args, **kwds)


def rectangular_core_grid(num_cores: int, device) -> ttnn.CoreGrid:
    """A rectangular ``x x y`` core grid of exactly ``num_cores`` cores on ``device``.

    ``num_cores`` is a core count, not a tensor shape; it should divide the width of whatever
    ``[H, W]`` tensor is sharded over the result. Finds the widest ``x`` that divides it and fits
    ``grid.x``, giving an ``(x, num_cores // x)`` rectangle; raises if no such rectangle fits the
    device grid.
    """
    grid = device.compute_with_storage_grid_size()
    x = grid.x
    while x > 0 and num_cores % x != 0:
        x -= 1
    y = num_cores // x if x > 0 else 0
    if x == 0 or y > grid.y:
        raise ValueError(f"cannot form a rectangular grid of {num_cores} cores within a {grid.x}x{grid.y} device grid")
    return ttnn.CoreGrid(y=y, x=x)


def rectangular_core_range_set(num_cores: int, device) -> ttnn.CoreRangeSet:
    """``CoreRangeSet`` covering the ``x x y`` rectangle from :func:`rectangular_core_grid`.

    ``num_cores`` is a core count, so it fits any tensor sharded over those cores -- typically
    the ``[H, W]`` activation whose columns :func:`width_sharded_l1_config` splits.
    """
    core_grid = rectangular_core_grid(num_cores, device)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))})


def with_shard_height(memory_config: ttnn.MemoryConfig, shard_height: int) -> ttnn.MemoryConfig:
    """Return a copy of ``memory_config`` with its shard height replaced by ``shard_height``.

    ``shard_height`` is in rows, so a ``[H, W]`` tensor's shards become ``[shard_height,
    W/cores]``. Core grid, shard width, layout and buffer type are preserved so a retile does
    not rebuild a different width-sharded core map.
    """
    shard_spec = memory_config.shard_spec
    if shard_spec is None:
        raise ValueError("memory_config has no shard_spec to override")
    new_shard_spec = ttnn.ShardSpec(
        shard_spec.grid,
        [shard_height, shard_spec.shape[1]],
        shard_spec.orientation,
    )
    return ttnn.MemoryConfig(memory_config.memory_layout, memory_config.buffer_type, new_shard_spec)


def with_tile_height(memory_config: ttnn.MemoryConfig, height: int, tile_height: int) -> ttnn.MemoryConfig:
    """Return a copy of ``memory_config`` whose shard height is ``height`` padded to ``tile_height``.

    Both are in rows: a ``[height, W]`` tensor's shards become ``[height_padded, W/cores]``.
    """
    height_padded = ((height + tile_height - 1) // tile_height) * tile_height
    return with_shard_height(memory_config, height_padded)


def width_sharded_l1_config(
    height: int, width: int, device, num_cores: int | None = None, tile_height: int = ttnn.TILE_SIZE
) -> ttnn.MemoryConfig:
    """Width-sharded L1 config for a ``[..., height, width]`` tensor.

    Each of ``num_cores`` cores holds a ``[height_padded, width // num_cores]`` shard, ROW_MAJOR
    orientation, with ``height_padded`` the ``height`` rounded up to a ``tile_height`` boundary.
    ``num_cores`` defaults to ``width // TILE_SIZE`` -- one 32-column tile-width per core -- and
    is halved until it fits the device grid, which widens every shard to match. ``width`` must be
    tile-aligned.
    """
    assert width % ttnn.TILE_SIZE == 0, f"width {width} must be tile-aligned"
    if num_cores is None:
        num_cores = width // ttnn.TILE_SIZE
    device_grid_size = device.compute_with_storage_grid_size()
    device_cores = device_grid_size.x * device_grid_size.y
    while num_cores > device_cores:
        num_cores //= 2
    shard_width = width // num_cores
    grid = rectangular_core_range_set(num_cores, device)
    height_padded = ((height + tile_height - 1) // tile_height) * tile_height
    shard_spec = ttnn.ShardSpec(grid, [height_padded, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
