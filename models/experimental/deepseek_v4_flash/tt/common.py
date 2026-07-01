import os
from contextlib import contextmanager
from typing import Any

import ttnn

# ``ttnn.ReadDeviceProfiler`` is a host call that syncs the device; it must never
# run inside a ``ttnn`` trace capture (which records device ops only and forbids
# host round-trips / syncs mid-capture). The traced decode path reuses several of
# the eager ``forward`` helpers below, so route every profiler read through this
# guard and silence it while a trace is being captured.
_IN_TRACE_CAPTURE = False


def _profile(device) -> None:
    if not _IN_TRACE_CAPTURE:
        ttnn.ReadDeviceProfiler(device)


# Tracy signposts let the (flat) device-op profile be sliced per decoder-layer
# sub-module (attention, MoE router/experts/shared, hyper-connection, norms): each
# ``_region`` emits a ``<NAME>_START`` / ``<NAME>_END`` host marker around the ops
# it issues. ``tracy`` only imports on a profiler-enabled build, so degrade to a
# no-op otherwise, and stay silent during a ttnn trace capture (no host calls).
try:
    from tracy import signpost as _tracy_signpost
except Exception:  # pragma: no cover - tracy missing on non-profiling builds
    _tracy_signpost = None

# Master switch for the per-module signposts. Defaults on (they are a no-op unless
# the run is captured under the Tracy profiler), but can be disabled to drop even
# the host-side call overhead: set ``DEEPSEEK_V4_SIGNPOSTS=0`` or call
# :func:`set_signposts_enabled(False)` at runtime.
_SIGNPOSTS_ENABLED = os.environ.get("DEEPSEEK_V4_SIGNPOSTS", "1") not in ("0", "", "false", "False")


def set_signposts_enabled(enabled: bool) -> None:
    """Enable/disable the per-module Tracy signposts at runtime."""
    global _SIGNPOSTS_ENABLED
    _SIGNPOSTS_ENABLED = bool(enabled)


def _signpost(header: str) -> None:
    if _SIGNPOSTS_ENABLED and _tracy_signpost is not None and not _IN_TRACE_CAPTURE:
        _tracy_signpost(header=header)


@contextmanager
def _region(name: str):
    """Wrap the enclosed ttnn ops in a Tracy ``<name>_START`` / ``<name>_END`` pair."""
    _signpost(f"{name}_START")
    try:
        yield
    finally:
        _signpost(f"{name}_END")


@contextmanager
def _trace_capture_guard():
    """Silence :func:`_profile` for the duration of a trace capture."""
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
        return self.forward(*args, **kwds)


def rectangular_core_grid(num_cores: int, device) -> ttnn.CoreGrid:
    """A rectangular ``x x y`` core grid of exactly ``num_cores`` cores on ``device``.

    Finds the widest ``x`` that divides ``num_cores`` and fits ``grid.x``, giving a
    ``(x, num_cores // x)`` rectangle. Raises if no such rectangle fits the device grid.
    """
    grid = device.compute_with_storage_grid_size()
    x = grid.x
    while x > 0 and num_cores % x != 0:
        x -= 1
    y = num_cores // x if x > 0 else 0
    if x == 0 or y > grid.y:
        raise ValueError(f"cannot form a rectangular grid of {num_cores} cores within a {grid.x}x{grid.y} device grid")
    return ttnn.CoreGrid(y=y, x=x)


def print_l1_tensors(device):
    device_info = ttnn._ttnn.reports.get_buffers(device)
    l1_buffers = [b for b in device_info if b.buffer_type == ttnn.BufferType.L1]
    if not l1_buffers:
        print("No L1 buffers found.")
        return

    headers = ("#", "Address", "Size (bytes)", "Layout")
    rows = [
        (str(i), f"{b.address}", f"{b.max_size_per_bank:,}", str(b.buffer_layout)) for i, b in enumerate(l1_buffers)
    ]
    total_l1_size = sum(b.max_size_per_bank for b in l1_buffers)
    rows.append(("", "Total", f"{total_l1_size:,}", ""))

    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]

    def _row(cells):
        return " | ".join(
            cells[i].rjust(widths[i]) if i == 2 else cells[i].ljust(widths[i]) for i in range(len(headers))
        )

    print(_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(_row(row))
    print("\n\n")


def rectangular_core_range_set(num_cores: int, device) -> ttnn.CoreRangeSet:
    """``CoreRangeSet`` for :func:`rectangular_core_grid`."""
    core_grid = rectangular_core_grid(num_cores, device)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))})


def width_sharded_l1_config(height: int, width: int, device, num_cores: int | None = None) -> ttnn.MemoryConfig:
    """Width-sharded L1 config: one tile-width (32 cols) per core over ``width // TILE_SIZE`` cores.

    Mirror of :func:`_rope_height_sharded_config` along the width axis: for a
    ``[..., height, width]`` tensor each core holds a ``[height_padded, TILE_SIZE]`` shard
    (``height_padded`` is ``height`` rounded up to a tile boundary), ROW_MAJOR orientation.
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
    height_padded = ((height + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    shard_spec = ttnn.ShardSpec(grid, [height_padded, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
