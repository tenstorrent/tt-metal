# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device initialization for qualified TP4 cache buffers."""

import ttnn


def require_cache_initialization_ready(owner):
    if owner._cache_initialization_pending is not None:
        raise RuntimeError("Cache initialization cleanup is unresolved; close the model mesh before reuse")


def device_zero_eligible(shape, dtype, layout, device):
    shape = tuple(shape)
    if layout != ttnn.TILE_LAYOUT or any(type(n) is not int for n in shape):
        return False
    supported = (shape == (8192, 1, 32, 256) and dtype in (ttnn.bfloat8_b, ttnn.bfloat16)) or (
        shape in ((1, 12, 128, 128), (2, 12, 128, 128)) and dtype == ttnn.float32
    )
    return (
        supported
        and device.get_num_devices() == 4
        and device.arch() == ttnn.Arch.BLACKHOLE
        and tuple(device.shape) == (1, 4)
        and ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.P300_X2
    )


def allocate_cache_zero(owner, shape, *, dtype):
    """Return completed storage, retaining it on the decoder if cleanup fails.

    This is a setup operation, outside trace capture. TTNN's tiled zeros_like
    uses device fill and writes into optional_tensor without allocating output.
    """
    require_cache_initialization_ready(owner)
    if not device_zero_eligible(shape, dtype, ttnn.TILE_LAYOUT, owner.device):
        raise ValueError("Device cache initialization requires qualified TP4 geometry")
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, owner.device, ttnn.DRAM_MEMORY_CONFIG
    )
    try:
        ttnn.zeros_like(output, optional_tensor=output)
        # The caller owns the buffer only after return. Complete the fill even
        # if subsequent decoder/TP/model-cache construction raises.
        ttnn.synchronize_device(owner.device)
    except BaseException:
        try:
            # Enqueue can fail after submitting work. Drain before releasing
            # storage; a failed first fence also gets one cleanup attempt.
            ttnn.synchronize_device(owner.device)
            ttnn.deallocate(output)
        except BaseException as cleanup_error:
            # Retain the buffer independently of the exception's lifetime.
            owner._cache_initialization_pending = output
            raise RuntimeError(
                "Cache initialization cleanup failed; close the model mesh before reuse"
            ) from cleanup_error
        raise
    return output
