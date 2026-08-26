# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


def clamp(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    return value


def is_default_value(values, default):
    """Check if values match a default, handling None, scalar, and iterable inputs."""
    if values is None:
        return True
    if isinstance(values, (int, float)):
        return values == default
    return all(value == default for value in values)


def filter_none(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


def split_list(lst, n):
    """Split list into n equal parts."""
    chunk_size = len(lst) // n
    return [list(lst[i * chunk_size : (i + 1) * chunk_size]) for i in range(n)]


def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def upper_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# ttnn.topk large-k routing predicate (call-site mirror).
#
# Mirror of should_route_to_topk_large_indices
# (ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp) as evaluated for a
# call site that passes NO indices_tensor, NO preallocated outputs, NO
# sub_core_grids and stable=False, on a 4D tensor reduced over its LAST dim
# with largest=True (every production sampling call satisfies the shape/dim/
# largest part already). True iff dropping those arguments makes ttnn.topk take
# the Blackhole topk_large_indices composite for `x`.
#
# KEEP IN SYNC with the C++ gate. Drift is fail-safe in both directions:
#  - False while C++ would route: the call keeps today's arguments bit-for-bit
#    (stock path, no change).
#  - True while C++ would not route: the call runs the stock factory without
#    indices_tensor/stable; the stock op then generates the same 0-based
#    positions itself (dtype handled by the caller's typecast restore), and
#    greedy determinism is owned by _adjust_values_for_tiebreak downstream.
# ---------------------------------------------------------------------------
_TOPK_ROUTE_MIN_K_EXCLUSIVE = 64  # large_k_route_min_k_exclusive
_TOPK_ROUTE_SMALL_K_MIN_PADDED_WIDTH = 4096  # small_k_route_min_padded_width
_TOPK_ROUTE_MAX_K = 2048  # large_k_route_max_k
_TOPK_ROUTE_K_MULTIPLE = 16  # large_k_route_k_multiple (as merged in #53464)
_TOPK_ROUTE_MAX_WIDTH = 1 << 19  # large_k_route_max_width (as merged in #53464)
_TOPK_ROUTE_UINT16_MAX = 65535  # std::numeric_limits<uint16_t>::max()
_TOPK_ROUTE_MULTI_CORE_MIN_WIDTH = 8192  # ttnn::prim::constants::multi_core_min_width
# NOTE: merged topk.cpp has no MoE-gate arm; if one lands later, mirror it here
# in the same PR (KEEP IN SYNC).


def topk_would_route_to_large_indices(x, k, mesh_device) -> bool:
    """True iff ttnn.topk(x, k, dim=-1) with no indices_tensor / sub_core_grids /
    preallocated outputs and stable=False routes to the Blackhole
    topk_large_indices composite (see module comment above)."""
    import ttnn

    if not ttnn.device.is_blackhole(mesh_device):  # topk.cpp:311
        return False
    if k > _TOPK_ROUTE_MAX_K:  # topk.cpp:285
        return False
    if x.dtype != ttnn.bfloat16:  # topk.cpp:302
        return False
    if x.layout != ttnn.TILE_LAYOUT:  # topk.cpp:305
        return False
    if x.memory_config().is_sharded():  # topk.cpp:308
        return False
    padded_width = x.padded_shape[-1]
    if k <= _TOPK_ROUTE_MIN_K_EXCLUSIVE:  # small-k arm (wide structurally-ineligible)
        pow2 = padded_width > 0 and (padded_width & (padded_width - 1)) == 0
        structurally_ineligible = (
            padded_width >= _TOPK_ROUTE_UINT16_MAX or not pow2 or padded_width < _TOPK_ROUTE_MULTI_CORE_MIN_WIDTH
        )
        wide_arm = structurally_ineligible and padded_width >= _TOPK_ROUTE_SMALL_K_MIN_PADDED_WIDTH
        if not wide_arm:
            return False
    width = x.shape[-1]
    k_rounded = ((k + _TOPK_ROUTE_K_MULTIPLE - 1) // _TOPK_ROUTE_K_MULTIPLE) * _TOPK_ROUTE_K_MULTIPLE
    return k_rounded <= width <= _TOPK_ROUTE_MAX_WIDTH  # width envelope + k_rounded fit
