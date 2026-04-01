# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Serialize / deserialize OverlappedTensor view metadata for fusion-group cache entries."""

from __future__ import annotations

import ttnn

_DTYPE_TO_STR: dict[ttnn.DataType, str] = {
    ttnn.bfloat16: "bfloat16",
    ttnn.bfloat8_b: "bfloat8_b",
    ttnn.bfloat4_b: "bfloat4_b",
    ttnn.float32: "float32",
    ttnn.uint32: "uint32",
    ttnn.uint16: "uint16",
    ttnn.uint8: "uint8",
}

_STR_TO_DTYPE: dict[str, ttnn.DataType] = {v: k for k, v in _DTYPE_TO_STR.items()}


def core_range_set_to_list(crs: ttnn.CoreRangeSet) -> list[list[list[int]]]:
    """Serialize CoreRangeSet to JSON-serializable list of [[sx, sy], [ex, ey]].

    Range order follows ``crs.ranges()`` (not sorted). Fingerprint canonicalization sorts ranges
    for deterministic hashing; here we preserve spec order for round-trip view metadata.
    """
    result = []
    for r in crs.ranges():
        start, end = r.start, r.end
        result.append([[start.x, start.y], [end.x, end.y]])
    return result


def core_range_set_from_list(lst: list[list[list[int]]]) -> ttnn.CoreRangeSet:
    """Deserialize list of [[sx, sy], [ex, ey]] to CoreRangeSet."""
    ranges = [
        ttnn.CoreRange(
            ttnn.CoreCoord(pair[0][0], pair[0][1]),
            ttnn.CoreCoord(pair[1][0], pair[1][1]),
        )
        for pair in lst
    ]
    return ttnn.CoreRangeSet(ranges)


def overlapped_tensor_to_view_dict(ot: "OverlappedTensor") -> dict:
    """Serialize one OverlappedTensor's metadata (no fused_tensor) for metadata.json."""
    dtype_str = _DTYPE_TO_STR.get(ot.dtype)
    if dtype_str is None:
        dtype_str = str(ot.dtype)
    return {
        "tensor_shape": list(ot.tensor_shape),
        "shard_shape": list(ot.shard_shape),
        "core_range_set": core_range_set_to_list(ot.core_range_set),
        "dtype": dtype_str,
        "tile_shape": list(ot.tile_shape),
        "byte_offset": ot.byte_offset,
        "total_size": ot.total_size,
    }


def overlapped_tensor_from_view_dict(fused_tensor: ttnn.Tensor, d: dict) -> OverlappedTensor:
    """Reconstruct OverlappedTensor from loaded fused tensor and view dict."""
    from models.demos.deepseek_v3_b1.blitz_decode_weights import OverlappedTensor

    dtype = _STR_TO_DTYPE.get(d["dtype"])
    if dtype is None:
        raise ValueError(f"Unknown dtype in fusion view metadata: {d['dtype']}")
    total_size = d.get("total_size", 0)
    if total_size == 0:
        raise ValueError("fusion view metadata missing 'total_size'")
    return OverlappedTensor(
        fused_tensor=fused_tensor,
        tensor_shape=tuple(d["tensor_shape"]),
        shard_shape=tuple(d["shard_shape"]),
        core_range_set=core_range_set_from_list(d["core_range_set"]),
        dtype=dtype,
        tile_shape=tuple(d["tile_shape"]),
        byte_offset=d["byte_offset"],
        total_size=total_size,
    )


def views_dict_from_overlapped(views: dict[str, OverlappedTensor]) -> dict[str, dict]:
    return {name: overlapped_tensor_to_view_dict(ot) for name, ot in views.items()}
