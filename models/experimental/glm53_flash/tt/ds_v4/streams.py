"""Residual-stream helpers of the DeepSeek-V4-Flash decoder blocks.

The hyper-connected residual is a stack of ``hc`` streams ``[B, S, hc, D]`` (TILE), one token per
``(B, S)`` position; the sublayers see ``[1, 1, T, D]`` rows with ``T = B * S``.
"""

from typing import Optional

import ttnn

from .weight_cache import _load_weight, _materialize


def _strip_prefix(weights: dict, prefix: str) -> dict:
    """Sub-dict of ``weights`` whose keys start with ``prefix.`` (prefix stripped).

    The values are passed through untouched, so a weight keeps its own shape (``[K, N]`` for a
    projection) and a lazy thunk stays lazy.
    """
    p = f"{prefix}."
    return {k[len(p) :]: v for k, v in weights.items() if k.startswith(p)}


def flatten_streams(streams: ttnn.Tensor) -> ttnn.Tensor:
    """``[B, S, hc, D]`` TILE -> ``[1, 1, B*S, hc*D]`` TILE: each token's streams laid side by side.

    ``hc`` is far below a tile row, so this is not a view: it goes through ROW_MAJOR, where the
    merge of ``hc`` rows into one wide row is a plain page reshape.
    """
    b, s, hc, d = streams.shape
    rows = ttnn.to_layout(streams, ttnn.ROW_MAJOR_LAYOUT)
    rows = ttnn.reshape(rows, [1, 1, b * s, hc * d])
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)


def load_norm_gamma(weight, device: ttnn.MeshDevice, cache_file_name: Optional[str] = None) -> ttnn.Tensor:
    """A ``[D]`` RMSNorm weight (tensor or thunk) as a bf16 TILE ``[1, 1, 1, D]`` device tensor."""
    w = _materialize(weight, cache_file_name, ttnn.bfloat16)
    return _load_weight(
        w.detach().float().reshape(1, 1, 1, -1) if w is not None else None,
        device,
        cache_file_name=cache_file_name,
    )


def rows_from_tokens(x: ttnn.Tensor) -> ttnn.Tensor:
    """``[B, S, 1, D]`` TILE (one token per ``(B, S)`` position, the hyper-connection's layout) -> ``[1, 1, B*S, D]`` TILE.

    The token axes move to the row axis. In TILE layout that would repack every token's one-row
    tile, so it goes through ROW_MAJOR where it is a page-preserving reshape.
    """
    b, s, _, d = x.shape
    rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    rows = ttnn.reshape(rows, [1, 1, b * s, d])
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)


def tokens_from_rows(x: ttnn.Tensor, b: int, s: int) -> ttnn.Tensor:
    """``[1, 1, B*S, D]`` TILE (a sublayer output) -> ``[B, S, 1, D]`` TILE, the inverse of :func:`rows_from_tokens`."""
    d = x.shape[-1]
    rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    rows = ttnn.reshape(rows, [b, s, 1, d])
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)
