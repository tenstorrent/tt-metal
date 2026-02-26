# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from .metrics import metric_value, pearson_corr

MIXED_TILE_FORMATS = ["bf16", "bfp8", "bfp4", "bfp2"]
MIXED_TILE_BYTES_PER_ELEM = {
    "bf16": 2.0,
    "bfp8": 1.088,
    "bfp4": 0.50097,
    "bfp2": 0.25097,
}


def mixed_tile_total_bytes(counts: dict[str, int], tile_hw: int = 32) -> float:
    total = 0.0
    elems_per_tile = float(tile_hw * tile_hw)
    for fmt, count in counts.items():
        total += float(count) * elems_per_tile * MIXED_TILE_BYTES_PER_ELEM.get(fmt, 0.0)
    return total


def tile_metrics(ref_tiles: np.ndarray, q_tiles: np.ndarray, metric: str) -> np.ndarray:
    if metric == "pcc":
        scores = []
        for i in range(ref_tiles.shape[0]):
            scores.append(pearson_corr(ref_tiles[i], q_tiles[i]))
        return np.asarray(scores, dtype=np.float32)
    diff = np.abs(ref_tiles - q_tiles)
    if metric == "mae":
        return diff.reshape(diff.shape[0], -1).mean(axis=1)
    if metric == "atol":
        return diff.reshape(diff.shape[0], -1).max(axis=1)
    raise ValueError(f"Unsupported metric: {metric}")


def reshape_to_2d_with_padding(xf: np.ndarray) -> tuple[np.ndarray, tuple, tuple]:
    xf = np.asarray(xf, dtype=np.float32)
    if xf.ndim == 0:
        data2d = xf.reshape(1, 1)
        shape_info = ("scalar", xf.shape)
    elif xf.ndim == 1:
        n = xf.shape[0]
        h = int(np.ceil(n / 32.0))
        w = 32
        data2d = np.zeros((h, w), dtype=np.float32)
        data2d.reshape(-1)[:n] = xf.reshape(-1)
        shape_info = ("vector", n)
    else:
        w = xf.shape[-1]
        h = int(np.prod(xf.shape[:-1]))
        data2d = xf.reshape(h, w)
        shape_info = ("nd", xf.shape)

    h, w = data2d.shape
    h_pad = int(np.ceil(h / 32.0)) * 32
    w_pad = int(np.ceil(w / 32.0)) * 32
    padded = np.zeros((h_pad, w_pad), dtype=np.float32)
    padded[:h, :w] = data2d
    pad_info = (h, w, h_pad, w_pad)
    return padded, shape_info, pad_info


def reconstruct_from_tiles(tiles: np.ndarray, shape_info: tuple, pad_info: tuple, tile_hw: int = 32) -> np.ndarray:
    h, w, h_pad, w_pad = pad_info
    tiles_h = h_pad // tile_hw
    tiles_w = w_pad // tile_hw
    padded = tiles.reshape(tiles_h, tiles_w, tile_hw, tile_hw).transpose(0, 2, 1, 3).reshape(h_pad, w_pad)
    data2d = padded[:h, :w]
    if shape_info[0] == "scalar":
        return np.array(data2d[0, 0], dtype=np.float32)
    if shape_info[0] == "vector":
        n = shape_info[1]
        return data2d.reshape(-1)[:n].astype(np.float32)
    if shape_info[0] == "nd":
        orig_shape = shape_info[1]
        return data2d.reshape(orig_shape).astype(np.float32)
    raise ValueError("Invalid shape_info")


def global_metric(xf: np.ndarray, tiles: np.ndarray, shape_info: tuple, pad_info: tuple, metric: str) -> float:
    y = reconstruct_from_tiles(tiles, shape_info, pad_info)
    return metric_value(xf, y, metric)
