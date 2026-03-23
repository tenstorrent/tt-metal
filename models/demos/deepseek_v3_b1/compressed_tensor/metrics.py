# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return 1.0
    am = a - np.mean(a)
    bm = b - np.mean(b)
    denom = float(np.linalg.norm(am) * np.linalg.norm(bm))
    if denom == 0.0:
        return 1.0 if np.max(np.abs(a - b)) == 0.0 else 0.0
    return float(np.dot(am, bm) / denom)


def metric_value(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "pcc":
        return pearson_corr(a, b)
    diff = np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))
    if metric == "mae":
        return float(np.mean(diff))
    if metric == "atol":
        return float(np.max(diff))
    raise ValueError(f"Unsupported metric: {metric}")


def metric_is_good(value: float, metric: str, threshold: float) -> bool:
    if metric == "pcc":
        return value >= threshold
    return value <= threshold


def metric_better(a: float, b: float, metric: str) -> bool:
    if metric == "pcc":
        return a > b
    return a < b
