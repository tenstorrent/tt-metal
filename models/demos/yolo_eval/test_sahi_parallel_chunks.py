# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for SAHI / TT slice batching (no device)."""

from models.demos.yolo_eval.sahi_ultralytics_eval import parallel_slice_chunk_bounds


def test_parallel_slice_chunk_bounds_exact_multiple():
    assert list(parallel_slice_chunk_bounds(8, 4)) == [(0, 4), (4, 4)]


def test_parallel_slice_chunk_bounds_remainder():
    assert list(parallel_slice_chunk_bounds(10, 4)) == [(0, 4), (4, 4), (8, 2)]


def test_parallel_slice_chunk_bounds_single():
    assert list(parallel_slice_chunk_bounds(3, 1)) == [(0, 1), (1, 1), (2, 1)]


def test_parallel_slice_chunk_bounds_empty():
    assert list(parallel_slice_chunk_bounds(0, 4)) == []
