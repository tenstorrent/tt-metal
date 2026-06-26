# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import sys

import pytest
import torch
import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "operation"))
from lreg_layout_op import lreg_layout_op


def _make_input(device):
    """Single 32x32 uint32 tile filled with 1024."""
    a_torch = torch.full((32, 32), 1024, dtype=torch.int32)
    return a_torch, ttnn.from_torch(
        a_torch,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_lreg_layout_ids(device):
    """Stage 2: SFPU writes lane/iter IDs. All values should be < 1000 (IDs are 0..255)."""
    _, a = _make_input(device)
    result = lreg_layout_op(a)
    result_torch = ttnn.to_torch(result)
    torch.set_printoptions(linewidth=200, threshold=10000)
    print(f"output =\n{result_torch.to(torch.int64)}")

    # Save as CSV with row/column indices.
    csv_path = os.path.join(os.path.dirname(__file__), "lreg_layout.csv")
    rows, cols = result_torch.shape
    values = result_torch.to(torch.int64).tolist()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row\\col"] + list(range(cols)))
        for r in range(rows):
            writer.writerow([r] + values[r])
    print(f"saved CSV to {csv_path}")
    # uint32 doesn't support .max() in torch; cast to int64 for the check.
    max_val = result_torch.to(torch.int64).max().item()
    assert max_val < 1000, f"Expected all IDs < 1000 (input fill 1024 should be overwritten), max={max_val}"
