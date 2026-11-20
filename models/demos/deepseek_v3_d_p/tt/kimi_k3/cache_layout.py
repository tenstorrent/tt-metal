# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where the Kimi-K3 family keeps a weight cache inside its model directory: ``<checkpoint id>/mesh{R}x{C}.tpaxis{T}``.

The checkpoint id keeps one checkpoint from loading another's tensors; the mesh shape and the TP axis fix each
tensorbin's per-device placement, which differs between the two axis choices even on a square mesh. Standard library
only, so the CI launcher can resolve the same directory the runner reads without importing ttnn.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def checkpoint_identity(checkpoint_dir: Path | str) -> str:
    """A short stable id for a checkpoint, from its index rather than its 5.5 TB of tensors."""
    index = Path(checkpoint_dir) / "model.safetensors.index.json"
    digest = hashlib.sha256()
    digest.update(Path(checkpoint_dir).resolve().name.encode())
    stat = index.stat()
    digest.update(str(stat.st_size).encode())
    return digest.hexdigest()[:16]


def checkpoint_cache_key(checkpoint_dir: Path | str, mesh_shape: tuple[int, int], tp_axis: int) -> Path:
    """The cache's path below the model directory for this (checkpoint, mesh, TP axis)."""
    assert tp_axis in (0, 1), f"tp_axis must be 0 or 1, got {tp_axis}"
    rows, columns = tuple(mesh_shape)
    return Path(checkpoint_identity(checkpoint_dir)) / f"mesh{rows}x{columns}.tpaxis{tp_axis}"


if __name__ == "__main__":
    # The CI launcher's view: python3 -m ...cache_layout <checkpoint dir> <rows> <columns> <tp axis>
    import sys

    checkpoint, rows, columns, tp_axis = sys.argv[1:]
    print(checkpoint_cache_key(checkpoint, (int(rows), int(columns)), int(tp_axis)))
