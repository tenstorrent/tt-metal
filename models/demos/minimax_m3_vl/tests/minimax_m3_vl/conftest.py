# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Test fixtures for MiniMax-M3-VL submodule PCC tests.

The `mesh_device` fixture comes from tt-metal's root conftest. Tests
parametrize it `indirect=True` with a `(rows, cols)` tuple — for the v1
single-Blackhole prototype this is `(1, 1)` (MESH_DEVICE=N150).

Fixtures here:
  - `ensure_gc`: autouse GC sweep between tests.
  - `model_args`: MiniMaxM3VLModelArgs bound to the mesh device.
  - `reference`: the torch weight-module tree from `_m3_loader` (built from
    the checkpoint shards; main-env-safe, no transformers).
  - `goldens`: loader for the per-grid reference activations on disk.
"""
from __future__ import annotations

import gc
import os

import pytest


@pytest.fixture(autouse=True)
def ensure_gc():
    gc.collect()


@pytest.fixture(scope="function")
def model_args(mesh_device):
    import ttnn
    from models.demos.minimax_m3_vl.tt.model_config import MiniMaxM3VLModelArgs

    return MiniMaxM3VLModelArgs(mesh_device=mesh_device, dtype=ttnn.bfloat16)


@pytest.fixture(scope="session")
def reference():
    """Torch reference-module tree holding the checkpoint vision weights."""
    from models.demos.minimax_m3_vl.tt._m3_loader import build_reference
    from models.demos.minimax_m3_vl.tt.model_config import MiniMaxM3VLModelArgs

    return build_reference(MiniMaxM3VLModelArgs())


@pytest.fixture(scope="session")
def goldens():
    """Return load_golden(grid_tag) -> dict[str, torch.Tensor] from tests/goldens/."""
    from safetensors.torch import load_file

    from models.demos.minimax_m3_vl.tt.model_config import MiniMaxM3VLModelArgs

    gdir = MiniMaxM3VLModelArgs().goldens_dir

    def load_golden(tag: str):
        path = os.path.join(gdir, f"{tag}.safetensors")
        if not os.path.exists(path):
            pytest.skip(f"golden {path} not found — run tests/gen_goldens.py in the transformers-5.12 venv")
        return load_file(path)

    return load_golden
