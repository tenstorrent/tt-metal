# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the `Qwen/Qwen3-Coder-Next` e2e gate.

THE FIXTURE IS THE DEVICE OWNER.  `qwen_mesh` opens the mesh exactly once per pytest session --
through `device_harness.open_mesh()`, the package's only opener, with `num_command_queues=1` and a
trace region -- and every test in this directory builds on THAT device.  Nothing under `tt/` opens
a device of its own, so there is never a second mesh with a different command-queue count to
collide with the trace lever.

The reference checkpoint and the built pipeline are session-scoped for the same reason: one HF
load, one weight upload, shared by `test_e2e_pipeline.py` and `test_captured_parity.py`.
"""
from __future__ import annotations

import os

import pytest

from models.demos.qwen3_coder_next import device_harness
from models.demos.qwen3_coder_next.tt.pipeline import DEFAULT_CAPACITY, build_pipeline
from models.demos.qwen3_coder_next.tt.reference import DEFAULT_LAYERS, load_reference

LAYERS = int(os.environ.get("TT_QWEN3_LAYERS", DEFAULT_LAYERS))
CAPACITY = int(os.environ.get("TT_QWEN3_CAPACITY", DEFAULT_CAPACITY))


@pytest.fixture(scope="session")
def reference():
    """SOURCE A: the depth-capped real checkpoint plus its tokenizer."""
    return load_reference(LAYERS)


@pytest.fixture(scope="session")
def qwen_mesh():
    """The ONE device open of this session. Yields `(mesh_device, (rows, cols))`."""
    device, shape = device_harness.open_mesh()
    try:
        yield device, shape
    finally:
        device_harness.close_mesh(device)


@pytest.fixture(scope="session")
def pipeline(reference, qwen_mesh):
    """The resident TT pipeline, built on the fixture's device -- the same object the demo runs."""
    model, tokenizer = reference
    device, shape = qwen_mesh
    pipe = build_pipeline(device, model=model, layers=LAYERS, tokenizer=tokenizer, capacity=CAPACITY)
    pipe._mesh_shape = shape
    return pipe
