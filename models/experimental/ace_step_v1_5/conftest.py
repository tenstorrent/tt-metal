# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for AceStep v1.5 (``tests/``, ``perf/``, and future subtrees).

Prefer the repo Python environment (``./create_venv.sh`` → ``./python_env``):

- ``source python_env/bin/activate`` then ``python -m pytest ...``
- or ``./python_env/bin/python -m pytest ...``

Conv-heavy TTNN kernels (patchify / Oobleck VAE) rely on adequate ``l1_small_size`` when opening
device (matches ``ign/ACE_perf``; override via ``ACE_STEP_L1_SMALL_SIZE``).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# This conftest lives at:
#   tt-metal/models/experimental/ace_step_v1_5/conftest.py
# We need the repo root `tt-metal/` on sys.path, and also `tt-metal/ttnn/`
# so `import ttnn` resolves to `tt-metal/ttnn/ttnn/__init__.py` (not the
# namespace package at `tt-metal/ttnn/`).
_TT_METAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_TTNN_ROOT = os.path.join(_TT_METAL_ROOT, "ttnn")
# NOTE: Do not add `tt-metal/tools` to sys.path here. Some environments contain an optional
# `tools/tracy` package that depends on extra plotting libs (e.g. seaborn). Importing TTNN
# should not require those extras for unit tests.
for _p in (_TT_METAL_ROOT, _TTNN_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Default aligns with TTNN conv1d unit tests needing non-trivial L1 small space (Blackhole / Wormhole).
_DEFAULT_L1_SMALL = int(os.environ.get("ACE_STEP_L1_SMALL_SIZE", "98304"))

# All ACE-Step session devices open with 2 CQs by default. The trace+2CQ perf tests
# (``tests/test_*_trace_2cq.py`` and ``perf/conftest.py``) need CQ 1 for host->device
# copies; if any device in the same pytest session opens with a different CQ count,
# TT's dispatch state becomes inconsistent and ``close_device``'s implicit
# ``synchronize_device`` trips ``Could not find the dispatch core for 1``. Keep the
# count uniform across the whole session to avoid that cross-fixture failure.
# Override with ``ACE_STEP_NUM_CQS=1`` to restore the legacy single-CQ behaviour.
_DEFAULT_NUM_CQS = int(os.environ.get("ACE_STEP_NUM_CQS", "2"))


def _open_kwargs(*, include_num_cqs: bool = True) -> dict:
    """Common kwargs for ``ttnn.open_device`` / ``ttnn.open_mesh_device``.

    Centralised so ``device`` and ``mesh_device`` always agree on ``num_command_queues``
    and ``trace_region_size``.
    """
    kw = dict(
        device_id=int(os.environ.get("TT_DEVICE_ID", "0")),
        l1_small_size=_DEFAULT_L1_SMALL,
        trace_region_size=128 << 20,
    )
    if include_num_cqs and _DEFAULT_NUM_CQS > 1:
        kw["num_command_queues"] = _DEFAULT_NUM_CQS
    return kw


@pytest.fixture
def torch_seed():
    torch.manual_seed(42)
    yield 42
    torch.manual_seed(42)


@pytest.fixture(scope="session")
def device():
    ttnn = require_ttnn()
    dev = ttnn.open_device(**_open_kwargs())

    if hasattr(dev, "enable_program_cache"):
        dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def require_ttnn():
    # TTNN native extension may fail to initialize on hosts without a proper runtime.
    # Treat that as a skip for demo tests.
    return pytest.importorskip("ttnn", exc_type=ImportError)


@pytest.fixture(scope="session")
def mesh_device():
    """
    Single mesh for the whole test session.

    A function-scoped open/close loop exhausts Metal context slots and breaks
    subsequent ``open_mesh_device`` calls (invalid context_id / MAX_CONTEXT_COUNT).
    Remote-only meshes can also abort if many MeshDevice instances are torn down.
    """
    ttnn = require_ttnn()
    # Prefer a mesh device when supported.
    if hasattr(ttnn, "open_mesh_device") and hasattr(ttnn, "MeshShape") and os.environ.get("MESH_DEVICE"):
        from models.experimental.ace_step_v1_5.tt_device import ace_step_mesh_shape, resolve_ace_step_mesh_sku

        mesh_sku = resolve_ace_step_mesh_sku()
        rows, cols = ace_step_mesh_shape(mesh_sku)
        mesh_kw = {k: v for k, v in _open_kwargs().items() if k != "device_id"}
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(int(rows), int(cols)),
            **mesh_kw,
        )
        if hasattr(mesh, "enable_program_cache"):
            mesh.enable_program_cache()
        try:
            yield mesh
        finally:
            ttnn.close_mesh_device(mesh)
        return

    # Fallback: some TTNN builds only expose single-device APIs.
    if hasattr(ttnn, "open_device"):
        dev = ttnn.open_device(**_open_kwargs())

        if hasattr(dev, "enable_program_cache"):
            dev.enable_program_cache()
        try:
            yield dev
        finally:
            ttnn.close_device(dev)
        return

    pytest.skip("No TT device API available (missing open_mesh_device/open_device).")


# Note: the ``trace_device`` fixture lives in ``perf/conftest.py`` now (it overrides this scope
# so module-trace tests under ``perf/module_trace/`` share the perf-level session ``device``
# instead of opening a second handle on the same physical hardware). Don't add a sibling here.
