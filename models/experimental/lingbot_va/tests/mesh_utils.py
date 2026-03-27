# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mesh shape from ``MESH_DEVICE`` env, aligned with tt_transformers / experimental LLM tests."""

from __future__ import annotations

import os

import ttnn

_MESH_DEVICE_SHAPES = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}


def mesh_shape_request_param():
    """
    Value for pytest ``@pytest.mark.parametrize("mesh_device", [...], indirect=True)`` or for
    matching root ``conftest`` ``mesh_device`` semantics: a ``(rows, cols)`` tuple or an int
    ``N`` for a ``(1, N)`` mesh when falling back to ``len(ttnn.get_device_ids())``.
    """
    return _MESH_DEVICE_SHAPES.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))


def ttnn_mesh_shape_from_env() -> ttnn.MeshShape:
    """``MeshShape`` for ``ttnn.open_mesh_device``, same rules as ``mesh_shape_request_param``."""
    param = mesh_shape_request_param()
    if isinstance(param, tuple):
        return ttnn.MeshShape(*param)
    return ttnn.MeshShape(1, param)


def ttnn_mesh_shape_for_inference_demo() -> ttnn.MeshShape:
    """
    Mesh passed to ``ttnn.open_mesh_device`` for the Lingbot **demo** process.

    - ``MESH_DEVICE=N150`` → ``(1, 1)``. ``MESH_DEVICE=N300`` → ``(1, 2)`` (same mapping as pytest).
    - When ``MESH_DEVICE`` is **unset**, infers from ``len(ttnn.get_device_ids())``: one device →
      ``(1, 1)`` (N150-style), two devices → ``(1, 2)`` (N300-style). More devices without
      ``MESH_DEVICE`` defaults to ``(1, 1)`` so T3K/TG require an explicit env (see
      ``mesh_shape_request_param``).

    The demo matches tensor / sequence / VAE parallel factors to ``mesh_device.shape``; use
    ``LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH=1`` to force a ``(1, 1)`` submesh on a multi-chip open.
    """
    if os.environ.get("MESH_DEVICE"):
        return ttnn_mesh_shape_from_env()
    n = len(ttnn.get_device_ids())
    if n <= 1:
        return ttnn.MeshShape(1, 1)
    if n == 2:
        return ttnn.MeshShape(1, 2)
    return ttnn.MeshShape(1, 1)


def inference_work_mesh_from_opened(opened_mesh: ttnn.MeshDevice) -> tuple[ttnn.MeshDevice, ttnn.MeshDevice | None]:
    """
    Return ``(work_mesh, parent_mesh)`` for Lingbot demo / ``run_inference`` / ``run_generate``.

    **Default:** ``work_mesh`` is the full ``opened_mesh`` so tensor-parallel (column mesh axis),
    sequence-parallel (row axis), and VAE H/W parallel match the grid (same layout as
    ``test_transformer_wan._make_parallel_config``).

    **Opt-in single chip:** set ``LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH=1`` to use a ``(1,1)``
    submesh on a multi-device open (TP=1/SP=1 on one die; useful for debugging or if sharding fails).
    """
    single_chip = os.environ.get("LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if single_chip and opened_mesh.get_num_devices() > 1:
        sub = opened_mesh.create_submesh(ttnn.MeshShape(1, 1))
        return sub, opened_mesh
    return opened_mesh, None
