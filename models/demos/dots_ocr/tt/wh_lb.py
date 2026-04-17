# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Deprecated alias for :mod:`models.demos.dots_ocr.tt.mesh`.

Earlier versions of this demo used "WH LB" to mean "Wormhole Low Batch"
(single chip, 1×1 mesh). That conflicts with the rest of tt-metal where
"WH LB" / "WH LLMBox" refers to multi-chip Wormhole hardware (T3K-class).

This module is kept as a thin forwarder so external imports keep working.
New code should import from :mod:`models.demos.dots_ocr.tt.mesh` instead.
"""

from __future__ import annotations

from loguru import logger

from models.demos.dots_ocr.tt.mesh import default_mesh_shape as _default_mesh_shape


def default_mesh_shape_wh_lb():
    """Deprecated — use ``tt.mesh.default_mesh_shape`` instead."""
    logger.warning(
        "dots_ocr.tt.wh_lb.default_mesh_shape_wh_lb() is deprecated; " "use dots_ocr.tt.mesh.default_mesh_shape()."
    )
    return _default_mesh_shape()


def assert_single_wormhole_device(mesh_device) -> None:
    """Deprecated — retained for compatibility with older call sites that
    explicitly required a single-chip configuration.

    T3K and other multi-chip meshes are now supported; callers that still want
    to enforce single-chip should do so explicitly (or check
    ``mesh_device.get_num_devices() == 1``).
    """
    n = mesh_device.get_num_devices()
    assert n == 1, f"assert_single_wormhole_device expected 1 device, got {n}"
