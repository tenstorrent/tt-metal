# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.tt_dit.utils.conv3d import _blocking_mesh_override

# 4x8 1080p per-chip shape with a swept entry; 544x960 on 2x4 has the same per-chip shape.
SHAPE = (128, 128, (3, 3, 3), 155, 68, 60)


def test_unset_keeps_mesh_factors(monkeypatch):
    monkeypatch.delenv("LTX_CONV3D_BLOCKING_MESH", raising=False)
    assert _blocking_mesh_override(2, 4, *SHAPE) == (2, 4)


def test_override_maps_to_4x8_entry(monkeypatch):
    monkeypatch.setenv("LTX_CONV3D_BLOCKING_MESH", "4,8")
    assert _blocking_mesh_override(2, 4, *SHAPE) == (4, 8)


def test_override_without_matching_entry_keeps_mesh_factors(monkeypatch):
    monkeypatch.setenv("LTX_CONV3D_BLOCKING_MESH", "4,8")
    assert _blocking_mesh_override(2, 4, 128, 128, (3, 3, 3), 155, 999, 999) == (2, 4)
