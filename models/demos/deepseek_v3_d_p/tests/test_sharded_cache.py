# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the 2D weight-cache names and layout stamps (``utils/sharded_cache.py``)."""

from __future__ import annotations

import itertools

from models.demos.deepseek_v3_d_p.utils.sharded_cache import (
    STAMP_SUFFIX,
    LayoutCacheGuard,
    layout_descriptor,
    layout_tag,
)


def test_tags_name_the_split():
    assert layout_tag((8, 4), {"sp": 0, "tp": 1}) == "8x4_sp8ax0_tp4ax1"
    assert layout_tag((8, 4), {"h": 1, "i": 0}) == "8x4_h4ax1_i8ax0"


def test_different_layouts_never_share_a_tag(expect_error):
    meshes = [(8, 4), (4, 8), (4, 4), (2, 4), (8, 8)]
    tags = {}
    for mesh, sp_axis in itertools.product(meshes, (0, 1)):
        tag = layout_tag(mesh, {"sp": sp_axis, "tp": 1 - sp_axis})
        assert tag not in tags, f"{(mesh, sp_axis)} and {tags[tag]} share {tag}"
        tags[tag] = (mesh, sp_axis)
    # The two SC1 splits on the 8x4 Galaxy: hidden over the 4-chip axis (now) vs over the 8-chip axis (before).
    assert layout_tag((8, 4), {"h": 1, "i": 0}) != layout_tag((8, 4), {"h": 0, "i": 1})
    # Square mesh: the axis, not only the size, is in the tag.
    assert layout_tag((4, 4), {"sp": 0, "tp": 1}) != layout_tag((4, 4), {"sp": 1, "tp": 0})
    with expect_error(ValueError, "exactly one role"):
        layout_tag((8, 4), {"sp": 0, "tp": 0})


def _guard(tmp_path, mesh=(8, 4), axes=None, fmt=1, dtype="BFLOAT8_B"):
    axes = axes or {"sp": 0, "tp": 1}
    name = f"layer_0.mla2d_{layout_tag(mesh, axes)}"
    desc = layout_descriptor("ttMLA2DTP", fmt, mesh, axes, {"q_a_proj": dtype})
    return LayoutCacheGuard(tmp_path, name, f"{name}.*.tensorbin", desc)


def test_stamp_round_trip(tmp_path):
    g = _guard(tmp_path)
    g.check()  # cold: no shards, no stamp
    (tmp_path / f"{g.name}.q_a_proj_dtype_BFLOAT8_B_layout_TILE.tensorbin").touch()
    g.write()
    assert (tmp_path / f"{g.name}{STAMP_SUFFIX}").is_file()
    assert g.stamped()
    g.check()


def test_mismatched_or_missing_stamp_raises(tmp_path, expect_error):
    g = _guard(tmp_path)
    (tmp_path / f"{g.name}.q_a_proj_dtype_BFLOAT8_B_layout_TILE.tensorbin").touch()
    with expect_error(RuntimeError, "no layout stamp"):
        g.check()
    g.write()
    # Same name, other content: a format bump or another dtype must not load these shards.
    for other in (_guard(tmp_path, fmt=2), _guard(tmp_path, dtype="BFLOAT16")):
        assert other.name == g.name and not other.stamped()
        with expect_error(RuntimeError, "records layout"):
            other.check()
    # Corrupt stamp.
    g.stamp.write_text("not json")
    with expect_error(RuntimeError, "records layout"):
        g.check()


def test_no_cache_path_is_a_no_op(tmp_path):
    g = LayoutCacheGuard(None, "x", "x.*", {})
    g.check()
    g.write()
    assert not g.stamped() and g.shards() == []
