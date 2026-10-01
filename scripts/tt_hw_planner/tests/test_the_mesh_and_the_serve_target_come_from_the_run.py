# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""--mesh reads the same in both spellings, and publish-hf serves on the hardware the run detected.

WH Galaxy, 2026-09-29: `optimize --mesh 4,8` split on 'x' only, so the comma form -- the one the
CLI's own parser and help text use -- read as no mesh at all. And publish-hf, given no --box, wrote a
Blackhole p300x2 serve block and a `blackhole` card tag for a model measured on 32 Wormhole chips.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import optimize  # noqa: E402
from tt_hw_planner.commands import publish_hf as ph  # noqa: E402
from tt_hw_planner.hardware import HARDWARE  # noqa: E402


@pytest.mark.parametrize("mesh,chips", [("4,8", 32), ("4x8", 32), ("4X8", 32), ("8", 8), ("2x2x2", 8), ("", 0)])
def test_both_spellings_count_the_same_chips(mesh, chips):
    assert optimize._chip_count_from_mesh(mesh) == chips


def test_nothing_that_parsed_before_parses_differently():
    for mesh in ("1x1", "1x2", "1x4", "1x8", "2x2", "2x4", "4x8", "8x4", "abc", "x", "4x", None):
        old = 0
        if mesh:
            try:
                old = 1
                for tok in str(mesh).lower().split("x"):
                    old *= int(tok)
            except Exception:  # noqa: BLE001
                old = 0
        assert optimize._chip_count_from_mesh(mesh) == max(old, 0), mesh


def _env(arch, chips):
    return {"arch": arch, "device_count": chips}


def test_the_serve_target_is_the_detected_box():
    for b in HARDWARE:
        if b.name not in ph._BOX_TARGET:
            continue
        same = [o for o in HARDWARE if o.arch == b.arch and o.chips == b.chips and o.name in ph._BOX_TARGET]
        got = ph._box_from_env(_env(b.arch.lower(), b.chips))
        assert got == (b.name if len(same) == 1 else None), b.name


def test_an_explicit_box_still_wins():
    for name, target in ph._BOX_TARGET.items():
        assert ph._serve_target(name, _env("other", 999)) == target


def test_undetected_hardware_is_no_guess():
    assert ph._serve_target(None, None) == (None, None, None)
    assert ph._serve_target("NoSuchBox", _env("other", 3)) == (None, None, None)


def test_no_serve_target_refuses_to_write(tmp_path, expect_error):
    with expect_error(ValueError, "no serve target"):
        ph._write_tt_model_yaml(
            tmp_path / "tt-model.yaml",
            state={},
            slug="s",
            repo_id="r",
            checkout=str(tmp_path),
            weights=None,
            box=None,
            arch=None,
            hardware=None,
            mesh_device=None,
            kind="k",
            plugin_ref="p",
            vllm_version="v",
            extra_models_dir="e",
            commit=None,
        )
    assert not (tmp_path / "tt-model.yaml").exists()


def test_the_card_tags_the_detected_arch():
    card = ph._build_card({"env": _env("wormhole", 32)}, "s", None, None)
    head = card.split("---")[1]
    assert "  - wormhole" in head and "blackhole" not in head
    assert "  - blackhole" in ph._build_card({"env": _env("blackhole", 4)}, "s", None, None).split("---")[1]
    untagged = ph._build_card({}, "s", None, None).split("---")[1]
    assert "wormhole" not in untagged and "blackhole" not in untagged
