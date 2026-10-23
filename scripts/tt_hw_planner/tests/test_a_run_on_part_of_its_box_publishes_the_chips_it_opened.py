# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""publish-hf names the hardware for the chips the run opened, not the whole box.

minicpm5_2b, 2026-10-23: optimized on 1 chip of a QB2, so the serve mesh came from the run as (1, 1)
while the hardware label stayed the box's p300x2 (4 chips). tt-model's manifest check refused it --
"mesh_device '(1, 1)' opens a 1x1 mesh (1 chips) but hardware 'p300x2' implies 4" -- and every
container publish of the model stopped before the build.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402


@pytest.mark.parametrize(
    "hardware,arch,chips,want",
    [
        ("p300x2", "blackhole", 1, "p150"),
        ("p300x2", "blackhole", 2, "p300"),
        ("p300x2", "blackhole", 4, "p300x2"),
        ("n300x4", "wormhole_b0", 1, "n150"),
        ("n300x4", "wormhole_b0", 2, "n300"),
        ("n300x4", "wormhole_b0", 4, "n300x2"),
        ("n300x4", "wormhole_b0", 8, "n300x4"),
        ("p150", "blackhole", 1, "p150"),
        ("galaxy", "wormhole_b0", 1, "galaxy"),
        ("p300x2", "unknown", 1, "p300x2"),
    ],
)
def test_the_label_counts_the_chips_opened(hardware, arch, chips, want):
    assert ph._hardware_for_chips(hardware, arch, chips) == want


def _chips(label):
    m = re.fullmatch(r"([a-z]+\d+)[a-z]?(?:x(\d+))?", label)
    return ph._BOARD_CHIPS[m.group(1)] * int(m.group(2) or 1)


def _yaml(tmp_path, mesh, hardware=None):
    path = tmp_path / "tt-model.yaml"
    ph._write_tt_model_yaml(
        path,
        state={"throughput": {"current": 99.3}, "batch": 32},
        slug="s",
        repo_id="r",
        checkout=str(tmp_path),
        weights=None,
        box="QB2",
        arch=None,
        hardware=hardware,
        mesh_device=mesh,
        kind="vllm-plugin",
        plugin_ref="main",
        vllm_version="0.24.0",
        extra_models_dir="e",
        commit=None,
    )
    text = path.read_text()
    return re.search(r"hardware: (\S+)", text).group(1), text


def test_one_chip_of_a_qb2_serves_as_one_chip(tmp_path):
    hardware, text = _yaml(tmp_path, "(1, 1)")
    assert hardware == "p150" and _chips(hardware) == 1
    assert "Validated on 1 Tenstorrent Blackhole chip of a p300x2." in text
    assert "on 1 chip of a p300x2." in text


def test_the_whole_box_is_unchanged(tmp_path):
    hardware, text = _yaml(tmp_path, "(1, 4)")
    assert hardware == "p300x2"
    assert "Validated on Tenstorrent p300x2." in text


def test_an_explicit_hardware_still_wins(tmp_path):
    hardware, _ = _yaml(tmp_path, "(1, 1)", hardware="p300x2")
    assert hardware == "p300x2"
