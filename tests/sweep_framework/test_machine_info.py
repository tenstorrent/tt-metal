# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only unit tests for board-type detection in model_tracer.generic_ops_tracer."""

import sys

import pytest

from model_tracer.generic_ops_tracer import _infer_board_type_from_series, get_machine_info


@pytest.fixture(autouse=True)
def _no_pyluwen(monkeypatch):
    # A None entry makes `from pyluwen import ...` raise ImportError, forcing the tt-smi path.
    monkeypatch.setitem(sys.modules, "pyluwen", None)


def _snapshot(board_series, arch=None):
    device = {"board_info": {"board_type": board_series}}
    if arch is not None:
        device["arch"] = arch
    return {"device_info": [device]}


@pytest.mark.parametrize(
    "series,expected",
    [
        ("n150", "Wormhole"),
        ("n300", "Wormhole"),
        ("p100a", "Blackhole"),
        ("p150b", "Blackhole"),
        ("p300a", "Blackhole"),
        ("tt-galaxy-wh", "Wormhole"),
        ("tt-galaxy-bh", "Blackhole"),
        ("e75", None),
        ("", None),
        (None, None),
    ],
)
def test_infer_board_type_from_series(series, expected):
    assert _infer_board_type_from_series(series) == expected


@pytest.mark.parametrize(
    "board_series,expected_series,expected_board_type",
    [
        ("n150 L", "n150", "Wormhole"),
        ("n300 L", "n300", "Wormhole"),
        ("n300 R", "n300", "Wormhole"),
        ("p100a", "p100a", "Blackhole"),
        ("p150b", "p150b", "Blackhole"),
        ("p300a", "p300a", "Blackhole"),
        ("tt-galaxy-wh", "tt-galaxy-wh", "Wormhole"),
        ("tt-galaxy-bh", "tt-galaxy-bh", "Blackhole"),
        ("e75", "e75", None),
    ],
)
def test_get_machine_info_infers_board_type_from_series(board_series, expected_series, expected_board_type):
    info = get_machine_info(tt_smi_snapshot=_snapshot(board_series))

    assert info["device_series"] == expected_series
    assert info["board_type"] == expected_board_type
    assert info["card_count"] == 1
    assert info["device_count"] == 1


def test_get_machine_info_arch_takes_precedence_over_series():
    # A Wormhole series with a Blackhole arch: arch must win.
    info = get_machine_info(tt_smi_snapshot=_snapshot("n150 L", arch="Blackhole"))

    assert info["board_type"] == "Blackhole"
    assert info["device_series"] == "n150"
