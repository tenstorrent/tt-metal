# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the TP foundation (``ttnn_impl/tp_config.py``).

Assert the **default-OFF** contract: nothing changes until ``ACE_STEP_TP`` is set, even on a
multi-chip mesh; collectives are strict pass-throughs when off; config resolves to the
tt_transformers per-axis convention when on. No TT device is opened here — the ON-path weight
mapper / collectives are covered by the device-gated gate-G0 test (see docs/TP4_PLAN.md).
"""

from __future__ import annotations

import pytest

from models.experimental.ace_step_v1_5.ttnn_impl import tp_config as tp


class _FakeMesh:
    """Minimal stand-in for a ttnn.MeshDevice (shape + device count only)."""

    def __init__(self, rows: int, cols: int) -> None:
        self.shape = (rows, cols)
        self._n = rows * cols

    def get_num_devices(self) -> int:
        return self._n


def test_off_by_default_single_chip():
    cfg = tp.resolve_tp_config(None)
    assert cfg.enabled is False
    assert cfg.degree == 1


def test_off_by_default_even_on_2x2_mesh(monkeypatch):
    monkeypatch.delenv("ACE_STEP_TP", raising=False)
    cfg = tp.resolve_tp_config(_FakeMesh(2, 2))
    assert cfg.enabled is False, "TP must stay OFF on multi-chip until explicitly enabled"
    assert cfg.degree == 1
    assert cfg.mesh_shape == (2, 2)


def test_off_path_mapper_is_replicate_none_for_none_device():
    assert tp.tp_weight_mesh_mapper(None, shard_dim=0) is None


def test_off_path_collectives_are_passthrough():
    sentinel = object()
    assert tp.tp_all_reduce(sentinel, None) is sentinel
    assert tp.tp_all_gather(sentinel, None, dim=-1) is sentinel


@pytest.mark.parametrize("flag", ["on", "1", "true", "yes", "auto"])
def test_enabled_flags_turn_on_for_multichip(monkeypatch, flag):
    monkeypatch.setenv("ACE_STEP_TP", flag)
    cfg = tp.resolve_tp_config(_FakeMesh(2, 2))
    assert cfg.enabled is True
    assert cfg.degree == 2  # cols axis (tt_transformers head-shard convention)


def test_enabled_flag_ignored_on_single_chip(monkeypatch):
    monkeypatch.setenv("ACE_STEP_TP", "on")
    cfg = tp.resolve_tp_config(_FakeMesh(1, 1))
    assert cfg.enabled is False
    assert cfg.degree == 1


def test_tp_axis_env_selects_rows(monkeypatch):
    monkeypatch.setenv("ACE_STEP_TP", "on")
    monkeypatch.setenv("ACE_STEP_TP_AXIS", "0")
    cfg = tp.resolve_tp_config(_FakeMesh(2, 4))
    assert cfg.axis == 0
    assert cfg.degree == 2  # rows

    monkeypatch.setenv("ACE_STEP_TP_AXIS", "1")
    cfg2 = tp.resolve_tp_config(_FakeMesh(2, 4))
    assert cfg2.axis == 1
    assert cfg2.degree == 4  # cols
