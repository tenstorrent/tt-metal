# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The per-run PCC pytest wall must be adaptive, not a flat 1800s.

A big model or a wide (sharded) mesh legitimately needs longer than the flat base for
reference load + first-run kernel compile. With a flat wall the sharded run is hard-killed
with rc=124 mid-compile and then misread as an ``OTHER`` failure, so the component never
graduates and is re-queued forever. `_adaptive_pcc_timeout` scales the base by model size
and, in shard mode, mesh degree (mirroring `cli._agent_complexity_timeout`)."""
import importlib
import json

import pytest


@pytest.fixture()
def bmcp(tmp_path, monkeypatch):
    monkeypatch.setenv("BRINGUP_MCP_DEMO_DIR", str(tmp_path))
    monkeypatch.setenv("BRINGUP_MCP_MODEL_ID", "test/model")
    monkeypatch.setenv("BRINGUP_MCP_STATE", str(tmp_path / "state.json"))
    monkeypatch.delenv("BRINGUP_MCP_TIMEOUT_MODE", raising=False)
    import scripts.tt_hw_planner.bringup_mcp as m

    importlib.reload(m)
    (tmp_path / "_stubs").mkdir(parents=True, exist_ok=True)
    return m, tmp_path


def _write_status(tmp, shape):
    (tmp / "bringup_status.json").write_text(
        json.dumps({"components": [{"name": "decoder", "new_shape": shape}]})
    )


def test_small_single_device_keeps_base(bmcp, monkeypatch):
    """A small model on a single device gets bonus 0 — the flat base is preserved, so
    nothing that already passes within 1800s is slowed down."""
    m, tmp = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    _write_status(tmp, {"hidden_size": 512, "num_hidden_layers": 4})
    assert m._adaptive_pcc_timeout(shard=False) == 1800


def test_unknown_shape_keeps_base(bmcp, monkeypatch):
    """No status doc / unknown shape must never break the run — fall back to the base."""
    m, _ = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    assert m._adaptive_pcc_timeout(shard=False) == 1800


def test_fixed_mode_restores_flat_wall(bmcp, monkeypatch):
    """The BRINGUP_MCP_TIMEOUT_MODE=fixed escape hatch restores the old flat behaviour
    even for a big sharded model."""
    m, tmp = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 4)
    monkeypatch.setattr(m, "_SHARD_DP", 1)
    _write_status(tmp, {"hidden_size": 4096, "num_hidden_layers": 64})
    monkeypatch.setenv("BRINGUP_MCP_TIMEOUT_MODE", "fixed")
    assert m._adaptive_pcc_timeout(shard=True) == 1800


def test_non_positive_base_is_untouched(bmcp, monkeypatch):
    m, _ = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 0)
    assert m._adaptive_pcc_timeout(shard=True) == 0


def test_size_bonus_buckets(bmcp, monkeypatch):
    m, tmp = bmcp
    # ~30B-class: hidden*layers >= 4096*48 -> bonus 3
    _write_status(tmp, {"hidden_size": 4096, "num_hidden_layers": 48})
    assert m._model_size_bonus() == 3
    # ~7-13B-class: >= 4096*32 but < 4096*48 -> bonus 2
    _write_status(tmp, {"hidden_size": 4096, "num_hidden_layers": 32})
    assert m._model_size_bonus() == 2
    # ~1-3B-class: >= 2048*24 -> bonus 1
    _write_status(tmp, {"hidden_size": 2048, "num_hidden_layers": 24})
    assert m._model_size_bonus() == 1
    # small -> bonus 0
    _write_status(tmp, {"hidden_size": 512, "num_hidden_layers": 4})
    assert m._model_size_bonus() == 0


def test_shard_adds_one_unit_per_extra_chip(bmcp, monkeypatch):
    """A big model (size bonus 3) sharded at TP=2 DP=1 (one extra chip) -> bonus 4:
    1800 + 600*4 = 4200s, comfortably above the ~30 min a sharded 30B run needs."""
    m, tmp = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 2)
    monkeypatch.setattr(m, "_SHARD_DP", 1)
    _write_status(tmp, {"hidden_size": 4096, "num_hidden_layers": 48})
    assert m._adaptive_pcc_timeout(shard=True) == 1800 + 600 * 4


def test_hard_cap_never_exceeds_base_plus_hour(bmcp, monkeypatch):
    """Even an absurdly wide mesh is capped at base + 60 min, so a genuine hang still
    dies in bounded time."""
    m, tmp = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 8)
    monkeypatch.setattr(m, "_SHARD_DP", 4)
    _write_status(tmp, {"hidden_size": 4096, "num_hidden_layers": 64})
    assert m._adaptive_pcc_timeout(shard=True) == 1800 + 6 * 600
