# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The mesh/TP drives whether Phase 2 (shard-aware graduation) runs: run_bringup_cc derives the shard
degree from select_parallelism and enables Phase 2 only when TP>1. No mesh / 1 chip / TP=1 => Phase 2
stays off (single-device bring-up unchanged). This is the 'mesh decides, not a manual flag' wiring."""
from scripts.tt_hw_planner._cli_helpers import bringup_cc


def test_mesh_chip_count_parsing():
    assert bringup_cc._mesh_chips(None) == 1
    assert bringup_cc._mesh_chips("") == 1
    assert bringup_cc._mesh_chips("2x2") == 4
    assert bringup_cc._mesh_chips("1x8") == 8
    assert bringup_cc._mesh_chips("junk") == 1


def test_single_chip_no_shard(monkeypatch):
    assert bringup_cc._derive_shard_tp("any/model", None) == 1
    assert bringup_cc._derive_shard_tp("any/model", "1x1") == 1


def test_derive_uses_selector(monkeypatch):
    class _KR:
        tp_grid = [1, 2, 4, 8]

        def has_blockers(self, tp=None):
            return tp in (4, 8)

    class _Probe:
        raw_config = {"num_key_value_heads": 2}

    import scripts.tt_hw_planner.cli as cli

    monkeypatch.setattr(cli, "probe_model", lambda m: _Probe(), raising=False)
    monkeypatch.setattr(cli, "evaluate_kernels", lambda cfg, tp_grid=None: _KR(), raising=False)
    assert bringup_cc._derive_shard_tp("m", "2x2") == 2
    assert bringup_cc._derive_shard_tp("m", "1x2") == 2


def test_derive_falls_to_one_on_probe_failure(monkeypatch):
    import scripts.tt_hw_planner.cli as cli

    def _boom(*a, **k):
        raise RuntimeError("no config")

    monkeypatch.setattr(cli, "probe_model", _boom, raising=False)
    assert bringup_cc._derive_shard_tp("m", "2x2") == 1
