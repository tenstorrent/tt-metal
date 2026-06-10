"""Tests for lib/parallelism.py — placement decision procedure."""

from __future__ import annotations

from skills.orchestrator.lib.parallelism import plan_parallelism

GB = 1024**3


def _rows(result):
    return {r["name"]: r for r in result["plan"]}


def test_single_device_replicates_everything():
    out = plan_parallelism([{"name": "lm", "cadence": "per_token", "param_bytes": GB}], 1, 32 * GB)
    assert _rows(out)["lm"]["placement"] == "replicate"


def test_per_token_shards_per_input_replicates_when_fitting():
    comps = [
        {"name": "lm", "cadence": "per_token", "param_bytes": 3 * GB, "q_heads": 16, "kv_heads": 8},
        {"name": "vision", "cadence": "per_input", "param_bytes": GB},
    ]
    out = plan_parallelism(comps, 4, 32 * GB)
    rows = _rows(out)
    assert rows["lm"]["placement"] == "shard"
    assert rows["vision"]["placement"] == "replicate"
    assert out["judgments"] == []


def test_kv_heads_below_tp_forces_replication_factor():
    out = plan_parallelism(
        [{"name": "lm", "cadence": "per_token", "param_bytes": GB, "q_heads": 16, "kv_heads": 2}], 4, 32 * GB
    )
    assert _rows(out)["lm"]["kv_replication"] == 2


def test_q_heads_not_divisible_raises_judgment():
    out = plan_parallelism(
        [{"name": "lm", "cadence": "per_token", "param_bytes": GB, "q_heads": 14, "kv_heads": 2}], 4, 32 * GB
    )
    assert any("q_heads=14" in j for j in out["judgments"])


def test_oversized_model_forces_encoder_shard():
    comps = [
        {"name": "lm", "cadence": "per_token", "param_bytes": 100 * GB},
        {"name": "vision", "cadence": "per_input", "param_bytes": 40 * GB},
    ]
    out = plan_parallelism(comps, 4, 32 * GB)
    assert _rows(out)["vision"]["placement"] == "shard"
    assert not out["fits_replicated"]


def test_unknown_cadence_is_a_judgment_not_a_plan_row():
    out = plan_parallelism([{"name": "mystery", "param_bytes": GB}], 4, 32 * GB)
    assert "mystery" not in _rows(out)
    assert any("mystery" in j for j in out["judgments"])
