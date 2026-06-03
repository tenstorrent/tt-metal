from models.experimental.opt_transfer.schema import PlacementObservation, MemoryPlacement, KBEntry, PatternKind
from models.experimental.opt_transfer.config import CONFIG


def test_placement_observation_roundtrips():
    o = PlacementObservation(
        op="ttnn.matmul",
        tensor_role="activation",
        size_descriptor={"dims": "[seq, hidden]", "dtype": "bf16", "bytes_expr": "seq*hidden*2"},
        memory_config={"buffer": "L1", "layout": "interleaved", "shard_spec_template": None},
        program_config=None,
        condition={"var": "seq", "op": "<=", "value": 1024},
        source="models/x.py:10",
    )
    assert PlacementObservation.from_dict(o.to_dict()) == o


def test_kbentry_carries_placement_observations():
    e = KBEntry(
        id="m",
        fused_op="ttnn.matmul",
        category="mlp",
        pattern_kind=PatternKind.CHAIN,
        torch_pattern=["linear"],
        signature={},
        config_template={},
        weight_transform=None,
        source="x",
    )
    assert e.placement_observations == []  # defaults empty
    e2 = KBEntry.from_dict(e.to_dict())
    assert e2.placement_observations == []


def test_memory_placement_defaults_interleaved():
    p = MemoryPlacement(buffer="L1")
    assert p.layout == "interleaved" and p.shard_spec is None


def test_config_has_l1_budget_and_placement_gate():
    assert CONFIG.l1_budgets["blackhole"]["per_core_bytes"] > 0
    assert CONFIG.l1_budgets["blackhole"]["num_cores"] > 0
    assert CONFIG.gates["placement_min_gain_pct"] > 0
