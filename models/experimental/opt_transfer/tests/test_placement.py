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


from models.experimental.opt_transfer.placement import tensor_bytes, L1Budget, eval_condition, decide_placement


def test_tensor_bytes():
    assert tensor_bytes([64, 1024], "bf16") == 64 * 1024 * 2
    assert tensor_bytes([4891, 1536], "bf16") == 4891 * 1536 * 2


def test_l1budget_fits():
    b = L1Budget(per_core_bytes=1_000_000, num_cores=100)  # 100 MB aggregate
    assert b.fits(50_000_000) is True
    assert b.fits(150_000_000) is False


def test_eval_condition_seq_threshold():
    cond = {"var": "seq", "op": "<=", "value": 1024}
    assert eval_condition(cond, {"seq": 512}) is True
    assert eval_condition(cond, {"seq": 4891}) is False
    assert eval_condition(None, {"seq": 4891}) is True  # no condition = always applies


def _obs(buffer, condition=None):
    return PlacementObservation(
        op="ttnn.linear",
        tensor_role="activation",
        size_descriptor={},
        memory_config={"buffer": buffer, "layout": "interleaved"},
        program_config=None,
        condition=condition,
        source="x",
    )


def test_decide_prefers_L1_when_small_and_donor_says_L1():
    budget = L1Budget(1_000_000, 100)
    p = decide_placement(
        [_obs("L1", {"var": "seq", "op": "<=", "value": 1024})],
        size_bytes=64 * 1024 * 2,
        dims={"seq": 64},
        l1_budget=budget,
    )
    assert p.buffer == "L1"


def test_decide_forces_DRAM_when_over_budget_even_if_donor_says_L1():
    budget = L1Budget(1_000_000, 100)  # 100 MB
    p = decide_placement(
        [_obs("L1", None)],  # donor unconditionally prefers L1
        size_bytes=150_000_000,
        dims={"seq": 4891},
        l1_budget=budget,
    )
    assert p.buffer == "DRAM"  # budget backstop wins


def test_decide_respects_donor_condition_false():
    budget = L1Budget(1_000_000, 100)
    p = decide_placement(
        [_obs("L1", {"var": "seq", "op": "<=", "value": 1024})],
        size_bytes=4891 * 1536 * 2,
        dims={"seq": 4891},
        l1_budget=budget,
    )
    assert p.buffer == "DRAM"  # seq>1024 -> donor L1 rule doesn't apply


def test_decide_defaults_dram_when_no_observation():
    p = decide_placement([], size_bytes=1024, dims={"seq": 8}, l1_budget=L1Budget(1_000_000, 100))
    assert p.buffer == "DRAM"
