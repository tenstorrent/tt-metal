from models.experimental.opt_transfer.config import CONFIG


def test_source_roots_exist():
    for rel in CONFIG.kb_source_roots:
        assert (CONFIG.repo_root / rel).exists(), rel


def test_seamless_model_registered():
    m = CONFIG.models["seamless_m4t_v2"]
    assert m["embed_dim"] == 1024 and m["num_heads"] == 16


def test_gate_thresholds_present():
    assert CONFIG.gates["per_block_pcc"] >= 0.99
    assert CONFIG.gates["full_pcc"] >= 0.99
    assert CONFIG.gates["min_perf_gain_pct"] > 0
