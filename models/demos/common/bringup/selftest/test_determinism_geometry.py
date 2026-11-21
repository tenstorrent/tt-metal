"""F296: the determinism check the framework adds to a gate takes the gate's accuracy unit's geometry (BRINGUP_CHUNK),
even when another test file declares BRINGUP_CHUNK a per-run variable (test_no_rebuild's BATCH_ITEM_VARS): otherwise an
8192 gate's determinism check loaded the 5120 model and F283 refused the 8192 oracles in that process (PB.MB, PB.9X)."""

from models.demos.common.bringup.core import batch as B

R = "scripts/run_safe_pytest.sh --run-all --no-precompile"
ORACLE = "models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_oracle.py"


def test_the_added_determinism_check_keeps_the_gates_chunk(monkeypatch):
    monkeypatch.setattr(B, "DECLARED_ITEM_VARS", set(B.DECLARED_ITEM_VARS) | {"BRINGUP_CHUNK"})
    units = [("T#test_determinism", B.DETERMINISM_CMD), ("T#oracle", f"BRINGUP_CHUNK=8192 DS41_PACKED=on {R} {ORACLE}")]
    cfg = B._config_of(units)
    assert cfg.get("BRINGUP_CHUNK") == "8192", cfg


def test_per_run_variables_of_the_units_own_tests_stay_out(monkeypatch):
    monkeypatch.setattr(B, "declared_item_vars", lambda t, repo=None: {"BRINGUP_CHUNK"} if t == ORACLE else set())
    units = [("T#oracle", f"BRINGUP_CHUNK=8192 BRINGUP_RUNG=last {R} {ORACLE}")]
    cfg = B._config_of(units)
    assert "BRINGUP_CHUNK" not in cfg and "BRINGUP_RUNG" not in cfg, cfg
