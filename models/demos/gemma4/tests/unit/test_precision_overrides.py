# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression tests for Gemma4Precision.load key resolution.

Pins the HF_HUB_OFFLINE snapshot-path case: vLLM replaces the repo id with
the resolved snapshot directory, whose basename is the snapshot hash — the
variant lookup must still land on the repo basename, or every override in
precision_overrides.json is silently skipped (31B then loads all-bf16 and
OOMs the QB2 vLLM CI cell at 256k context).
"""

import json

import ttnn
from models.demos.gemma4.tt.precision import Gemma4Precision

SNAPSHOT_PATH = (
    "/mnt/MLPerf/huggingface/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"
)


def test_repo_id_resolves_overrides():
    p = Gemma4Precision.load("google/gemma-4-31B-it", (1, 4))
    assert p.get("shared_mlp") == ttnn.bfloat8_b
    assert p.get("attention") == ttnn.bfloat8_b


def test_hf_snapshot_path_resolves_same_overrides():
    direct = Gemma4Precision.load("google/gemma-4-31B-it", (1, 4))
    snapshot = Gemma4Precision.load(SNAPSHOT_PATH, (1, 4))
    assert snapshot._overrides == direct._overrides
    assert snapshot.get("shared_mlp") == ttnn.bfloat8_b


def test_hf_snapshot_path_trailing_slash():
    p = Gemma4Precision.load(SNAPSHOT_PATH + "/", (1, 4))
    assert p.get("shared_mlp") == ttnn.bfloat8_b


def test_unknown_model_still_empty():
    p = Gemma4Precision.load("/some/local/dir/my-finetune", (1, 4))
    assert p._overrides == {}
    # default fallback stays the caller-supplied dtype
    assert p.get("shared_mlp", ttnn.bfloat16) == ttnn.bfloat16


# ── The bfp8 context ceiling is mesh-scoped ─────────────────────────────────


def test_the_bfp8_ceiling_applies_on_the_mesh_it_was_measured_on():
    """31B/tp=8 at 262k degenerates with shared_mlp in bfp8, so it downgrades."""
    o = Gemma4Precision.load("/x/gemma-4-31B-it", (1, 8), max_seq_len=262144)
    assert o.get("shared_mlp") == ttnn.bfloat16


def test_the_bfp8_ceiling_does_not_apply_on_a_mesh_where_bf16_will_not_fit():
    """The downgrade DOUBLES those weights, so it may only be declared where
    bf16 fits. Applying it to every mesh hung Gemma4-31B on bh_quietbox_2
    (1x4) in model init -- 31B shared_mlp in bf16 at 262144 does not fit on
    four chips, and the leg passed on main, which had no ceiling at all. A mesh
    with no entry keeps bfp8, i.e. main's behaviour."""
    o = Gemma4Precision.load("/x/gemma-4-31B-it", (1, 4), max_seq_len=262144)
    assert o.get("shared_mlp") == ttnn.bfloat8_b


def test_below_the_ceiling_bfp8_is_kept_on_the_measured_mesh():
    o = Gemma4Precision.load("/x/gemma-4-31B-it", (1, 8), max_seq_len=131072)
    assert o.get("shared_mlp") == ttnn.bfloat8_b


def test_ccl_topology_is_model_wide():
    """31B pins Linear; every other bundled variant leaves the arch default."""
    assert Gemma4Precision.load("/x/gemma-4-31B-it", (1, 8)).ccl_topology == "linear"
    for name in ("gemma-4-12B-it", "gemma-4-26B-A4B-it", "gemma-4-E2B-it"):
        assert Gemma4Precision.load(f"/x/{name}", (1, 8)).ccl_topology is None
    # Model-wide, so the mesh key must not change it.
    assert Gemma4Precision.load("/x/gemma-4-31B-it", (1, 2)).ccl_topology == "linear"


def test_ccl_topology_rejects_a_mesh_entry(tmp_path, monkeypatch, expect_error):
    """A mesh entry replaces 'default' wholesale, so this flag must not live there."""
    overrides = {"m": {"1x8": {"ccl_topology": "ring"}}}
    path = tmp_path / "precision_overrides.json"
    path.write_text(json.dumps(overrides))
    monkeypatch.setattr("models.demos.gemma4.tt.precision._PATH", str(path))
    with expect_error(ValueError, "model-wide"):
        Gemma4Precision.load("/x/m", (1, 8))


def test_ccl_topology_rejects_an_unknown_name(tmp_path, monkeypatch, expect_error):
    overrides = {"m": {"ccl_topology": "torus"}}
    path = tmp_path / "precision_overrides.json"
    path.write_text(json.dumps(overrides))
    monkeypatch.setattr("models.demos.gemma4.tt.precision._PATH", str(path))
    with expect_error(ValueError, "expected 'ring' or 'linear'"):
        Gemma4Precision.load("/x/m", (1, 8))


# --- single_tile_dest_acc: per-arch form -------------------------------------
# 31B turns the flag off as a workaround for Wormhole #38306 (HiFi3 with fp32
# dest-accumulation). Written model-wide it also turned it off on Blackhole,
# which main neither needs nor validates. The object form says which arch the
# workaround is for; an arch it does not name keeps the default.


def _write(tmp_path, monkeypatch, overrides):
    path = tmp_path / "precision_overrides.json"
    path.write_text(json.dumps(overrides))
    monkeypatch.setattr("models.demos.gemma4.tt.precision._PATH", str(path))


def test_dest_acc_per_arch_object_applies_only_to_the_named_arch(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"m": {"single_tile_dest_acc": {"wormhole_b0": False}}})
    monkeypatch.setattr("models.common.utility_functions.is_blackhole", lambda: False)
    assert Gemma4Precision.load("/x/m", (1, 8)).single_tile_dest_acc is False
    # Blackhole is not named, so it keeps the default rather than the workaround.
    monkeypatch.setattr("models.common.utility_functions.is_blackhole", lambda: True)
    assert Gemma4Precision.load("/x/m", (1, 8)).single_tile_dest_acc is True


def test_dest_acc_plain_boolean_still_applies_everywhere(tmp_path, monkeypatch):
    """A genuine model preference stays a bare bool and holds on every arch."""
    _write(tmp_path, monkeypatch, {"m": {"single_tile_dest_acc": False}})
    for blackhole in (False, True):
        monkeypatch.setattr("models.common.utility_functions.is_blackhole", lambda bh=blackhole: bh)
        assert Gemma4Precision.load("/x/m", (1, 8)).single_tile_dest_acc is False


def test_dest_acc_rejects_an_unknown_arch_key(tmp_path, monkeypatch, expect_error):
    """Raise rather than silently take the default, matching ccl_topology."""
    _write(tmp_path, monkeypatch, {"m": {"single_tile_dest_acc": {"grayskull": False}}})
    with expect_error(ValueError, "unknown arch"):
        Gemma4Precision.load("/x/m", (1, 8))


def test_31b_keeps_the_workaround_on_wormhole(monkeypatch):
    """The shipped entry must still resolve False on the arch it was measured on."""
    monkeypatch.setattr("models.common.utility_functions.is_blackhole", lambda: False)
    assert Gemma4Precision.load("/x/gemma-4-31B-it", (1, 8)).single_tile_dest_acc is False
