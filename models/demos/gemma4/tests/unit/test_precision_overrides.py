# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression tests for Gemma4Precision.load key resolution.

Pins the HF_HUB_OFFLINE snapshot-path case: vLLM replaces the repo id with
the resolved snapshot directory, whose basename is the snapshot hash — the
variant lookup must still land on the repo basename, or every override in
precision_overrides.json is silently skipped (31B then loads all-bf16 and
OOMs the QB2 vLLM CI cell at 256k context).
"""

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


def test_ccl_topology_pinned_for_31b():
    """31B JSON still pins Linear (applied at max_seq_len >= 128k in create_tt_model).
    Model-wide, so it is read off the model entry and must not depend on the mesh key."""
    for mesh in ((1, 4), (1, 8)):
        assert Gemma4Precision.load("google/gemma-4-31B-it", mesh).ccl_topology == "linear"
    assert Gemma4Precision.load(SNAPSHOT_PATH, (1, 8)).ccl_topology == "linear"


def test_ccl_topology_unset_leaves_arch_default():
    """Models without the key return None so default_ccl_topology still picks."""
    assert Gemma4Precision.load("google/gemma-4-12B-it", (1, 8)).ccl_topology is None
    assert Gemma4Precision.load("/some/local/dir/my-finetune", (1, 8)).ccl_topology is None


def test_ccl_topology_rejects_unknown_value(tmp_path, monkeypatch):
    """A typo must fail loudly rather than silently falling back to the arch
    default -- the same contract single_tile_dest_acc has."""
    import json

    from models.demos.gemma4.tt import precision as precision_mod

    bad = tmp_path / "precision_overrides.json"
    bad.write_text(json.dumps({"my-model": {"ccl_topology": "rng", "default": {}}}))
    monkeypatch.setattr(precision_mod, "_PATH", str(bad))
    try:
        Gemma4Precision.load("my-model", (1, 8))
    except ValueError as e:
        assert "ccl_topology" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown ccl_topology")
