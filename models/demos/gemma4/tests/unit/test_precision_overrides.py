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
