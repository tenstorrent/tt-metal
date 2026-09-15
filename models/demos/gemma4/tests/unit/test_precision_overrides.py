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
