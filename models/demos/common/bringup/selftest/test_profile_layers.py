# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-op profiling never runs over the whole model: one representative layer per block type, or an explicit list."""

from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.testing.profile import op_layers

SPEC = "models/demos/xing40_a4b_d_p/bringup/spec.yaml"


def test_default_is_representatives(monkeypatch):
    monkeypatch.delenv("BRINGUP_PROFILE_LAYERS", raising=False)
    s = Spec.load(SPEC)
    reps = sorted({s.representative_layer(bt) for bt in s.data["block_types"]})
    assert op_layers(s, s.layers()) == reps
    assert len(reps) < len(s.layers())


def test_env_list(monkeypatch):
    s = Spec.load(SPEC)
    monkeypatch.setenv("BRINGUP_PROFILE_LAYERS", "2")
    assert op_layers(s, s.layers()) == [2]


def test_whole_model_refused(monkeypatch):
    s = Spec.load(SPEC)
    monkeypatch.setenv("BRINGUP_PROFILE_LAYERS", ",".join(map(str, s.layers())))
    try:
        op_layers(s, s.layers())
    except AssertionError as e:
        assert "minute per layer" in str(e)
    else:
        raise AssertionError("op mode over the whole model was not refused")
