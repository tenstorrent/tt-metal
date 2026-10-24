# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the decode fusion defaults (tt/mesh_orientation.py): the fused decode step is the default
of a Blackhole MoE build on the 4-chip (1,4) / (4,1) meshes only, every env var still wins, and without a recorded
profile the modules keep the old paths. No device, no ttnn import.

    pytest models/demos/gemma4/tests/unit/test_decode_fusion_defaults_host.py --noconftest -p no:cacheprovider
"""

import pytest

from models.demos.gemma4.tt import mesh_orientation as mo


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in mo.DECODE_FUSION_ON:
        monkeypatch.delenv(name, raising=False)
    mo.clear_decode_fusion_profile()
    yield
    mo.clear_decode_fusion_profile()


def test_four_chip_blackhole_moe_is_fused():
    for shape in ((1, 4), (4, 1)):
        assert mo.fused_decode_build(shape, blackhole=True, moe=True)
        assert mo.decode_fusion_defaults(shape, blackhole=True, moe=True) == mo.DECODE_FUSION_ON
    assert mo.DECODE_FUSION_ON["GEMMA4_NORM_FUSED"] == "2"
    assert set(mo.DECODE_FUSION_ON) == {
        "GEMMA4_MLP_DECODE_1D", "GEMMA4_ATTN_DECODE_1D", "GEMMA4_ROUTER_DECODE_1D", "GEMMA4_HEAD_NORM_4D",
        "GEMMA4_MOE_FUSED", "GEMMA4_MLP_GEGLU_FUSED", "GEMMA4_SOFTCAP_FUSED", "GEMMA4_CONCAT_HEADS_FUSED",
        "GEMMA4_QKV_L1_OUT", "GEMMA4_NORM_FUSED", "GEMMA4_LAYER_NORM_FUSED", "GEMMA4_QKV_NORM_ROPE_FUSED",
    }  # fmt: skip


def test_other_builds_keep_the_old_defaults():
    assert not mo.fused_decode_build((1, 4), blackhole=False, moe=True)  # Wormhole
    assert not mo.fused_decode_build((1, 4), blackhole=True, moe=False)  # dense 31B / 12B
    for shape in ((1, 1), (1, 8), (2, 4), (4, 8), (8, 4), (1, 32)):  # one chip, the Galaxy lane engine
        assert not mo.fused_decode_build(shape, blackhole=True, moe=True)
    assert mo.decode_fusion_defaults((4, 8), blackhole=True, moe=True) == mo.DECODE_FUSION_OFF
    assert mo.decode_fusion_defaults((1, 4), blackhole=False, moe=True) == mo.DECODE_FUSION_OFF


def test_profile_and_env_override(monkeypatch):
    # no profile: the old paths, unless the env var asks
    assert mo.decode_flag("GEMMA4_MOE_FUSED") == "0"
    monkeypatch.setenv("GEMMA4_MOE_FUSED", "1")
    assert mo.decode_flag("GEMMA4_MOE_FUSED") == "1"
    monkeypatch.delenv("GEMMA4_MOE_FUSED")
    # the 4-chip Blackhole MoE profile turns everything on ...
    assert mo.set_decode_fusion_profile((1, 4), blackhole=True, moe=True) == mo.DECODE_FUSION_ON
    for name, value in mo.DECODE_FUSION_ON.items():
        assert mo.decode_flag(name) == value
    # ... and a set env var still wins, in either direction
    monkeypatch.setenv("GEMMA4_QKV_NORM_ROPE_FUSED", "0")
    assert mo.decode_flag("GEMMA4_QKV_NORM_ROPE_FUSED") == "0"
    monkeypatch.setenv("GEMMA4_NORM_FUSED", "1")
    assert mo.decode_flag("GEMMA4_NORM_FUSED") == "1"
    monkeypatch.setenv("GEMMA4_MOE_FUSED", "")  # empty = unset
    assert mo.decode_flag("GEMMA4_MOE_FUSED") == "1"
    # a Wormhole profile turns everything off again, env still wins
    assert mo.set_decode_fusion_profile((1, 4), blackhole=False, moe=True) == mo.DECODE_FUSION_OFF
    assert mo.decode_flag("GEMMA4_MLP_DECODE_1D") == "0"
    monkeypatch.setenv("GEMMA4_MLP_DECODE_1D", "1")
    assert mo.decode_flag("GEMMA4_MLP_DECODE_1D") == "1"


def test_four_chip_blackhole_dense_runs_the_combined_dense_configuration():
    for shape in ((1, 4), (4, 1)):
        assert mo.dense_fused_decode_build(shape, blackhole=True, moe=False)
        assert not mo.fused_decode_build(shape, blackhole=True, moe=False)
        d = mo.decode_fusion_defaults(shape, blackhole=True, moe=False)
        assert d == mo.DENSE_DECODE_FUSION_ON
        for name in mo.DECODE_FUSION_ON:
            if name not in mo.DENSE_DECODE_FUSION_ON:
                assert (
                    d.get(name, "0") == "0"
                )  # the MoE-only fusions and the fused norm family stay off on dense widths
    # the MoE profile keeps every MoE flag on and none of the dense-only ones
    d = mo.decode_fusion_defaults((1, 4), blackhole=True, moe=True)
    assert d == mo.DECODE_FUSION_ON
    for name in mo.DENSE_DECODE_FUSION_ON:
        assert d.get(name, "0") == mo.DECODE_FUSION_ON.get(name, "0")
    # Wormhole, one chip and the (1,8) line keep the old dense defaults
    assert not mo.dense_fused_decode_build((1, 4), blackhole=False, moe=False)
    for shape in ((1, 1), (1, 8), (4, 8)):
        assert not mo.dense_fused_decode_build(shape, blackhole=True, moe=False)
        assert mo.decode_fusion_defaults(shape, blackhole=True, moe=False) == mo.DECODE_FUSION_OFF


def test_dense_profile_env_override(monkeypatch):
    monkeypatch.delenv("GEMMA4_QKV_HEADS_FUSED", raising=False)
    mo.set_decode_fusion_profile((1, 4), blackhole=True, moe=False)
    try:
        assert mo.decode_flag("GEMMA4_QKV_HEADS_FUSED") == "1"
        assert mo.decode_flag("GEMMA4_NORM_FUSED") == "0"
        monkeypatch.setenv("GEMMA4_QKV_HEADS_FUSED", "0")
        assert mo.decode_flag("GEMMA4_QKV_HEADS_FUSED") == "0"  # a set env var wins
    finally:
        mo.clear_decode_fusion_profile()
