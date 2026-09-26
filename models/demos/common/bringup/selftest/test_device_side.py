# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F4: component, swap and ladder helpers and the rendered test templates, against the fixture's fake device hooks
(the CPU reference plus deterministic noise). CPU only: the real device runs happen in a model's own gates."""

import importlib.util

import pytest

from models.demos.common.bringup.core.runs import IMPL_ENV
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.reference import generate_golden, prompt
from models.demos.common.bringup.selftest import fixture_model
from models.demos.common.bringup.selftest.conftest import got
from models.demos.common.bringup.testing.component import run_component_test, run_swap_test
from models.demos.common.bringup.testing.ladder import run_ladder
from models.demos.common.bringup.testing.templates import render_component_test, render_swap_test


@pytest.fixture
def gspec(fx, monkeypatch):
    """Fixture spec with the s256 (full dumps) and s512 goldens generated."""

    def make(**over):
        p = fx(**over)
        monkeypatch.setenv("BRINGUP_SPEC", p)
        prompt.build(Spec.load(p))
        generate_golden.main(["--spec", p, "--rung", "s256"])
        generate_golden.main(["--spec", p, "--rung", "s512"])
        return Spec.load(p)

    return make


@pytest.fixture
def noise(monkeypatch):
    def set_(v):
        monkeypatch.setitem(fixture_model.NOISE, "value", v)

    return set_


@pytest.mark.parametrize(
    "impl,noise_level,passes",
    [
        ("device", 1e-3, True),
        ("device", 0.5, False),
        ("reference", 0.5, True),
        ("stub", 0.0, False),
    ],
)
def test_component_stateful_step(gspec, noise, monkeypatch, impl, noise_level, passes):
    s = gspec()
    noise(noise_level)
    monkeypatch.setenv(IMPL_ENV, impl)
    assert run_component_test(s, "attention", layer=1) is passes
    assert "pcc_attention_L01" in got()


def test_swap_order(gspec, noise, monkeypatch):
    s = gspec()
    noise(1e-3)
    assert run_swap_test(s, "blk", ["attn_norm", "attention"])
    m = got()
    assert m["pcc_swap_out"] > 0.99 and "pcc_swap_attn_out" in m and "pcc_swap_in" not in m
    noise(0.5)
    assert not run_swap_test(s, "blk", ["attn_norm", "attention", "attn_residual", "ffn_norm", "mlp"])


def test_ladder_all_chunks(gspec, noise):
    s = gspec()
    noise(1e-3)
    out = run_ladder(s, "s256", None)
    m = got()
    assert not out["failed"], out
    assert sorted(k for k in m if k.startswith("pcc_layer_")) == ["pcc_layer_L00", "pcc_layer_L01", "pcc_layer_L02"]
    assert m["pcc_final_hidden"] > 0.99 and m["top5_overlap"] >= 0.9 and m["pcc_state_min"] > 0.99
    assert m["subset"] == 0 and "chunk_seconds_c03" in m


def test_ladder_last_chunk_after_golden_prefix(gspec, noise):
    s = gspec()
    noise(1e-3)
    out = run_ladder(s, "last", None)
    m = got()
    assert not out["failed"] and "chunk_seconds_c03" in m and "chunk_seconds_c00" not in m


def test_ladder_catches_a_bad_layer(gspec, noise):
    s = gspec()
    noise(0.5)
    assert run_ladder(s, "s256", None)["failed"]


def test_ladder_on_a_layer_subset_restarts_each_run_from_the_golden(gspec, noise):
    s = gspec(layers=[0, 2])
    noise(0.0)  # exact fake: any mismatch would come from wiring, not noise
    out = run_ladder(s, "s512", None)
    m = got()
    assert not out["failed"], out
    assert m["subset"] == 1 and m["covered_layers"] == 2 and "pcc_layer_L01" not in m
    assert m["pcc_layer_L02"] > 0.9999 and m["pcc_final_hidden"] > 0.9999


def _load(path):
    spec_ = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("impl,passes", [("reference", True), ("stub", False)])
def test_rendered_templates(gspec, noise, monkeypatch, impl, passes):
    s = gspec()
    monkeypatch.setenv(IMPL_ENV, impl)
    comp = render_component_test(s, "blk", "mlp")
    swap = render_swap_test(s, "blk", ["attn_norm", "attention"])
    assert comp.name == "test_c_blk_mlp.py" and swap.name == "test_swap_blk_02_attention.py"
    assert (comp.parent / "__init__.py").exists()
    for path, fn in ((comp, "test_component"), (swap, "test_swap")):
        test = getattr(_load(path), fn)
        if passes:
            test(None)
        else:
            with pytest.raises(AssertionError):
                test(None)
    before = comp.read_text()
    render_component_test(s, "blk", "mlp", thr=0.5)  # never overwrites an existing (possibly frozen) test
    assert comp.read_text() == before


def test_device_tests_outlive_the_repo_pytest_timeout(fx):
    """F25: the ladder, contract and profile tests carry their own pytest timeout (spec box.test_timeout_s, default
    3600 s); the repo's pytest.ini 300 s killed a full-target rung that had passed its checks."""
    from pathlib import Path

    from models.demos.common.bringup.testing.harness import DEVICE_TEST_TIMEOUT_S, device_timeout

    assert device_timeout(Spec.load(fx())).args == (DEVICE_TEST_TIMEOUT_S,)
    assert device_timeout(Spec.load(fx(box={"mesh": [1, 4], "test_timeout_s": 7}))).args == (7,)
    tests = Path(__file__).parents[1] / "tests"
    for name in ("test_ladder.py", "test_contract.py", "test_profile.py"):
        assert "pytestmark = device_timeout(S)" in (tests / name).read_text(), name
