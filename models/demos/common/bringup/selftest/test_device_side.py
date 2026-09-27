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


def test_host_transfers_counts_round_trips_and_restores_ttnn():
    """F27: the counter behind host_transfers_per_layer (agent rule 5)."""
    import types

    from models.demos.common.bringup.testing.host_transfers import HostTransfers

    fake = types.SimpleNamespace(
        from_torch=lambda t: ("dev", t), to_torch=lambda t: t[1], synchronize_device=lambda d: None
    )
    orig = fake.from_torch
    with HostTransfers(fake) as h:
        fake.to_torch(fake.from_torch(1))
        fake.from_torch(2)
    assert h.total == 3 and h.calls == {"from_torch": 2, "to_torch": 1}
    assert fake.from_torch is orig


def test_ladder_records_warm_host_transfers(gspec, noise):
    """F27: multi-chunk rungs record host_transfers_per_layer from warm chunks; the fixture model does no host work."""
    from models.demos.common.bringup.plan.ledger_gen import generate

    s = gspec()
    noise(1e-3)
    run_ladder(s, "s256", None)
    assert got()["host_transfers_per_layer"] == 0
    tasks = generate(s, fixture_model.Reference())["tasks"]
    gates = {t["id"]: t["gate"]["metrics"] for t in tasks if t["id"].startswith("L.")}
    assert gates["L.s256"]["host_transfers_per_layer"] == "== 0" and "host_transfers_per_layer" not in gates["L.last"]


def test_run_block_marks_a_profile_section_per_step(monkeypatch):
    """F33: X.1 got no device times because nothing marked sections; run_block now signposts every step."""
    from models.demos.common.bringup.reference.interface import Ctx, Step, run_block
    from models.demos.common.bringup.testing import profiler

    seen = []
    monkeypatch.setattr(profiler, "signpost", seen.append)
    steps = [Step("a", ["in"], "x"), Step("b", ["x"], "out")]
    out = run_block(steps, lambda n: (lambda ctx, v: v + 1), Ctx(0, 0, 1, None, {}), 0)
    assert out == 2 and seen == ["a", "b"]


def test_the_ledger_assembles_the_all_device_model_before_the_ladder(gspec):
    """F33: M.1 (role assemble) sits between the last swap test and the first ladder rung, gated on zero host transfers."""
    from models.demos.common.bringup.plan.ledger_gen import generate

    tasks = {t["id"]: t for t in generate(gspec(), fixture_model.Reference())["tasks"]}
    m = tasks["M.1"]
    assert m["step"] == "assemble" and m["gate"]["metrics"]["host_transfers_per_layer"] == "== 0"
    assert all(d.startswith("S.") for d in m["deps"]) and "M.1" in tasks["L.s256"]["deps"]


def test_contract_serves_the_layer_subset(fx, monkeypatch):
    """F41: the serving contract expects acks and read-back for the brought-up layers only (MiMo 0-5 of 48 expected
    96 acks for 2 chunks, 12 were right)."""
    from models.demos.common.bringup.testing import contract

    monkeypatch.setattr(contract.os, "environ", {})
    s = Spec.load(fx(layers=[0, 1]))
    assert contract.served_layers(s) == (0, 2)
    assert contract.engine_env(s)["PREFILL_NUM_LAYERS"] == "2"
    assert contract.served_layers(Spec.load(fx())) == (0, 3)
    with pytest.raises(ValueError, match="contiguous"):
        contract.served_layers(Spec.load(fx(layers=[0, 2])))


def test_full_prefill_times_a_prefix_subset(fx, monkeypatch):
    """F42: MiMo layers 0-5 of 48 recorded no prefill_ms_full; a prefix subset is timed (no final norm), a gapped one
    is not, the full stack still ends with the final norm."""
    import torch

    from models.demos.common.bringup.testing import profile

    rec = {}
    monkeypatch.setattr(profile.metrics, "record", lambda k, v: rec.__setitem__(k, v))

    class Fake:
        def __init__(self):
            self.calls = []

        def embed(self, t):
            return "h"

        def layer(self, i, h, s0, st):
            self.calls.append(i)
            return "h"

        def final_norm(self, h):
            self.calls.append("norm")
            return "h"

        def free(self, h):
            pass

        def sync(self):
            pass

    rung = {"seq": 256, "chunk": 128}
    tokens = torch.zeros(256, dtype=torch.long)
    m = Fake()
    assert profile.full_prefill(Spec.load(fx(layers=[0, 1])), m, None, [0, 1], rung, tokens) is not None
    assert "norm" not in m.calls and rec["prefill_layers"] == 2 and rec["prefill_ms_full"] >= 0
    assert profile.full_prefill(Spec.load(fx(layers=[0, 2])), Fake(), None, [0, 2], rung, tokens) is None
    m = Fake()
    profile.full_prefill(Spec.load(fx()), m, None, [0, 1, 2], rung, tokens)
    assert m.calls.count("norm") == 3  # compile pass, timed pass, per-chunk pass
