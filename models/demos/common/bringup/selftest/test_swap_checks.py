# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F49: per-step swap checks, the mistake-injection impl mode, and swap tests frozen without a review. CPU only,
against the synthetic fixture model (its fake device is the CPU reference plus deterministic noise)."""

import json

import pytest
import torch

from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.runs import IMPL_ENV
from models.demos.common.bringup.selftest import fixture_model
from models.demos.common.bringup.selftest.conftest import got
from models.demos.common.bringup.selftest.test_device_side import _load, gspec, noise  # noqa: F401 (fixtures)
from models.demos.common.bringup.testing import mutate as MU
from models.demos.common.bringup.testing.component import (
    _isolated,
    component_error,
    component_test_limits,
    run_swap_test,
    step_errors,
)
from models.demos.common.bringup.testing.templates import render_component_test, render_swap_test

SWAPPED = ["attn_norm", "attention", "attn_residual"]


def _results(s, task, metrics):
    p = s.bringup_dir / "results" / f"{task}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"task": task, "metrics": {k: {"value": v} for k, v in metrics.items()}}))


def test_checks_none_is_unchanged_and_steps_does_not_change_the_block(gspec, noise):
    s = gspec()
    noise(1e-3)
    assert run_swap_test(s, "blk", SWAPPED)
    old = {k: v for k, v in got().items() if k.startswith(("pcc_swap_", "swap_"))}
    assert not [k for k in old if k.startswith("swap_")], "checks=None records no per-step metrics"
    M.reset("T")
    assert run_swap_test(s, "blk", SWAPPED, checks="steps")
    new = got()
    # the CPU runs on the same inputs (the stateful attention's on a copied state) leave the block itself unchanged
    assert {k: v for k, v in new.items() if k.startswith("pcc_swap_")} == old
    assert (
        new["swap_attention_rel"] < 0.01 and new["swap_attention_finite"] == 1 and "swap_attention_out_rel" not in new
    )
    assert new["swap_attn_residual_out_rel"] < 0.01  # the last swapped step's effect on the block out
    noise(0.5)
    assert not run_swap_test(s, "blk", SWAPPED)


def test_steps_pass_with_the_reference_and_fail_with_the_stub(gspec, monkeypatch):
    s = gspec()
    monkeypatch.setenv(IMPL_ENV, "reference")
    assert run_swap_test(s, "blk", SWAPPED, checks="steps")
    m = got()
    assert m["swap_attention_rel"] == 0 and m["swap_attention_vs_cpu"] > 0.999999 and m["swap_attention_bias"] == 0
    monkeypatch.setenv(IMPL_ENV, "stub")
    assert not run_swap_test(s, "blk", SWAPPED, checks="steps")


@pytest.mark.parametrize("step", ["attention", "attn_residual"])
@pytest.mark.parametrize("kind", MU.FLOAT_KINDS)
def test_steps_catch_every_mutation_of_a_swapped_step(gspec, monkeypatch, kind, step):
    s = gspec()
    layer = s.representative_layer("blk")
    for n in SWAPPED:  # the device component gates as they would have recorded (fixture noise 1e-3: pcc ~1 - 1e-7)
        _results(s, f"C.blk.{n}", {f"pcc_{n}_L{layer:02d}": 0.9999999})
    monkeypatch.setenv(IMPL_ENV, f"mutate:{kind}")
    monkeypatch.setenv(MU.MUTATE_STEP_ENV, step)
    assert not run_swap_test(s, "blk", SWAPPED, checks="steps")
    m = got()
    assert m["swap_attn_norm_rel"] == 0, "only the named step is mutated"
    assert m[f"swap_{step}_rel_limit"] == 0.005  # calibrated to the component error, never below the floor


def test_the_bf16_control_passes_and_limits_fall_back_without_a_component_result(gspec, monkeypatch):
    s = gspec()
    monkeypatch.setenv(IMPL_ENV, "mutate:bf16")  # every swapped step on bf16-rounded inputs, bf16 output
    assert run_swap_test(s, "blk", SWAPPED, checks="steps")
    m = got()
    assert 0 < m["swap_attention_rel"] < 0.01 and m["swap_attention_rel_limit"] == 0.02
    assert "swap_attention_component_err" not in m


def test_steps_catch_a_cpu_bridge(gspec, noise, monkeypatch):
    s = gspec()
    noise(1e-3)
    real = fixture_model.device_component

    def bridged(mesh, spec, layer, name):
        fn = real(mesh, spec, layer, name)
        fn.cpu_bridge = name == "attention"
        return fn

    monkeypatch.setattr(fixture_model, "device_component", bridged)
    assert run_swap_test(s, "blk", SWAPPED), "checks=None never looked at the bridge"
    assert not run_swap_test(s, "blk", SWAPPED, checks="steps")
    assert got()["swap_attention_cpu_bridge"] == 1


def test_component_limits_are_read_from_the_frozen_component_test(gspec, monkeypatch):
    s = gspec()
    assert component_test_limits(s, "blk", "mlp") == (None, None)
    render_component_test(s, "blk", "mlp", mode="pcc", thr=0.999)
    assert component_test_limits(s, "blk", "mlp") == ("pcc", 0.999)
    monkeypatch.setenv(IMPL_ENV, "reference")
    # a component threshold the reference cannot meet on the golden fails the swap check (vs golden)
    render_component_test(s, "blk", "attn_norm", thr=1.01)
    assert run_swap_test(s, "blk", ["attn_norm"])
    assert not run_swap_test(s, "blk", ["attn_norm"], checks="steps")
    assert got()["swap_attn_norm_vs_cpu"] > 0.999999
    _results(s, "C.blk.mlp", {"pcc_mlp_L00": 0.99995})
    assert abs(component_error(s, "blk", "mlp", 0) - 0.01) < 1e-6 and component_error(s, "blk", "attention", 0) is None


def test_mutations():
    x = torch.randn(64, 16)
    ids = torch.randint(0, 100, (64, 8))
    for kind in MU.FLOAT_KINDS:
        y = MU.mutate(x, kind)
        assert y.shape == x.shape and not torch.equal(y, x), kind
        assert MU.mutate(ids, kind) is ids, f"{kind} leaves integer outputs alone"
    assert torch.equal(x, x.clone()) and MU.mutate(x, "idxshift") is x
    s = MU.mutate(ids, "idxshift")
    assert s.max() == ids.max() and ((s == ids + 1) | (ids == ids.max())).all()
    assert (MU.mutate(x, "quarterzero")[:16] == 0).all() and torch.equal(MU.mutate(x, "quarterzero")[16:], x[16:])
    assert torch.equal(MU.mutate(x, "halfswap"), torch.cat([x[:, 8:], x[:, :8]], 1))
    with pytest.raises(ValueError):
        MU.kind_of("mutate:bogus")
    assert MU.kind_of("stub") is None and MU.kind_of("mutate:sign") == "sign"


def test_selection_rows_that_move_support_are_counted_not_scored():
    w = torch.zeros(100, 32)
    w[torch.arange(100)[:, None], torch.arange(4)[None] + torch.arange(100)[:, None] % 8] = 0.25
    g = w.clone()
    g[3] = torch.roll(w[3], 1)  # a near-tie flip in one row
    e = step_errors(g, w)
    assert e["reselected"] == pytest.approx(0.01) and e["rel"] == 0 and e["row"] == 0
    e = step_errors(MU.mutate(w, "rowshift"), w)
    assert e["reselected"] > 0.5
    assert "reselected" not in step_errors(torch.randn(10, 8), torch.randn(10, 8))


def test_isolated_context_owns_its_state(gspec):
    s = gspec()
    ref = s.hooks().reference(s, layers=[0])
    from models.demos.common.bringup.reference.interface import Ctx

    ctx = Ctx(0, 0, 4, ref.new_state(8), {"a": 1})
    c = _isolated(ctx)
    c.state.k[0][0] = 5.0
    c.extra["b"] = 2
    assert ctx.state.k[0][0].abs().sum() == 0 and "b" not in ctx.extra


def test_rendered_swap_template_gates_the_steps(gspec, monkeypatch):
    s = gspec()
    path = render_swap_test(s, "blk", ["attn_norm", "attention"])
    assert 'CHECKS = "steps"' in path.read_text()
    monkeypatch.setenv(IMPL_ENV, "mutate:scale1.02")
    monkeypatch.setenv(MU.MUTATE_STEP_ENV, "attention")
    with pytest.raises(AssertionError):
        _load(path).test_swap(None)


# ---- orchestrator: swap tests are frozen without the test-role review unless the spec opts back in
from models.demos.common.bringup.orchestrator import DONE  # noqa: E402
from models.demos.common.bringup.selftest.test_orchestrator import CHECK, impl_task, orch  # noqa: E402,F401


def _swap_task():
    return impl_task(id="S.blk.01", title="swap", brief={"block_type": "blk", "swapped": ["a"], "layer": 0})


IMPLEMENT = {"write": {"src/impl.txt": "0.999"}, "bash": ["scripts/run_safe_pytest.sh tests/check.py"]}


@pytest.mark.parametrize(
    "review,reviewed",
    [(None, False), ("none", False), (["other"], False), (["blk"], True), ("all", True)],
)
def test_swap_tasks_skip_the_review_unless_the_spec_names_them(orch, sandbox, review, reviewed):
    extra = {"agents": {"swap_review": review}} if review is not None else {}
    sandbox.write_spec(block_types={"blk": {"layers": [0], "representative": 0}}, **extra)
    o = orch([_swap_task(), impl_task()], {"S.blk.01.implement.1.md": IMPLEMENT, "C.1.implement.1.md": IMPLEMENT})
    assert o.run() == DONE
    calls = orch.calls()
    assert ("S.blk.01.test.1.md bringup-engineer" in calls) is reviewed
    assert "C.1.test.1.md bringup-engineer" in calls, "component tasks keep their test agent"
    assert any("swap test frozen without review (F49)" in line for line in orch.lines) is not reviewed
    assert o.led.task("S.blk.01")["frozen"]["reference"] == "PASS"
    assert sandbox.git("log", "--format=%s").count("[toy][S.blk.01][freeze] swap") == 1


def test_a_swap_test_that_cannot_freeze_unreviewed_goes_to_the_test_role(orch, sandbox):
    sandbox.write_spec(block_types={"blk": {"layers": [0], "representative": 0}})
    bad = CHECK.replace('"reference": 1.0', '"reference": 0.5')
    (sandbox.repo / "tests/check.py").write_text(bad)
    o = orch(
        [_swap_task()],
        {"S.blk.01.test.1.md": {"write": {"tests/check.py": CHECK}}, "S.blk.01.implement.1.md": IMPLEMENT},
    )
    assert o.run() == DONE
    assert orch.calls()[0] == "S.blk.01.test.1.md bringup-engineer"
    assert any("freeze without review failed, starting the test role" in line for line in orch.lines)
    assert "test fails with the CPU reference" in (o.run_dir / "briefs" / "S.blk.01.test.1.md").read_text()
