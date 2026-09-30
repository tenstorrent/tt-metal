# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F56: built-in component checks (run_component_test(..., checks="auto")), the freeze sweep, and component tests
frozen without a review. CPU only, against the synthetic fixture model (its fake device is the CPU reference plus
deterministic noise)."""

import pytest
import torch

from models.demos.common.bringup.core.runs import IMPL_ENV, FreezeError, freeze_task
from models.demos.common.bringup.selftest.conftest import got
from models.demos.common.bringup.selftest.test_device_side import _load, gspec, noise  # noqa: F401 (fixtures)
from models.demos.common.bringup.testing import component_checks as CC
from models.demos.common.bringup.testing import mutate as MU
from models.demos.common.bringup.testing.component import run_component_test
from models.demos.common.bringup.testing.templates import render_component_test

STEPS = ["attn_norm", "attention", "attn_residual", "ffn_norm", "mlp", "mlp_residual"]


def test_checks_none_is_unchanged(gspec, noise):
    s = gspec()
    noise(1e-3)
    assert run_component_test(s, "mlp", 0)
    assert not [k for k in got() if k.startswith(("auto_", "sweep_"))], "checks=None records no built-in checks"
    noise(0.5)
    assert not run_component_test(s, "mlp", 0)


@pytest.mark.parametrize("step", STEPS)
def test_auto_passes_with_the_reference_and_fails_with_the_stub(gspec, monkeypatch, step):
    s = gspec()
    monkeypatch.setenv(IMPL_ENV, "reference")
    assert run_component_test(s, step, 0, checks="auto")
    assert got()[f"auto_{step}_L00_vs_cpu_rel"] == 0
    monkeypatch.setenv(IMPL_ENV, "stub")
    assert not run_component_test(s, step, 0, checks="auto")


@pytest.mark.parametrize("kind", MU.FLOAT_KINDS)
@pytest.mark.parametrize("step", ["attention", "mlp", "mlp_residual"])
def test_auto_catches_every_standard_mistake(gspec, monkeypatch, kind, step):
    if (step, kind) == ("attention", "noise1e-2"):
        pytest.skip("below the fixture attention's bf16 precision: its sweep sends it to a review (test below)")
    s = gspec()
    monkeypatch.setenv(IMPL_ENV, f"mutate:{kind}")
    assert not run_component_test(s, step, 0, checks="auto")


def test_auto_passes_a_near_exact_device_and_fails_a_noisy_one(gspec, noise):
    s = gspec()
    noise(1e-4)
    assert run_component_test(s, "attention", 0, checks="auto")
    assert got()["auto_attention_L00_vs_cpu_rel"] > 0
    noise(3e-2)
    assert not run_component_test(s, "attention", 0, checks="auto")


def test_second_inputs(gspec):
    s = gspec()
    from models.demos.common.bringup.testing.harness import component_golden

    g, c = component_golden(s)
    ref = s.hooks().reference(s, layers=[0], dtype=torch.float32)
    names = {st.name: st for st in ref.block_graph(0)}
    att = [x.name for x in CC.cases(s, ref, 0, names["attention"], g, c, "float")]
    assert att[0] == "golden" and "mixed" in att and "big" in att
    assert ("chunk0" in att) is (c != 0), "a stateful step also runs on chunk 0"
    assert any(n.startswith("layer") for n in att), "the fixture golden has another layer"
    assert "chunk0" not in [x.name for x in CC.cases(s, ref, 0, names["mlp"], g, c, "float")]


def test_a_mistake_below_the_steps_precision_fails_the_sweep(gspec, monkeypatch, capsys):
    # the fixture attention (tiny head dim) has a bf16 error of ~0.6 %: 1 % noise is within its limit, so the sweep
    # fails and the orchestrator sends the test to a review
    s = gspec()
    monkeypatch.setenv(IMPL_ENV, "mutations")
    monkeypatch.setattr(CC, "NARROW", 0)  # the fixture's hidden size is narrow; real hidden sizes are not
    assert not run_component_test(s, "attention", 0, checks="auto")
    out = capsys.readouterr().out
    assert "noise1e-2                    SLIPPED" in out and "bf16 everywhere              pass" in out


@pytest.mark.parametrize("step", [s for s in STEPS if s != "attention"])
def test_the_sweep_passes_for_the_fixture_steps(gspec, monkeypatch, capsys, step):
    s = gspec()
    monkeypatch.setenv(IMPL_ENV, "mutations")
    assert run_component_test(s, step, 0, checks="auto")
    out = capsys.readouterr().out
    assert "SLIPPED" not in out and f"ok   sweep {step}" in out
    assert got()[f"sweep_{step}_L00_slipped"] == 0 and got()[f"sweep_{step}_L00_caught"] == len(MU.FLOAT_KINDS)


def test_the_sweep_fails_when_a_mistake_slips_through(gspec, monkeypatch, capsys):
    s = gspec()
    monkeypatch.setenv(IMPL_ENV, "mutations")
    loose = dict(CC.COMPONENT_DEFAULTS, component_bias=1.0, component_ratio=1.0, component_floor=0.5, component_rel=0.5)
    monkeypatch.setattr(CC, "COMPONENT_DEFAULTS", loose)
    monkeypatch.setattr(CC, "NARROW", 0)
    assert not run_component_test(s, "mlp", 0, checks="auto")
    assert "scale1.02                    SLIPPED" in capsys.readouterr().out


def test_output_kinds():
    assert CC.output_kind(torch.randn(8, 16)) == "float"
    w = torch.zeros(8, 32)
    w[:, :2] = 0.5
    assert CC.output_kind(w) == "selection"
    idx = torch.stack([torch.randperm(64)[:16] for _ in range(8)])
    assert CC.output_kind(idx) == "index"
    assert CC.output_kind(torch.randint(0, 4, (8, 16))) == "int"  # repeats within rows: not a set
    padded = idx.clone()
    padded[:, 10:] = -1
    assert CC.output_kind(padded) == "index"
    assert (CC.normalize_index(torch.tensor([[3, 0xFFFFFFFF, -1]])) == torch.tensor([[3, -1, -1]])).all()


def _causal_topk(rows=64, start=64, k=16):
    g = torch.Generator().manual_seed(0)
    out = []
    for r in range(rows):
        p = start + r
        others = torch.randperm(p, generator=g)[: k - 1]
        out.append(torch.cat([torch.tensor([p]), others]).sort().values)
    return torch.stack(out)


def test_index_checks():
    want, lim = _causal_topk(), CC.limits(_Spec())
    shuffled = torch.gather(want, 1, torch.argsort(torch.rand(want.shape), 1))
    assert not CC.index_fails(shuffled, want, 64, lim, 0.99)[1], "order does not matter"
    sentinel = want.clone().to(torch.int64)
    assert not CC.index_fails(sentinel, want, 64, lim, 0.99)[1]
    assert CC.index_fails(MU.mutate(want, "idxshift"), want, 64, lim, 0.99)[1]
    future = want.clone()
    future[:, -1] = 64 + torch.arange(64) + 1  # the own position (the largest) moved one past the row
    bad = " ".join(CC.index_fails(future, want, 64, lim, 0.99)[1])
    assert "non-causal" in bad and "own position" in bad
    dup = want.clone()
    dup[:, 1] = dup[:, 2]
    assert "repeated" in " ".join(CC.index_fails(dup, want, 64, lim, 0.99)[1])
    padded = want.clone()
    padded[:, -4:] = -1  # a reference with its pads at the end of every row
    valid_shuffled = padded.clone()
    valid_shuffled[:, :-4] = valid_shuffled[:, :-4].flip(1)
    assert not CC.index_fails(valid_shuffled, padded, 64, lim, 0.99)[1], "order within the valid part is free"
    mid = padded.clone()
    mid[:, [0, -1]] = mid[:, [-1, 0]]  # the same set, one pad moved to the front
    assert CC.index_fails(mid, padded, 64, lim, 0.99)[1] == ["64 rows have a pad before a valid position"]
    fewer = want.clone()
    fewer[:, -1] = -1
    assert "number of valid" in " ".join(CC.index_fails(fewer, want, 64, lim, 0.99)[1])


def test_selection_checks():
    lim = CC.limits(_Spec())
    w = torch.zeros(400, 32)
    for r in range(400):
        w[r, torch.randperm(32, generator=torch.Generator().manual_seed(r))[:4]] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    assert not CC.selection_fails(w, w, lim)[1]
    flip = w.clone()
    flip[3] = torch.roll(w[3], 1)  # one near-tie row: counted in the overlap, not failed by itself
    assert not CC.selection_fails(flip, w, lim)[1]
    for kind in MU.FLOAT_KINDS:
        assert CC.selection_fails(MU.mutate(w, kind), w, lim)[1], kind
    assert CC.selection_fails(-w, w, lim)[1]


class _Spec:
    def threshold(self, k, v):
        return v

    def get(self, k, d=None):
        return d


def test_rendered_component_template_uses_the_checks(gspec, monkeypatch):
    s = gspec()
    path = render_component_test(s, "blk", "mlp", overwrite=True)
    assert 'CHECKS = "auto"' in path.read_text()
    mod = _load(path)
    for impl, passes in (("reference", True), ("stub", False), ("mutate:scale1.02", False), ("mutations", True)):
        monkeypatch.setenv(IMPL_ENV, impl)
        if passes:
            mod.test_component(None)
        else:
            with pytest.raises(AssertionError):
                mod.test_component(None)


# ---- freeze and orchestrator
from models.demos.common.bringup.orchestrator import DONE  # noqa: E402
from models.demos.common.bringup.selftest.test_orchestrator import CHECK, impl_task, orch  # noqa: E402,F401

# the toy check: the sweep passes unless src/sweep.txt says it slips
SWEEP_CHECK = CHECK.replace(
    'impl = os.environ.get("BRINGUP_IMPL", "device")',
    'impl = os.environ.get("BRINGUP_IMPL", "device")\n'
    'if impl == "mutations":\n'
    '    raise SystemExit(1 if Path("src/sweep.txt").exists() else 0)',
)
IMPLEMENT = {"write": {"src/impl.txt": "0.999"}, "bash": ["scripts/run_safe_pytest.sh tests/check.py"]}


def _comp_task():
    return impl_task(id="C.blk.a", title="component", brief={"block_type": "blk", "step": "a", "layer": 0})


@pytest.mark.parametrize(
    "review,reviewed", [(None, True), ("all", True), (["blk"], True), (["other"], False), ("none", False)]
)
def test_component_tasks_skip_the_review_only_when_the_spec_says_so(orch, sandbox, review, reviewed):
    extra = {"agents": {"component_review": review}} if review is not None else {}
    sandbox.write_spec(block_types={"blk": {"layers": [0], "representative": 0}}, **extra)
    (sandbox.repo / "tests/check.py").write_text(SWEEP_CHECK)
    o = orch([_comp_task()], {"C.blk.a.implement.1.md": IMPLEMENT})
    assert o.run() == DONE
    assert ("C.blk.a.test.1.md bringup-engineer" in orch.calls()) is reviewed
    assert any("component test frozen without review (F56)" in line for line in orch.lines) is not reviewed
    frozen = o.led.task("C.blk.a")["frozen"]
    assert frozen["reference"] == "PASS" and (frozen.get("mutations") == "PASS") is not reviewed


def test_a_component_test_whose_sweep_fails_goes_to_the_test_role(orch, sandbox):
    sandbox.write_spec(block_types={"blk": {"layers": [0], "representative": 0}}, agents={"component_review": "none"})
    (sandbox.repo / "tests/check.py").write_text(SWEEP_CHECK)
    (sandbox.repo / "src/sweep.txt").write_text("slips")
    o = orch(
        [_comp_task()],
        {"C.blk.a.test.1.md": {"bash": ["rm src/sweep.txt"]}, "C.blk.a.implement.1.md": IMPLEMENT},
    )
    assert o.run() == DONE
    assert orch.calls()[0] == "C.blk.a.test.1.md bringup-engineer"
    assert any("freeze without review failed, starting the test role" in line for line in orch.lines)
    assert "mistake sweep failed" in (o.run_dir / "briefs" / "C.blk.a.test.1.md").read_text()


def test_freeze_with_mutations_requires_the_sweep(sandbox):
    (sandbox.repo / "tests").mkdir(exist_ok=True)
    (sandbox.repo / "src").mkdir(exist_ok=True)
    (sandbox.repo / "tests/check.py").write_text(SWEEP_CHECK)
    led = sandbox.tasks(impl_task())
    assert freeze_task(sandbox.spec, led, "C.1", commit=False, mutations=True)["mutations"] == "PASS"
    (sandbox.repo / "src/sweep.txt").write_text("slips")
    with pytest.raises(FreezeError, match="mistake sweep failed"):
        freeze_task(sandbox.spec, led, "C.1", commit=False, mutations=True)
    assert "mutations" not in freeze_task(sandbox.spec, led, "C.1", commit=False)


def test_a_narrow_output_is_checked_per_column():
    # iHC gates: one column near 0 among columns near 1; its sign flipped moves the whole-matrix error by ~1e-5
    want = torch.ones(256, 8)
    want[:, 0] = 1e-5 * torch.rand(256)
    lim = CC.limits(_Spec())
    L = CC.float_limits(lim, 1.0, 0.004, None)
    flipped = want.clone()
    flipped[:, 0] *= -1
    e, bad = CC.float_fails(flipped, want, L)
    assert e["rel"] < 1e-4 and bad == [f"worst column {e['col']:.5f} > 0.0150"]
    assert not CC.float_fails(want * (1 + 1e-4), want, L)[1]
    assert "col" not in CC.float_errors(torch.ones(4, 128), torch.ones(4, 128)), "wide outputs: no column check"
