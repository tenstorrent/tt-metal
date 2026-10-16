# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F80: the noise floor = max(the official arithmetic, the input nudge); selections need explained flips."""

import json

import pytest
import torch

from models.demos.common.bringup.selftest.conftest import got
from models.demos.common.bringup.selftest.test_device_side import gspec  # noqa: F401 (fixture)
from models.demos.common.bringup.testing import noise_floor as NF


def test_the_nudge_stays_inside_the_rounding_cell():
    x = torch.randn(64, 64).to(torch.bfloat16).float()
    for d, cell in (("bf16", 2.0**-7), ("fp32", 2.0**-23)):
        y = NF.nudge(x, 0, d)
        assert (
            ((y.double() - x.double()).abs() <= cell * x.double().abs() + 1e-30).all()
            and not torch.equal(x, y)
            or d == "fp32"
        )
    assert torch.equal(NF.nudge(x, 0, "fixed"), x)
    idx = torch.arange(8)
    assert NF.nudge(idx, 0) is idx
    assert torch.equal(NF.nudge(x, 3), NF.nudge(x, 3)) and not torch.equal(NF.nudge(x, 3), NF.nudge(x, 4))


def test_input_dtypes_come_from_the_official_taps_map(tmp_path):
    class S:
        bringup_dir = tmp_path

    assert NF.input_dtypes(S, "swa", "router", 2) == ["bf16", "bf16"]
    (tmp_path / NF.TAPS_FILE).write_text(
        "steps:\n  attention: {inputs: [bf16, fixed]}\n  swa.router: {inputs: [fp32]}\n"
    )
    assert NF.input_dtypes(S, "swa", "attention", 2) == ["bf16", "fixed"]
    assert NF.input_dtypes(S, "swa", "router", 1) == ["fp32"] and NF.input_dtypes(S, "mla", "router", 1) == ["bf16"]
    (tmp_path / NF.TAPS_FILE).write_text("steps:\n  router: {inputs: [fp8]}\n")
    with pytest.raises(ValueError):
        NF.input_dtypes(S, "swa", "router", 1)


def test_a_flip_is_explained_only_near_the_cutoff():
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(256, 16, generator=g, dtype=torch.float64)
    scores[0, :] = torch.linspace(1, 0, 16, dtype=torch.float64)
    scores[0, 4] = scores[0, 3] - 1e-9  # a near tie of the 4th and 5th: the cutoff for k = 4 sits between them
    k = 4
    nudged = [scores + 1e-6 * torch.randn(scores.shape, generator=g, dtype=torch.float64) for _ in range(3)]
    exact = NF.support(scores, k)
    assert NF.flips(scores, k, exact, nudged) == {"rows": 0, "unexplained": 0, "frac": 0.0}
    near = exact.clone()
    order = scores[0].argsort(descending=True)
    near[0, order[k - 1]], near[0, order[k]] = False, True  # swap the k-th and k+1-th: a near tie
    far = exact.clone()
    far[1, scores[1].argmax()] = False
    far[1, scores[1].argmin()] = True  # the best expert out, the worst in
    f_near = NF.flips(scores, k, near, nudged)
    assert f_near == {"rows": 1, "unexplained": 0, "frac": 1 / 256}  # a 1e-9 gap, the nudge spreads ~1e-6
    f_far = NF.flips(scores, k, far, nudged)
    assert f_far == {"rows": 1, "unexplained": 1, "frac": 1 / 256}


def test_a_selection_without_topk_has_no_scores():
    x = torch.randn(8, 16)
    out, sc, k = NF.tapped(lambda a: torch.zeros_like(a).scatter_(1, a.argsort(-1)[:, -2:], 1.0), [x])
    assert sc is None and k is None
    out, sc, k = NF.tapped(lambda a: a.topk(3, dim=-1)[1], [x])
    assert k == 3 and torch.equal(sc, x.double())


def test_the_official_arithmetic_raises_a_floor(gspec, monkeypatch):  # noqa: F811
    """A cached official_floor.json (the official code's error vs the fp32 step) above the nudge sets the limit."""
    from models.demos.common.bringup.core.runs import IMPL_ENV
    from models.demos.common.bringup.testing.component import run_component_test
    from models.demos.common.bringup.testing.harness import component_golden

    s = gspec()
    monkeypatch.setenv(IMPL_ENV, "reference")
    assert run_component_test(s, "mlp", 0, checks="auto")
    base = got()["auto_mlp_L00_vs_cpu_rel_limit"]
    g, c = component_golden(s)
    big = {"rel": 0.05, "row": 0.1, "ratio_min": 0.95, "ratio_max": 1.05, "bias": 0.0}
    (g.dir / NF.FLOOR_FILE).write_text(json.dumps({"steps": {"L00.mlp": {str(c): {"vs_step": big, "vs_golden": big}}}}))
    CC_cache = __import__("models.demos.common.bringup.testing.component", fromlist=["_EXPECT"])._EXPECT
    CC_cache.clear()
    assert NF.arith(s, 0, "mlp", c)["vs_step"]["rel"] == 0.05
    assert run_component_test(s, "mlp", 0, checks="auto")
    lim = got()["auto_mlp_L00_vs_cpu_rel_limit"]
    assert lim == pytest.approx(2.0 * 0.05) and lim > base


def test_an_fp32_output_has_no_bf16_resolution_floor(tmp_path):
    class S:
        bringup_dir = tmp_path

    x = torch.randn(64, 64)
    assert NF.output_dtype(S, "swa", "attn_hc") == "bf16" and NF.bf16_res("pcc", x) < 1.0
    (tmp_path / NF.TAPS_FILE).write_text("steps:\n  attn_hc: {inputs: [bf16], out: fp32}\n")
    assert NF.output_dtype(S, "swa", "attn_hc") == "fp32" and NF.bf16_res("pcc", x, "fp32") == pytest.approx(
        1.0, abs=1e-12
    )


def _fake_official(s, wrong: str | None = None, skip: str | None = None, state: str | None = None):
    """official_taps for the fixture: each step on the golden's inputs rounded to bf16, its output in bf16 (the
    'official code' at bf16); ``wrong`` hands that step another step's output, ``skip`` leaves a step out."""
    from models.demos.common.bringup.testing.harness import reference_ctx

    def taps(spec, layers, tokens, chunk, chunk_size):
        from models.demos.common.bringup.reference.golden import Golden  # noqa: F401
        from models.demos.common.bringup.testing.harness import component_golden

        g, c = component_golden(spec)
        ref = spec.hooks().reference(spec, layers=layers, dtype=torch.float32)
        out = {}
        for L in layers:
            gl, out[L] = g.layer(c, L), {}
            for st in ref.block_graph(L):
                if st.name == skip:
                    continue
                xs = [
                    gl[i].float().to(torch.bfloat16).float() if gl[i].is_floating_point() else gl[i] for i in st.inputs
                ]
                y = ref.component(L, st.name)(reference_ctx(ref, L, g, c, spec), *xs)
                out[L][st.name] = {"inputs": xs, "output": y.to(torch.bfloat16) if y.is_floating_point() else y}
            if state is not None:  # the official run's state: the golden's ("same") or another chunk's ("stale")
                st0 = g.state(L, at=c * g.chunk)
                if state != "same":  # another run's state: the prefix the step reads is not the official one
                    st0 = {
                        k: (v + v.float().std() * torch.randn_like(v.float())).to(v.dtype)
                        if v.is_floating_point()
                        else v
                        for k, v in st0.items()
                    }
                out[L]["_state"] = {"start": st0, "end": g.state(L, at=(c + 1) * g.chunk)}
            if wrong:
                other = next(
                    n for n in out[L] if n != wrong and out[L][n]["output"].shape == out[L][wrong]["output"].shape
                )
                out[L][wrong] = dict(out[L][wrong], output=out[L][other]["output"])
        return out

    return taps


def test_the_official_floor_is_generated_and_checks_the_tap_map(gspec, monkeypatch):  # noqa: F811
    from models.demos.common.bringup.testing import official_floor as OF
    from models.demos.common.bringup.testing.harness import component_golden

    s = gspec()
    hooks = s.hooks()
    monkeypatch.setattr(hooks, "official_taps", _fake_official(s), raising=False)
    doc = OF.generate(s)
    assert not doc["errors"] and not doc["unmapped"] and doc["steps"]
    g, c = component_golden(s)
    assert (g.dir / NF.FLOOR_FILE).exists()
    some = next(iter(doc["steps"]))
    L, step = int(some[1:3]), some.split(".", 1)[1]
    assert NF.arith(s, L, step, c)["vs_step"]["rel"] >= 0
    steps = [k.split(".", 1)[1] for k in doc["steps"]]
    monkeypatch.setattr(
        hooks, "official_taps", _fake_official(s, wrong="attn_norm" if "attn_norm" in steps else steps[0])
    )
    assert OF.generate(s, write=False)["errors"], "a mis-mapped output fails the check"
    monkeypatch.setattr(hooks, "official_taps", _fake_official(s, skip=steps[0]))
    assert OF.generate(s, write=False)["unmapped"]


def test_the_official_floor_command_reads_the_spec_from_the_gate_env(gspec, monkeypatch):  # noqa: F811
    """F80: the R.5 gate command has no --spec; the gate runner sets BRINGUP_SPEC."""
    from models.demos.common.bringup.testing import official_floor as OF

    s = gspec()
    monkeypatch.setattr(s.hooks(), "official_taps", _fake_official(s), raising=False)
    monkeypatch.setenv("BRINGUP_SPEC", str(s.path))
    assert OF.main([]) == 0


def test_a_stateful_step_reads_the_official_runs_state(gspec, monkeypatch):  # noqa: F811
    """The official taps' own state (here: the golden's, then a wrong one) is what the fp32 step reads."""
    from models.demos.common.bringup.testing import official_floor as OF

    s = gspec()
    hooks = s.hooks()
    monkeypatch.setattr(hooks, "official_taps", _fake_official(s, state="same"), raising=False)
    doc = OF.generate(s, write=False)
    assert not doc["errors"] and all(v[str(doc["chunk"])]["state"] == "official" for v in doc["steps"].values())
    monkeypatch.setattr(hooks, "official_taps", _fake_official(s, state="stale"), raising=False)
    bad = OF.generate(s, write=False)["errors"]
    assert bad and all("attention" in e for e in bad), bad  # the stateful step read a prefix that is not its own
