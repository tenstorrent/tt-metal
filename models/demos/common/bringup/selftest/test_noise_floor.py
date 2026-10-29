# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F80: the noise floor = max(the official arithmetic, the input nudge); selections need explained flips."""

import json

import pytest
import torch

from models.demos.common.bringup.reference.golden import reference_at
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


def test_an_fp4_input_is_nudged_by_the_size_of_its_quantizers_rounding():
    """F223: an fp4/<block> input (an e2m1 quantizer output, scale per block) moves inside its e2m1 cell: the nudge's
    error matches an fp4 quant-dequant's (scale amax / 6, round to the e2m1 grid), not bf16's ~100x smaller one."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 512, generator=g) * torch.pow(2.0, torch.randint(-4, 4, (256, 1), generator=g).float())
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    xg = x.view(256, -1, 16)
    s = xg.abs().amax(-1, keepdim=True) / 6.0
    m = (xg.abs() / s).unsqueeze(-1)
    q = (grid[(m - grid).abs().argmin(-1)] * xg.sign() * s).view_as(x)
    rel = lambda a: ((a.double() - x.double()).norm() / x.double().norm()).item()  # noqa: E731
    r_q, r_n, r_b = rel(q), rel(NF.nudge(x, 0, "fp4/16")), rel(NF.nudge(x, 0, "bf16"))
    assert 0.5 < r_n / r_q < 2.0, (r_n, r_q)
    assert r_b < r_q / 30, (r_b, r_q)
    y = NF.nudge(q, 1, "fp4/16")  # on the grid: within half a step of it
    step = torch.where(
        (q.view(256, -1, 16).abs() / s) < 2, 0.5, torch.where((q.view(256, -1, 16).abs() / s) < 4, 1.0, 2.0)
    )
    assert ((y - q).view(256, -1, 16).abs() <= 0.5 * step * s + 1e-6).all()
    with pytest.raises(ValueError):
        NF.nudge(torch.randn(4, 24), 0, "fp4/16")


def test_a_pow2_scale_fp4_input_is_nudged_by_the_size_of_its_quantizers_rounding():
    """F224: an fp4e8/<block> input (e2m1 with a power-of-two scale 2^ceil(log2(amax / 6)), the indexer's keys / query)
    moves inside its own e2m1 cell: the nudge's error matches that quantizer's (0.5-2x), its cell is the grid step on
    the power-of-two scale (not amax / 6), and a grid value stays within half a step of itself."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 512, generator=g) * torch.pow(2.0, torch.randint(-4, 4, (256, 1), generator=g).float())
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    xg = x.view(256, -1, 32)
    s = torch.pow(2.0, torch.ceil(torch.log2(xg.abs().amax(-1, keepdim=True) / 6.0)))
    m = (xg.abs() / s).unsqueeze(-1)
    q = (grid[(m - grid).abs().argmin(-1)] * xg.sign() * s).view_as(x)
    rel = lambda a: ((a.double() - x.double()).norm() / x.double().norm()).item()  # noqa: E731
    r_q, r_n = rel(q), rel(NF.nudge(x, 0, "fp4e8/32"))
    assert 0.5 < r_n / r_q < 2.0, (r_n, r_q)
    qg = q.view(256, -1, 32).abs() / s
    step = torch.where(qg < 2, 0.5, torch.where(qg < 4, 1.0, 2.0)) * s
    d = (NF.nudge(q, 1, "fp4e8/32") - q).view(256, -1, 32).abs()
    assert (d <= 0.5 * step + 1e-6).all()
    # the cell follows the power-of-two scale: on a block whose amax / 6 is just above a power of two the scale doubles
    # and so does the cell, which an amax / 6 scale (fp4/32) would not show
    y = torch.zeros(1, 32)
    y[0, 0] = 6.0 * 1.01
    y[0, 1:] = 0.9
    dy = lambda dt: (NF.nudge(y, 2, dt) - y)[0, 1:].abs().max().item()  # noqa: E731
    assert dy("fp4e8/32") > 1.5 * dy("fp4/32"), (dy("fp4e8/32"), dy("fp4/32"))
    with pytest.raises(ValueError):
        NF.nudge(torch.randn(4, 48), 0, "fp4e8/32")
    with pytest.raises(ValueError):
        NF.nudge(x, 0, "fp4e8/x")


def test_nudge_inner_moves_a_steps_own_quantizer_only_in_its_marked_nudged_runs(tmp_path):
    """F224: a step whose official code quantizes inside (e.g. the fp4 index query) marked ``nudge_inner: true``: in its
    nudged runs the reference's quantizer (calling nudge_inner) moves inside its cell; the exact run, an unmarked step
    and any call outside a nudged run see the value unchanged."""

    class S:
        bringup_dir = tmp_path

    x = torch.randn(8, 64, generator=torch.Generator().manual_seed(0))
    step = lambda a: NF.nudge_inner(a * 2.0, "fp4e8/32")  # noqa: E731
    assert torch.equal(step(x), x * 2.0)
    (tmp_path / NF.TAPS_FILE).write_text("steps:\n  indexer: {inputs: [fixed], nudge_inner: true}\n")
    d = NF.input_dtypes(S, "csa", "indexer", 1)
    assert d == ["fixed"] and d.inner and not NF.input_dtypes(S, "csa", "router", 1).inner
    outs = NF.nudged(step, [x], d, seeds=2)
    assert all(not torch.equal(o, x * 2.0) for o in outs) and not torch.equal(outs[0], outs[1])
    # within half a cell: the top e2m1 step 2 s, s = 2^ceil(log2(amax / 6)) <= amax / 3
    assert all(((o - x * 2.0).abs() <= (x * 2.0).abs().amax() / 3.0 + 1e-6).all() for o in outs)
    assert torch.equal(NF.nudged(step, [x], d, seeds=2)[1], outs[1])  # seeded
    plain = NF.Dtypes(["fixed"])
    assert all(torch.equal(o, x * 2.0) for o in NF.nudged(step, [x], plain, seeds=2))
    assert torch.equal(step(x), x * 2.0)  # the context is gone after the run


def test_input_dtypes_come_from_the_official_taps_map(tmp_path):
    class S:
        bringup_dir = tmp_path

    assert NF.input_dtypes(S, "swa", "router", 2) == ["bf16", "bf16"]
    (tmp_path / NF.TAPS_FILE).write_text(
        "steps:\n  attention: {inputs: [bf16, fixed]}\n  swa.router: {inputs: [fp32]}\n"
    )
    assert NF.input_dtypes(S, "swa", "attention", 2) == ["bf16", "fixed"]
    assert NF.input_dtypes(S, "swa", "router", 1) == ["fp32"] and NF.input_dtypes(S, "mla", "router", 1) == ["bf16"]
    (tmp_path / NF.TAPS_FILE).write_text("steps:\n  attention: {inputs: [bf16, fp4/16]}\n")
    assert NF.input_dtypes(S, "swa", "attention", 2) == ["bf16", "fp4/16"]
    (tmp_path / NF.TAPS_FILE).write_text("steps:\n  indexer: {inputs: [bf16, bf16, fp4e8/32]}\n")
    assert NF.input_dtypes(S, "swa", "indexer", 3) == ["bf16", "bf16", "fp4e8/32"]
    (tmp_path / NF.TAPS_FILE).write_text("steps:\n  indexer: {inputs: [fp4e8/0]}\n")
    with pytest.raises(ValueError):
        NF.input_dtypes(S, "swa", "indexer", 1)
    (tmp_path / NF.TAPS_FILE).write_text("steps:\n  attention: {inputs: [fp4/x]}\n")
    with pytest.raises(ValueError):
        NF.input_dtypes(S, "swa", "attention", 1)
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
    pick = lambda d: {x: d[x] for x in ("rows", "unexplained", "frac")}  # noqa: E731
    assert pick(NF.flips(scores, k, exact, nudged)) == {"rows": 0, "unexplained": 0, "frac": 0.0}
    near = exact.clone()
    order = scores[0].argsort(descending=True)
    near[0, order[k - 1]], near[0, order[k]] = False, True  # swap the k-th and k+1-th: a near tie
    far = exact.clone()
    far[1, scores[1].argmax()] = False
    far[1, scores[1].argmin()] = True  # the best expert out, the worst in
    f_near = NF.flips(scores, k, near, nudged)
    assert pick(f_near) == {"rows": 1, "unexplained": 0, "frac": 1 / 256}  # a 1e-9 gap, the nudge spreads ~1e-6
    assert f_near["recall_tie"] == 1.0 and f_near["recall"] == 1 - 1 / (4 * 256)
    wide = [scores + 0.5 * torch.randn(scores.shape, generator=g, dtype=torch.float64) for _ in range(3)]
    f_wide = NF.flips(scores, k, far, wide)  # a nudge that moves the scores by ~1 explains the far swap ...
    assert f_wide["unexplained"] == 0 and f_wide["recall_tie"] < 1  # ... the tie-aware recall (rtol only) does not
    f_far = NF.flips(scores, k, far, nudged)
    assert pick(f_far) == {"rows": 1, "unexplained": 1, "frac": 1 / 256}
    assert f_far["recall_tie"] == f_far["recall"] < 1 and f_far["worst_recall_tie"] == 0.75


def test_a_selection_without_topk_has_no_scores():
    x = torch.randn(8, 16)
    out, sc, k = NF.tapped(lambda a: torch.zeros_like(a).scatter_(1, a.argsort(-1)[:, -2:], 1.0), [x])
    assert sc is None and k is None
    out, sc, k = NF.tapped(lambda a: a.topk(3, dim=-1)[1], [x])
    assert k == 3 and torch.equal(sc, x.double())


def _causal_topk(x, keys, k, block=4):
    """An indexer-like step: per block of query rows, the k best visible keys (-inf beyond the row), pads -1; then a
    second top-k of another width (block scores) the output does not hold."""
    s, n = x.shape[0], keys.shape[0]
    out = torch.full((s, k), -1)
    for t0 in range(0, s, block):
        sc = x[t0 : t0 + block] @ keys.T
        sc = sc.masked_fill(torch.arange(n) > torch.arange(t0, t0 + sc.shape[0])[:, None], float("-inf"))
        sc.unflatten(-1, (-1, 2)).amax(-1).topk(2, dim=-1)  # candidate blocks: a top-k of another width
        v, i = sc.topk(k, dim=-1)
        out[t0 : t0 + block] = torch.where(torch.isfinite(v), i, -1)
    return out


def test_scores_match_only_when_they_hold_every_id():
    """F101: a top-k of another width (k = width, all ids < width) never stands in for the output's scores."""
    x = torch.randn(8, 64)
    out, sc, k = NF.tapped(lambda a: (a[:, :16].topk(16, dim=-1), a.topk(4, dim=-1)[1])[1], [x])
    assert k == 4 and sc.shape == (8, 64)
    out, sc, k = NF.tapped(lambda a: (a[:, :16].topk(16, dim=-1), a.topk(4, dim=-1)[1] + 64)[1], [x])
    assert sc is None  # ids beyond every tapped width


def test_an_index_list_is_checked_by_tie_aware_recall():
    """F101: an index output (top-k positions per row, in row blocks, after another top-k) gets its scores; a swap
    at a near tie is explained and keeps the tie-aware recall at 1, a masked or a far position is not."""
    g = torch.Generator().manual_seed(1)
    x, keys = torch.randn(16, 8, generator=g), torch.randn(16, 8, generator=g)
    want, sc, k = NF.tapped(lambda a: _causal_topk(a, keys, 4), [x])
    row5 = (x[5:6] @ keys.T).masked_fill(torch.arange(16) > 5, float("-inf"))[0].double()
    assert k == 4 and sc.shape == (16, 16) and torch.allclose(sc[5], row5)  # the position scores, row blocks stacked
    w = NF.support(want, k, 16)
    assert w.sum(1).tolist() == [min(r + 1, 4) for r in range(16)]  # pads dropped
    r = 10
    order = sc[r].argsort(descending=True)
    sc_tie = sc.clone()
    sc_tie[r, order[k]] = sc[r, order[k - 1]] * (1 - 5e-4)  # the k+1-th 0.05 % below the k-th: a near tie
    swapped = want.clone()
    swapped[r, (want[r] == order[k - 1]).nonzero()[0, 0]] = order[k]
    f = NF.flips(sc_tie, k, NF.support(swapped, k, 16), [], w)
    assert f["rows"] == 1 and f["unexplained"] == 0 and f["recall_tie"] == 1.0 and f["recall"] < 1.0
    masked = want.clone()
    masked[3, 0] = 12  # a position after its row: -inf in the reference, never a tie
    f = NF.flips(sc, k, NF.support(masked, k, 16), [], w)
    assert f["unexplained"] == 1 and f["recall_tie"] < 1.0
    wrong = want.clone()
    wrong[r, (want[r] == order[0]).nonzero()[0, 0]] = order[-1]  # the best position out, a far one in
    assert NF.flips(sc, k, NF.support(wrong, k, 16), [], w)["unexplained"] == 1


def test_a_near_tie_by_rtol_without_a_nudge_spread():
    """F101: without a nudge spread, a swap whose scores agree to tie_rtol is a tie (Momcilo's rule); 1 % apart is not."""
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float64)
    want = NF.support(scores, 2)
    got = torch.tensor([[True, False, True, False]])
    for gap, ok in ((NF.TIE_RTOL * 0.5, True), (0.01, False)):
        sc = scores.clone()
        sc[0, 2] = 3.0 * (1 - gap)
        f = NF.flips(sc, 2, got, [], want)
        assert (f["unexplained"] == 0) == ok and (f["recall_tie"] == 1.0) == ok


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


def test_an_fp32_steps_golden_limit_is_one_fp32_rounding_not_bf16(gspec, monkeypatch):  # noqa: F811
    """F133: an `out: fp32` step with nothing to nudge (fixed inputs) is limited vs the full-precision golden by one
    fp32 rounding of its output (~1e-7), not by a bf16 one (~4e-3)."""
    from models.demos.common.bringup.selftest import fixture_model
    from models.demos.common.bringup.testing.component import run_component_test

    s = gspec()
    monkeypatch.setitem(fixture_model.NOISE, "value", 0.0)
    s.bringup_dir.mkdir(parents=True, exist_ok=True)
    (s.bringup_dir / NF.TAPS_FILE).write_text("steps:\n  attn_residual: {inputs: [fixed, fixed], out: fp32}\n")
    assert run_component_test(s, "attn_residual", 0, checks="auto")
    lim = got()["err_attn_residual_L00_limit"]
    assert 0 < lim < 1e-5, lim
    x = torch.randn(256, 64)
    assert 1e-8 < float((NF.round_to(x, "fp32") - x.double()).norm() / x.double().norm()) < 1e-6


def _fake_official(s, wrong: str | None = None, skip: str | None = None, state: str | None = None):
    """official_taps for the fixture: each step on the golden's inputs rounded to bf16, its output in bf16 (the
    'official code' at bf16); ``wrong`` hands that step another step's output, ``skip`` leaves a step out."""
    from models.demos.common.bringup.testing.harness import reference_ctx

    def taps(spec, layers, tokens, chunk, chunk_size):
        from models.demos.common.bringup.reference.golden import Golden  # noqa: F401
        from models.demos.common.bringup.testing.harness import component_golden

        g, c = component_golden(spec)
        ref = reference_at(spec, "official", layers=layers)  # the fixture's official code
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
    assert OF.generate(s, write=False)["official_ref_errors"], "a mis-mapped output fails the check (F136 / F145)"
    monkeypatch.setattr(hooks, "official_taps", _fake_official(s, skip=steps[0]))
    assert OF.generate(s, write=False)["unmapped"]


def test_r5_checks_the_references_official_numerics_against_the_taps(gspec, monkeypatch):  # noqa: F811
    """F136 (V13): the reference's own official numerics (the end-to-end margin's source) must reproduce each tapped
    official step; one that does not (here: 1 % off) is an error of R.5."""
    from models.demos.common.bringup.selftest import fixture_model
    from models.demos.common.bringup.testing import official_floor as OF

    s = gspec()
    monkeypatch.setattr(s.hooks(), "official_taps", _fake_official(s), raising=False)
    doc = OF.generate(s, write=False)
    assert not doc["official_ref_errors"] and got()["official_ref_vs_taps_errors"] == 0
    native = fixture_model.reference

    def off_by_one_percent(spec, layers=None, dtype=torch.float32, numerics="native"):
        ref = native(spec, layers, dtype, numerics)
        if numerics == "official" and dtype != torch.float32:
            comp = ref.component
            ref.component = lambda L, n: (lambda ctx, *x: 1.01 * comp(L, n)(ctx, *x))
        return ref

    monkeypatch.setattr(fixture_model, "reference", off_by_one_percent)
    doc = OF.generate(s, write=False)
    assert doc["official_ref_errors"] and not doc["errors"] and got()["official_ref_vs_taps_errors"] > 0


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
    doc = OF.generate(s, write=False)
    bad = doc["errors"] + doc["official_ref_errors"]  # F145: the official numerics on the stale state miss the tap
    assert bad and all("attention" in e for e in bad), bad  # the stateful step read a prefix that is not its own


def test_an_explanation_never_hides_a_mis_mapped_tap(gspec, monkeypatch):  # noqa: F811
    """F145: the tap map is checked by the official numerics (F136), so an `explained` entry in official_taps.yaml
    still records why the official step is far from full precision but cannot hide a mis-mapped output."""
    from models.demos.common.bringup.testing import official_floor as OF

    s = gspec()
    hooks = s.hooks()
    monkeypatch.setattr(hooks, "official_taps", _fake_official(s, wrong="attn_norm"), raising=False)
    assert OF.generate(s, write=False)["official_ref_errors"]
    s.bringup_dir.mkdir(parents=True, exist_ok=True)
    (s.bringup_dir / NF.TAPS_FILE).write_text(
        "steps:\n  attn_norm: {explained: 'the official code rounds inside; emulated, it reproduces the output'}\n"
    )
    doc = OF.generate(s, write=False)
    assert doc["official_ref_errors"] and any(
        "explained" in v[str(doc["chunk"])]["check"] for v in doc["steps"].values()
    )


def _floor_case(seed=0, rows=2048, cols=256):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g, dtype=torch.float64)
    w[5] *= 1e-5  # a near-zero row
    w[:, 3] *= 1e-6  # a near-zero column
    outs = [w + 1e-3 * torch.randn(w.shape, generator=g, dtype=torch.float64) * w.abs() for _ in range(3)]
    dev = w + 1e-3 * torch.randn(w.shape, generator=g, dtype=torch.float64) * w.abs()
    return w, outs, dev


def test_each_row_and_column_is_judged_against_its_own_floor():
    """F83: a correct output (the floor's own noise) passes, near-zero rows and columns included; one row scaled by
    1 %, one tiny row with a large error, and an additive offset on one column each fail, at their own place."""
    for seed in range(5):
        w, outs, dev = _floor_case(seed)
        r = NF.own_floor(dev, w, outs, "fp32")
        assert max(r["row_floor"], r["col_floor"], r["bias_floor"]) < 2.0, (seed, r)
    w, outs, dev = _floor_case()
    bad = dev.clone()
    bad[7] *= 1.01
    r = NF.own_floor(bad, w, outs, "fp32")
    assert r["row_floor"] > 2 and r["row_at"] == 7
    tiny = dev.clone()
    tiny[5] += 1e-3 * w.abs().mean()  # an error the size of a normal row's noise, on a row 1e-5 its size
    assert NF.own_floor(tiny, w, outs, "fp32")["row_at"] == 5
    off = dev.clone()
    off[:, 9] += 3e-4 * w[:, 9].pow(2).mean().sqrt()
    r = NF.own_floor(off, w, outs, "fp32")
    assert r["bias_floor"] > 2 and r["bias_at"] == 9 and r["bias_signed"] > 0
    exact = NF.own_floor(w.clone(), w, outs, "fp32")
    assert exact["row_floor"] == exact["col_floor"] == exact["bias_floor"] == 0  # an exact output counts 0


def test_an_owner_waiver_passes_a_steps_noise_floor_checks_and_records_it(gspec, monkeypatch):  # noqa: F811
    """F85: a waived step's noise-floor failures are printed WAIVED and recorded; the test and the gate pass."""
    from models.demos.common.bringup.core.gate import check_metrics
    from models.demos.common.bringup.core.runs import IMPL_ENV
    from models.demos.common.bringup.testing import mutate as MU
    from models.demos.common.bringup.testing.component import run_component_test

    s = gspec()
    monkeypatch.setenv(IMPL_ENV, "mutate:scale1.02")
    monkeypatch.setenv(MU.MUTATE_STEP_ENV, "attn_norm")
    assert not run_component_test(s, "attn_norm", 0, checks="auto")
    s.data.setdefault("tests", {})["waivers"] = {"attn_norm": "device arithmetic; fix later (owner)"}
    assert run_component_test(s, "attn_norm", 0, checks="auto")
    m = {k: {"value": v} for k, v in got().items()}
    assert m["err_attn_norm_L00_waived"]["value"] == 1
    ok, lines = check_metrics({"err_attn_norm_L00": "<= limit"}, m)
    assert ok and any("WAIVED" in x for x in lines)


def test_a_waiver_never_passes_a_wrong_module(gspec, monkeypatch):  # noqa: F811
    """F97: a waiver covers a precision gap only: a zero stub (rel 1) of a waived step still fails, so the freeze's
    stub check still works (DeepSeek-V4.1 ffn_hc: the waived test passed with a zero stub)."""
    from models.demos.common.bringup.core.runs import IMPL_ENV
    from models.demos.common.bringup.testing.component import run_component_test

    s = gspec()
    s.data.setdefault("tests", {})["waivers"] = {"attn_norm": "device arithmetic; fix later (owner)"}
    monkeypatch.setenv(IMPL_ENV, "stub")
    assert not run_component_test(s, "attn_norm", 0, checks="auto")
    monkeypatch.setenv(IMPL_ENV, "reference")
    assert run_component_test(s, "attn_norm", 0, checks="auto")


def test_an_official_step_that_rounds_inside_is_its_margin_not_an_error(gspec, monkeypatch):  # noqa: F811
    """F145: DeepSeek-V4.1 R.5 failed 40 steps (q_a, kv, attention, experts, shared expert: 1-5 % from the
    full-precision step) because the official code rounds to fp8 / fp4 inside them; that distance is the margin."""
    from models.demos.common.bringup.testing import official_floor as OF

    s = gspec()
    hooks = s.hooks()
    base = _fake_official(s)

    def rounding(*a, **k):
        out = base(*a, **k)
        for L, steps in out.items():
            for name, t in steps.items():
                if name != "_state" and t["output"].is_floating_point():
                    steps[name] = dict(t, output=t["output"].bfloat16().float() * 1.02)
        return out

    monkeypatch.setattr(hooks, "official_taps", rounding, raising=False)
    doc = OF.generate(s, write=False)
    assert not doc["errors"]
    assert any(v[str(doc["chunk"])]["check"].get("in_step_rounding") for v in doc["steps"].values())
