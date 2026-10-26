# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests: the host pipeline (``common/host.py``, fp64) against the reference's goldens.

- pre-processing: the decode inputs ``predict_batch`` built, and the token features and running RevIN
  statistics (masked cumsums vs the Chan loop) against the fp64 reference (``host_fp64`` golden); the gap to
  the fp32 reference is printed: its own rounding of (x - mu) / sigma reaches ~4e-5 when level / std is large;
- post-processing: the reference's own head output through ``host.postprocess`` reproduces the
  forecaster's final quantiles (CPM refine, stitching, trend, sort, symmetric averaging, make_positive).

Tolerance 1e-5 (the contract's rtol): max|a - b| / max|b|; the running mean relative to each row's largest std.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.timesfm3_forecast.common import host
from models.experimental.timesfm3_forecast.tests.golden_io import load_golden

RTOL = 1e-5
FULL = ["single_2048_h64", "full16_2048_h64", "full16_2048_h128", "mv4x8_2048_h64"]
ALL = FULL + ["gift64_h48", "bench_paper", "bench_long_context", "bench_multivariate"]


def _rel(a, b) -> float:
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return float((a - b).abs().max() / b.abs().max())


def _sigma_rel(a, b, sigma) -> float:
    """max over rows of max|a - b| / max(sigma): the error in units of the row's data scale (rows, n)."""
    a, b, sigma = torch.as_tensor(a).double(), torch.as_tensor(b).double(), torch.as_tensor(sigma).double()
    scale = sigma.max(-1).values
    keep = scale > 0
    return float(((a - b).abs().max(-1).values[keep] / scale[keep]).max())


def _prepare(g: dict) -> host.HostInputs:
    if "context" in g:  # the single-series golden
        return host.prepare([g["context"].numpy()], g["horizon"])
    contexts = g["contexts"]
    contexts = [c.numpy() for c in contexts] if isinstance(contexts, list) else list(contexts.numpy())
    return host.prepare(contexts, g["horizon"], **g.get("settings", {}))


@pytest.mark.parametrize("name", FULL)
def test_preprocessing(name):
    g = load_golden(name)
    hi = _prepare(g)
    tok = g["resblock_input"]
    b, v, n, f = tok.shape
    if "target" in g:
        assert torch.equal(torch.from_numpy(hi.batch.target), g["target"]), "decode target differs"
        assert torch.equal(torch.from_numpy(hi.batch.mask), g["mask"]), "decode mask differs"
    g64 = load_golden("host_fp64")[name]
    sigma64 = g64["revin_sigma"].reshape(hi.running_sigma.shape)
    for label, key, got in (
        ("token features", "resblock_input", hi.tokens),
        ("running mean (row scale)", "revin_mu", hi.running_mu),
        ("running std", "revin_sigma", hi.running_sigma),
    ):
        ref64, ref32 = g64[key].reshape(got.shape), g[key].reshape(got.shape)
        # The running mean relative to the row's data scale (its largest running std): a detrended mean
        # sits near 0, so max|mu| is an ill-conditioned scale.
        metric = (lambda a, b: _sigma_rel(a, b, sigma64)) if key == "revin_mu" else _rel
        r = metric(got, ref64)
        print(
            f"[host] {name} {label}: {r:.2e} vs fp64 reference ({metric(got, ref32):.2e} vs fp32, "
            f"whose gap to fp64 is {metric(ref32, ref64):.2e})"
        )
        assert r <= RTOL, f"{name} {label}: {r:.2e} > {RTOL}"
    eff = torch.cumprod(tok[..., 96:].eq(1).all(-1).int(), dim=2).reshape(b * v, n).sum(-1)
    assert torch.equal(hi.num_front_masked, eff), "leading-pad patch counts differ"


@pytest.mark.parametrize("name", ALL)
def test_postprocessing(name):
    g = load_golden(name)
    hi = _prepare(g)
    head = (
        g["head_readout"]
        if "head_readout" in g
        else g["head_raw"].reshape(hi.rows, hi.num_patches, -1)[:, hi.readout_tokens.start : hi.readout_tokens.stop]
    )
    assert head.shape[1] == hi.num_forecast_patches
    q = host.postprocess(head, hi)
    ref = g["quantiles"]
    ref = ref[None] if ref.dim() == 2 else ref  # the single-series golden stores (H, 9)
    r = _rel(torch.from_numpy(q), ref)
    print(f"[host] {name} postprocess: rel {r:.2e}, shape {tuple(q.shape)}")
    assert tuple(q.shape) == tuple(ref.shape)
    assert r <= RTOL, f"{name} postprocess: rel {r:.2e} > {RTOL}"


@pytest.mark.parametrize("name", ALL)
def test_broken_bands_fail_the_gates(name):
    """PLAN section 9: a forecast with all 9 quantiles collapsed to their mean must fail every lower-bound gate
    (accuracy), and one with the band doubled around the quantile mean must fail every upper-bound gate
    (``spread_ratio_p99`` / ``max``). ``pcc_final`` is the exception and is only printed: data-space levels keep it
    near 1 for a collapsed forecast, which is why ``pcc_final_spread`` exists. On goldens with few rows the per-row
    metrics degenerate (one row: row PCC = pcc_normalized), so there only the global metrics must fail."""
    from models.experimental.timesfm3_forecast.tests.golden_io import (
        GATES,
        collapse_quantiles,
        final_metrics,
        gate_failures,
        head_metrics,
        inflate_quantiles,
    )

    g = load_golden(name)
    hi = _prepare(g)
    head = (
        g["head_readout"]
        if "head_readout" in g
        else g["head_raw"].reshape(hi.rows, hi.num_patches, -1)[:, hi.readout_tokens.start : hi.readout_tokens.stop]
    )
    ref = g["quantiles"]
    ref = ref[None] if ref.dim() == 2 else ref
    per_row = name.startswith("bench_")
    for label, broken, op in (
        ("COLLAPSED", collapse_quantiles(head), ">="),
        ("INFLATED x2", inflate_quantiles(head), "<="),
    ):
        m = {**head_metrics(head, broken), **final_metrics(ref, torch.from_numpy(host.postprocess(broken, hi)))}
        print(f"[{label}] {name}: " + ", ".join(f"{k} {v:.5f}" for k, v in m.items()))
        for precision in ("accuracy", "performance"):
            failing = gate_failures(m, precision)
            must_fail = [
                k
                for k, (o, _) in GATES[precision].items()
                if o in (op, "in")
                and k != "pcc_final"
                and (per_row or not (k.startswith("pcc_row") or k.startswith("spread_ratio")))
            ]
            passing = [k for k in must_fail if k not in failing]
            assert not passing, f"{name}: the {label.lower()} output passes the {precision} {passing}"


@pytest.mark.parametrize("name", [n for n in ALL if n.startswith("bench_")])
def test_v19_broken_outputs_fail_a_floor(name):
    """PLAN v1.9: inactive-row bands collapsed, ~1 % of active rows at a 0.3x band, and every band at 0.88x / 1.12x
    must each fail at least one floor in each precision (head metrics; the scores are printed for the gate)."""
    from models.experimental.timesfm3_forecast.tests.golden_io import (
        active_rows,
        gate_failures,
        head_metrics,
        scale_bands,
    )

    g = load_golden(name)
    hi = _prepare(g)
    head = (
        g["head_readout"]
        if "head_readout" in g
        else g["head_raw"].reshape(hi.rows, hi.num_patches, -1)[:, hi.readout_tokens.start : hi.readout_tokens.stop]
    )
    active = active_rows(head)
    act_idx = torch.nonzero(active).flatten()
    cases = [
        (
            "active 1% at 0.3x",
            scale_bands(
                head,
                0.3,
                act_idx[
                    torch.randperm(len(act_idx), generator=torch.Generator().manual_seed(0))[
                        : max(1, len(act_idx) // 100)
                    ]
                ],
            ),
        ),
        ("all x0.88", scale_bands(head, 0.88)),
        ("all x1.12", scale_bands(head, 1.12)),
    ]
    if (~active).any():
        cases.insert(0, ("inactive collapsed", scale_bands(head, 0.0, torch.nonzero(~active).flatten())))
    for label, broken in cases:
        m = head_metrics(head, broken, active)
        print(f"[BROKEN {label}] {name}: " + ", ".join(f"{k} {v:.5f}" for k, v in m.items()))
        for precision in ("accuracy", "performance"):
            fails = gate_failures(m, precision)
            print(f"   {precision}: fails {sorted(fails)}")
            assert fails, f"{name}: '{label}' passes every {precision} floor"


@pytest.mark.parametrize("horizon", [64, 128])
def test_dropped_trailing_tokens(horizon):
    """R1-1: the device runs tokens up to the last read-out token only; the ROLLS tokens after it are dropped."""
    g = load_golden("full16_2048_h64" if horizon == 64 else "full16_2048_h128")
    hi = _prepare(g)
    assert g["horizon"] == horizon
    assert hi.num_kept_patches == hi.readout_tokens.stop
    assert hi.num_patches - hi.num_kept_patches == host.ROLLS == 2
