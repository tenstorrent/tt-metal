# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Stage-A CPU tests of the dataflow (reference/dataflow.py) against the test and B0 goldens.

CONTRACT v1.1 §3 "host front end and back end" (rtol 1e-5, index logic exact) and §6 A (the dataflow within 1e-5
relative of the reference in fp32 on the test goldens; route S emulated in fp32 and float64). Each check has a
positive control that must fail (D0 item 4). Run in the reference venv as a ttq cpu job:
  ttq run --model flowstate --kind cpu -- env -u PYTHONPATH /home/andy/bringup/venvs/flowstate/bin/python -m pytest \
      models/experimental/flowstate_forecast/tests/golden/test_dataflow.py
"""

import json
import os
from pathlib import Path

import pytest
import torch

from models.experimental.flowstate_forecast.reference import dataflow as dfl

GOLDENS = Path("/home/andy/bringup/goldens/flowstate")
TEST_CASES = ("single", "model16", "single_long", "nan_mix", "mask0", "daily", "monthly", "fivemin", "cpm_edge")
FULL_CASES = ("single", "nan_mix", "mask0", "monthly", "cpm_edge")
REPORT = Path(os.environ.get("FLOWSTATE_SCAN_REPORT", "/tmp/flowstate_scan_report.json"))
RTOL = 1e-5

pytestmark = pytest.mark.skipif(not (GOLDENS / "single.pt").exists(), reason="test goldens not generated yet")


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()


@pytest.fixture(scope="module")
def model():
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from models.experimental.flowstate_forecast.tests.golden.make_goldens import load_model
    except ImportError:  # python_env has no granite-tsfm: the same checkpoint through safetensors
        from models.experimental.flowstate_forecast.reference import weights

        return weights.load()
    return load_model()


_cache = {}


def golden(case):
    if case not in _cache:
        _cache[case] = torch.load(GOLDENS / f"{case}.pt", weights_only=True)
    return _cache[case]


def run(model, case, **kw):
    g = golden(case)
    return g, dfl.forward(model, g["inputs"].clone(), g["s"], g["horizon"], **kw)


@pytest.mark.parametrize("case", TEST_CASES)
def test_bookkeeping(case):
    """fl / mask_n / max_context / min_context / L exact against the golden's recorded values."""
    g = golden(case)
    bk = dfl.bookkeeping(g["inputs"].shape[1], g["horizon"], g["s"])
    for k in ("fl", "mask_n", "max_context", "L", "min_context"):
        assert bk[k] == g[k], (k, bk[k], g[k])


@pytest.mark.parametrize("case", TEST_CASES)
def test_causal_revin(case):
    """NaN mask, appended [0, 1] steps, running mean / stdev and normalized tokens, rtol 1e-5."""
    g = golden(case)
    tok, mean, stdev, _ = dfl.front_end(g["inputs"].clone(), g["s"], g["horizon"])
    assert rel(mean, g["revin_mean"][..., :1]) < RTOL
    assert rel(stdev, g["revin_stdev"][..., :1]) < RTOL
    assert rel(tok, g["normalized"]) < RTOL
    # control: RevIN that ignores the missing-value flag must fail wherever there are NaNs or appended steps
    if g["mask_n"] or g["inputs"].isnan().any():
        n = torch.arange(1, tok.shape[1] + 1).view(1, -1, 1)
        v = torch.nan_to_num(g["inputs"][:, -g["max_context"] :].float())
        v = torch.cat((v, torch.zeros(v.shape[0], g["mask_n"], 1)), 1)
        assert rel(torch.cumsum(v, 1) / n, g["revin_mean"][..., :1]) > RTOL


def test_constants(model):
    """Lbar, Bbar per layer and Legendre W from the reference expressions; control: float64 constants must fail."""
    g = golden("single")
    for i, lay in enumerate(model.model.encoder.layers):
        log_lbar, Bbar, K = dfl.ssm_constants(lay.ssm, g["s"], g["L"])
        assert torch.equal(K[1], g[f"Lbar{i}"]), i
        assert torch.equal(Bbar, g[f"Bbar{i}"]), i
        # control: the same constants in float64 differ from the complex64 reference expressions
        lam = (-torch.exp(lay.ssm.log_Lambda_real.double()) + 1j * lay.ssm.Lambda_imag.double()) * torch.exp(
            lay.ssm.log_Delta.double()
        )
        assert not torch.equal(torch.exp(g["s"] * lam).to(torch.complex64), g[f"Lbar{i}"]), i
    assert rel(dfl.legendre_w(g["s"], g["fl"]), g["W"]) < RTOL


@pytest.mark.parametrize("case", FULL_CASES)
def test_dataflow_fft(model, case):
    """The device decomposition (FFT at pow-2 N) within 1e-5 relative; control: circular conv at N = L must fail."""
    g, d = run(model, case)
    keys = ["normalized", "decoder_pre_denorm", "combined_normalized", "quantile_outputs", "prediction_outputs"]
    keys += [k for k in g if k.startswith(("layer", "ssm", "mlp")) or k == "embed"]
    errs = {k: rel(d[k], g[k]) for k in keys}
    print(case, json.dumps({k: f"{v:.2e}" for k, v in errs.items()}))
    assert all(v < RTOL for v in errs.values()), errs
    _, c = run(model, case, N=g["L"])
    assert rel(c["layer0"], g["layer0"]) > 10 * RTOL if "layer0" in g else rel(c["layer5"], g["layer5"]) > 10 * RTOL


def test_modules(model):
    """Module goldens on random inputs: one layer at L = 96 / 70 (first and last layer)."""
    g = golden("modules")
    layers = model.model.encoder.layers
    for L, s in ((96, 1.0), (70, 6.0)):
        u = g[f"u_L{L}"]
        out, parts = dfl.layer_forward(layers[0], u, s, 0, False)
        assert rel(parts["ssm"], g[f"ssm0_L{L}"]) < RTOL and rel(out, g[f"layer0_L{L}"]) < RTOL
        out, parts = dfl.layer_forward(layers[5], u, s, L - 7, True)
        assert rel(parts["ssm"], g[f"ssm5_L{L}"]) < RTOL and rel(out, g[f"layer5_L{L}"]) < RTOL


@pytest.mark.parametrize("case", TEST_CASES)
def test_back_end(model, case):
    """CPM combine, de-normalization, torch.quantile, weighted mean; cpm_edge returns H - fl + 1 steps."""
    g = golden(case)
    q, p = dfl.back_end(g["decoder_pre_denorm"], g["revin_mean"][..., :1], g["revin_stdev"][..., :1], g["horizon"])
    assert rel(q, g["quantile_outputs"]) < RTOL and rel(p, g["prediction_outputs"]) < RTOL
    assert rel(dfl.combine_cpm(g["decoder_pre_denorm"], g["horizon"]), g["combined_normalized"]) < RTOL
    if case == "cpm_edge":
        assert q.shape[2] == g["horizon"] - g["fl"] + 1
    # control: denormalizing with the first row's stats instead of the read rows' must fail
    rows = g["decoder_pre_denorm"].shape[0]
    if g["revin_stdev"].shape[1] > rows:
        bad, _ = dfl.back_end(
            g["decoder_pre_denorm"],
            g["revin_mean"][:, :1].expand_as(g["revin_mean"])[..., :1],
            g["revin_stdev"][:, :1].expand_as(g["revin_stdev"])[..., :1],
            g["horizon"],
        )
        assert rel(bad, g["quantile_outputs"]) > RTOL


B0_CASES = ("b0_s6_L682", "b0_s1_L4096", "b0_s0.25_L16384", "b0_s1_12_L16384")


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_scan_emulation(model, dtype):
    """Route S chunked scan error per layer and channel against the reference and the float64 truth (B0 goldens).

    A report, not a gate: the bound is B0's business (§6 A). Asserts only finiteness and that the float64 scan
    tracks the truth; writes per-(case, layer) whole-tensor rel-L2 and per-channel p99 / max to REPORT."""
    cdt = torch.complex64 if dtype == "float32" else torch.complex128
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    layers = model.model.encoder.layers
    for case in B0_CASES:
        g = golden(case)
        L = g["L"]
        for i in g["layers"]:
            u = g[f"u{i}"].float()
            ref_x = dfl.fft_conv(*_bu_k(layers[i], g, i, u), N=2 * L)  # the reference's own n = 2L
            Bu, K = _bu_k(layers[i], g, i, u)
            truth = dfl.recurrence(Bu, g[f"Lbar{i}"])
            row = {"ref_vs_truth": rel(ref_x, truth)}
            for c in (32, 64):
                x, bounded = dfl.scan_conv(Bu, g[f"Lbar{i}"], c=c, dtype=cdt)
                assert torch.isfinite(torch.view_as_real(x)).all(), (case, i, c)
                ch = ((x - truth).abs().pow(2).sum((0, 1)).sqrt() / truth.abs().pow(2).sum((0, 1)).sqrt()).double()
                row[f"c{c}"] = {
                    "vs_ref": rel(x, ref_x),
                    "vs_truth": rel(x, truth),
                    "ch_p99": torch.quantile(ch, 0.99).item(),
                    "ch_max": ch.max().item(),
                    "bounded_modes": int(bounded.sum()),
                }
                if dtype == "float64":
                    assert row[f"c{c}"]["vs_truth"] < 1e-9, (case, i, row)
            report.setdefault(dtype, {}).setdefault(case, {})[str(i)] = row
            print(dtype, case, i, json.dumps(row))
    REPORT.write_text(json.dumps(report, indent=1))


def _bu_k(lay, g, i, u):
    """Bu in fp32 from the stored bf16-rounded u and the golden Bbar; the reference-expression kernel."""
    _, _, K = dfl.ssm_constants(lay.ssm, g["s"], g["L"])
    Bbar = g[f"Bbar{i}"]
    bu = u @ torch.cat((Bbar.real, Bbar.imag), 0).T
    return torch.complex(bu[..., :512], bu[..., 512:]), K
