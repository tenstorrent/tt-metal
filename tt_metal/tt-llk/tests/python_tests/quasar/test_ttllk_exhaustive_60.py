# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
FULL 60-activation tt-llk EXHAUSTIVE bf16 sweep — the tt-llk analog of the DFB
test_dfb_sweep_60.py.

For each of the 60 deployed activations the fitter records the TRUE bf16 shipping
pick in best.csv (best_ulp_* columns). We resolve that pick to a coefficient CSV
with the SAME 3-tier tolerant glob the DFB sweep + quasar_sweep.sh use, drive the
tt-llk SFPI generic-LUT kernel (polynomial / rational / newton_root) on craq-sim
Quasar with EVERY representable bf16 in the activation's full fit domain, and
compute the SAME DFB metrics: bf16 bit-distance ULP (max/mean/p99) + ml_pass
(Torch tolerance 1e-3 + 1e-3*|truth|) vs the fitter ground_truth.

Out-of-scope: a pick whose method the tt-llk kernel cannot evaluate (e.g.
newton_root params missing from the CSV, or a parse/build/run failure) is recorded
+ skipped, never hard-failing the run, so all 60 are collected.

Run (craq-sim Quasar) — via run_quasar.sh env, from tt_metal/tt-llk/tests/python_tests:
  TT_METAL_HOME=/localdev/nkapre/tt-metal CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_SIMULATOR=/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so \
  <py> -m pytest --run-simulator quasar/test_ttllk_exhaustive_60.py -s -q
"""

import csv as _csv
import glob
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import ttllk_exhaustive_driver as drv  # noqa: E402

_FITTER = Path(
    os.environ.get("TT_POLY_FITTER", "/localdev/nkapre/tt-polynomial-fitter")
)
_BEST = _FITTER / "best.csv"
_COEFFS = _FITTER / "data" / "coefficients"
_PRECROW = "bf16"
_SEL = "ulp"

_TXT_OUT = Path("/tmp/ttllk_exhaustive.txt")
_RESULTS = []


# ---- deployment-truth resolution (VERBATIM from test_dfb_sweep_60._resolve_deployed) --
def _resolve_deployed(act):
    row = None
    with open(_BEST) as f:
        for r in _csv.DictReader(f):
            if r.get("activation") == act and r.get("precision") == _PRECROW:
                row = r
                break
    if row is None:
        return None

    def g(field):
        return (row.get(f"best_{_SEL}_{field}") or "").strip()

    deg = g("degree")
    nseg = g("num_segments") or g("segments")
    fit = g("fitting")
    if not (deg and nseg):
        return None
    kind = "rational" if (fit == "rational" or "/" in deg) else "polynomial"
    if kind == "rational":
        num, den = deg.split("/", 1) if "/" in deg else (deg, deg)
        core = f"n{num.strip()}d{den.strip()}"
    else:
        core = f"p{deg.strip()}"

    cands = sorted(glob.glob(f"{_COEFFS}/{act}_{core}_s{nseg}_*.csv"))
    if cands:
        return kind, core, nseg, cands[0]
    pat = re.compile(rf"{re.escape(act)}_{re.escape(core)}_s(\d+)_.*")
    matches = []
    for p in glob.glob(f"{_COEFFS}/{act}_{core}_s*_*.csv"):
        m = pat.search(os.path.basename(p))
        if m:
            matches.append((int(m.group(1)), p))
    if matches:
        matches.sort(key=lambda t: (t[0], t[1]))
        return kind, core, nseg, matches[0][1]
    cands = sorted(glob.glob(f"{_COEFFS}/{act}_{core}*_*.csv"))
    if cands:
        return kind, core, nseg, cands[0]
    return kind, core, nseg, None


def _all_bf16_activations():
    acts = []
    with open(_BEST) as f:
        for r in _csv.DictReader(f):
            if r.get("precision") == _PRECROW:
                acts.append(r["activation"])
    seen, out = set(), []
    for a in acts:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


_ACTS = _all_bf16_activations()


def _record(activation, eval_method, res, status, reason=""):
    _RESULTS.append(
        dict(
            activation=activation,
            eval_method=eval_method,
            ulp_max=res.get("ulp_max", float("nan")) if res else float("nan"),
            ulp_mean=res.get("ulp_mean", float("nan")) if res else float("nan"),
            ulp_p99=res.get("ulp_p99", float("nan")) if res else float("nan"),
            ml_pass=res.get("ml_pass", float("nan")) if res else float("nan"),
            status=status,
            reason=reason,
        )
    )


@pytest.mark.quasar
@pytest.mark.parametrize("activation", _ACTS, ids=_ACTS)
def test_ttllk_exhaustive_60(activation):
    res = _resolve_deployed(activation)
    if res is None or res[3] is None:
        _record(activation, "-", None, "out-of-scope", "no deployed CSV resolved")
        pytest.skip("no deployed CSV resolved")

    kind, core, segs, csv_path = res
    method = drv.detect_method(csv_path)

    try:
        r = drv.run_exhaustive(str(csv_path), activation, method=method)
    except Exception as e:  # parse/build/run failure -> out of scope, named.
        _record(activation, method, None, "out-of-scope", f"{type(e).__name__}: {e}")
        print(
            f"\n[{activation:>16} {method:>12}]  OUT-OF-SCOPE: {type(e).__name__}: {e}"
        )
        pytest.skip(f"out-of-scope: {e}")

    _record(activation, r["eval_method"], r, "measured")
    print(
        f"\n[{activation:>16} {r['eval_method']:>12} n={r['n_bf16']}]  "
        f"bf16 ULP max/mean/p99={r['ulp_max']:.0f}/{r['ulp_mean']:.3f}/{r['ulp_p99']:.1f} "
        f"ml_pass={r['ml_pass']:.4f}"
    )


# ---- session-end table ---------------------------------------------------------------
def _f(v, w, p):
    return f"{v:>{w}.{p}f}" if v == v else f"{'n/a':>{w}}"


@pytest.fixture(scope="session", autouse=True)
def _write_summary():
    yield
    if not _RESULTS:
        return
    nmeas = sum(1 for r in _RESULTS if r["status"] == "measured")
    noos = sum(1 for r in _RESULTS if r["status"] == "out-of-scope")
    total = len(_RESULTS)

    header = "activation       | eval_method  | bf16_ULP_max | bf16_ULP_mean | bf16_ULP_p99 | ml_pass | status"
    sep = "-" * len(header)
    lines = [header, sep]
    order = {"measured": 0, "out-of-scope": 1}
    for r in sorted(
        _RESULTS, key=lambda r: (order.get(r["status"], 2), r["activation"])
    ):
        lines.append(
            f"{r['activation']:<16} | {str(r['eval_method']):<12} | "
            f"{_f(r['ulp_max'],12,0)} | {_f(r['ulp_mean'],13,3)} | {_f(r['ulp_p99'],12,1)} | "
            f"{_f(r['ml_pass'],7,4)} | {r['status']}"
            + (f"  ({r['reason']})" if r["reason"] else "")
        )
    table = "\n".join(lines)

    tally = (
        f"TALLY: {nmeas}/{total} measured exhaustively  |  {noos} out-of-scope  "
        f"(tt-llk generic-LUT kernel on craq-sim Quasar; EVERY representable bf16 in the full fit "
        f"domain; bf16 bit-distance ULP excludes zero-reference points; ml_pass = fraction within "
        f"1e-3 + 1e-3*|truth|; same metric defs as the DFB sweep)"
    )
    _TXT_OUT.write_text(tally + "\n\n" + table + "\n")
    print("\n\n" + "=" * len(header))
    print(tally)
    print("=" * len(header))
    print(table)
    print(f"\nwrote {_TXT_OUT}")
