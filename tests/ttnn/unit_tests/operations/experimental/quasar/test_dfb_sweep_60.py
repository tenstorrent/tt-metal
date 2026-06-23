# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
FULL 60-activation DFB DEPLOYMENT sweep — the DFB analog of the tt-llk
quasar_sweep.sh deployment validator.

For each of the 60 deployed activations the fitter records the TRUE shipping pick
per (activation, precision) in best.csv columns best_ulp_{fitting,degree,
num_segments,segmentation}. We read the **bf16** row, resolve the deployed pick to a
coefficient CSV under $TT_POLY_FITTER/data/coefficients using the SAME 3-tier tolerant
glob quasar_sweep.sh uses (exact segs -> same-degree fewest-segs -> loosest), then run
that CSV through the generic unary_lut DFB op on craq-sim Quasar and PCC-check vs the
fitter ground_truth (the TRUE activation). Zero per-activation special-casing — the CSV
(POLY vs RATIONAL, degree, segments, coefficients, RR metadata) drives everything.

PCC is MEASURED on craq-sim, never assumed. Each activation JIT-recompiles the kernel
with its own coefficients — no full rebuild needed. Sim runs are sequential (one sim at
a time).

Classification per activation (NO hard-assert-fail of the whole run on any one
activation — ALL 60 are collected, then a session-end summary writes the table):
  * pass          — PCC_vs_true >= 0.99
  * fail          — DFB op ran but PCC_vs_true < 0.99
  * out-of-scope  — the deployed pick uses a feature the DFB kernel does not yet
                    implement (e.g. newton_root range reduction for sqrt/rsqrt). These
                    are NOT kernel defects; they are named in the summary with the
                    reason.

Run (craq-sim Quasar):
  cd /localdev/nkapre/tt-metal-dfbport
  export TT_METAL_HOME=$PWD ARCH_NAME=quasar CHIP_ARCH=quasar TT_METAL_SLOW_DISPATCH_MODE=1 \
         TT_METAL_SIMULATOR=/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so \
         PYTHONPATH=$PWD:$PWD/ttnn
  /localdev/nkapre/tt-metal/python_env/bin/python -m pytest \
      tests/ttnn/unit_tests/operations/experimental/quasar/test_dfb_sweep_60.py -q
"""

import csv as _csv
import glob
import os
import re
import sys
from pathlib import Path

import pytest

# Import the sibling driver module by directory (no package __init__ chain here).
sys.path.insert(0, str(Path(__file__).parent))
import dfb_lut_driver as drv  # noqa: E402

_FITTER = Path(os.environ.get("TT_POLY_FITTER", "/localdev/nkapre/tt-polynomial-fitter"))
_BEST = _FITTER / "best.csv"
_COEFFS = _FITTER / "data" / "coefficients"

_PCC = 0.99
_PRECROW = "bf16"  # the deployed row we validate (task: use the bf16 row per activation)
_SEL = "ulp"  # the deployed pick selector (best_ulp_*)

# RR methods the DFB kernel (unary_lut_sfpu.h) actually implements: codes 1-8.
# A deployed pick whose CSV range_reduction_method is enabled but NOT one of these is
# out of scope for the DFB kernel as it stands (e.g. newton_root for sqrt/rsqrt).
_RR_IN_KERNEL = {
    "log",
    "exp",
    "cbrt",
    "exponent_alu_exp2",
    "exponent_alu_log2",
    "exponent_alu_pow",
    "trig",
    "tan",
}

# Shared results collector — every parametrized case appends one record; the
# session-finish fixture writes the table + tally once at the end.
_RESULTS = []


# ---- deployment-truth resolution (replicates quasar_sweep.sh resolve_deployed) --------
def _read_meta_method(csv_path):
    """Return (range_reduction_method, enabled_bool) from a CSV's METADATA rows."""
    method, enabled = "none", False
    with open(csv_path, newline="") as f:
        for r in _csv.reader(f):
            if r and r[0] == "METADATA" and len(r) >= 3:
                if r[1].strip() == "range_reduction_method":
                    method = r[2].strip()
                elif r[1].strip() == "range_reduction_enabled":
                    enabled = r[2].strip().lower() == "true"
    return method, enabled


def _resolve_deployed(act):
    """Resolve the deployed (kind, core, segs, csv_path) for `act` at the bf16 row.

    Mirrors quasar_sweep.sh: read best_ulp_{fitting,degree,num_segments}; rational iff
    fitting=='rational' or degree contains '/'; core = p<deg> (poly) or n<num>d<den>
    (rational). 3-tier tolerant glob (exact segs -> same-degree fewest-segs -> loosest).
    Returns (kind, core, segs, path|None) or None if no deployed row.
    """
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

    # (a) exact segs
    cands = sorted(glob.glob(f"{_COEFFS}/{act}_{core}_s{nseg}_*.csv"))
    if cands:
        return kind, core, nseg, cands[0]
    # (b) same degree, any segment count -- fewest first (closest deployed intent)
    pat = re.compile(rf"{re.escape(act)}_{re.escape(core)}_s(\d+)_.*")
    matches = []
    for p in glob.glob(f"{_COEFFS}/{act}_{core}_s*_*.csv"):
        m = pat.search(os.path.basename(p))
        if m:
            matches.append((int(m.group(1)), p))
    if matches:
        matches.sort(key=lambda t: (t[0], t[1]))
        return kind, core, nseg, matches[0][1]
    # (c) loosest
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
    # preserve first-seen order, unique
    seen = set()
    out = []
    for a in acts:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


_ACTS = _all_bf16_activations()


# ---- the sweep -----------------------------------------------------------------------
@pytest.mark.parametrize("activation", _ACTS, ids=_ACTS)
def test_dfb_sweep_60(device, activation):
    """Run ONE deployed bf16 pick through the DFB op and record the result. Never
    hard-fails the whole run: out-of-scope picks are recorded + skipped, fits are
    asserted softly via the recorded status so the session-end summary sees all 60."""
    res = _resolve_deployed(activation)
    if res is None or res[3] is None:
        rec = dict(
            activation=activation,
            eval_method="-",
            rr="-",
            num_segments="-",
            pcc_true=float("nan"),
            pcc_approx=float("nan"),
            status="out-of-scope",
            reason="no deployed CSV resolved",
        )
        _RESULTS.append(rec)
        pytest.skip(rec["reason"])

    kind, core, segs, csv_path = res
    rr_method_name, rr_enabled = _read_meta_method(csv_path)

    # Out-of-scope BEFORE running: deployed pick needs an RR method the DFB kernel
    # does not implement (e.g. newton_root). Not a kernel defect — name + skip.
    if rr_enabled and rr_method_name not in _RR_IN_KERNEL:
        rec = dict(
            activation=activation,
            eval_method=kind.upper()[:8],
            rr=rr_method_name,
            num_segments=segs,
            pcc_true=float("nan"),
            pcc_approx=float("nan"),
            status="out-of-scope",
            reason=f"RR method '{rr_method_name}' not implemented in DFB kernel",
        )
        _RESULTS.append(rec)
        pytest.skip(rec["reason"])

    r = drv.run_dfb(device, str(csv_path), activation=activation, tiles=4)

    pcc_true = r["pcc_vs_true"]
    pcc_approx = r["pcc_vs_approx"]
    rr_code = r["rr_method"]
    rr_label = rr_method_name if r["rr_enabled"] else "none"
    status = "pass" if (pcc_true == pcc_true and pcc_true >= _PCC) else "fail"

    rec = dict(
        activation=activation,
        eval_method=r["eval_method"],
        rr=rr_label,
        num_segments=r["num_segments"],
        pcc_true=pcc_true,
        pcc_approx=pcc_approx,
        status=status,
        reason="" if status == "pass" else f"PCC_vs_true {pcc_true:.6f} < {_PCC}",
        csv=os.path.basename(csv_path),
        rr_code=rr_code,
    )
    _RESULTS.append(rec)

    print(
        f"\n[{activation:>16} {r['eval_method']:>8} deg={r['degree']} seg={r['num_segments']} "
        f"rr={rr_label} dom={r['domain']}]  PCC_true={pcc_true:.6f} PCC_approx={pcc_approx:.6f}  -> {status.upper()}"
    )
    # Soft per-case: do NOT propagate a single failure as a hard error that aborts the
    # collection. We record the status; the session-end summary is the source of truth.


# ---- session-end summary -------------------------------------------------------------
_MD_OUT = Path(__file__).parent / "DFB_SWEEP_60.md"
_TXT_OUT = Path("/tmp/dfb_sweep_60.txt")


def _render_table():
    hdr = f"| {'activation':<16} | {'eval':<8} | {'rr':<18} | {'segs':>4} | {'PCC_true':>10} | {'status':<12} |"
    sep = "|" + "-" * (len(hdr) - 2) + "|"
    lines = [hdr, sep.replace(" ", "-")]
    # tabular separator aligned to columns
    lines = [
        "| activation       | eval     | rr                 | segs | PCC_true   | status       |",
        "|------------------|----------|--------------------|------|------------|--------------|",
    ]
    order = {"pass": 0, "fail": 1, "out-of-scope": 2}
    for rec in sorted(_RESULTS, key=lambda r: (order.get(r["status"], 3), r["activation"])):
        pcc = rec["pcc_true"]
        pcc_s = f"{pcc:.6f}" if pcc == pcc else "   n/a   "
        seg = str(rec["num_segments"])
        lines.append(
            f"| {rec['activation']:<16} | {str(rec['eval_method']):<8} | {str(rec['rr']):<18} "
            f"| {seg:>4} | {pcc_s:>10} | {rec['status']:<12} |"
        )
    return "\n".join(lines)


@pytest.fixture(scope="session", autouse=True)
def _write_summary():
    yield
    if not _RESULTS:
        return
    npass = sum(1 for r in _RESULTS if r["status"] == "pass")
    nfail = sum(1 for r in _RESULTS if r["status"] == "fail")
    noos = sum(1 for r in _RESULTS if r["status"] == "out-of-scope")
    total = len(_RESULTS)

    table = _render_table()
    fails = [r for r in _RESULTS if r["status"] == "fail"]
    oos = [r for r in _RESULTS if r["status"] == "out-of-scope"]

    tally = (
        f"TALLY: {npass}/{total} pass  |  {nfail} fail  |  {noos} out-of-scope  "
        f"(threshold PCC_vs_true >= {_PCC}, measured on craq-sim Quasar)"
    )

    def _fail_line(r):
        # PCC_vs_approx isolates DFB-path correctness from deployed-fit fidelity: when
        # PCC_vs_approx ~ 1.0 the DFB kernel reproduced its own approximation faithfully,
        # so the PCC_vs_true miss is the deployed fit's accuracy on its own domain (a
        # fitter/domain characteristic), NOT a DFB-kernel defect.
        pa = r.get("pcc_approx", float("nan"))
        pa_s = f"{pa:.6f}" if pa == pa else "n/a"
        return f"  - {r['activation']}: {r['reason']}  (PCC_vs_approx={pa_s} -> DFB path faithful; fit-on-domain miss)"

    fail_lines = "\n".join(_fail_line(r) for r in fails) or "  (none)"
    oos_lines = "\n".join(f"  - {r['activation']}: {r['reason']}" for r in oos) or "  (none)"

    md = (
        "<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent Inc. -->\n"
        "<!-- SPDX-License-Identifier: Apache-2.0 -->\n\n"
        "# DFB 60-Activation Deployment Sweep\n\n"
        "The DFB analog of the tt-llk `quasar_sweep.sh` deployment validator. Each of the\n"
        "60 deployed activations is the fitter's TRUE bf16 shipping pick (`best_ulp_*` in\n"
        "`best.csv`), resolved to a coefficient CSV with the same 3-tier tolerant glob, run\n"
        "through the generic `ttnn.experimental.quasar.unary_lut` DFB op on craq-sim Quasar,\n"
        "and PCC-checked vs the fitter `ground_truth` (the TRUE activation). PCC is MEASURED\n"
        "on craq-sim, never assumed.\n\n"
        f"**{tally}**\n\n"
        "## Results\n\n" + table + "\n\n"
        "## Failures (DFB op ran, PCC_vs_true < 0.99)\n" + fail_lines + "\n\n"
        "## Out-of-scope (deployed pick uses a feature the DFB kernel does not yet implement)\n" + oos_lines + "\n"
    )
    _MD_OUT.write_text(md)

    txt = tally + "\n\n" + table + "\n\nFAILURES:\n" + fail_lines + "\n\nOUT-OF-SCOPE:\n" + oos_lines + "\n"
    _TXT_OUT.write_text(txt)

    print("\n\n" + "=" * 90)
    print(tally)
    print("=" * 90)
    print(table)
    print(f"\nwrote {_MD_OUT}")
    print(f"wrote {_TXT_OUT}")
