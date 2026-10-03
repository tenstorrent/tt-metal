# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-variant accuracy summaries and the baseline compare.

Pure pandas, no hardware. Spec:
docs/superpowers/specs/2026-09-07-sfpu-accuracy-ci-gate-design.md

    summarize(per_element_df)  -> one row per (op, formats, config) variant
    compare(baseline, current) -> CompareResult (regressions / improvements /
                                  schema errors, with the CI exit code)
    save_baseline / load_baseline -> the committed per-arch CSV
    render_report              -> markdown for the job summary / artifact

Only regressions fail. Improvements are listed so someone can decide to
refresh the baseline by hand; the gate never forces that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# CHIP_ARCH value -> per-arch output folder / baseline file stem. Shared with
# accuracy_harness (which imports it) so the CLI and the sweep agree on paths.
ARCH_ABBR: Dict[str, str] = {"wormhole": "wh", "blackhole": "bh", "quasar": "qsr"}

# Same knobs as accuracy_harness.variant_name, so one baseline row == one shard.
IDENTITY_COLUMNS: List[str] = [
    "op",
    "input_format",
    "output_format",
    "approx_mode",
    "fast_mode",
    "dest_acc",
]

INT_METRICS: List[str] = ["n_rows", "n_finite_mismatch", "n_exact"]
FLOAT_METRICS: List[str] = [
    "max_abs_ulp",
    "mean_abs_ulp",
    "max_abs_error",
    "max_rel_error",
]
METRIC_COLUMNS: List[str] = INT_METRICS + FLOAT_METRICS
BASELINE_COLUMNS: List[str] = IDENTITY_COLUMNS + METRIC_COLUMNS

# ULP is only meaningful when the output grid matches the SFPU's ~16-bit
# internal precision. The per-element column is populated for fp32 too, but
# it explodes by construction there, so the summary blanks it.
ULP_OUTPUT_FORMATS = ("bf16", "fp16")

# Gated metrics and which direction is worse. n_rows is a schema property,
# not a gated metric (a change means the sweep itself changed).
WORSE_IF_HIGHER: Dict[str, bool] = {
    "n_finite_mismatch": True,
    "n_exact": False,
    "max_abs_ulp": True,
    "mean_abs_ulp": True,
    "max_abs_error": True,
    "max_rel_error": True,
}


# ── summarize ─────────────────────────────────────────────────────────────────


def _nan_max(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else math.nan


def _nan_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-element rows into one summary row per variant.

    *df* is any concatenation of the harness's per-op frames (compact schema).
    Output is sorted by IDENTITY_COLUMNS so the CSV is stable across shard order.
    """
    if df.empty:
        return pd.DataFrame(columns=BASELINE_COLUMNS)

    rows = []
    for key, grp in df.groupby(IDENTITY_COLUMNS, sort=True):
        golden = grp["golden_result"].to_numpy(dtype=np.float64)
        hw = grp["hardware_result"].to_numpy(dtype=np.float64)
        fin_g = grp["is_finite_golden"].to_numpy(dtype=bool)
        fin_h = grp["is_finite_hw"].to_numpy(dtype=bool)
        finite = fin_g & fin_h

        exact = (hw == golden) | (np.isnan(hw) & np.isnan(golden))

        signed = grp["signed_error"].to_numpy(dtype=np.float64)
        abs_err = np.abs(np.where(finite, signed, np.nan))

        rel = grp["rel_error"].to_numpy(dtype=np.float64)
        rel = np.where(finite, rel, np.nan)

        out_fmt = key[IDENTITY_COLUMNS.index("output_format")]
        if out_fmt in ULP_OUTPUT_FORMATS:
            ulp = np.abs(grp["signed_ulp_error"].to_numpy(dtype=np.float64))
            max_ulp, mean_ulp = _nan_max(ulp), _nan_mean(ulp)
        else:
            max_ulp, mean_ulp = math.nan, math.nan

        row = dict(zip(IDENTITY_COLUMNS, key))
        row.update(
            n_rows=int(len(grp)),
            n_finite_mismatch=int((fin_g != fin_h).sum()),
            n_exact=int(exact.sum()),
            max_abs_ulp=max_ulp,
            mean_abs_ulp=mean_ulp,
            max_abs_error=_nan_max(abs_err),
            max_rel_error=_nan_max(rel),
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=BASELINE_COLUMNS)


def summarize_dir(parquet_dir: Path) -> pd.DataFrame:
    """summarize() over every merged per-op Parquet in *parquet_dir* (not _shards)."""
    files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no per-op .parquet files in {parquet_dir}")
    return summarize(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))


# ── io ────────────────────────────────────────────────────────────────────────


def save_baseline(df: pd.DataFrame, path: Path) -> Path:
    """Write the baseline CSV. Floats use full repr so load(save(x)) == x exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[BASELINE_COLUMNS].sort_values(IDENTITY_COLUMNS, kind="stable")
    out.to_csv(path, index=False, na_rep="")
    return path


def load_baseline(path: Path) -> pd.DataFrame:
    """Read a baseline CSV. Identity columns stay strings ("0"/"1"), floats float64."""
    df = pd.read_csv(
        path,
        dtype={c: str for c in IDENTITY_COLUMNS},
        keep_default_na=True,
        # The compare is exact; pandas' default (fast) float parser is only
        # ~15 significant digits and would report parse noise as regressions.
        float_precision="round_trip",
    )
    missing = [c for c in BASELINE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: baseline is missing column(s) {missing}")
    for c in INT_METRICS:
        df[c] = df[c].astype("int64")
    for c in FLOAT_METRICS:
        df[c] = df[c].astype("float64")
    return df[BASELINE_COLUMNS].reset_index(drop=True)


# ── compare ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Finding:
    variant: str
    metric: str
    baseline: float
    current: float


@dataclass
class CompareResult:
    regressions: List[Finding] = field(default_factory=list)
    improvements: List[Finding] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        """0 pass (improvements included), 1 regression, 2 schema mismatch."""
        if self.schema_errors:
            return 2
        if self.regressions:
            return 1
        return 0


def variant_label(key) -> str:
    """Human label for one identity tuple, e.g. 'exp bf16->fp16 approx=1 fast=0 dest=0'."""
    op, in_fmt, out_fmt, approx, fast, dest = key
    return f"{op} {in_fmt}->{out_fmt} approx={approx} fast={fast} dest={dest}"


def compare(baseline: pd.DataFrame, current: pd.DataFrame) -> CompareResult:
    """Compare *current* summary against *baseline* with exact equality.

    Variant-set differences and n_rows changes are schema errors (the sweep or
    the baseline changed shape). Everything else follows WORSE_IF_HIGHER.
    """
    result = CompareResult()
    base_i = baseline.set_index(IDENTITY_COLUMNS)
    cur_i = current.set_index(IDENTITY_COLUMNS)

    for key in base_i.index.difference(cur_i.index):
        result.schema_errors.append(
            f"in baseline but not in current run: {variant_label(key)}"
        )
    for key in cur_i.index.difference(base_i.index):
        result.schema_errors.append(
            f"in current run but not in baseline: {variant_label(key)}"
        )

    for key in base_i.index.intersection(cur_i.index):
        b, c = base_i.loc[key], cur_i.loc[key]
        label = variant_label(key)

        if int(b["n_rows"]) != int(c["n_rows"]):
            result.schema_errors.append(
                f"{label}: n_rows {int(b['n_rows'])} -> {int(c['n_rows'])} "
                "(sweep density or domain changed; regenerate the baseline)"
            )
            continue

        for metric, higher_is_worse in WORSE_IF_HIGHER.items():
            bv, cv = float(b[metric]), float(c[metric])
            b_nan, c_nan = math.isnan(bv), math.isnan(cv)
            if b_nan and c_nan:
                continue
            if b_nan != c_nan:
                result.schema_errors.append(
                    f"{label}: {metric} is defined on only one side "
                    f"(baseline={bv!r}, current={cv!r})"
                )
                continue
            if cv == bv:
                continue
            worse = (cv > bv) if higher_is_worse else (cv < bv)
            finding = Finding(label, metric, bv, cv)
            (result.regressions if worse else result.improvements).append(finding)

    return result


# ── report ────────────────────────────────────────────────────────────────────


def _findings_table(findings: List[Finding]) -> str:
    lines = ["| variant | metric | baseline | current |", "|---|---|---|---|"]
    for f in findings:
        lines.append(f"| {f.variant} | {f.metric} | {f.baseline!r} | {f.current!r} |")
    return "\n".join(lines)


def render_report(result: CompareResult, arch: str) -> str:
    """Markdown report for the job summary and the uploaded artifact."""
    n_r, n_i, n_s = (
        len(result.regressions),
        len(result.improvements),
        len(result.schema_errors),
    )
    out = [f"# SFPU accuracy vs baseline ({arch})", ""]

    if n_s:
        out.append(f"**FAIL (schema):** {n_s} mismatch(es) between baseline and sweep.")
    elif n_r:
        out.append(f"**FAIL:** {n_r} regression(s), {n_i} improvement(s).")
    else:
        out.append(f"**PASS:** no regressions, {n_i} improvement(s).")
    out.append("")

    if result.schema_errors:
        out += ["## Schema errors", ""]
        out += [f"- {e}" for e in result.schema_errors]
        out.append("")
    if result.regressions:
        out += ["## Regressions", "", _findings_table(result.regressions), ""]
    if result.improvements:
        out += [
            "## Improvements (informational)",
            "",
            "Not a failure. Refresh the baseline manually with "
            "`--update-baseline` if these should become the new reference.",
            "",
            _findings_table(result.improvements),
            "",
        ]
    return "\n".join(out)
