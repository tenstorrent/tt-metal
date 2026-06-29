#!/usr/bin/env python3
"""Build a CRAQ-friendly GBDT training table from frontier sweep shards.

The frontier shard CSV is the silicon source of truth: one coefficient artifact,
one precision, one measured Tracy runtime. This tool enriches those rows with
stable numeric features parsed from the coefficient CSV itself and writes a TSV
that can be fed directly to craq-sim's ``scripts/perf/fit.py`` using
``--target target_runtime_ns``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


FREQ_MHZ = 1350.0
IDENTITY_COLUMNS = [
    "key",
    "nodeid",
    "suite",
    "family",
    "module",
    "source",
    "coeff_csv",
    "activation",
    "precision",
    "method",
    "metric",
    "range_reduction_method",
    "segmentation",
    "approximation_type",
]
NUMERIC_COLUMNS = [
    "target_runtime_us",
    "target_runtime_ns",
    "target_cycles_1350mhz",
    "max_ulp",
    "tile_count",
    "degree",
    "segments",
    "num_degree",
    "den_degree",
    "is_rational",
    "is_polynomial",
    "is_lowering",
    "has_range_reduction",
    "range_min",
    "range_max",
    "range_width",
    "coeff_count",
    "nonzero_coeff_count",
    "max_abs_coeff",
    "sum_abs_coeff",
    "avg_abs_coeff",
    "log10_max_abs_coeff",
    "log10_sum_abs_coeff",
    "log10_avg_abs_coeff",
    "avg_effective_degree",
    "max_effective_degree",
]
FIELDNAMES = IDENTITY_COLUMNS + NUMERIC_COLUMNS


def fnum(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def bounded_feature(value: float, limit: float = 1.0e30) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-limit, min(limit, value))


def log10p_feature(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return math.log10(1.0 + abs(value))


def find_fitter() -> Path:
    candidates = []
    if os.environ.get("TT_POLY_FIT_DIR"):
        candidates.append(Path(os.environ["TT_POLY_FIT_DIR"]))
    candidates += [
        Path.home() / "tt-polynomial-fitter",
        Path.home() / "workspace" / "tt-polynomial-fitter",
        Path("/localdev") / os.environ.get("USER", "") / "tt-polynomial-fitter",
    ]
    for path in candidates:
        if (path / "data" / "coefficients").exists():
            return path
    raise SystemExit("frontier_gbdt_dataset: could not locate tt-polynomial-fitter; set TT_POLY_FIT_DIR")


def import_ttp(fit_dir: Path):
    sys.path.insert(0, str(fit_dir))
    try:
        from ttpoly.spec.csv_io import parse_csv_artifact
    except ImportError as exc:
        raise SystemExit(
            "frontier_gbdt_dataset: tt-polynomial-fitter must expose " "ttpoly.spec.csv_io.parse_csv_artifact"
        ) from exc
    return parse_csv_artifact


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(m) for m in (matches or [pattern]))
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"frontier_gbdt_dataset: input CSV not found: {missing[0]}")
    return paths


def coeff_path(coeff_dir: Path, row: dict[str, str]) -> Path:
    raw = (row.get("coeff_csv") or row.get("csv") or "").strip()
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return path
    candidate = coeff_dir / path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"frontier_gbdt_dataset: coefficient CSV not found for row: {raw}")


def parse_range(row: dict[str, str], fit_dir: Path, activation: str) -> tuple[float | None, float | None]:
    lo = fnum(row.get("range_min"))
    hi = fnum(row.get("range_max"))
    if lo is not None and hi is not None:
        return lo, hi
    rng = (row.get("range") or "").strip().strip("[]")
    if rng:
        parts = [p.strip() for p in rng.split(",", 1)]
        if len(parts) == 2:
            lo, hi = fnum(parts[0]), fnum(parts[1])
            if lo is not None and hi is not None:
                return lo, hi
    act_json = fit_dir / "activations" / f"{activation}.json"
    if act_json.exists():
        data = json.loads(act_json.read_text(encoding="utf-8"))
        dom = data.get("domain") or {}
        return fnum(dom.get("min")), fnum(dom.get("max"))
    return None, None


def coeff_stats(path: Path) -> dict[str, float]:
    coeff_count = nonzero = 0
    total_abs = 0.0
    max_abs = 0.0
    effective: list[int] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        coeff_cols = [c for c in fields if len(c) > 1 and c[0] in {"c", "n", "d"} and c[1:].isdigit()]
        for row in reader:
            if str(row.get("segment_id", "")).upper() == "METADATA":
                continue
            seg_eff = 0
            for col in coeff_cols:
                v = fnum(row.get(col))
                if v is None:
                    continue
                coeff_count += 1
                av = abs(v)
                total_abs += av
                max_abs = max(max_abs, av)
                if av != 0.0:
                    nonzero += 1
                    seg_eff = max(seg_eff, int(col[1:]))
            effective.append(seg_eff)
    avg_abs = total_abs / coeff_count if coeff_count else 0.0
    return {
        "coeff_count": float(coeff_count),
        "nonzero_coeff_count": float(nonzero),
        "max_abs_coeff": bounded_feature(max_abs),
        "sum_abs_coeff": bounded_feature(total_abs),
        "avg_abs_coeff": bounded_feature(avg_abs),
        "log10_max_abs_coeff": log10p_feature(max_abs),
        "log10_sum_abs_coeff": log10p_feature(total_abs),
        "log10_avg_abs_coeff": log10p_feature(avg_abs),
        "avg_effective_degree": sum(effective) / len(effective) if effective else 0.0,
        "max_effective_degree": float(max(effective) if effective else 0),
    }


def normalize_degree(value) -> float:
    text = str(value or "").strip()
    if "d" in text:
        parts = [fnum(p) for p in text.split("d", 1)]
        return max(p for p in parts if p is not None) if any(p is not None for p in parts) else 0.0
    return fnum(text) or 0.0


def enrich(row: dict[str, str], path: Path, identity: dict, fit_dir: Path) -> dict[str, str]:
    activation = str(row.get("activation") or identity.get("activation") or "")
    precision = str(row.get("precision") or row.get("dtype") or "")
    method = str(row.get("method") or identity.get("eval_method") or "")
    metric = str(row.get("metric") or identity.get("metric") or "")
    approx_type = str(identity.get("approximation_type") or "")
    rr = str(identity.get("range_reduction_method") or "none")
    runtime_us = fnum(row.get("target_runtime_us")) or fnum(row.get("runtime_us"))
    compiles = str(row.get("compiles") or "1").strip()
    if compiles not in {"1", "true", "True"}:
        runtime_us = None
    lo, hi = parse_range(row, fit_dir, activation)
    stats = coeff_stats(path)
    num_degree = fnum(identity.get("num_degree"))
    den_degree = fnum(identity.get("den_degree"))
    degree = normalize_degree(row.get("degree") or identity.get("degree"))
    segments = fnum(row.get("segments") or identity.get("segments") or identity.get("depth")) or 0.0
    is_rational = 1.0 if (approx_type == "rational" or (den_degree or 0.0) > 0.0) else 0.0
    is_lowering = 1.0 if method in {"identity", "clamped_affine", "threshold_identity", "algebraic_lowering"} else 0.0
    key = f"{activation}/{precision}/{path.name}"
    out = {
        "key": key,
        "nodeid": key,
        "suite": "tt-metal-embedded-frontier",
        "family": "generic_lut_activation_embedded",
        "module": "frontier_sweep",
        "source": str(path),
        "coeff_csv": path.name,
        "activation": activation,
        "precision": precision,
        "method": method,
        "metric": metric,
        "range_reduction_method": rr,
        "segmentation": str(identity.get("segmentation") or ""),
        "approximation_type": approx_type,
        "target_runtime_us": runtime_us,
        "target_runtime_ns": runtime_us * 1000.0 if runtime_us is not None else "",
        "target_cycles_1350mhz": runtime_us * FREQ_MHZ if runtime_us is not None else "",
        "max_ulp": fnum(row.get("bf16_maxulp") or row.get("max_ulp")) or "",
        "tile_count": fnum(row.get("tile_count")) or 256.0,
        "degree": degree,
        "segments": segments,
        "num_degree": num_degree if num_degree is not None else degree,
        "den_degree": den_degree if den_degree is not None else 0.0,
        "is_rational": is_rational,
        "is_polynomial": 0.0 if is_rational else 1.0,
        "is_lowering": is_lowering,
        "has_range_reduction": 0.0 if rr in {"", "none"} else 1.0,
        "range_min": lo if lo is not None else "",
        "range_max": hi if hi is not None else "",
        "range_width": (hi - lo) if lo is not None and hi is not None else "",
        **stats,
    }
    return {k: "" if out.get(k) is None else str(out.get(k, "")) for k in FIELDNAMES}


def load_rows(paths: list[Path], coeff_dir: Path, fit_dir: Path) -> list[dict[str, str]]:
    parse_csv_artifact = import_ttp(fit_dir)
    rows: list[dict[str, str]] = []
    cache: dict[Path, tuple[dict, dict[str, str]]] = {}
    for shard in paths:
        with shard.open(newline="") as f:
            for row in csv.DictReader(f):
                path = coeff_path(coeff_dir, row)
                if path not in cache:
                    cache[path] = (parse_csv_artifact(path), {})
                identity, _ = cache[path]
                enriched = enrich(row, path, identity, fit_dir)
                if enriched["target_runtime_ns"]:
                    rows.append(enriched)
    rows.sort(key=lambda r: (r["activation"], r["precision"], r["coeff_csv"]))
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def maybe_train(args, out: Path) -> None:
    if not args.train:
        return
    craq = Path(args.craq_sim).expanduser()
    fit = craq / "scripts" / "perf" / "fit.py"
    if not fit.exists():
        raise SystemExit(f"frontier_gbdt_dataset: craq-sim fit.py not found: {fit}")
    train_out = Path(args.train_out) if args.train_out else out.parent / "frontier_gbdt_model"
    cmd = [
        sys.executable,
        str(fit),
        "--table",
        str(out),
        "--target",
        "target_runtime_ns",
        "--out-dir",
        str(train_out),
        "--save-model",
        "--final-full-fit",
    ]
    if args.tune:
        cmd += ["--tune", "--tune-iters", str(args.tune_iters)]
    subprocess.run(cmd, check=True)
    if args.cv_folds > 1:
        run_cross_validation(out, craq, train_out, args.cv_folds, args.seed)


def abs_pct_error(actual, pred):
    import numpy as np

    return np.abs(pred - actual) / np.maximum(actual, 1e-9) * 100.0


def run_cross_validation(table_path: Path, craq: Path, out_dir: Path, folds: int, seed: int) -> None:
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    sys.path.insert(0, str(craq))
    from scripts.perf import fit as craq_fit

    table = pd.read_csv(table_path, sep="\t")
    X, y, sub = craq_fit.build_xy(table, "target_runtime_ns")
    if len(X) < max(20, folds):
        summary = {
            "table": str(table_path),
            "target": "target_runtime_ns",
            "rows": int(len(X)),
            "folds": folds,
            "skipped": "not enough usable rows",
        }
    else:
        idx = np.arange(len(X))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        splits = [part for part in np.array_split(idx, folds) if len(part)]
        rows = []
        pred_all = []
        for fold, te_idx in enumerate(splits):
            tr_idx = np.array([i for i in idx if i not in set(te_idx)], dtype=int)
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "seed": seed + fold,
                "nthread": 1,
                "max_depth": 8,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 2,
            }
            dtrain = xgb.DMatrix(Xtr, label=np.log1p(ytr), feature_names=list(Xtr.columns))
            dtest = xgb.DMatrix(Xte, feature_names=list(Xtr.columns))
            model = xgb.train(params, dtrain, num_boost_round=600, verbose_eval=False)
            pred = np.expm1(model.predict(dtest))
            ape = abs_pct_error(yte.to_numpy(), pred)
            for j, actual, predicted, err in zip(te_idx, yte.to_numpy(), pred, ape):
                pred_all.append(
                    {
                        "fold": fold,
                        "key": sub.iloc[int(j)].get("key", ""),
                        "nodeid": sub.iloc[int(j)].get("nodeid", ""),
                        "actual_runtime_ns": float(actual),
                        "pred_runtime_ns": float(predicted),
                        "abs_pct_error": float(err),
                    }
                )
            rows.append(
                {
                    "fold": fold,
                    "train_count": int(len(tr_idx)),
                    "holdout_count": int(len(te_idx)),
                    "median_abs_pct_error": float(np.median(ape)),
                    "p90_abs_pct_error": float(np.percentile(ape, 90)),
                }
            )
        all_err = np.array([r["abs_pct_error"] for r in pred_all], dtype=float)
        summary = {
            "table": str(table_path),
            "target": "target_runtime_ns",
            "rows": int(len(X)),
            "folds": int(len(splits)),
            "feature_count": int(X.shape[1]),
            "median_abs_pct_error": round(float(np.median(all_err)), 2),
            "p90_abs_pct_error": round(float(np.percentile(all_err, 90)), 2),
            "p99_abs_pct_error": round(float(np.percentile(all_err, 99)), 2),
            "folds_detail": rows,
        }
        pred_path = out_dir / "frontier_gbdt_cv_predictions.tsv"
        pd.DataFrame(pred_all).sort_values("abs_pct_error", ascending=False).to_csv(pred_path, sep="\t", index=False)
        summary["predictions"] = str(pred_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "frontier_gbdt_cv.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"frontier_gbdt_dataset: wrote CV summary -> {summary_path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frontier_csv", nargs="+", help="frontier_chip*.csv paths or globs")
    parser.add_argument("--coeff-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", action="store_true", help="train craq-sim GBDT after writing the TSV")
    parser.add_argument("--craq-sim", default=str(Path.home() / "craq-sim"))
    parser.add_argument("--train-out", default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-iters", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fit_dir = find_fitter()
    coeff_dir = args.coeff_dir or (fit_dir / "data" / "coefficients")
    rows = load_rows(expand_inputs(args.frontier_csv), coeff_dir, fit_dir)
    write_tsv(args.out, rows)
    print(f"frontier_gbdt_dataset: wrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    maybe_train(args, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
