# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the ACE-Step v1.5 prompt-to-wav demo across a device/duration/LM matrix,
parse the ``[ace_step_v1_5][perf] KEY METRICS`` block from the demo output, and
emit a BENCHMARK_RESULTS.md-style Markdown file with **P150 and BH_QB side by side**.

RTF matches upstream ACE-Step hardware-performance:
``RTF = audio_duration / wall`` (higher = faster). See
https://github.com/ace-step/ACE-Step#hardware-performance

Because a P150 and a BH_QB board usually live on *different hosts*, results are
persisted to a JSON store (``--results-json``). Run this once per host (each pass
fills in its own device column) pointing at the same JSON, then the Markdown is
regenerated from the merged store every time. Use ``--regenerate-only`` to rebuild
the Markdown from an existing JSON without running anything.

Default matrix:
    variant       = {acestep-v15-turbo, acestep-v15-base, acestep-v15-sft}
    lm_variant    = {0.6B, 1.7B, 4B}
    duration_sec  = {15, 30, 60}
    infer_steps / guidance_scale = per-variant (8/1 turbo, 50/7 base & sft),
                    overridable with --infer-steps / --guidance-scale
    --no-use-cot-caption
    devices       = {P150, BH_QB}

Example:
    # On the P150 host:
    python models/experimental/ace_step_v1_5/perf/run_benchmarks.py --devices P150
    # On the BH_QB host (same JSON copied over / on shared storage):
    python models/experimental/ace_step_v1_5/perf/run_benchmarks.py --devices BH_QB
    # Rebuild the side-by-side .md from whatever is in the JSON:
    python models/experimental/ace_step_v1_5/perf/run_benchmarks.py --regenerate-only
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DEVICES = ["P150", "BH_QB"]
DEFAULT_DURATIONS = [15, 30, 60]
DEFAULT_LM_VARIANTS = [
    "acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-4B",
]
DEFAULT_VARIANTS = [
    "acestep-v15-turbo",
    "acestep-v15-base",
    "acestep-v15-sft",
]
DEFAULT_PROMPT = "guitar, saxophone and prominent drums with clear kick and snare"

# Per-variant inference defaults. These mirror the demo's own auto-resolution
# (run_prompt_to_wav.py): turbo is distilled for 8 steps / guidance 1, while
# base and sft are trained for ~50 steps / guidance 7. Benchmarking base/sft at
# turbo's 8/1 would produce degenerate audio and unrepresentative timings, so
# each variant runs at its intended settings unless overridden on the CLI.
_VARIANT_INFER_DEFAULTS: Dict[str, Dict[str, float]] = {
    "acestep-v15-turbo": {"infer_steps": 8, "guidance_scale": 1.0},
    "acestep-v15-base": {"infer_steps": 50, "guidance_scale": 7.0},
    "acestep-v15-sft": {"infer_steps": 50, "guidance_scale": 7.0},
}
_FALLBACK_INFER_DEFAULT = {"infer_steps": 8, "guidance_scale": 1.0}


def variant_infer_defaults(variant: str) -> Dict[str, float]:
    """Intended (infer_steps, guidance_scale) for a DiT variant."""
    return dict(_VARIANT_INFER_DEFAULTS.get(variant, _FALLBACK_INFER_DEFAULT))


# Repo root = .../tt-metal (this file is at models/experimental/ace_step_v1_5/perf/)
REPO_ROOT = Path(__file__).resolve().parents[4]
DEMO_REL = "models/experimental/ace_step_v1_5/demo/run_prompt_to_wav.py"

# ---------------------------------------------------------------------------
# Output-parsing regexes  (lines are prefixed with "[ace_step_v1_5][perf]  ")
# ---------------------------------------------------------------------------

_RE = {
    "wall_time_s": re.compile(r"Wall Time\s+([0-9]+\.?[0-9]*)\s*s\b"),
    "lm_total_time_s": re.compile(r"LM Total Time\s+([0-9]+\.?[0-9]*)\s*s\b"),
    "dit_total_time_s": re.compile(r"DiT Total Time\s+([0-9]+\.?[0-9]*)\s*s\b"),
    "vae_decode_time_s": re.compile(r"VAE Decode Time\s+([0-9]+\.?[0-9]*)\s*s\b"),
    "tokens_per_sec": re.compile(r"Tokens/sec\s+([0-9]+\.?[0-9]*)\b"),
    "lm_num_tokens": re.compile(r"\(([0-9]+) new tokens\)"),
    # Upstream RTF line in KEY METRICS: "RTF  0.13×  audio_duration / wall ..."
    "rtf": re.compile(r"^\s*\[ace_step_v1_5\]\[perf\]\s+RTF\s+([0-9]+\.?[0-9]*)\s*[×x]", re.MULTILINE),
}


def _last_float(pattern: re.Pattern, text: str) -> Optional[float]:
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except (TypeError, ValueError):
        return None


def _last_int(pattern: re.Pattern, text: str) -> Optional[int]:
    v = _last_float(pattern, text)
    return int(v) if v is not None else None


def parse_metrics(log_text: str, *, duration_sec: float) -> Dict[str, Any]:
    """Extract the final KEY METRICS values from a demo run's stdout/stderr.

    RTF follows upstream ACE-Step: ``audio_duration / wall`` (higher = faster).
    """
    m: Dict[str, Any] = {}
    for key in ("wall_time_s", "lm_total_time_s", "dit_total_time_s", "vae_decode_time_s"):
        m[key] = _last_float(_RE[key], log_text)
    m["tokens_per_sec"] = _last_float(_RE["tokens_per_sec"], log_text)
    m["lm_num_tokens"] = _last_int(_RE["lm_num_tokens"], log_text)
    m["rtf"] = _last_float(_RE["rtf"], log_text)
    # Fallback if the KEY METRICS RTF line was missing: recompute from wall.
    wall = m.get("wall_time_s")
    if m["rtf"] is None and wall is not None and duration_sec > 0 and float(wall) > 0:
        m["rtf"] = round(float(duration_sec) / float(wall), 3)
    return m


# ---------------------------------------------------------------------------
# Command construction / execution
# ---------------------------------------------------------------------------


def build_command(
    *,
    python: str,
    variant: str,
    lm_variant: str,
    device: str,
    duration_sec: int,
    prompt: str,
    infer_steps: int,
    guidance_scale: float,
    out_wav: Path,
) -> List[str]:
    return [
        python,
        DEMO_REL,
        "--mesh-device",
        device,
        "--variant",
        variant,
        "--lm_variant",
        lm_variant,
        "--duration_sec",
        str(duration_sec),
        "--prompt",
        prompt,
        "--infer_steps",
        str(infer_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--no-use-cot-caption",
        "--out",
        str(out_wav),
    ]


def run_one(
    cmd: List[str],
    *,
    log_path: Path,
    cwd: Path,
    timeout: int,
) -> Dict[str, Any]:
    """Run a single demo invocation, tee output to log_path, return status dict."""
    env = dict(os.environ)
    env.setdefault("ACE_STEP_DEMO_PERF_LOG", "1")  # force KEY METRICS + RTF output
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now().isoformat(timespec="seconds")
    with open(log_path, "w") as lf:
        lf.write(f"# started={started}\n# cmd={shlex.join(cmd)}\n\n")
        lf.flush()
        try:
            proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
            return {"status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            lf.write("\n# TIMEOUT\n")
            return {"status": "timeout", "returncode": None}
        except Exception as exc:  # noqa: BLE001 - record any launch failure
            lf.write(f"\n# EXCEPTION: {exc}\n")
            return {"status": "error", "returncode": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Results store (JSON) — key = "<duration>|<lm_variant>", value = {device: record}
# ---------------------------------------------------------------------------


def _migrate_legacy_keys(store: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade legacy ``"<dur>|<lm>"`` run keys (single-variant era, always turbo)
    to the variant-scoped ``"<variant>|<dur>|<lm>"`` form used now."""
    runs = store.get("runs", {})
    migrated: Dict[str, Any] = {}
    changed = False
    for k, v in runs.items():
        if k.count("|") == 1:  # legacy: "<dur>|<lm>" -> assume the old default (turbo)
            k = f"acestep-v15-turbo|{k}"
            changed = True
        migrated[k] = v
    if changed:
        store["runs"] = migrated
    return store


def load_store(path: Path) -> Dict[str, Any]:
    if path.is_file():
        try:
            return _migrate_legacy_keys(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"[warn] {path} is not valid JSON; starting fresh", file=sys.stderr)
    return {"runs": {}, "meta": {}}


def save_store(path: Path, store: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2, sort_keys=True))


def cell_key(variant: str, duration: int, lm_variant: str) -> str:
    return f"{variant}|{duration}|{lm_variant}"


# ---------------------------------------------------------------------------
# Markdown generation (P150 vs BH_QB side by side)
# ---------------------------------------------------------------------------

_LM_LABEL = {
    "acestep-5Hz-lm-0.6B": "0.6B",
    "acestep-5Hz-lm-1.7B": "1.7B",
    "acestep-5Hz-lm-4B": "4B",
}

_VARIANT_LABEL = {
    "acestep-v15-turbo": "turbo",
    "acestep-v15-base": "base",
    "acestep-v15-sft": "sft",
}


def _fmt_s(v: Optional[float]) -> str:
    return f"{v:.2f} s" if isinstance(v, (int, float)) else "—"


def _fmt_tps(rec: Optional[Dict[str, Any]]) -> str:
    if not rec:
        return "—"
    m = rec.get("metrics") or {}
    tps = m.get("tokens_per_sec")
    if tps is None:
        return "—"
    ntok = m.get("lm_num_tokens")
    return f"{tps:.1f} tok/s ({ntok} tok)" if ntok is not None else f"{tps:.1f} tok/s"


def _metric_cell(rec: Optional[Dict[str, Any]], key: str, fmt=_fmt_s) -> str:
    if not rec:
        return "—"
    status = rec.get("status")
    if status not in ("ok", None):
        # Surface non-ok runs instead of a silently blank cell.
        return f"_({status})_"
    return fmt((rec.get("metrics") or {}).get(key))


def _rtf_cell(rec: Optional[Dict[str, Any]]) -> str:
    if not rec:
        return "—"
    v = (rec.get("metrics") or {}).get("rtf")
    return f"{v:.2f}×" if isinstance(v, (int, float)) else "—"


def generate_markdown(store: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    devices = cfg["devices"]
    durations = cfg["durations"]
    lm_variants = cfg["lm_variants"]
    variants = cfg["variants"]
    variant_settings = cfg.get("variant_settings", {})
    runs = store.get("runs", {})

    # Always show P150 then BH_QB when present; otherwise whatever devices exist.
    ordered_devices = [d for d in ["P150", "BH_QB"] if d in devices]
    for d in devices:
        if d not in ordered_devices:
            ordered_devices.append(d)

    lines: List[str] = []
    lines.append("<!-- Generated by perf/run_benchmarks.py — do not edit by hand -->")
    lines.append("")
    lines.append(f"# ACE-Step benchmark results — {' vs '.join(ordered_devices)} (side by side)")
    lines.append("")
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python {DEMO_REL} \\")
    lines.append(f"  --mesh-device <{'|'.join(ordered_devices)}> \\")
    lines.append(f"  --variant <{'|'.join(variants)}> \\")
    lines.append("  --lm_variant <acestep-5Hz-lm-0.6B|1.7B|4B> \\")
    lines.append("  --duration_sec <15|30|60> \\")
    lines.append(f'  --prompt "{cfg["prompt"]}" \\')
    lines.append("  --infer_steps <per-variant> \\")
    lines.append("  --guidance_scale <per-variant> \\")
    lines.append("  --no-use-cot-caption \\")
    lines.append("  --out /tmp/<variant>_<dur>_<lm>.wav")
    lines.append("```")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Prompt | {cfg['prompt']} |")
    lines.append(f"| Branch | ign/ACE_demo_modified |")
    lines.append("")
    # Per-variant inference settings (they differ: turbo is distilled for few steps).
    lines.append("Per-variant inference settings:")
    lines.append("")
    lines.append("| Variant | Steps | Guidance scale |")
    lines.append("| --- | --- | --- |")
    for v in variants:
        s = variant_settings.get(v, variant_infer_defaults(v))
        lines.append(f"| {v} | {int(s['infer_steps'])} | {float(s['guidance_scale']):g} |")
    lines.append("")
    lines.append(
        "> **RTF** matches upstream ACE-Step hardware-performance: "
        "`RTF = audio_duration / wall` (higher = faster; e.g. 27.27× ⇒ 60 s audio in ~2.2 s). "
        "See https://github.com/ace-step/ACE-Step#hardware-performance. "
        "Audio quality is a manual listen — fill in the last row after review."
    )
    lines.append("")

    metric_rows = [
        ("Wall Time", lambda rec: _metric_cell(rec, "wall_time_s")),
        ("LM Total Time", lambda rec: _metric_cell(rec, "lm_total_time_s")),
        ("DiT Total Time", lambda rec: _metric_cell(rec, "dit_total_time_s")),
        ("VAE Decode Time", lambda rec: _metric_cell(rec, "vae_decode_time_s")),
        (
            "Tokens/sec",
            lambda rec: _fmt_tps(rec)
            if (rec and rec.get("status") in ("ok", None))
            else _metric_cell(rec, "tokens_per_sec"),
        ),
        ("RTF", _rtf_cell),
        ("Audio GOOD/BAD", lambda rec: "TBD"),
    ]

    for variant in variants:
        v_label = _VARIANT_LABEL.get(variant, variant)
        lines.append(f"# Variant: {variant} (`{v_label}`)")
        lines.append("")
        for dur in durations:
            lines.append(f"## {dur}s generation — {v_label}")
            lines.append("")
            for lm in lm_variants:
                key = cell_key(variant, dur, lm)
                per_dev = runs.get(key, {})
                lines.append(f"### {variant} · lm {_LM_LABEL.get(lm, lm)} (`{lm}`)")
                lines.append("")
                header = "| Metric | " + " | ".join(ordered_devices) + " |"
                sep = "| --- | " + " | ".join(["---"] * len(ordered_devices)) + " |"
                lines.append(header)
                lines.append(sep)
                for name, fn in metric_rows:
                    cells = [fn(per_dev.get(dev)) for dev in ordered_devices]
                    lines.append(f"| {name} | " + " | ".join(cells) + " |")
                lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=DEFAULT_DEVICES, help="Mesh SKUs to run (e.g. P150 BH_QB).")
    ap.add_argument("--durations", nargs="+", type=int, default=DEFAULT_DURATIONS)
    ap.add_argument("--lm-variants", nargs="+", default=DEFAULT_LM_VARIANTS)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help="DiT variants to run (e.g. acestep-v15-turbo acestep-v15-base acestep-v15-sft).",
    )
    ap.add_argument(
        "--variant", default=None, help="Deprecated single-variant alias for --variants (kept for back-compat)."
    )
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument(
        "--infer-steps",
        type=int,
        default=None,
        help="Override inference steps for ALL variants. Default: per-variant " "(8 turbo, 50 base/sft).",
    )
    ap.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override DiT CFG guidance for ALL variants. Default: per-variant " "(1 turbo, 7 base/sft).",
    )
    ap.add_argument("--python", default=sys.executable, help="Python interpreter for the demo.")
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help="tt-metal repo root (cwd for the demo).")
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "models/experimental/ace_step_v1_5/perf/bench_out"),
        help="Directory for per-run logs and .wav outputs.",
    )
    ap.add_argument("--results-json", default=None, help="Merged results store (default: <out-dir>/results.json).")
    ap.add_argument(
        "--out-md", default=str(REPO_ROOT / "models/experimental/ace_step_v1_5/perf/BENCHMARK_RESULTS_P150_vs_BHQB.md")
    )
    ap.add_argument("--timeout", type=int, default=1800, help="Per-run timeout in seconds.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands; do not run.")
    ap.add_argument("--regenerate-only", action="store_true", help="Only rebuild the .md from the JSON store.")
    ap.add_argument(
        "--skip-existing", action="store_true", help="Skip matrix cells already present (status ok) in the store."
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"
    out_md = Path(args.out_md)
    repo_root = Path(args.repo_root)

    # --variant (singular, deprecated) overrides --variants when given.
    variants = [args.variant] if args.variant else list(args.variants)

    # Resolve per-variant (infer_steps, guidance_scale): CLI overrides win for all
    # variants; otherwise each variant uses its intended defaults.
    variant_settings: Dict[str, Dict[str, float]] = {}
    for v in variants:
        d = variant_infer_defaults(v)
        if args.infer_steps is not None:
            d["infer_steps"] = args.infer_steps
        if args.guidance_scale is not None:
            d["guidance_scale"] = args.guidance_scale
        variant_settings[v] = d

    cfg = {
        "devices": list(args.devices),
        "durations": list(args.durations),
        "lm_variants": list(args.lm_variants),
        "variants": variants,
        "variant_settings": variant_settings,
        "prompt": args.prompt,
    }

    store = load_store(results_json)
    store.setdefault("runs", {})
    store.setdefault("meta", {})
    store["meta"].update({"config": cfg})

    if args.regenerate_only:
        out_md.write_text(generate_markdown(store, cfg))
        print(f"[ok] regenerated {out_md} from {results_json}")
        return 0

    if not (repo_root / DEMO_REL).is_file():
        print(f"[error] demo not found at {repo_root / DEMO_REL}", file=sys.stderr)
        return 2

    total = len(args.devices) * len(variants) * len(args.durations) * len(args.lm_variants)
    idx = 0
    for device in args.devices:
        for variant in variants:
            v_label = _VARIANT_LABEL.get(variant, variant)
            settings = variant_settings[variant]
            infer_steps = int(settings["infer_steps"])
            guidance_scale = float(settings["guidance_scale"])
            for dur in args.durations:
                for lm in args.lm_variants:
                    idx += 1
                    key = cell_key(variant, dur, lm)
                    lm_short = _LM_LABEL.get(lm, lm)
                    tag = f"{device}_{v_label}_{dur}s_{lm_short}"
                    out_wav = out_dir / "wav" / f"{tag}.wav"
                    log_path = out_dir / "logs" / f"{tag}.log"
                    cmd = build_command(
                        python=args.python,
                        variant=variant,
                        lm_variant=lm,
                        device=device,
                        duration_sec=dur,
                        prompt=args.prompt,
                        infer_steps=infer_steps,
                        guidance_scale=guidance_scale,
                        out_wav=out_wav,
                    )

                    existing = store["runs"].get(key, {}).get(device)
                    if args.skip_existing and existing and existing.get("status") == "ok":
                        print(f"[{idx}/{total}] skip (already ok): {tag}")
                        continue

                    print(f"[{idx}/{total}] {tag}  (steps={infer_steps}, gs={guidance_scale:g})")
                    print("    " + shlex.join(cmd))
                    if args.dry_run:
                        continue

                    status = run_one(cmd, log_path=log_path, cwd=repo_root, timeout=args.timeout)
                    log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
                    metrics = parse_metrics(log_text, duration_sec=float(dur))
                    record = {
                        "device": device,
                        "duration_sec": dur,
                        "lm_variant": lm,
                        "variant": variant,
                        "infer_steps": infer_steps,
                        "guidance_scale": guidance_scale,
                        "status": status["status"],
                        "returncode": status.get("returncode"),
                        "log": str(log_path),
                        "wav": str(out_wav),
                        "metrics": metrics,
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    }
                    store["runs"].setdefault(key, {})[device] = record
                    save_store(results_json, store)  # persist after every run (crash-safe)
                    w = metrics.get("wall_time_s")
                    print(
                        f"    -> {status['status']}  wall={w if w is not None else 'n/a'}s  "
                        f"RTF={metrics.get('rtf')}"
                    )

    out_md.write_text(generate_markdown(store, cfg))
    print(f"\n[ok] results JSON: {results_json}")
    print(f"[ok] side-by-side markdown: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
