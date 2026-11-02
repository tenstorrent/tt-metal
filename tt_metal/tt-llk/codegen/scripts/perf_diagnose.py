#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Measure-only wave diagnosis of the kernel in the tree (one variant, one run type).

Runs the private ``perf-capture`` (through ``.claude/scripts/llk_wave_debug.py``) on one
perf-test variant, judges the breakdown with ``perf_headroom.py`` against the CSV that
same run produced, and prints ONE line:

    PERF_DIAG status=<ok|not_reconciled|failed|skipped> ... dir=<out-dir>

It never changes the kernel and never fails the caller (exit 0 on every outcome).
Space: perf-capture deletes the ~1 GB emulator dump; this script deletes the 2 MB
wave file after the verdict (``--keep-waves`` keeps it), cleans ``perf_data/`` in
the worktree, and leaves ~20 KB of JSON/CSV in ``--out-dir``. A capture is skipped
when the emulator's filesystem has less than ``--min-free-gb`` free.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PUBLIC_RATES = _HERE.parent / "references" / "quasar_issue_rates.json"
_PRIVATE_ENTRY = Path("tools/llk_wave_debug/codegen/scripts/llk_wave_debug.py")
_PRIVATE_RATES = Path(
    "tools/llk_wave_debug/codegen/llk_wave_debug/rates/quasar_issue_rates.private.json"
)
TOP_CAUSES = 5


def _headroom():
    spec = importlib.util.spec_from_file_location(
        "perf_headroom", _HERE / "perf_headroom.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def _newest_post_csv(llk: Path, module: str) -> Path | None:
    stem = Path(module).stem
    hits = sorted(
        (llk / "perf_data").rglob(f"{stem}.post.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return hits[0] if hits else None


def _finish(out: Path, diag: dict) -> int:
    out.mkdir(parents=True, exist_ok=True)
    diag["dir"] = str(out)
    (out / "diag.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n")
    fields = [f"status={diag['status']}"]
    if diag.get("run_type"):
        fields.append(f"run_type={diag['run_type']}")
    if "measured" in diag:
        fields += [
            f"measured={diag['measured']}",
            f"floor={diag['floor']}",
            f"stall_headroom={diag['stall_headroom_pct']}%",
            f"reconcile={diag['reconcile']['delta_pct']:+.2f}%",
            "top=" + ",".join(f"{name}:{value}" for name, value in diag["top_causes"]),
        ]
    if diag.get("reason"):
        fields.append(f"reason={diag['reason']!r}")
    fields.append(f"dir={out}")
    print("PERF_DIAG " + " ".join(fields))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--llk",
        type=Path,
        required=True,
        help="tt-llk dir of the worktree (kernel in the tree)",
    )
    p.add_argument(
        "--module",
        required=True,
        help="perf test file, e.g. perf_eltwise_unary_sfpu_quasar.py",
    )
    p.add_argument("--test-id", required=True, help="the one variant to capture")
    p.add_argument("--run-type", required=True, help="MATH_ISOLATE, L1_TO_L1, ...")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--launcher", type=Path, help="default: <llk>/.claude/scripts/llk_wave_debug.py"
    )
    p.add_argument(
        "--private-root",
        type=Path,
        default=Path(
            os.environ.get("LLK_CODEGEN_PRIVATE_ROOT", "/proj_sw/user_dev/llk_code_gen")
        ),
    )
    p.add_argument(
        "--space-path",
        type=Path,
        default=Path(
            os.environ.get("AETHER_WORKSPACE")
            or f"/proj_sw/user_dev/{os.environ.get('USER', '')}/work/aether"
        ),
        help="filesystem the emulator dump lands on (checked for --min-free-gb)",
    )
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("--keep-waves", action="store_true")
    p.add_argument("--timeout", type=int, default=3 * 3600)
    args = p.parse_args(argv)

    out = args.out_dir
    shutil.rmtree(out, ignore_errors=True)
    base = {"run_type": args.run_type, "test_id": args.test_id, "module": args.module}

    if not (args.private_root / _PRIVATE_ENTRY).is_file():
        return _finish(
            out,
            {
                **base,
                "status": "skipped",
                "reason": f"private wave tool not found under {args.private_root}",
            },
        )
    free = _free_gb(args.space_path)
    if free < args.min_free_gb:
        return _finish(
            out,
            {
                **base,
                "status": "skipped",
                "reason": f"only {free:.1f} GB free at {args.space_path} (< {args.min_free_gb} GB)",
            },
        )

    launcher = args.launcher or (args.llk / ".claude" / "scripts" / "llk_wave_debug.py")
    env = dict(
        os.environ,
        LLK_CODEGEN_PRIVATE_ROOT=str(args.private_root),
        LLK_CODEGEN_TT_LLK=str(args.llk),
    )
    cmd = [
        sys.executable,
        str(launcher),
        "perf-capture",
        "--test",
        args.module,
        "--test-id",
        args.test_id,
        "--worktree",
        str(args.llk),
        "--run-type",
        args.run_type,
        "--out-dir",
        str(out),
    ]
    shutil.rmtree(args.llk / "perf_data", ignore_errors=True)
    try:
        done = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=args.timeout
        )
        rc, err = done.returncode, done.stderr
    except subprocess.TimeoutExpired:
        rc, err = 124, f"perf-capture timed out after {args.timeout}s"
    csv = _newest_post_csv(args.llk, args.module)
    out.mkdir(parents=True, exist_ok=True)
    if csv is not None:
        shutil.copyfile(csv, out / "perf.post.csv")
    shutil.rmtree(args.llk / "perf_data", ignore_errors=True)
    (out / "perf_capture.stderr.txt").write_text(err[-4000:])

    report_path = out / f"perf_{args.run_type}.json"
    try:
        if rc == 2 or not report_path.is_file():
            last = next(
                (l for l in reversed(err.splitlines()) if l.strip()),
                f"perf-capture exited {rc}",
            )
            raise ValueError(last.strip()[:300])
        if csv is None:
            raise ValueError("the capture run left no perf CSV")
        ph = _headroom()
        rates = [_PUBLIC_RATES]
        if (args.private_root / _PRIVATE_RATES).is_file():
            rates.append(args.private_root / _PRIVATE_RATES)
        names_yaml = getattr(ph, "_DEFAULT_ASSEMBLY", None)
        names = (
            ph.load_opcode_names(names_yaml)
            if names_yaml and Path(names_yaml).exists()
            else None
        )
        verdict = ph.headroom_verdict(
            json.loads(report_path.read_text()),
            ph._select_row(out / "perf.post.csv", []),
            ph.load_rates(rates),
            run_type=args.run_type,
            opcode_names=names,
        )
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        return _finish(out, {**base, "status": "failed", "reason": str(exc)})
    finally:
        if not args.keep_waves:
            (out / "waves.vcd.gz").unlink(missing_ok=True)

    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    bottleneck = verdict["units"][verdict["bottleneck"]]
    return _finish(
        out,
        {
            **base,
            "status": "ok" if verdict["reconcile"]["ok"] else "not_reconciled",
            "measured": verdict["reconcile"]["csv_per_tile"],
            "floor": bottleneck["floor_per_tile"],
            "stall_headroom_pct": verdict["stall_headroom_pct"],
            "reconcile": verdict["reconcile"],
            "bottleneck": verdict["bottleneck"],
            "top_causes": verdict["causes"][:TOP_CAUSES],
            "capture_rc": rc,
        },
    )


if __name__ == "__main__":
    sys.exit(main())
