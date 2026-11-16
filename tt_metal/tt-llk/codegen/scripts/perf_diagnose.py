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
wave file after the verdict (``--keep-waves`` keeps it) and leaves ~20 KB of JSON/CSV
in ``--out-dir``. A capture is skipped when the emulator's filesystem has less than
``--min-free-gb`` free.

The capture's perf report goes to its own ``<llk>/perf_data/runs/perf-diagnose-*``
(``PERF_RUN_TAG``), which is removed afterwards; other runs under ``perf_data/`` are
never touched. ``--out-dir`` is replaced only when this script wrote it before.
On ``--timeout`` the whole capture process group (run_test.sh, the emulator) is
stopped and the emulator dump this host left is deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import perf_headroom as ph

_HERE = Path(__file__).resolve().parent
_PUBLIC_RATES = _HERE.parent / "references" / "quasar_issue_rates.json"
_PRIVATE_ENTRY = Path("tools/llk_wave_debug/codegen/scripts/llk_wave_debug.py")
_PRIVATE_RATES = Path(
    "tools/llk_wave_debug/codegen/llk_wave_debug/rates/quasar_issue_rates.private.json"
)
_MARKER = ".perf_diagnose"
TOP_CAUSES = 5
STDERR_TAIL = 4000
STOP_GRACE_SECS = 30


def _free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def _finish(out: Path, diag: dict) -> int:
    diag["dir"] = str(out)
    (out / "diag.json").write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n")
    fields = [f"status={diag['status']}"]
    if diag.get("run_type"):
        fields.append(f"run_type={diag['run_type']}")
    if "measured" in diag:
        bound = diag["bound"]
        hr = diag["stall_headroom_pct"]
        fields += [
            f"measured={diag['measured']}",
            f"floor={diag['floor']}",
            f"bound={bound['kind']}({bound['thread']})",
            f"stall_headroom={'n/a' if hr is None else f'{hr}%'}",
            f"reconcile={diag['reconcile']['delta_pct']:+.2f}%",
            "top=" + ",".join(f"{name}:{value}" for name, value in diag["top_causes"]),
        ]
    if diag.get("reason"):
        fields.append(f"reason={diag['reason']!r}")
    fields.append(f"dir={out}")
    print("PERF_DIAG " + " ".join(fields))
    return 0


def _claim_out_dir(out: Path) -> str | None:
    """Empty ``out`` for a new diagnosis; a reason instead when it is someone else's dir."""

    if out.is_dir() and any(out.iterdir()):
        if not ((out / _MARKER).exists() or (out / "diag.json").exists()):
            return f"{out} is not empty and was not written by perf_diagnose.py; not deleting it"
        shutil.rmtree(out)
    elif out.exists() and not out.is_dir():
        return f"{out} exists and is not a directory"
    out.mkdir(parents=True, exist_ok=True)
    (out / _MARKER).touch()
    return None


def _preflight(args: argparse.Namespace) -> str | None:
    """Why the capture cannot start, or None."""

    if not (args.private_root / _PRIVATE_ENTRY).is_file():
        return f"private wave tool not found under {args.private_root}"
    free = _free_gb(args.space_path)
    if free < args.min_free_gb:
        return f"only {free:.1f} GB free at {args.space_path} (< {args.min_free_gb} GB)"
    return None


# --- emulator dumps (same ownership rule as perf-capture's own cleanup) --------


def _zruns(space: Path) -> set[Path]:
    return set(space.glob("verification/emu/emu_out/*/*/zrun.*"))


def _owned_dump(path: Path, host: str) -> bool:
    """A wave dump whose run called back to this host."""

    if not (path / "waves.ztdb").exists():
        return False
    try:
        env = (path / "env.log").read_text(errors="replace")
    except OSError:
        return False
    pattern = rf"^NNG_SOCKET_ADDR=tcp://{re.escape(host)}:\d+"
    return re.search(pattern, env, re.M) is not None


def _remove_new_dumps(space: Path, before: set[Path]) -> list[Path]:
    host = socket.gethostname().split("-special-", 1)[0]
    removed = []
    for path in sorted(_zruns(space) - before):
        if path.is_dir() and _owned_dump(path, host):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed


# --- capture -------------------------------------------------------------------


def _stop_group(proc: subprocess.Popen) -> None:
    """SIGTERM the capture's process group, SIGKILL whatever is left after a grace period."""

    for sig, grace in ((signal.SIGTERM, STOP_GRACE_SECS), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def _capture(args: argparse.Namespace, out: Path, run_tag: str) -> int:
    """Run perf-capture in its own process group; stderr goes to perf_capture.stderr.txt."""

    launcher = args.launcher or (args.llk / ".claude" / "scripts" / "llk_wave_debug.py")
    env = dict(
        os.environ,
        LLK_CODEGEN_PRIVATE_ROOT=str(args.private_root),
        LLK_CODEGEN_TT_LLK=str(args.llk),
        PERF_RUN_TAG=run_tag,
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
    dumps_before = _zruns(args.space_path)
    with open(out / "perf_capture.stderr.txt", "w") as err:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=err,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            _stop_group(proc)
            removed = _remove_new_dumps(args.space_path, dumps_before)
            err.write(
                f"\nperf-capture timed out after {args.timeout}s; process group stopped, "
                f"removed {[str(p) for p in removed] or 'no'} emulator dump(s)\n"
            )
            return 124


def _stderr_tail(out: Path) -> str:
    """Trim perf_capture.stderr.txt to its last STDERR_TAIL bytes and return them."""

    path = out / "perf_capture.stderr.txt"
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - STDERR_TAIL))
        tail = handle.read().decode(errors="replace")
    path.write_text(tail)
    return tail


def _latest_target(llk: Path) -> str | None:
    latest = llk / "perf_data" / "latest"
    return os.readlink(latest) if latest.is_symlink() else None


def _take_perf_csv(
    llk: Path, module: str, run_tag: str, out: Path, latest_before: str | None
) -> Path | None:
    """Copy this capture's .post.csv into ``out``, then remove the capture's perf_data run.

    Only ``perf_data/runs/<run_tag>`` is removed, and ``perf_data/latest`` is pointed back
    where it was; perf_data/ and runs/ go only if that leaves them empty.
    """

    perf_data = llk / "perf_data"
    run_dir = perf_data / "runs" / run_tag
    hits = sorted(run_dir.rglob(f"{Path(module).stem}.post.csv"))
    taken = None
    if hits:
        taken = out / "perf.post.csv"
        shutil.copyfile(hits[0], taken)
    shutil.rmtree(run_dir, ignore_errors=True)
    latest = perf_data / "latest"
    if latest.is_symlink() and Path(os.readlink(latest)).name == run_tag:
        latest.unlink()
        if latest_before:
            latest.symlink_to(latest_before, target_is_directory=True)
    for empty in (perf_data / "runs", perf_data):
        try:
            empty.rmdir()
        except OSError:
            pass
    return taken


def _judge(args: argparse.Namespace, out: Path, rc: int, csv: Path | None, err: str):
    report_path = out / f"perf_{args.run_type}.json"
    if rc == 2 or not report_path.is_file():
        last = next(
            (l for l in reversed(err.splitlines()) if l.strip()),
            f"perf-capture exited {rc}",
        )
        raise ValueError(last.strip()[:300])
    if csv is None:
        raise ValueError("the capture run left no perf CSV")
    rates = [_PUBLIC_RATES]
    if (args.private_root / _PRIVATE_RATES).is_file():
        rates.append(args.private_root / _PRIVATE_RATES)
    return ph.headroom_verdict(
        json.loads(report_path.read_text()),
        ph.select_row(csv, []),
        ph.load_rates(rates),
        run_type=args.run_type,
        opcode_names=ph.default_opcode_names(),
    )


def _parse(argv: list[str] | None) -> argparse.Namespace:
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
        help="Aether workspace the emulator dump lands on (checked for --min-free-gb)",
    )
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("--keep-waves", action="store_true")
    p.add_argument("--timeout", type=int, default=3 * 3600)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    out = args.out_dir
    base = {"run_type": args.run_type, "test_id": args.test_id, "module": args.module}

    refused = _claim_out_dir(out)
    if refused:
        # Nothing may be written into a directory this script does not own.
        print(f"PERF_DIAG status=failed reason={refused!r} dir={out}")
        return 0
    why_not = _preflight(args)
    if why_not:
        return _finish(out, {**base, "status": "skipped", "reason": why_not})

    run_tag = f"perf-diagnose-{os.getpid()}-{int(time.time())}"
    latest_before = _latest_target(args.llk)
    rc = _capture(args, out, run_tag)
    csv = _take_perf_csv(args.llk, args.module, run_tag, out, latest_before)
    err = _stderr_tail(out)
    try:
        verdict = _judge(args, out, rc, csv, err)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the contract is one PERF_DIAG line on every outcome
        reason = (
            str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
        )
        return _finish(out, {**base, "status": "failed", "reason": reason})
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
            "bound": verdict["bound"],
            "occupancy_pct": verdict["occupancy_pct"],
            "top_causes": verdict["causes"][:TOP_CAUSES],
            "capture_rc": rc,
        },
    )


if __name__ == "__main__":
    sys.exit(main())
