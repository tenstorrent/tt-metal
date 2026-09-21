"""Run the unchanged, hardware-qualified v86 stock reproducer from any directory."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tt-metal", required=True, type=Path, help="Built tt-metal checkout")
    parser.add_argument("--device", default=11, type=int, help="UMD physical device ID; verify PCI identity first")
    parser.add_argument("--iterations", default=100000, type=int)
    parser.add_argument("--output", required=True, type=Path, help="New directory for report and fault capture")
    parser.add_argument("--keep-going", action="store_true", help="Retain all faults; otherwise stop at the first")
    parser.add_argument("--library-path", action="append", default=[], help="Additional native-library directory")
    parser.add_argument("--dry-run", action="store_true", help="Verify files and print command; do not import TTNN")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    integrity = verify(root)
    checkout, output = args.tt_metal.resolve(), args.output.resolve()
    if not (checkout / "ttnn/ttnn/__init__.py").is_file():
        parser.error("--tt-metal must point to a tt-metal checkout")
    if output.exists() or args.iterations <= 0 or args.device < 0:
        parser.error("Use a new output directory, positive iterations and a nonnegative device ID")
    for key in ("TT_METAL_SLOW_DISPATCH_MODE", "TT_VISIBLE_DEVICES"):
        if key in os.environ:
            parser.error(f"Unset {key} for the recorded fast-dispatch, unfiltered-device configuration")
    env = dict(os.environ)
    env["TT_METAL_HOME"] = str(checkout)
    env["PYTHONPATH"] = os.pathsep.join([str(checkout), str(checkout / "ttnn"), env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(checkout / "build/lib"), *args.library_path, env.get("LD_LIBRARY_PATH", "")]
    )
    env.setdefault("TT_METAL_CACHE", str(output.with_name(output.name + "-kernel-cache")))
    command = [
        sys.executable,
        str(root / "device_read_repro_v86.py"),
        "--fixture",
        str(root / "55041-read-fixture-v86.pt.gz"),
        "--device",
        str(args.device),
        "--iterations",
        str(args.iterations),
        "--route",
        "original",
        "--input-memory",
        "dram",
        "--output",
        str(output),
    ]
    if args.keep_going:
        command.append("--keep-going")
    metadata = dict(
        command=command,
        cwd=str(checkout),
        device=args.device,
        integrity=integrity,
        slow_dispatch=False,
        device_filter=None,
        dry_run=args.dry_run,
    )
    try:
        metadata["checkout_head"] = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["checkout_head"] = None
    print(json.dumps(metadata, indent=2), flush=True)
    if args.dry_run:
        return 0
    code = subprocess.run(command, cwd=checkout, env=env).returncode
    verify(root)
    # An import failure also exits Python with 1; require the actual fault report.
    path = output / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    qualified = report.get("stage") == "complete" and report.get("source_unchanged_end") is True
    qualified = qualified and code in (0, 1) and code == int(bool(report.get("faults")))
    metadata.update(child_exit_code=code, result_qualified=qualified)
    if output.is_dir():
        (output / "launcher.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not qualified:
        print(
            "Setup/runtime failure: inspect the log and report.json; this is not a qualified numerical result.",
            file=sys.stderr,
        )
        return 2
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Reproducer setup error: {exc}", file=sys.stderr)
        raise SystemExit(2)
