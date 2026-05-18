#!/usr/bin/env python3
"""Download master JSON files for all traces matching a given architecture.

Reads model_tracer/trace_selection_registry.yaml, filters entries by
hardware.board_type AND hardware.card_count (and optionally hardware.device_series), and runs
  python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-trace <id> <out>
for each match. Output files are written to generated/model_traces/ named
<model1>_<model2>_..._trace<id>.json.

The database URL is passed via --db and exported as TTNN_OPS_DATABASE_URL to the child env.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("architecture", help="Board type to filter on, e.g. Blackhole, Wormhole (case-insensitive).")
    p.add_argument("card_count", type=int, help="Card count to filter on, e.g. 1, 4, 32.")
    p.add_argument(
        "device_series",
        nargs="?",
        default=None,
        help="Optional device series filter, e.g. p150b, p100a, n300, tt-galaxy-wh. "
        "Omit to include every device_series matching the given arch+card_count.",
    )
    p.add_argument(
        "--repo-root",
        required=True,
        type=Path,
        help="tt-metal repo root (where model_tracer/, tests/, and generated/model_traces/ live).",
    )
    p.add_argument(
        "--db",
        required=True,
        help="TTNN_OPS_DATABASE_URL value (exported to the child env for load_ttnn_ops_data_v2.py).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands instead of running them.")
    args = p.parse_args()
    assert args.db, "--db must be a non-empty TTNN_OPS_DATABASE_URL string"
    return args


def load_registry(registry_path):
    with registry_path.open() as f:
        data = yaml.safe_load(f)
    return data.get("registry", []) or []


def matches_hw(entry, arch, card_count, device_series):
    hw = entry.get("hardware") or {}
    bt = hw.get("board_type")
    cc = hw.get("card_count")
    ds = hw.get("device_series")
    if not (isinstance(bt, str) and bt.lower() == arch.lower()):
        return False
    if not (isinstance(cc, int) and cc == card_count):
        return False
    if device_series is not None:
        if not (isinstance(ds, str) and ds.lower() == device_series.lower()):
            return False
    return True


def output_name(entry):
    models = entry.get("models") or ["unknown"]
    suffix = f"_trace{entry['trace_id']}.json"
    # Cap the total filename well under the 255-byte Linux limit.
    max_models_chars = 200
    parts = [str(m) for m in models]
    joined = "_".join(parts)
    if len(joined) <= max_models_chars:
        return joined + suffix
    kept = []
    running = 0
    for i, name in enumerate(parts):
        added = len(name) + (1 if kept else 0)
        if running + added > max_models_chars:
            break
        kept.append(name)
        running += added
    dropped = len(parts) - len(kept)
    return "_".join(kept) + f"_plus{dropped}more" + suffix


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    registry_path = repo_root / "model_tracer" / "trace_selection_registry.yaml"
    loader_path = repo_root / "tests" / "sweep_framework" / "load_ttnn_ops_data_v2.py"
    out_dir = repo_root / "generated" / "model_traces"

    if not registry_path.exists():
        sys.exit(f"Registry not found: {registry_path}")
    if not loader_path.exists():
        sys.exit(f"Loader script not found: {loader_path}")

    db_url = args.db.strip()

    registry = load_registry(registry_path)
    matches = [e for e in registry if matches_hw(e, args.architecture, args.card_count, args.device_series)]

    ds_label = args.device_series if args.device_series is not None else "*"
    label = f"{args.architecture}/{ds_label}/x{args.card_count}"
    if not matches:
        print(f"No registry entries match '{label}'.")
        return

    print(f"Found {len(matches)} trace(s) for '{label}'.")

    if not args.dry_run:
        out_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["TTNN_OPS_DATABASE_URL"] = db_url

    for entry in matches:
        trace_id = entry["trace_id"]
        out_path = out_dir / output_name(entry)
        cmd = [
            sys.executable,
            str(loader_path.relative_to(repo_root)),
            "reconstruct-trace",
            str(trace_id),
            str(out_path),
        ]
        printable = " ".join(cmd)

        if args.dry_run:
            print(f"[dry-run] TTNN_OPS_DATABASE_URL=<from --db> {printable}")
            continue

        print(f"\n=== trace_id={trace_id} -> {out_path.name} ===")
        print(f"$ {printable}")
        result = subprocess.run(cmd, cwd=repo_root, env=env)
        if result.returncode != 0:
            print(f"  ! trace_id={trace_id} failed with exit code {result.returncode}", file=sys.stderr)


if __name__ == "__main__":
    main()
