# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Preserve immutable report bytes in deterministic, repository-safe gzip files.

Originals stay available locally. This only packages existing evidence; it never
runs a device, changes measurements, or formats historical source snapshots.
"""

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

DOC = Path(__file__).resolve().parent
ROOT = DOC.parents[4]
START = "# BEGIN byte-preserving evidence archives\n"
END = "# END byte-preserving evidence archives\n"
# Executed historical scripts retain their recorded bytes, including formatting.
HISTORICAL_SCRIPTS = {
    "run_v6_remaining.py",
    "run_validated_v6.py",
    "summarize_validated_v6.py",
    "run_minimal_acceptance.py",
    "run_prefill_resume_controls.py",
    "run_minimal_stress.py",
    "run_trace_allocation_controls.py",
    "run_validated_v5.py",
    "run_prefill_boundaries.py",
    "run_operator_audits.py",
    "summarize_validated_v4.py",
    "summarize_validated_v5.py",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ignore = DOC / ".gitignore"
    original_ignore = ignore.read_text()
    if START in original_ignore:
        before, remainder = original_ignore.split(START, 1)
        _, after = remainder.split(END, 1)
        original_ignore = before + after
    ignore.write_text(original_ignore)
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", str(DOC)], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    # The repository ignores *.log; retain compact correctness/recovery logs
    # that the final evidence explicitly relies on, without raw profiler dumps.
    log_patterns = (
        "validated_v8*.log",
        "prefill_boundary_v8*.log",
        "prefill_router_v8/*.log",
        "tracy/*/*perf_report*.csv",
        "tracy/actual_*/*replays.csv",
        "tracy/actual_*/decode_one_replay.csv",
        "pytest_validated_v8.log",
        "tight_cache_v8*.log",
        "stress_validated_v8_defaults/**/*.log",
        "stress_validated_v8_defaults.log",
        "validated_v7*.log",
        "prefill_qkv_*v7*.log",
        "precommit_final*.log",
        "black_final*.log",
        "reader_cpu_final.log",
        "pytest_validated_v7.log",
        "tight_cache_v7*.log",
        "validated_v5*.log",
        "pytest_validated_v5.log",
        "trace_alloc_v[56]*.log",
        "tight_cache_v6*.log",
        "minimal_hifi2_acceptance_v6/*.log",
        "stress_validated_v7_defaults/**/*.log",
        "stress_validated_v7_defaults.log",
        "stress_validated_v5_defaults/**/*.log",
        "stress_validated_v5_defaults.log",
    )
    names.extend(str(path.relative_to(ROOT)) for pattern in log_patterns for path in DOC.glob(pattern))
    records = []
    for name in sorted(set(filter(None, names))):
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or path.suffix in {".gz", ".patch"}:
            continue
        data = path.read_bytes()
        whitespace = path.suffix in {".json", ".txt"} and (
            any(line.rstrip(b" \t\r") != line for line in data.split(b"\n"))
            or bool(data and (not data.endswith(b"\n") or data.endswith(b"\n\n")))
        )
        oversized = len(data) > 500 * 1024
        historical_script = path.parent == DOC and path.name in HISTORICAL_SCRIPTS
        execution_log = path.suffix == ".log"
        execution_table = path.suffix == ".csv"
        if not whitespace and not oversized and not historical_script and not execution_log and not execution_table:
            continue
        packed = gzip.compress(data, compresslevel=9, mtime=0)
        assert gzip.decompress(packed) == data
        assert len(packed) <= 500 * 1024, f"Archive remains too large: {path}"
        archive = path.with_name(path.name + ".gz")
        archive.write_bytes(packed)
        relative = str(path.relative_to(DOC))
        records.append(
            dict(
                original=relative,
                original_sha256=sha(data),
                original_bytes=len(data),
                archive=relative + ".gz",
                archive_sha256=sha(packed),
                archive_bytes=len(packed),
                reason="preserve exact evidence bytes; "
                + (
                    "report exceeds repository500KiB limit"
                    if oversized
                    else "historical executed source snapshot"
                    if historical_script
                    else "compact correctness/recovery log ignored by repository"
                    if execution_log
                    else "compact profiler CSV ignored by repository"
                    if execution_table
                    else "immutable report whitespace"
                ),
            )
        )
    ignore.write_text(
        original_ignore.rstrip() + "\n\n" + START + "".join("/" + row["original"] + "\n" for row in records) + END
    )
    manifest = dict(
        format="gzip; deterministic mtime0; exact byte-for-byte reconstruction",
        restore='From this directory: python -c "import gzip,json,pathlib; '
        "[(pathlib.Path(r['original']).write_bytes(gzip.decompress(pathlib.Path(r['archive']).read_bytes()))) "
        "for r in json.loads(pathlib.Path('evidence_archives.json').read_text())['artifacts']]\"",
        note="Originals remain local. Unpack before running CPU audits that verify historical file hashes.",
        artifacts=records,
    )
    (DOC / "evidence_archives.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Preserved {len(records)} immutable evidence files in verified gzip archives")


if __name__ == "__main__":
    main()
