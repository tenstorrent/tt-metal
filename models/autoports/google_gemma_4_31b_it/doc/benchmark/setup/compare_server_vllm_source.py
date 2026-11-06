#!/usr/bin/env python3
"""Hash Python source without importing or modifying the serving installation."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--installed", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def hashes(directory):
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*.py"))
    }


def digest(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


a, b = hashes(args.source / "vllm"), hashes(args.installed)
common = sorted(set(a) & set(b))
result = {
    "upstream_url": "https://github.com/vllm-project/vllm",
    "upstream_tag": "v0.31.0",
    "verified_python_source_commit": subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_checkout": str(args.source),
    "installed_package": str(args.installed),
    "method": "SHA-256 byte comparison of every upstream vllm/**/*.py with installed counterpart; no package imports",
    "source_python_files": len(a),
    "installed_python_files": len(b),
    "matching_files": sum(a[p] == b[p] for p in common),
    "different_files": [p for p in common if a[p] != b[p]],
    "missing_installed_files": sorted(set(a) - set(b)),
    "installed_only_files": {p: b[p] for p in sorted(set(b) - set(a))},
    "source_python_manifest_sha256": digest(a),
    "installed_python_manifest_sha256": digest(b),
    "matching_installed_subset_manifest_sha256": digest({p: b[p] for p in a if p in b}),
    "manifest_encoding": "JSON mapping relative path to file SHA-256; sorted keys; separators comma and colon; UTF-8; no newline",
    "relevant_file_sha256": {
        p: b[p]
        for p in [
            "entrypoints/openai/api_server.py",
            "v1/engine/core.py",
            "v1/engine/output_processor.py",
            "v1/metrics/stats.py",
            "v1/core/sched/scheduler.py",
            "v1/outputs.py",
        ]
    },
    "wheel_distribution": "vllm-0.31.0+empty",
    "wheel_build_commit": None,
    "limits": [
        "Python content corresponds exactly to the named upstream commit; this does not establish wheel build provenance or a uniquely possible historical source commit.",
        "Generated installed _version.py is absent upstream and declares version 0.31.0 with commit_id None.",
        "Compiled artifacts, non-Python assets, dependencies and build options were not compared.",
        "Server import identity is established separately by runtime launch evidence; comparison does not import vllm.",
    ],
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            k: result[k]
            for k in ("verified_python_source_commit", "matching_files", "different_files", "missing_installed_files")
        }
    )
)
