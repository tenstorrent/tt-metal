# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Snapshot source provenance without accessing credentials or user environment."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    selected = [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.cpp")]
    for rel in (
        "experiments/sdpa-l2/bfp4-lofi-v2",
        "experiments/sdpa-l2/bf16-denom-pair-v3/candidate",
        "experiments/sdpa-l2/hybrid-mixed-v1/candidate",
    ):
        selected += [p for p in (root / rel).rglob("*") if p.suffix in (".py", ".hpp", ".cpp", ".h")]
    selected += [
        root / "experiments/sdpa-l2/flux2-frontier-v1/device_attention.py",
        root / "models/tt_dit/models/transformers/wan2_2/attention_wan.py",
        root / "models/tt_dit/pipelines/wan/pipeline_wan.py",
    ]
    data = dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected},
    )
    with args.output.open("x") as handle:
        json.dump(data, handle, indent=2)


if __name__ == "__main__":
    main()
