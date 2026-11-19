# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qualify an explicit diagnostic snapshot without changing the selected source."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-length", type=int, default=8192)
    parser.add_argument("--continuation-length", type=int, default=33)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--oracle-cache")
    args = parser.parse_args()
    source = args.snapshot.resolve()
    assert source.is_file() and not args.output.exists(), "Use a snapshot and a fresh output path"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import stress_fused_decoder as stress

    stress.FusedDecoder = runpy.run_path(str(source))["FusedDecoder"]
    original_record = stress.record
    identity = dict(
        measured_decoder_source=str(source),
        measured_decoder_source_sha256=digest,
        candidate_selection="Explicit diagnostic snapshot; not final acceptance",
        acceptance_eligible=False,
    )

    def record(path):
        original_record(path)
        provenance = Path(str(path) + ".provenance.json")
        data = json.loads(provenance.read_text())
        data.update(identity)
        provenance.write_text(json.dumps(data, indent=2) + "\n")

    stress.record = record
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        stress.run(args)
    finally:
        stress.record = original_record
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, "Snapshot changed during measurement"
        if args.output.exists():
            data = json.loads(args.output.read_text())
            data.update(identity)
            args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
