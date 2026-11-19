# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Benchmark an explicitly named fused candidate with the standard layer harness.

This diagnostic never selects a candidate for production. Each invocation runs
one candidate in a fresh process; the result records its exact source hash.
"""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--prefill-repeats", type=int, default=10)
    parser.add_argument("--decode-repeats", type=int, default=60)
    args = parser.parse_args()
    source = args.snapshot.resolve()
    assert source.is_file() and not Path(args.output).exists(), "Use a snapshot and a fresh output path"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    loader = importlib.machinery.SourceFileLoader("measured_fused_candidate", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    candidate = importlib.util.module_from_spec(spec)
    loader.exec_module(candidate)
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import provenance, run_decoder

    run_decoder.FunctionalDecoder = candidate.FusedDecoder
    identity = dict(
        measured_decoder_source=str(source),
        measured_decoder_source_sha256=digest,
        candidate_selection="Explicit diagnostic snapshot; not production selection",
        acceptance_eligible=False,
    )
    original_record = provenance.record

    def record(path):
        original_record(path)
        target = Path(str(path) + ".provenance.json")
        value = json.loads(target.read_text())
        value.update(identity)
        target.write_text(json.dumps(value, indent=2) + "\n")

    provenance.record = record
    try:
        run_decoder.run(
            argparse.Namespace(
                layer=args.layer,
                length=args.length,
                context=args.context,
                seed=42,
                chunk_size=None,
                output=args.output,
                compare_functional=False,
                profile=False,
                prefill_repeats=args.prefill_repeats,
                decode_repeats=args.decode_repeats,
            )
        )
    finally:
        provenance.record = original_record
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        if Path(args.output).exists():
            path = Path(args.output)
            value = json.loads(path.read_text())
            value.update(identity)
            path.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
