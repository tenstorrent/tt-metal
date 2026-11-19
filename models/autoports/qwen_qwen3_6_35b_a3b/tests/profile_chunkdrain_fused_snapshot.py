# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile a fused snapshot, draining events after each existing prefill chunk.

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
    loader = importlib.machinery.SourceFileLoader("profiled_fused_candidate", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    candidate = importlib.util.module_from_spec(spec)
    loader.exec_module(candidate)
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import provenance, run_decoder

    original_forward = candidate.FusedDecoder._forward

    def forward_with_profile_drain(self, *call_args, **kwargs):
        result = original_forward(self, *call_args, **kwargs)
        if not kwargs["decode"]:
            # Profiling-only synchronization after an existing logical chunk.
            # No model arithmetic, padding, plan, or data path changes.
            import ttnn

            ttnn.synchronize_device(self.mesh_device)
            ttnn.ReadDeviceProfiler(self.mesh_device)
        return result

    candidate.FusedDecoder._forward = forward_with_profile_drain
    run_decoder.FunctionalDecoder = candidate.FusedDecoder
    identity = dict(
        measured_decoder_source=str(source),
        measured_decoder_source_sha256=digest,
        candidate_selection="Explicit diagnostic snapshot; not production selection",
        acceptance_eligible=False,
        profiling=True,
        profile_event_drain="after every existing prefill _forward chunk; additional standard phase/replay drains retained",
        profile_wrapper_source=str(Path(__file__).resolve()),
        profile_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
                profile=True,
                prefill_repeats=args.prefill_repeats,
                decode_repeats=args.decode_repeats,
            )
        )
    finally:
        provenance.record = original_record
        candidate.FusedDecoder._forward = original_forward
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        if Path(args.output).exists():
            path = Path(args.output)
            value = json.loads(path.read_text())
            value.update(identity)
            path.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
