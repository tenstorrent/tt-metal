# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
from pathlib import Path

from .provenance import provenance as functional_provenance


def provenance():
    p = functional_provenance()
    root = Path(__file__).resolve().parents[1]
    for name in [
        "tt/fused_decoder.py",
        "tests/fused_coverage.py",
        "tests/fused_profile.py",
        "tests/fused_expert_candidate.py",
        "tests/fused_final_candidates.py",
        "tests/fusion_candidates.py",
        "tests/fused_decoder_checks.py",
        "tests/fused_context_edges.py",
        "tests/test_fused_decoder.py",
    ]:
        p["source_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    p["environment"].update({n: os.environ.get(n) for n in ["FUSIONS", "FUSION_TAG", "FUSION_IMPL"]})
    snapshots = root / "doc/fused_decoder/source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    for name, digest in p["source_sha256"].items():
        target = snapshots / (digest + ".py.txt")
        if not target.exists():
            target.write_bytes((root / name).read_bytes())
    return p
