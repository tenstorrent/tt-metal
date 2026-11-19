# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit precision and geometry overrides; the normal acceptance runner is reused."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import validate_optimized
from models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_history import HistoryCandidate


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--policy", default="{}")
    ap.add_argument("--policy-file")
    args, rest = ap.parse_known_args()
    policy = json.loads(Path(args.policy_file).read_text() if args.policy_file else args.policy)

    class Candidate(HistoryCandidate):
        pass

    for key, value in policy.items():
        if key.endswith("_dtype"):
            value = getattr(ttnn, value)
        if key.endswith("_fidelity"):
            value = getattr(ttnn.MathFidelity, value)
        setattr(Candidate, key, value)
    validate_optimized.OptimizedDecoder = Candidate
    sys.argv = [sys.argv[0], *rest]
    validate_optimized.main()
    output = Path(rest[rest.index("--output") + 1])
    j = json.loads(output.read_text())
    j.update(policy=policy, candidate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    j["implementation_sha256"] = hashlib.sha256(
        (Path(__file__).parent / "optimized_reference.py").read_bytes()
    ).hexdigest()
    j["implementation_source"] = "tests/optimized_reference.py plus experiment subclasses"
    j["experiment_sources"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("optimized_*.py")
    }
    output.write_text(json.dumps(j, indent=2) + "\n")


if __name__ == "__main__":
    main()
