# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the measured fields of ``doc/context_contract.json`` from the suite run.

Same idea as the fusing stage's version, retargeted at this stage's labels and at
the two PCC populations the optimized stage has: the released checkpoint (the bar
the precision policy is selected on) and the i.i.d.-Gaussian synthetic harness.
Transcribing these by hand is how they go stale, so this does it and ``--check``
turns staleness into a non-zero exit rather than a review finding.

Usage::

    python refresh_context_contract.py            # rewrite the JSON in place
    python refresh_context_contract.py --check    # exit 1 if anything would change

Prose fields, the capability contract itself, the byte budget and the performance
block are left alone: this only touches values with a single source of truth in
the run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
CONTRACT = ROOT / "doc/context_contract.json"
JUNIT = ROOT / "doc/optimized_decoder/test_results.xml"
SUITE_LOG = ROOT / "doc/optimized_decoder/logs/full_test_run.log"

#: contract key -> the ``assert_pcc`` label prefix that carries it
PREFILL_PCC = {
    "sliding@131072_last32rows": "optimized prefill[sliding] full-context seq_len=131072 (last 32 rows)",
    "full@131072_last32rows": "optimized prefill[full] full-context seq_len=131072 (last 32 rows)",
    "sliding@130073_last32rows": "optimized prefill[sliding] full-context seq_len=130073 (last 32 rows)",
    "full@130073_last32rows": "optimized prefill[full] full-context seq_len=130073 (last 32 rows)",
    "sliding@131072_interior65536_32rows": (
        "optimized prefill[sliding] full-context seq_len=131072 (interior @65536 32 rows)"
    ),
    "full@131072_interior65536_32rows": (
        "optimized prefill[full] full-context seq_len=131072 (interior @65536 32 rows)"
    ),
}
DECODE_PCC = {
    "sliding@pos131071": "optimized decode[sliding] full-context pos=131071",
    "full@pos131071": "optimized decode[full] full-context pos=131071",
}
SLOT_PCC = {
    "prefill_sliding": "optimized multi-chunk prefill[sliding] user_id=2 seq_len=12345",
    "prefill_full": "optimized multi-chunk prefill[full] user_id=2 seq_len=12345",
    "decode_sliding": "optimized decode[sliding] user_id=2 pos=12345",
    "decode_full": "optimized decode[full] user_id=2 pos=12345",
}
REAL_PCC = {
    "prefill_sliding@12345": "optimized real-weight prefill[sliding] seq_len=12345",
    "prefill_full@12345": "optimized real-weight prefill[full] seq_len=12345",
    "prefill_sliding@1": "optimized real-weight prefill[sliding] seq_len=1",
    "prefill_full@1": "optimized real-weight prefill[full] seq_len=1",
    "traced_decode_sliding_batch8": "optimized real-weight traced decode[sliding] batch=8",
    "traced_decode_full_batch8": "optimized real-weight traced decode[full] batch=8",
}


def read_pccs() -> dict[str, float]:
    """``{assert_pcc label: value}`` from the committed suite log."""
    out: dict[str, float] = {}
    for line in SUITE_LOG.read_text(errors="ignore").splitlines():
        match = re.search(r"assert_pcc:\d+ - (.+?): ([0-9.]+)$", line.strip())
        if match:
            out[match.group(1)] = float(match.group(2))
    return out


def resolve(pccs: dict[str, float], label: str) -> float:
    """Exact label, else the unique label starting with it (window suffixes vary)."""
    if label in pccs:
        return pccs[label]
    hits = [v for key, v in pccs.items() if key.startswith(label)]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} log lines match {label!r}; cannot refresh unambiguously")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the contract is stale")
    args = ap.parse_args()

    contract = json.loads(CONTRACT.read_text())
    before = json.dumps(contract, indent=2, sort_keys=True)
    pccs = read_pccs()
    suite = ET.parse(JUNIT).getroot()
    if suite.tag != "testsuite":
        suite = suite.find("testsuite")

    total = int(suite.get("tests"))
    failures = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))

    tested = contract["tested"]
    for key, label in PREFILL_PCC.items():
        tested["prefill"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in DECODE_PCC.items():
        tested["decode"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in SLOT_PCC.items():
        tested["nonzero_cache_slot"]["pcc"][key] = round(resolve(pccs, label), 6)
    for key, label in REAL_PCC.items():
        tested["real_weights"]["pcc"][key] = round(resolve(pccs, label), 6)

    real = {k: v for k, v in pccs.items() if "real-weight" in k}
    synthetic = {k: v for k, v in pccs.items() if "real-weight" not in k}
    tests = contract["tests"]
    tests["total"] = total
    tests["passed"] = total - failures - skipped
    tests["skipped"] = skipped
    tests["asserted_pcc_checks"] = len(pccs)
    tests["real_weight_checks"] = len(real)
    tests["min_pcc_real_weights"] = round(min(real.values()), 6)
    tests["synthetic_checks"] = len(synthetic)
    tests["min_pcc_synthetic"] = round(min(synthetic.values()), 6)

    full_context = [v for k, v in pccs.items() if "full-context" in k]
    contract["optimized_decoder_capacity_evidence"]["result"] = (
        f"{len(full_context)} full-context checks, all >= {min(full_context):.6f} "
        "(see tested.prefill.pcc / tested.decode.pcc)"
    )

    after = json.dumps(contract, indent=2, sort_keys=True)
    if args.check:
        if before != after:
            print("context_contract.json is stale against the committed suite run", file=sys.stderr)
            return 1
        print("context_contract.json matches the committed suite run")
        return 0
    CONTRACT.write_text(json.dumps(contract, indent=2) + "\n")
    print(f"refreshed {CONTRACT} from {total} tests and {len(pccs)} asserted PCC checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
