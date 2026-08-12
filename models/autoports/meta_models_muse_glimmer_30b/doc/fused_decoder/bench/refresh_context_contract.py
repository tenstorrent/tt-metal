# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the measured fields of ``doc/context_contract.json`` from the suite run.

Every number under ``tested.*.pcc``, ``tests.*`` and
``fused_decoder_capacity_evidence.result`` is a transcription of
``doc/fused_decoder/test_results.xml`` and ``doc/fused_decoder/logs/full_test_run.log``.
Transcribing them by hand is how they went stale, so this script does it, and
``--check`` makes the staleness a non-zero exit instead of a review finding.

Usage::

    python refresh_context_contract.py            # rewrite the JSON in place
    python refresh_context_contract.py --check    # exit 1 if anything would change

Prose fields, the capability contract itself and the performance block are left
alone: this only touches values that have a single source of truth in the run.
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
JUNIT = ROOT / "doc/fused_decoder/test_results.xml"
SUITE_LOG = ROOT / "doc/fused_decoder/logs/full_test_run.log"

#: contract key -> the ``assert_pcc`` label that carries it
PREFILL_PCC = {
    "sliding@131072_last32rows": "fused prefill[sliding] full-context seq_len=131072 (last 32 rows)",
    "full@131072_last32rows": "fused prefill[full] full-context seq_len=131072 (last 32 rows)",
    "sliding@130073_last32rows": "fused prefill[sliding] full-context seq_len=130073 (last 32 rows)",
    "full@130073_last32rows": "fused prefill[full] full-context seq_len=130073 (last 32 rows)",
    "sliding@131072_interior65536_32rows": "fused prefill[sliding] full-context seq_len=131072 (interior @65536 32 rows)",
    "full@131072_interior65536_32rows": "fused prefill[full] full-context seq_len=131072 (interior @65536 32 rows)",
}
DECODE_PCC = {
    "sliding@pos131071": "fused decode[sliding] full-context pos=131071",
    "full@pos131071": "fused decode[full] full-context pos=131071",
}
SLOT_PCC = {
    "prefill_sliding": "fused multi-chunk prefill[sliding] user_id=2 seq_len=12345",
    "prefill_full": "fused multi-chunk prefill[full] user_id=2 seq_len=12345",
    "decode_sliding": "fused decode[sliding] user_id=2 pos=12345",
    "decode_full": "fused decode[full] user_id=2 pos=12345",
}


def read_pccs() -> dict[str, float]:
    """``{assert_pcc label: value}`` from the committed suite log."""
    out: dict[str, float] = {}
    for line in SUITE_LOG.read_text(errors="ignore").splitlines():
        m = re.search(r"assert_pcc:\d+ - (.+?): ([0-9.]+)$", line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def resolve(pccs: dict[str, float], label: str) -> float:
    """Exact label, else the unique label starting with it (window suffixes vary)."""
    if label in pccs:
        return pccs[label]
    hits = [v for k, v in pccs.items() if k.startswith(label)]
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

    hf = {k: v for k, v in pccs.items() if "vs unfused" not in k}
    eq = {k: v for k, v in pccs.items() if "vs unfused" in k}
    tests = contract["tests"]
    tests["total"] = total
    tests["passed"] = total - failures - skipped
    tests["asserted_pcc_checks"] = len(pccs)
    tests["hf_vs_ttnn_checks"] = len(hf)
    tests["min_pcc_vs_hf"] = round(min(hf.values()), 6)
    tests["fused_vs_unfused_checks"] = len(eq)
    tests["min_pcc_fused_vs_unfused"] = round(min(eq.values()), 6)

    full_context = [v for k, v in pccs.items() if "full-context" in k and "vs unfused" not in k]
    contract["fused_decoder_capacity_evidence"]["result"] = (
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
